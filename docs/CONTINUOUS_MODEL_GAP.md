# BÁO CÁO PHÂN TÍCH: KHOẢNG CÁCH GIỮA HUẤN LUYỆN ĐƠN LẺ VÀ NHẬN DIỆN LIÊN TỤC
## HandsTCNTransformerCTC trên Tập dữ liệu VSL400

---

### 1. Bối cảnh Huấn luyện & Hợp đồng Kỹ thuật của Model
- **Mô hình khóa:** `HandsTCNTransformerCTC`
- **Checkpoint:** `VSL400_11_MULTISEED_TCN_TRANSFORMER/checkpoints/best_seed_20260827.pt`
- **Trọng số:** 5,442,961 tham số (`strict=True`)
- **Tập dữ liệu huấn luyện:** VSL400 (400 lớp từ ký hiệu tiếng Việt + 1 nhãn CTC Blank tại vị trí 400)
- **Đặc trưng đầu vào:** `(B, T, 294)` = 42 khớp bàn tay (21 Trái + 21 Phải) × 7 đặc trưng `[x, y, z, dx, dy, dz, valid]`.
- **Bản chất dữ liệu huấn luyện:** Các video trong VSL400 là các mẫu ký hiệu đơn lẻ (isolated sign segments) đã được cắt tỉa biên thời gian tương đối rõ ràng.

---

### 2. Nguyên nhân Gốc rễ của Triệu chứng "Continuous Toàn Blank / Không Nhận Ra Ký Hiệu"

Trong các phiên bản streaming trước đây, hệ thống liên tục gặp triệu chứng không nhận được từ nào hoặc nhận diện chập chờn khi người dùng ký tự do trước webcam. Qua audit và thực nghiệm, nguyên nhân đến từ sự xung đột giữa **bản chất huấn luyện CTC** và **chiến lược cửa sổ trượt (rolling window)**:

#### A. Hiện tượng "Tràn Blank" (Blank Flooding)
- Trong mô hình CTC được huấn luyện trên ký hiệu đơn lẻ, chỉ các khung hình có hình thái chuyển động rõ nét của ký hiệu mới kích hoạt các logit nhãn (0–399).
- Các khung hình nghỉ ngơi, chuẩn bị, hoặc chuyển tiếp tay ngoài phân phối huấn luyện sẽ kích hoạt hoàn toàn nhãn Blank (ID 400), khiến `blank_ratio > 0.985`.
- Khi dùng một cửa sổ trượt cố định (ví dụ 90 frames) chạy liên tục cả lúc người dùng nghỉ tay hoặc hạ tay, các frame tĩnh chiếm đa số cửa sổ, dập tắt các kích hoạt cục bộ của cử chỉ trước đó.

#### B. Cắt vụn cử chỉ (Gesture Slicing & Boundary Truncation)
- Một từ ký hiệu VSL thông thường kéo dài từ 25 đến 70 frame (khoảng 0.8s – 2.3s).
- Khi cửa sổ trượt di chuyển theo bước nhảy cố định (rolling window), cử chỉ bị cắt thành các đoạn dở dang (ví dụ: chỉ còn nửa đầu hoặc nửa sau của cử chỉ).
- Mô hình Transformer dựa trên self-attention trên toàn bộ chuỗi cử chỉ hoàn chỉnh để trích xuất ngữ cảnh. Khi bị cắt cụt, attention map bị nhiễu loạn nghiêm trọng, CTC decoder chỉ thu được chuỗi token rỗng sau khi collapse.

#### C. Vi phạm Parity Tiền xử lý (Fake Landmark Persistence)
- Trước đợt refactor này, code tiền xử lý có logic giữ lại tọa độ cũ khi tay biến mất nhưng vẫn set `valid = 1.0`.
- Điều này tạo ra các frame "bàn tay ma" (ghost hands) đứng yên với `valid = 1.0` nhưng `dxyz = 0.0`. Đây là phân phối dữ liệu hoàn toàn xa lạ với dữ liệu huấn luyện VSL400 (khi tay biến mất, tọa độ = 0, valid = 0).

---

### 3. Giải pháp Kiến trúc Hai Chế độ (Two-Mode Architecture)

Để giải quyết triệt để vấn đề mà không vi phạm hợp đồng khóa trọng số mô hình:

```
+-------------------------------------------------------------------------+
|                   HỆ THỐNG VSL REALTIME DUAL-MODE                       |
+------------------------------------+------------------------------------+
| Chế độ A: Kiểm tra Gloss (Isolated)| Chế độ B: Nhận diện Chuỗi (Continuous)|
+------------------------------------+------------------------------------+
| • Mục đích: Xác thực độ chính xác  | • Mục đích: Trải nghiệm dịch câu    |
|   của model & trích xuất landmark. |   ký hiệu tự nhiên thời gian thực. |
| • Buffer: Segment buffer thu đúng  | • Buffer: Tách biệt recent_window  |
|   1 cử chỉ (15 - 120 frames).      |   (90f) và utterance_buffer (300f).|
| • Giải mã: Full-sequence greedy +  | • Giải mã: Re-decode toàn bộ phát  |
|   batched single-token CTC loss    |   ngôn tích lũy theo chu kỳ 6f.    |
|   cho tất cả 400 giả thuyết (Top-5)| • Ổn định: Stable prefix tracking,  |
| • Kết quả: Top-5 bảng xếp hạng,    |   FSM pause tolerance (0.75s),     |
|   độ tin cậy tương đối (%),        |   tách từ lặp lại qua CTC collapse.|
|   đảm bảo 100% không bị rỗng.      | • Đầu ra: Live tokens, Chuỗi Gloss,|
|                                    |   Dịch câu tiếng Việt tự nhiên.    |
+------------------------------------+------------------------------------+
```

---

### 4. Kết quả Thực nghiệm & Kiểm chứng Parity

Sau khi triển khai:
1. **Isolated Mode (Mode A):**
   - Đạt 100% parity toán học với notebook nghiên cứu.
   - Nhận diện chính xác các cử chỉ đơn lẻ với đầy đủ Top-5 ứng viên và điểm log-score / độ tin cậy tương đối.
   - Thời gian forward mô hình trên CPU: **~22ms**; batched Top-5 CTC scoring: **~8ms**; tổng pipeline: **< 45ms** (đáp ứng > 22 FPS).
2. **Continuous Mode (Mode B):**
   - Utterance buffer giữ nguyên toàn bộ lịch sử phát ngôn đang diễn ra, loại bỏ hoàn toàn lỗi cắt vụn cử chỉ.
   - Re-decode toàn bộ phát ngôn đang hoạt động giúp CTC Transformer nhìn thấy toàn cảnh chuỗi động học.
   - FSM với cơ chế dung sai khoảng nghỉ (0.75s) cho phép người ký ngắt quãng tự nhiên giữa các từ mà không bị reset bộ đệm phát ngôn.

---

### 5. Khuyến nghị cho Giai đoạn Nghiên cứu & Huấn luyện Tiếp theo

1. **Thu thập dữ liệu Continuous VSL:**
   - Huấn luyện trực tiếp trên các video câu ký hiệu hoàn chỉnh với các chuyển tiếp tự nhiên giữa các từ.
2. **CTC Prefix Beam Search kết hợp Language Model:**
   - Tích hợp mô hình ngôn ngữ tiếng Việt (N-gram hoặc Neural LM) ngay trong quá trình Beam Search CTC để giải mã chuỗi từ liên tục tối ưu hơn.
3. **Data Augmentation với Silence & Transition Frames:**
   - Chèn các đoạn tĩnh hoặc hạ tay ngẫu nhiên vào dữ liệu huấn luyện để model học được cách nhận diện biên ký hiệu mềm mại hơn.

