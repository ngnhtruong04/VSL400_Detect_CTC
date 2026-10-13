# Hệ Thống Nhận Diện Ngôn Ngữ Ký Hiệu Việt Nam (VSL) Realtime
## Tích hợp Locked TCN–Transformer–CTC Model từ VSL400

---

## 1. TỔNG QUAN (OVERVIEW)

Dự án này là hệ thống phần mềm hoàn chỉnh cho bài toán **nhận diện ngôn ngữ ký hiệu Việt Nam theo thời gian thực (Realtime Vietnamese Sign Language Recognition)**. Hệ thống chuyển giao mô hình nghiên cứu đã được khóa kiến trúc (**HandsTCNTransformerCTC**) thành sản phẩm ứng dụng thực tế hỗ trợ người điếc và cộng đồng giao tiếp dễ dàng thông qua webcam tiêu chuẩn.

### Khả năng cốt lõi:
- **Nhận diện liên tục qua Webcam**: Xử lý luồng hình ảnh trực tiếp từ người ký mới (unseen signer).
- **Trích xuất Landmark chuẩn xác**: Trích xuất Pose và 2 bàn tay thông qua MediaPipe.
- **Preprocessing Parity**: Đồng bộ 100% chuẩn tiền xử lý lúc huấn luyện (chuẩn hóa tọa độ theo tâm vai, vận tốc vi phân thời gian $dxyz$, và valid mask).
- **Mô hình TCN–Transformer–CTC**: Kiến trúc nén không gian - thời gian 4 tầng TCN + 4 tầng Transformer + CTC Classifier với 5,442,961 tham số đã huấn luyện.
- **CTC Greedy Decoding**: Giải mã tức thời chuỗi logits thành chuỗi từ vựng Gloss tiếng Việt (400 lớp từ đơn).
- **Endpoint FSM**: Máy trạng thái thông minh phát hiện điểm kết thúc phát ngôn, có khả năng chống ngắt câu sớm khi người ký tạm dừng ngắn giữa các từ.
- **Tầng ngôn ngữ tùy chọn (Optional LLM)**: Chuyển đổi chuỗi Gloss thành câu tiếng Việt tự nhiên, đúng ngữ pháp.
- **Giao diện Web hiện đại**: Thiết kế tối ưu cho trải nghiệm người dùng, hỗ trợ chế độ Nhà phát triển (Developer Diagnostics Mode).

---

## 2. NGỮ CẢNH NGHIÊN CỨU (RESEARCH CONTEXT)

Hệ thống được xây dựng trên kết quả nghiên cứu khoa học thuộc đề tài khóa luận tốt nghiệp với tập dữ liệu **VSL400**:
- **Tập dữ liệu**: 24,618 mẫu video từ 26 người ký (signers), góc quay chính diện (`front_view`), phân chia tập dữ liệu tách biệt người ký (`signer-disjoint split`).
- **Từ vựng**: 400 lớp Gloss tiếng Việt tiêu chuẩn.
- **Lựa chọn kiến trúc (Architecture Lock)**: Qua so sánh thực nghiệm đa mô hình (BiLSTM-CTC, Region-Conformer-CTC, TCN-Transformer-CTC), mô hình **TCN-Transformer-CTC** được lựa chọn chính thức nhờ đạt độ chính xác cao nhất (Validation WER ~ 0.1503 - 0.1595) và độ trễ xử lý tối ưu nhất (~5.8 ms trên GPU, ~9.5 ms trên CPU).
- **Checkpoint chính thức**: `best_seed_20260827.pt` (Multi-seed evaluation seed 20260827).

---

## 3. KIẾN TRÚC TOÀN HỆ THỐNG (SYSTEM ARCHITECTURE)

```
Webcam (Browser)
       │ (1280x720 @ 30 FPS)
       ▼
React Frontend
       │ (WebSocket /ws/recognition - JPEG Blob binary)
       ▼
FastAPI Server (Session Isolation)
       │
       ▼
MediaPipe Landmark Extractor (Pose + Hands)
       │
       ▼
Body-Centric Preprocessing & Delta Velocity
       │  (T, 42 hand joints, 7 features: [x, y, z, dx, dy, dz, valid])
       ▼
Flattened Sequence Representation: (T, 294)
       │
       ▼
Realtime Rolling Sequence Buffer (Sliding Window T frames)
       │
       ▼
HandsTCNTransformerCTC Model (Inference)
       │
       ▼
CTC Greedy Decoder
       │
       ├──► Live Gloss (Phản hồi tức thì lên UI)
       │
       ▼
Endpoint Finite State Machine (IDLE -> STARTING -> SIGNING -> POSSIBLE_END -> END_OF_UTTERANCE)
       │
       ▼
Final Gloss Sequence (Chuỗi chốt hoàn chỉnh)
       │
       ▼
Optional LanguageService (Gemini / Ollama / Rule-based)
       │
       ▼
Vietnamese Natural Sentence
```

---

## 4. CHI TIẾT MÔ HÌNH VÀ CHECKPOINT (MODEL & ASSETS)

### Cấu hình kiến trúc:
- **Tên lớp mô hình**: `HandsTCNTransformerCTC`
- **Kích thước đầu vào (Input Dim)**: `294` ($42 \text{ joints} \times 7 \text{ features}$)
- **Kích thước đặc trưng ẩn (Model Dim)**: `256`
- **Số khối TCN**: 4 khối (Mỗi khối: 2 tầng Conv1d $k=5$, dilation $d=2^i$, BatchNorm1d, ReLU, Dropout + Residual + LayerNorm)
- **Số tầng Transformer Encoder**: 4 tầng ($d_{\text{model}}=256$, $\text{nhead}=8$, $\text{dim\_feedforward}=768$, batch_first=True) + LayerNorm cuối
- **Bộ phân loại (Classifier)**: Linear($256 \to 401$)
- **Số lớp phân loại**: `401` (400 Glosses + 1 CTC Blank tại chỉ số `400`)
- **Khởi tạo bias CTC Blank**: `blank_bias_init = -1.5` (trọng số huấn luyện tại bias[400] là `-1.0256`)
- **Tổng số tham số có thể huấn luyện (Trainable Parameters)**: `5,442,961` tham số (khớp 100% với báo cáo `architecture_lock.json`)

### Checkpoint chính thức:
- Đường dẫn: `VSL400_System_Assets/VSL400_11_MULTISEED_TCN_TRANSFORMER/checkpoints/best_seed_20260827.pt`
- Xác thực: `load_state_dict(strict=True)` khớp toàn bộ 120/120 keys, 0 missing keys, 0 unexpected keys.

---

## 5. CHUẨN TIỀN XỬ LÝ (PREPROCESSING PARITY CONTRACT)

Để đảm bảo kết quả nhận diện chính xác như môi trường huấn luyện, luồng tiền xử lý tuân thủ tuyệt đối các nguyên tắc:
1. **Gốc tọa độ cơ thể (Body-Centric Center)**:
   $$\text{Center} = \frac{\text{Pose}_{11} + \text{Pose}_{12}}{2}$$
   (Tọa độ trung điểm giữa 2 khớp vai trái Pose 11 và vai phải Pose 12).
2. **Khoảng cách co dãn (Scale Normalization)**:
   $$\text{Dist} = \|\text{Pose}_{11} - \text{Pose}_{12}\|_2$$
   $$\text{Scale} = \max(\text{Dist} \times 1.0, 0.0001)$$
3. **Chuẩn hóa tọa độ**:
   $$\text{Coords}_{\text{norm}} = \text{clip}\left(\frac{\text{Coords} - \text{Center}}{\text{Scale}}, -6.0, 6.0\right)$$
4. **Trích xuất 42 khớp bàn tay (Hands-Only)**:
   - Khớp 0 đến 20: Bàn tay trái (21 landmarks)
   - Khớp 21 đến 41: Bàn tay phải (21 landmarks)
5. **Vận tốc vi phân thời gian ($dxyz$)**:
   $$dx[t] = x[t] - x[t-1],\quad dy[t] = y[t] - y[t-1],\quad dz[t] = z[t] - z[t-1]$$
   *(Tại frame đầu tiên hoặc khi bàn tay bị khuất/invalid: $dxyz = 0.0$)*.
6. **Mặt nạ hợp lệ (Valid Mask)**:
   - Feature thứ 7 mang giá trị $1.0$ nếu khớp được phát hiện, và $0.0$ nếu bị khuất.
7. **Vector đặc trưng**: Mỗi frame được làm phẳng thành vector $42 \times 7 = 294$ chiều kiểu `float32`.

---

## 6. MÁY TRẠNG THÁI KẾT THÚC PHÁT NGÔN (ENDPOINT FSM)

Để tránh hiện tượng ngắt câu sai khi người ký tạm dừng chuyển đổi giữa các ký hiệu:
- **`IDLE`**: Người ký chưa thực hiện động tác.
- **`STARTING`**: Phát hiện chuyển động bàn tay vượt ngưỡng `motion_threshold` (0.015).
- **`SIGNING`**: Ký hiệu kéo dài trên 400ms. Hệ thống liên tục phát Live Gloss lên giao diện.
- **`POSSIBLE_END`**: Động tác tạm dừng. Hệ thống chờ trong khoảng đệm 600ms – 1200ms.
  - *Nếu người ký chuyển động tiếp*: FSM quay lại ngay trạng thái `SIGNING`.
  - *Nếu người ký dừng hẳn quá 1200ms*: Chuyển sang `END_OF_UTTERANCE`.
- **`END_OF_UTTERANCE`**: Chốt chuỗi Final Gloss hoàn chỉnh, gửi sang tầng xử lý ngôn ngữ tự nhiên và chuyển về `COOLDOWN` (500ms) trước khi sẵn sàng cho câu tiếp theo.

---

## 7. CÀI ĐẶT VÀ KHỞI ĐỘNG (QUICK START)

### Yêu cầu tiên quyết:
- Hệ điều hành: Windows 10/11 64-bit.
- Python: Phiên bản 3.11 (đã có sẵn môi trường `.venv` trong dự án).
- Node.js: Phiên bản 18+ (Node v24 đã cài trên hệ thống).
- Webcam hoặc camera tích hợp.

### Khởi động 1-Click trên Windows:
Chạy file script sau từ thư mục dự án:
```cmd
start_system.bat
```
Script sẽ tự động:
1. Mở cửa sổ Backend server tại `http://localhost:8000`
2. Mở cửa sổ Frontend web tại `http://localhost:5173`
3. Tự động bật trình duyệt web tới giao diện hệ thống.

---

### Khởi động thủ công từng phần:

#### 1. Khởi động Backend Server:
```cmd
cd C:\Users\Admin\Desktop\UniSchl\KLTN\VSL_System
.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```
- API Health Check: `http://localhost:8000/health`
- API Model Info: `http://localhost:8000/model/info`
- Swagger UI tài liệu API: `http://localhost:8000/docs`

#### 2. Khởi động Frontend Web:
```cmd
cd C:\Users\Admin\Desktop\UniSchl\KLTN\VSL_System\frontend
npm run dev
```
Truy cập giao diện tại: `http://localhost:5173`

---

## 8. CHẠY KIỂM THỬ VÀ BENCHMARK (TESTING & BENCHMARK)

### 1. Chạy toàn bộ Unit Tests:
```cmd
.venv\Scripts\python.exe -m pytest backend\tests -v
```
Kiểm thử tự động 8 kịch bản:
- `test_model_architecture_parameter_count`: Xác nhận 5,442,961 tham số.
- `test_checkpoint_strict_load`: Xác nhận nạp khớp 100% trọng số checkpoint.
- `test_model_dummy_forward_contract`: Xác nhận shape đầu vào (B, T, 294) và đầu ra (B, T, 401).
- `test_preprocessor_shape_and_dtype`: Xác nhận chuẩn hóa vai, dxyz và valid mask.
- `test_missing_hand_handling`: Xác nhận xử lý khi khuất 1 tay.
- `test_ctc_decoder_collapse_and_blank_removal`: Xác nhận thuật toán gộp lặp và loại bỏ blank token 400.
- `test_ctc_decoder_empty_and_all_blank`: Xác nhận xử lý chuỗi rỗng.
- `test_fsm_signing_lifecycle_and_pause_resilience`: Xác nhận khả năng chống ngắt câu sớm khi dừng ngắn.

### 2. Chạy kiểm thử Offline Inference:
```cmd
.venv\Scripts\python.exe scripts\offline_inference.py
```

### 3. Đo lường Benchmark hiệu năng:
```cmd
.venv\Scripts\python.exe scripts\benchmark_inference.py
```
**Kết quả đo lường thành phần:**
| Thành phần | Độ trễ trung bình | Đơn vị |
| :--- | :--- | :--- |
| **MediaPipe Landmark Extraction** | 25.43 ms | Khung hình |
| **Preprocessing & Delta Velocity** | 0.354 ms | Khung hình |
| **TCN-Transformer Model Forward (T=60)** | 9.55 ms | Chuỗi 60 frames |
| **CTC Greedy Decoder** | 0.091 ms | Chuỗi |
| **Tổng độ trễ Per-Frame** | **~35.69 ms** | **~28 - 30+ FPS Realtime** |

---

## 9. CHẾ ĐỘ NHÀ PHÁT TRIỂN (DEVELOPER MODE)

Nhấn vào biểu tượng Code (`</>`) trên thanh Header để bật bảng chẩn đoán kỹ thuật trực tiếp:
- Chi tiết thông số checkpoint và số lượng tham số mô hình.
- Biểu đồ thời gian xử lý chi tiết (MediaPipe, Preprocessing, Model, CTC).
- Chỉ số chuyển động (Motion Energy), Tỷ lệ Blank (Blank Ratio), Số frames trong Buffer.
- Danh sách Raw Token IDs và Collapsed Token IDs sinh ra từ mạng nơ-ron.

---

## 10. TÍNH NĂNG BẢO MẬT VÀ QUYỀN RIÊNG TƯ (PRIVACY)

- **Xử lý hoàn toàn cục bộ**: Toàn bộ luồng nhận diện video, trích xuất landmarks và suy luận mô hình diễn ra trực tiếp trên máy chủ nội bộ.
- **Không lưu trữ hình ảnh**: Hệ thống không lưu trữ video, không lưu ảnh khuôn mặt, không nhận diện danh tính người dùng.
- **Tầng LLM tùy chọn**: Tầng dịch tự nhiên chỉ gửi danh sách tên Gloss (không gửi hình ảnh/video), có thể tắt hoàn toàn trong mục Cài đặt.

---

## 11. GIỚI HẠN KHOA HỌC ĐÃ BIẾT (KNOWN LIMITATIONS)

1. **Phạm vi từ vựng**: Giới hạn trong 400 từ đơn (Gloss) thuộc tập dữ liệu VSL400.
2. **Dữ liệu huấn luyện**: Dữ liệu liên tục (continuous) của mô hình gốc được tạo dựng từ việc ghép nối có kiểm soát các đoạn clip từ đơn, chưa phải tập câu ký hiệu tự nhiên ngoài đời thực.
3. **Người ký mới (Unseen Signer)**: Hệ thống sử dụng chuẩn hóa hình học theo tỉ lệ vai giúp người ký mới có thể sử dụng được; tuy nhiên độ chính xác thực tế phụ thuộc vào góc quay, khoảng cách camera và điều kiện ánh sáng môi trường.

