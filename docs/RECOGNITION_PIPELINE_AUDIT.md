# BÁO CÁO AUDIT TOÀN BỘ PIPELINE NHẬN DIỆN VSL REALTIME
**Tài liệu kỹ thuật Phase 1 theo yêu cầu Master Prompt**
**Ngày thực hiện:** 2026-09-22

---

## 1. Tổng Quan Kiến Trúc & Hiện Trạng Hệ Thống

Dự án sử dụng checkpoint mô hình đã khóa:
- **Model Architecture:** `HandsTCNTransformerCTC` (5,442,961 trainable parameters)
- **Checkpoint:** `VSL400_System_Assets/VSL400_11_MULTISEED_TCN_TRANSFORMER/checkpoints/best_seed_20260827.pt`
- **Input Contract:** `[B, T, 294]` với 42 khớp tay (0–20: Left Hand, 21–41: Right Hand) $\times$ 7 đặc trưng `[x, y, z, dx, dy, dz, valid]`.
- **Output Contract:** `[B, T, 401]` gồm 400 VSL Glosses (0–399) và 1 CTC Blank token (index 400).
- **Trọng số Blank Bias Init:** -1.5.

Qua phân tích metadata từ checkpoint nghiên cứu (`best_meta`), mô hình đạt:
- WER trên tập kiểm thử nội bộ: `~15.03% - 15.96%`.
- Tỷ lệ CTC blank trung bình: `98.50% - 98.56%`.
- Trung bình số token phát ra trên mỗi utterance: `~3.01` tokens.
- Tỷ lệ chuỗi không rỗng (`nonempty_ratio`): `99.96% - 99.98%`.

Điều này chứng minh: **Mô hình nghiên cứu hoàn toàn có khả năng nhận diện chính xác chuỗi Gloss khi được cung cấp đúng định dạng dữ liệu chuỗi.**

Tuy nhiên, trong môi trường production realtime, hệ thống gặp hiện tượng nghiêm trọng: **Continuous Mode không nhận ra bất kỳ Gloss nào (hoặc toàn rỗng).**

---

## 2. Phân Tích Đường Đi Hiện Tại (Current Codepaths)

### 2.1. Đường dẫn "Tương đương Isolated" (Current Isolated-Equivalent Path)
Hiện tại, hệ thống **CHƯA CÓ** một mode độc lập cho Isolated / Single-Gloss recognition. Cả hai trường hợp ký một từ hay ký nhiều từ đều bị ép chạy chung qua một pipeline duy nhất:
1. Webcam gửi frame JPEG qua WebSocket `/ws/recognition`.
2. `RealtimeSession.process_frame` nhận frame.
3. Trích xuất landmark bằng MediaPipe.
4. Preprocessing tạo vector 294 chiều.
5. Đẩy vào `RealtimeSequenceBuffer` có độ dài cố định `max_frames=90`.
6. Cứ mỗi `inference_stride=2` frame, buffer trích xuất tensor `(1, T, 294)` (với $T \le 90$).
7. Model forward qua `HandsTCNTransformerCTC`.
8. Greedy decode qua `CTCDecoder.decode_greedy`.
9. `EndpointFSM` cập nhật trạng thái.

### 2.2. Đường dẫn Continuous Hiện Tại (Current Continuous Path)
Continuous mode hiện tại phụ thuộc vào việc sliding window 90 frames trượt liên tục trên dòng thời gian. Khi người dùng ký:
- Các frame nghỉ (zeros) đẩy dần các frame ký hiệu ra khỏi buffer.
- Mỗi lần suy luận, model chỉ nhìn thấy một cửa sổ cục bộ 90 frame (tương đương ~3 giây ở 30 FPS).
- Decoder chạy độc lập trên từng cửa sổ trượt và cố gắng gán `live_gloss = last_live_gloss`.
- Khi FSM phát hiện khoảng lặng (`POSSIBLE_END` -> `END_OF_UTTERANCE`), hệ thống lấy `fsm.last_final_gloss` (vốn chỉ là snapshot cuối cùng của `live_gloss` lúc người dùng vừa hạ tay).

---

## 3. Các Mô-đun Dùng Chung (Shared Modules)

1. `backend/app/ml/model.py`:
   - Lớp `HandsTCNTransformerCTC`: 4 TCN blocks với dilation $1, 2, 4, 8$, kernel size 5, và 4 Transformer layers.
   - Receptive field của TCN: $1 + 2 \times (5-1) \times (1 + 2 + 4 + 8) = 121$ frames.
2. `backend/app/ml/model_loader.py`:
   - Singleton loader, nạp checkpoint với `strict=True`.
3. `backend/app/ml/ctc_decoder.py`:
   - Greedy decoder: `argmax -> collapse consecutive repeats -> remove blank (400) -> map to Gloss`.
4. `backend/app/ml/landmark_extractor.py`:
   - Trích xuất bàn tay và tư thế.
5. `backend/app/ml/preprocessing.py`:
   - Chuẩn hóa vai (body-centric shoulder normalization) và tính delta vận tốc $dxyz$.

---

## 4. Kiểm Toán Chi Tiết Các Lớp Lỗi (Forensic Layer-by-Layer Audit)

### Nghi Vấn 1: Vi phạm Hard Preprocessing Parity Contract (Tạo Landmark Giả)
- **Vị trí:** `backend/app/ml/preprocessing.py`, các dòng 138–160.
- **Vấn đề:** Trong phiên bản trước, một cơ chế "persistence hold" được thêm vào:
  ```python
  if eff_left is None and self.prev_left_lms is not None and self.left_lost_count < self.max_lost_frames:
      eff_left = self.prev_left_lms
  ```
  Khi bàn tay bị mất dấu ở frame $t$, code đã tự động sao chép lại tọa độ của frame $t-1$ và đánh dấu `valid = 1.0`!
- **Hậu quả:** Điều này vi phạm nghiêm trọng **Hard Contract** mục III:
  *"Missing landmark: coordinates = 0, valid = 0. Không được biến missing point thành một landmark giả hợp lệ."*
  Trong tập huấn luyện VSL400, khi không thấy tay, toàn bộ 21 khớp tay nhận giá trị 0 và cột `valid` bằng 0. Việc tự chế ra landmark giả có `valid=1` làm phân phối đầu vào bị lệch so với dữ liệu huấn luyện (Out-of-Distribution), khiến mạng Transformer chú ý vào các điểm giả và dự đoán sai.

### Nghi Vấn 2: Xung Đột Ngữ Nghĩa Bộ Đệm (Rolling Window vs. Utterance Buffer)
- **Vị trí:** `backend/app/services/realtime_session.py`, dòng 112 và `backend/app/ml/sequence_buffer.py`.
- **Vấn đề:** Hệ thống chỉ có DUY NHẤT một `RealtimeSequenceBuffer` với `max_frames=90`.
  - Khi người dùng bắt đầu ký một câu có 2–4 từ (thời lượng từ 4 đến 7 giây, tương đương 120–210 frames), cửa sổ 90 frame bị tràn.
  - Các frame của từ thứ nhất (Gloss 1) bị đẩy ra khỏi buffer (`popleft`) trước khi từ thứ ba (Gloss 3) hoàn thành!
  - Cắt vụn chuỗi thời gian khiến mô hình TCN-Transformer mất ngữ cảnh đầu câu.
  - Khi người dùng hạ tay xuống nghỉ, buffer 90 frame nhanh chóng bị lấp đầy bởi các frame rỗng (resting frames). Model suy luận trên 90 frame rỗng -> ra 100% CTC blank -> `live_gloss` trở thành rỗng `[]`!
- **Hậu quả:** Lúc FSM kích hoạt `END_OF_UTTERANCE`, `live_gloss` đã bị ghi đè bởi rỗng, dẫn đến `final_gloss` luôn rỗng!

### Nghi Vấn 3: Lỗi Greedy CTC Trên Sliding Window (Streaming CTC Semantics Mismatch)
- **Vị trí:** `backend/app/services/realtime_session.py`, dòng 128–134.
- **Vấn đề:** Thay vì lưu toàn bộ chuỗi ký hiệu từ lúc bắt đầu phát ngôn (`utterance_buffer`) và giải mã trên toàn bộ chuỗi cử chỉ thực, hệ thống lại decode từng lát cắt 90 frame rời rạc.
  - TCN có receptive field 121 frames. Một cửa sổ 90 frame thậm chí còn nhỏ hơn receptive field tối đa của 4 tầng TCN!
  - Khi một từ bị cắt ngang ở biên cửa sổ, các đặc trưng chuyển động bị cụt, mạng không kích hoạt đủ xác suất vượt ngưỡng blank, khiến toàn bộ output là CTC Blank (index 400).

### Nghi Vấn 4: Thiếu Chế Độ Độc Lập Cho Kiểm Tra Từ Đơn (Isolated Mode)
- **Vị trí:** Cả backend và frontend.
- **Vấn đề:** Không có cách nào để cô lập kiểm tra xem model có nhận diện được một từ đơn lẻ (như "XIN_CHÀO", "CẢM_ƠN", "TÔI") hay không. Mọi thử nghiệm đều bị phụ thuộc vào FSM continuous, gây nhiễu và không thể xác định lỗi nằm ở MediaPipe, Preprocessing, Model, hay FSM.

### Nghi Vấn 5: Đồng Bộ Hóa Frontend / Backend WebSocket Payload
- **Vị trí:** `backend/app/api/realtime.py` và `frontend/src/hooks/useRecognitionSocket.ts`.
- **Vấn đề:** WebSocket chỉ phục vụ một loại session chung chung `/ws/recognition`. Không có tham số `mode=isolated` hay `mode=continuous`. Client không thể yêu cầu session xử lý theo đúng ngữ cảnh của từng tác vụ.

---

## 5. Kế Hoạch Sửa Đổi Kỹ Thuật (Remediation Architecture)

```
                       ┌───────────────────────────────────────────────┐
                       │                  SHARED CORE                  │
                       │ - LandmarkExtractor (Holistic + Wrist Dist)   │
                       │ - VSLPreprocessor (Strict Parity: 0 if miss)  │
                       │ - ModelRuntime (HandsTCNTransformerCTC)       │
                       │ - CTCDecoder (Greedy + Top-5 Single Hypotheses│
                       │ - LabelMap (400 VSL classes)                  │
                       └───────────────────────┬───────────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │                                               │
                       ▼                                               ▼
         ┌───────────────────────────┐                   ┌───────────────────────────┐
         │          MODE A           │                   │          MODE B           │
         │ IsolatedRecognitionSession│                   │ContinuousRecognitionSession│
         │ - Segment Recorder        │                   │ - Utterance Buffer (Full) │
         │ - Complete Sign Inference │                   │ - Rolling Diagnostic Win  │
         │ - Top-5 Diagnostic Probs  │                   │ - Strategy 1: Re-decode   │
         │ - Instant Single Gloss    │                   │ - Endpoint FSM Refined    │
         └───────────────────────────┘                   └───────────────────────────┘
                       │                                               │
                       ▼                                               ▼
         ┌───────────────────────────┐                   ┌───────────────────────────┐
         │ /ws/recognition?mode=iso  │                   │/ws/recognition?mode=cont  │
         └───────────────────────────┘                   └───────────────────────────┘
```

1. **Khôi phục 100% Preprocessing Parity**:
   - Loại bỏ hoàn toàn persistence hold giả mạo. Landmark missing phải strictly là `coords = 0, valid = 0, dxyz = 0`.
2. **Tách Thành 2 Session Độc Lập**:
   - `IsolatedRecognitionSession`: Quản lý chu trình thu thập cử chỉ đơn (`IDLE -> RECORDING -> COLLECTED -> INFER -> RESULT`), chạy suy luận trên toàn bộ segment, xuất Predicted Gloss + Top-5 hypotheses bằng batched CTC loss.
   - `ContinuousRecognitionSession`: Tách biệt hoàn toàn `recent_window` (dành cho metric realtime) và `utterance_buffer` (lưu toàn bộ chuỗi từ khi bắt đầu ký đến khi kết thúc). Sử dụng **Strategy 1 (Full Active Utterance Re-Decode)** để đảm bảo ngữ cảnh thời gian toàn vẹn.
3. **Cung Cấp 2 WebSocket Endpoints**:
   - `/ws/recognition?mode=isolated` (hoặc `/ws/recognition/isolated`).
   - `/ws/recognition?mode=continuous` (hoặc `/ws/recognition/continuous`).
4. **Nâng Cấp Giao Diện Frontend 2 Tab Riêng Biệt**:
   - Tab 1: "Kiểm tra Gloss" (Nhận diện từ đơn + Top-5 bảng xếp hạng).
   - Tab 2: "Kiểm tra Continuous" (Nhận diện chuỗi ký hiệu + FSM + Câu hoàn chỉnh).
   - Tích hợp Developer Diagnostics Panel chi tiết cho từng mode.

