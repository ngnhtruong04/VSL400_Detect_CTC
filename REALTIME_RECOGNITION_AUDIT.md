# BÁO CÁO KIỂM THỬ VÀ PHÂN TÍCH LỖI HỆ THỐNG NHẬN DIỆN NGÔN NGỮ KÝ HIỆU VIỆT NAM REALTIME (VSL400)

- **Ngày thực hiện:** 2026-11-24
- **Phạm vi:** Kiểm tra, chạy đối chứng, phân tích nguyên nhân gốc và đề xuất giải pháp (CHỈ kiểm định — KHÔNG chỉnh sửa mã nguồn chính thức, KHÔNG thay đổi checkpoint/kiến trúc mô hình).
- **Tệp dữ liệu và biểu đồ đính kèm:**
  - Bảng kết quả từng video: [`audit_diagnostics/test_results.csv`](audit_diagnostics/test_results.csv)
  - Bảng so sánh tổng hợp Offline vs Realtime: [`audit_diagnostics/offline_vs_realtime_comparison.csv`](audit_diagnostics/offline_vs_realtime_comparison.csv)
  - Bảng phân tích nguyên nhân gốc: [`audit_diagnostics/root_cause_analysis.csv`](audit_diagnostics/root_cause_analysis.csv)
  - Kế hoạch sửa lỗi chi tiết: [`audit_diagnostics/proposed_fix_plan.md`](audit_diagnostics/proposed_fix_plan.md)
  - Dữ liệu log kiểm thử gốc: [`audit_diagnostics/full_audit_results.json`](audit_diagnostics/full_audit_results.json), [`audit_diagnostics/component_probe_results.json`](audit_diagnostics/component_probe_results.json), [`audit_diagnostics/deep_dive_evidence.json`](audit_diagnostics/deep_dive_evidence.json)
  - Biểu đồ kiểm thử (300 DPI): [`audit_diagnostics/figures/`](audit_diagnostics/figures/)

---

## 1. Tóm tắt hiện trạng hệ thống

Hệ thống `VSL400_Detect_CTC` được triển khai theo kiến trúc Dual-Mode (FastAPI + WebSocket ở backend, React + Vite ở frontend) tích hợp mô hình nghiên cứu khóa **`HandsTCNTransformerCTC`** (`best_seed_20260827.pt`, 5.442.961 tham số, 400 nhãn gloss + 1 CTC blank tại ID `400`).

Qua kiểm thử thực nghiệm trên toàn bộ **8 video mẫu ký hiệu đơn thực tế** được trích xuất từ notebook kiểm chứng (`Document_for_checking/vsl400-notebook-21-video-mau-ky-hieu-don.ipynb`), chúng tôi xác nhận kết luận cốt lõi:

1. **Mô hình học sâu và checkpoint hoạt động hoàn toàn chính xác:**
   - Khi đối chiếu định nghĩa mô hình trong Notebook 21 với `backend/app/ml/model.py` trên cùng tensor đầu vào, sai lệch logit tuyệt đối trung bình là **`9.69 × 10⁻⁷`** (tối đa `1.14 × 10⁻⁵`, tương đương sai số float32).
   - Khi chạy **Offline Inference (Test A)** bằng chính `LandmarkExtractor` + `VSLPreprocessor` + `HandsTCNTransformerCTC` của hệ thống trên 8 video gốc (`640×640 @ 25 FPS`), hệ thống đạt **8/8 (100.0%) CTC Greedy Exact Match** và **8/8 Top-1** với độ tin cậy `0.95–1.00`.
2. **Hệ thống suy sụp nghiêm trọng khi tái hiện đúng điều kiện triển khai Realtime Webcam (Test B & C):**
   - Trong điều kiện mô phỏng đúng khung hình webcam chuẩn của frontend (`4:3` tại `640×480` từ `useWebcam.ts` + nén xuống `384×288` JPEG quality `0.6` từ `App.tsx`/`useRecognitionSocket.ts`), tỷ lệ nhận diện đúng sụp đổ từ **8/8 (100.0%)** xuống chỉ còn **1/8 (12.5%)** ở cả **Mode A (Isolated Auto)** lẫn **Mode B (Continuous)**.
   - Trên webcam tỷ lệ `16:9` (như cấu hình `1280×720` trong `runtime.yaml` hoặc đa số laptop HD hiện nay, bị nén xuống `384×216` JPEG `0.6`), hệ thống sụp đổ hoàn toàn xuống **0/8 (0.0%)** ở cả Mode A và Mode B.
3. **Nguyên nhân gốc nằm 100% ở pipeline triển khai (Capture/Preprocessing/Endpoint FSM), KHÔNG phải lỗi mô hình học sâu:**
   - **3 lỗi Critical** cộng hưởng trực tiếp: (i) **Biến dạng tỷ lệ khung hình** 4:3 và 16:9 so với tập huấn luyện 1:1 vuông (`ERR-01`), (ii) **Ép giảm độ phân giải 384px + JPEG q=0.6** làm mất tới 30–38% landmark bàn tay và khiến MediaPipe chạy *chậm hơn* do phải chạy lại Palm Detector (`ERR-02`), và (iii) **Kẹt vĩnh viễn trạng thái `SIGNING`/`RECORDING`** vì `motion_energy` dùng nhiễu 3D thô (gồm dao động chiều sâu `dz` và dao động ROI của MediaPipe trên tay đứng yên lên tới `0.032–0.056`) vượt xa ngưỡng `0.008`, trong khi `EndpointFSM` tính `smoothed_motion` nhưng *không sử dụng* (`ERR-03`).
   - Kèm theo **2 lỗi High** về kích hoạt giả `STARTING → SIGNING` tạo từ ảo **`Cái quần`** trên 3/4 video đứng yên (`ERR-04`) và ngưỡng `check_signing_space` (`shoulder_y + 0.35`) luôn đánh dấu tay hạ ngang hông/đùi là `in_signing_space=True` (`ERR-05`).
4. **Chứng minh khả năng khắc phục không cần huấn luyện lại mô hình:**
   - Chỉ riêng việc phục hồi hình học đẳng hướng (`y * H/W`) và nâng độ phân giải gửi lên `480–640px` JPEG `0.75–0.80` đã khôi phục độ chính xác Offline trên webcam `4:3` từ **4/8 (50.0%) → 8/8 (100.0%)** và trên `16:9` từ **0/8 (0.0%) → 7/8 (87.5%)**, **đồng thời giảm thời gian MediaPipe CPU từ 30.2 ms xuống 25.8–26.2 ms/frame**.

---

## 2. Danh sách tài nguyên đã kiểm tra

| STT | Tài nguyên / Đường dẫn | Nội dung & Kết quả xác minh |
|---|---|---|
| 1 | `Document_for_checking/vsl400-notebook-21-video-mau-ky-hieu-don.ipynb` | 21 cells (3.96 MB); chứa catalog 144 bản ghi (37 gloss) và **8 video MP4 H.264 nhúng thực tế** (`640×640 @ 25 FPS`) kèm kết quả suy luận gốc (`8/8 Exact`, `blank_ratio 98.11–98.82%`). |
| 2 | `VSL400_System_Assets/VSL400_11_MULTISEED_TCN_TRANSFORMER/checkpoints/best_seed_20260827.pt` | Khớp 100% checkpoint khóa trong nghiên cứu (`seed=20260827`, `global_epoch=26`, `val_wer=0.1503`, `val_blank_ratio=0.9856`, `test_used=False`, 5.442.961 tham số, `strict=True`). |
| 3 | `VSL400_System_Assets/VSL400_COMBINE_5GROUPS_Shared_Keypoints/label_map.json` | 400 lớp gloss (ID `0..399`), CTC blank ID `400` (`num_classes=401`). |
| 4 | `VSL400_System_Assets/VSL400_COMBINE_5GROUPS_Shared_Keypoints/extractor_config.json` | Ghi nhận huấn luyện dùng `MediaPipe Tasks: Pose + Hands + Face`, 108 joints (`34:76` là 42 khớp tay), `pose_visibility_threshold=0.4`, `pose_presence_threshold=0.2`, `min_scale=0.0001`, `clip_value=6.0`. |
| 5 | `VSL400_System_Assets/VSL400_10_MODEL_COMPARISON_LOCK/reports/architecture_lock.json` & `VSL400_08.../final_report.json` | Xác nhận kiến trúc `HandsTCNTransformerCTC`, input `(T, 42, 7)` flatten `294`, `model_dim=256`, `tcn_layers=4`, `tcn_kernel_size=5`, `nhead=8`, `transformer_layers=4`, `ff_dim=768`, `blank_bias_init=-1.5`. |
| 6 | `config/runtime.yaml` & `backend/app/config/settings.py` | Cấu hình runtime chính thức: `motion_threshold=0.008`, `min_sign_duration_ms=300`, `possible_end_delay_ms=1200`, `silence_ms=2500`, `cooldown_ms=400`, `inference_stride=4`, `camera=1280x720`. |
| 7 | `backend/app/ml/` (`model.py`, `model_loader.py`, `landmark_extractor.py`, `preprocessing.py`, `ctc_decoder.py`, `endpoint_fsm.py`, `sequence_buffer.py`) | Đã đọc toàn bộ và đối chiếu từng dòng. Lưu ý: `model.py` dùng `dilation=1` (khớp 100% mã huấn luyện Notebook 21, dù docstring ghi nhầm "dilated TCN 1,2,4,8"). |
| 8 | `backend/app/services/` (`isolated_session.py`, `continuous_session.py`, `realtime_session.py`, `language_service.py`) & `backend/app/api/realtime.py` | Đã kiểm tra đầy đủ luồng Mode A (`IsolatedRecognitionSession`), Mode B (`ContinuousRecognitionSession`), WebSocket single-slot drop (`latest_frame_bytes`) và legacy `RealtimeSession`. |
| 9 | `frontend/src/` (`App.tsx`, `hooks/useWebcam.ts`, `hooks/useRecognitionSocket.ts`, `components/CameraPanel.tsx`) | Đã phát hiện `useWebcam.ts` yêu cầu `640×480` (4:3), `App.tsx` ép resize `targetW=384` (không lật ảnh, đúng Mirror Rule), `useRecognitionSocket.ts` nén `image/jpeg` quality `0.6` với cơ chế 1-in-flight watchdog `350 ms`. |
| 10 | `docs/RECOGNITION_PIPELINE_AUDIT.md`, `docs/CONTINUOUS_MODEL_GAP.md` | Các tài liệu audit trước đó (đã giải quyết phần fake landmark persistence và tách 2 chế độ Mode A/B). |

### Danh sách 8 video mẫu trích xuất từ Notebook 21 (`audit_diagnostics/extracted_videos/`)

| Test ID | Video ID | Signer | Split | Gloss thực tế | Label ID | Độ phân giải & FPS | Frames (Raw / Recipe) | Ref Valid (L / R) |
|---|---|---|---|---|---|---|---|---|
| T01 | `024582` | `028` | test | **Con bò** | 56 | 640×640 @ 25.0 FPS | 60 / 69 (2.40s) | 0.800 / 0.817 |
| T02 | `010504` | `014` | test | **Trường học** | 332 | 640×640 @ 25.0 FPS | 77 / 85 (3.08s) | 0.883 / 0.857 |
| T03 | `010556` | `014` | test | **Mưa** | 197 | 640×640 @ 25.0 FPS | 66 / 73 (2.64s) | 0.803 / 0.848 |
| T04 | `000715` | `001` | test | **Bố** | 33 | 640×640 @ 25.0 FPS | 47 / 53 (1.88s) | 1.000 / 0.766 |
| T05 | `000004` | `001` | test | **Em** | 107 | 640×640 @ 25.0 FPS | 65 / 71 (2.60s) | 1.000 / 0.800 |
| T06 | `024410` | `028` | test | **Giúp đỡ** | 116 | 640×640 @ 25.0 FPS | 69 / 75 (2.76s) | 0.942 / 0.928 |
| T07 | `010532` | `014` | test | **Thời tiết** | 317 | 640×640 @ 25.0 FPS | 70 / 77 (2.80s) | 1.000 / 0.800 |
| T08 | `000096` | `001` | test | **Nhà trọ** | 228 | 640×640 @ 25.0 FPS | 71 / 78 (2.84s) | 0.859 / 0.803 |

*(Ghi chú sử dụng dữ liệu: 8 video trên là các mẫu minh họa có sẵn trong `Document_for_checking`, được sử dụng thuần túy để chẩn đoán và tái hiện lỗi pipeline; không sử dụng tập Test chính thức để tinh chỉnh/tối ưu tham số hay huấn luyện lại mô hình.)*

---

## 3. Quy trình kiểm thử đã thực hiện

Toàn bộ script kiểm thử được tách biệt hoàn toàn trong thư mục `audit_diagnostics/`:
1. `extract_and_verify.py`: Trích xuất 8 video MP4 nhúng, đối chiếu cấu trúc checkpoint và kiểm tra độ khớp logit trực tiếp giữa định nghĩa `HandsTCNTransformerCTC` của Notebook 21 và `backend/app/ml/model.py`.
2. `probe_components.py` (**Test C — Cô lập thành phần vòng 1**): Chạy 11 biến thể cô lập trên cùng 8 video (`pose_stride=1/2`, không có edge hold, đảo slot Trái/Phải, handedness trực tiếp vs đảo, lật gương khung hình, nén frontend 384px JPEG q=0.6, `static_image_mode=True`, `min_det=0.3`, xóa chiều sâu `z=0`).
3. `run_realtime_and_robustness.py` (**Test A, Test B, Test C mở rộng & Test D**):
   - **Test A (Offline Inference):** Đọc đủ frame video gốc `640×640`, trích xuất landmark + chuẩn hóa + thêm `edge_hold` đúng recipe, suy luận trực tiếp và đo thời gian từng giai đoạn.
   - **Test B (Realtime Pipeline Replay — 11 kịch bản B1–B11):** Đưa **từng khung hình một** với đồng hồ thời gian thực (virtual clock bước `1/25s = 40 ms/frame`) kèm pre-idle 15 frames (0.6s) và post-idle 60–90 frames (2.4–3.6s) vào trực tiếp `IsolatedRecognitionSession` (Mode A Auto & Manual) và `ContinuousRecognitionSession` (Mode B), bao gồm: gốc `640×640`, nén frontend `384px JPEG 0.6`, mô phỏng webcam `4:3` (`384×288`), webcam `16:9` (`384×216`), và mô phỏng WebSocket 1-in-flight frame dropping (`B11`).
   - **Test C mở rộng (C1–C11):** Tách biệt thuần túy tác động của tỷ lệ khung hình `4:3` & `16:9` (cả pad mở rộng FOV và crop gần), độ phân giải `384px JPEG 0.6`, và phép chứng minh nhân quả khôi phục đẳng hướng `y * (H/W)`.
   - **Test D (Robustness Testing):**
     - **D1 (Tốc độ):** 0.50×, 0.75×, 1.00×, 1.50×, 2.00× trên cả Mode A và Mode B.
     - **D2 (Mất landmark):** Mất ngẫu nhiên 10%, 25%, mất liên tiếp 5 frame và 15 frame giữa ký hiệu.
     - **D3 (Chuỗi nhiều ký hiệu & lặp ký hiệu):** 9 chuỗi ghép cùng người ký (2 từ với khoảng nghỉ 0.2s / 0.6s / 1.5s, 3 từ liên tiếp, và 3 chuỗi lặp cùng ký hiệu `Con bò – Con bò`, `Mưa – Mưa`, `Bố – Bố`).
     - **D4 (Tay nghỉ không ký hiệu):** 100 khung hình (4 giây) giữ nguyên tư thế đứng yên kèm nhiễu cảm biến camera `±3` pixel.
4. `deep_dive_evidence.py`: Đo chi tiết dao động `motion_energy` (phân rã `dxy_2d` vs `dz_1d`) trên 60 khung hình tĩnh hoàn toàn, ma trận 12 cấu hình `Aspect Ratio × Resolution × JPEG Quality × Isotropic Fix` kèm thời gian MediaPipe CPU, và kiểm tra độ nhạy theo FPS hiệu dụng (`12, 15, 20, 25, 30 FPS`).

---

## 4. Kết quả kiểm thử từng video

Bảng dưới đây đối chiếu trực tiếp kết quả trên cả **8 video mẫu** giữa **Offline Inference (Test A)** và **Realtime Replay (Test B)** ở các chế độ chính (gồm điều kiện lý tưởng `640×640` và điều kiện webcam thực tế `4:3 @ 384×288 JPEG 0.6` của frontend hiện tại):

| Test ID | Video | Gloss thực tế | Offline Prediction (Test A, 640×640) | Realtime Prediction — Mode A Auto (640 / Web 4:3 384) | Realtime Prediction — Mode B Continuous (640 / Web 4:3 384) | Trạng thái (Webcam 4:3) | Thành phần nghi ngờ |
|---|---|---|---|---|---|---|---|
| T01 | `024582` | **Con bò** | `Con bò` (1.00) | `Con bò` / **`Con thỏ`** | `Con bò` / **`Con thỏ`** | **FAIL** (Nhận nhầm) | Aspect Ratio 4:3 (`ERR-01`) + Downscale 384px (`ERR-02`) |
| T02 | `010504` | **Trường học** | `Trường học` (1.00) | **`<no_stop>`** / **`<no_stop>`** | **`<none>`** (live: `Trường học \| Ca sĩ`) / **`<none>`** (live: `Trường học`) | **FAIL** (Bỏ sót / Kẹt FSM) | Jitter `motion_energy` 0.032 (`ERR-03`) + `signing_space` (`ERR-05`) |
| T03 | `010556` | **Mưa** | `Mưa` (1.00) | `Mưa` / `Mưa` (0.72) | `Mưa` / `Mưa` | **PASS** (4:3) / **FAIL** (16:9 → `Laptop`) | Đạt ở 4:3; thất bại ở 16:9 do `ERR-01` + `ERR-02` |
| T04 | `000715` | **Bố** | `Bố` (1.00) | `Bố` / **`<no_stop>`** | `Bố` / **`<none>`** (live: `Bố`) | **FAIL** (Bỏ sót / Kẹt FSM) | Jitter 384px (`post>0.008`: 41/60f, `p95=0.049`, `ERR-03`) + `ERR-05` |
| T05 | `000004` | **Em** | `Em` (1.00) | `Em` / **`<no_stop>`** | `Em` / **`<none>`** (live: `Em`) | **FAIL** (Bỏ sót / Kẹt FSM) | Jitter 384px (`post>0.008`: 60/60f, `p95=0.056`, `ERR-03`) + `ERR-05` |
| T06 | `024410` | **Giúp đỡ** | `Giúp đỡ` (1.00) | `Giúp đỡ` / **`<no_stop>`** | `Giúp đỡ` / **`<none>`** (live: `Vòng tay`) | **FAIL** (Kẹt FSM + Sai live) | Jitter 384px (`39/60f`, `ERR-03`) + 4:3 & 384px (`ERR-01`, `ERR-02`) |
| T07 | `010532` | **Thời tiết** | `Thời tiết` (0.95) | `Thời tiết` / **`Gió`** (greedy: `Gió \| Gió`) | `Thời tiết` / **`Gió \| Con heo`** | **FAIL** (Nhận nhầm + Lặp/Thừa) | Downscale 384px (`vR: 0.81→0.63`, `ERR-02`) + 4:3 (`ERR-01`) + Buffer đuôi tĩnh (`ERR-06`) |
| T08 | `000096` | **Nhà trọ** | `Nhà trọ` (1.00) | `Nhà trọ` / **`Ngân hàng`** | `Nhà trọ` / **`Ngân hàng`** | **FAIL** (Nhận nhầm) | Aspect Ratio 4:3 (`ERR-01`) + Downscale 384px (`ERR-02`) + MediaPipe tracking (`ERR-08`) |

*(Xem đầy đủ cả 11 kịch bản B1–B11 bao gồm Mode A Manual và Webcam 16:9 tại [`audit_diagnostics/test_results.csv`](audit_diagnostics/test_results.csv).)*

---

## 5. So sánh Offline Inference và Realtime Inference

### 5.1. Thống kê tổng hợp trên 8 video đơn (Test A vs Test B1–B11)

| Kịch bản | Khung hình / Điều kiện | Đúng hoàn toàn (EM) | Bỏ sót (`<none>`) | Nhận nhầm | Từ phát sinh thừa | Từ lặp | Tỷ lệ Blank TB | Tỷ lệ mất Landmark TB | MediaPipe TB (ms/f) | Model Forward TV (ms) | Độ trễ xác nhận TB (s) | FPS thực tế (CPU) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Test A (Offline)** | 1:1 `640×640` Raw | **8/8 (100.0%)** | 0 | 0 | 0 | 0 | 98.60% | 14.7% | 33.9 | 10.4 | 0.00 | 29.5 |
| **B1: Iso Auto** | 1:1 `640×640` Raw | 7/8 (87.5%) | 1 | 0 | 0 | 0 | 99.06% | 14.7% | 29.9 | 13.9 | 1.50 | 32.4 |
| **B2: Iso Auto** | 1:1 `384×384` JPEG60 | 7/8 (87.5%)* | 1 | 0 | 1 | 0 | 98.91% | 19.9% | 34.1 | 14.1 | 1.51 | 28.5 |
| **B3: Iso Manual (Tay trong khung)** | 1:1 `384×384` JPEG60 | 7/8 (87.5%)** | 0 | 1 | **3** | 0 | 99.08% | 19.9% | 26.0 | 13.9 | 2.40 | 37.6 |
| **B4: Iso Manual (Hạ tay hẳn)** | 1:1 `384×384` JPEG60 | 7/8 (87.5%) | 0 | 1 | 0 | 0 | 98.86% | 19.9% | 24.3 | 11.2 | 0.48 | 39.8 |
| **B5: Iso Auto (Webcam 4:3)** | 4:3 `384×288` JPEG60 | **1/8 (12.5%)** | **4** | **3** | 1 | 1 | 98.97% | **26.9%** | 23.6 | 10.6 | 1.51 | 41.1 |
| **B6: Iso Auto (Webcam 16:9)** | 16:9 `384×216` JPEG60 | **0/8 (0.0%)** | **4** | **4** | 0 | 0 | 98.89% | **42.4%** | 27.2 | 13.0 | 0.70 | 35.9 |
| **B7: Continuous** | 1:1 `640×640` Raw | 7/8 (87.5%) | 1 | 0 | 0 | 0 | 99.06% | 14.7% | 28.7 | 15.5 | 2.30 | 34.2 |
| **B8: Continuous** | 1:1 `384×384` JPEG60 | 5/8 (62.5%) | 1 | 1 | 1 | 0 | 98.99% | 19.9% | 28.1 | 11.8 | 2.31 | 35.3 |
| **B9: Continuous (Webcam 4:3)** | 4:3 `384×288` JPEG60 | **1/8 (12.5%)** | **4** | **3** | 1 | 0 | 99.16% | **26.9%** | 27.1 | 12.1 | 2.25 | 36.0 |
| **B10: Continuous (Webcam 16:9)** | 16:9 `384×216` JPEG60 | **0/8 (0.0%)** | **5** | **3** | 0 | 0 | 99.46% | **42.4%** | 30.3 | 11.7 | 2.04 | 32.7 |
| **B11: Continuous + WS Drop** | 1:1 `384×384` JPEG60 | 7/8 (87.5%) | 1 | 0 | 0 | 0 | 98.97% | 19.9% | 28.0 | 11.6 | 2.29 | 35.3 |

*\*B2: Greedy Exact chỉ đạt 6/8 do `Trường học` giải mã thành `Trường học | Ca sĩ` nhưng Mode A lấy `gloss_sequence[0]`.*  
*\*\*B3: Greedy Exact chỉ đạt 5/8 do 3/8 mẫu phát sinh từ ảo ở đuôi tĩnh (`Trường học | Ca sĩ`, `Em | Đẹp (người)`, `Gió | Ca sĩ`).*

### 5.2. Phân biệt rõ Độ trễ suy luận mô hình vs Độ trễ toàn bộ hệ thống

- **Độ trễ suy luận thuần của mô hình (`HandsTCNTransformerCTC` forward trên CPU):** chỉ **10.4–15.5 ms** (trung vị), kèm giải mã CTC greedy + Top-5 khoảng **2–8 ms**. Đây KHÔNG phải nút cổ chai.
- **Độ trễ xử lý từng frame của toàn pipeline backend:** **24–34 ms/frame** trên CPU, trong đó MediaPipe Pose + Hands chiếm **~90%** (`23.6–34.1 ms/frame`, có các đỉnh spike lên `58–68 ms` khi mất track và chạy lại Palm Detector). Tốc độ thực tế đạt khoảng **28–40 FPS** trên máy thử nghiệm, tuy nhiên ở những frame vượt ngân sách `33.3 ms` (30 FPS), cơ chế 1-in-flight WebSocket (`B11`) đã loại bỏ **7–54 frame/clip** (tối đa **29.7%** trên `Trường học`).
- **Độ trễ từ lúc kết thúc ký hiệu đến lúc hệ thống xác nhận kết quả (Confirmation Latency):**
  - Mode A Auto: trung bình **1.50–1.51 giây** (do `pause_timeout_sec = 1.6s`), hoặc **vô hạn (kẹt không trả kết quả)** nếu có jitter tĩnh.
  - Mode A Manual (hạ tay hoàn toàn ra khỏi khung hình): **0.48 giây** (`hands_absent_stop_sec = 0.45s`).
  - Mode B Continuous: trung bình **2.25–2.56 giây** (do `silence_ms = 2500` trong `runtime.yaml`), hoặc **vô hạn (kẹt `SIGNING`)** khi có jitter tĩnh.

### 5.3. Kết quả kiểm thử chuỗi nhiều ký hiệu & lặp ký hiệu (Test D3 — 9 chuỗi tham chiếu)

| Nhóm đánh giá (9 chuỗi, tổng 20 từ tham chiếu) | WER trung bình | Exact Sequence Match | Ghi chú sai lỗi chính |
|---|---:|---:|---|
| Offline `640×640` | 11.11% | **7/9** | M3 (nghỉ 1.5s: `Cái áo`), M9 (`Bố Bố` → `Bố`) |
| Realtime Continuous `640×640` | **5.56%** | **8/9** | Duy nhất M3 (nghỉ 1.5s) bị chèn từ thừa `Bác` |
| Realtime Continuous **Webcam 4:3 + 384px** | **35.18%** | **4/9** | M4 (`Thời tiết→Gió`), M5 (`Nhà trọ→Ngân hàng`), M6 (`[]`, WER 100%), M7 (`Con thỏ Con bò`), M9 (`[]`, WER 100%) |

![So sánh độ chính xác Offline vs Realtime và bằng chứng khôi phục](audit_diagnostics/figures/fig1_offline_vs_realtime_accuracy.png)

---

## 6. Danh sách lỗi và nguyên nhân gốc

| Mã lỗi | Hiện tượng | Nguyên nhân | File / Hàm liên quan | Bằng chứng kiểm thử | Mức độ | Trạng thái xác minh |
|---|---|---|---|---|---|---|
| **ERR-01** | Nhận nhầm hàng loạt trên webcam tỷ lệ `4:3` và `16:9` (`Con bò→Con thỏ`, `Mưa→Gội đầu`, `Bố→Thơm`, `Nhà trọ→Xin/Cháu`) | **Biến dạng tỷ lệ khung hình (Aspect-Ratio Distortion):** Video huấn luyện VSL400 là vuông `1:1` (`640×640`) nên tọa độ chuẩn hóa MediaPipe `(x=px/W, y=py/H)` đẳng hướng. Trên webcam `4:3` (`useWebcam.ts`) hoặc `16:9` (`runtime.yaml`), `scale` vai tính chủ yếu theo phương ngang `W` còn `y` chuẩn hóa theo `H`, làm trục `Y` và vận tốc `dy` bị kéo giãn **1.333×** (4:3) và **1.778×** (16:9). | `backend/app/ml/preprocessing.py` (`compute_shoulder_anchor`, `normalize_hand_landmarks`); `backend/app/ml/landmark_extractor.py` (`extract`); `frontend/src/hooks/useWebcam.ts` | Test C: `1:1` đạt **8/8** → `4:3` tụt **6/8** → `16:9` tụt **4/8** (Mean Rank `16.4`). Chỉ cần nhân `y * (H/W)` (C9/C10), cả `4:3` và `16:9` lập tức phục hồi lên **7/8**! | **Critical** | **Đã xác minh nguyên nhân** |
| **ERR-02** | Mất landmark bàn tay phải tới **30–38%**, sụp đổ nhận diện (`0/8` ở `16:9`, `4/8` ở `4:3`) và làm MediaPipe chạy **chậm hơn** | **Ép giảm độ phân giải quá mức + nén JPEG mạnh ở Frontend:** `App.tsx` ép `targetW = 384` (còn `384×288` ở 4:3 và `384×216` ở 16:9) kết hợp `useRecognitionSocket.ts` nén JPEG `0.6`. Bàn tay chỉ còn ~30–45px khiến MediaPipe Hands liên tục mất track và phải chạy lại Palm Detector nặng nề mỗi frame. | `frontend/src/App.tsx:120` (`targetW = 384`); `frontend/src/hooks/useRecognitionSocket.ts:404` (`0.6`) | Forensic 2: `4:3 w=384 q=60` đạt **4/8** (`vR=0.64`, `mp=30.2 ms`); nâng lên `4:3 w=480 q=75 + IsoFix` đạt **8/8 (100%)** (`vR=0.76`, `mp=26.2 ms` — nhanh hơn 4 ms). `16:9 w=384 q=60` đạt **0/8** (`vR=0.46`) vs `w=854 q=80 + IsoFix` đạt **7/8** (`vR=0.70`, `mp=26.2 ms`). | **Critical** | **Đã xác minh nguyên nhân** |
| **ERR-03** | Hệ thống kẹt vĩnh viễn ở `RECORDING` (Mode A Auto) hoặc `SIGNING` (Mode B), không bao giờ trả kết quả (`<none>`) dù người ký đã dừng hoàn toàn | **Nhiễu `motion_energy` 3D thô + bỏ quên bộ lọc EMA:** `motion_energy` tính chuẩn 3D `dxyz` (gồm nhiễu chiều sâu `dz` lớn và dao động ROI ±0.0006 của MediaPipe trên tay tĩnh), tạo `motion_energy = 0.024–0.056` ngay trên 60 khung hình giống hệt nhau (gấp 3–7× ngưỡng `0.008`). `EndpointFSM` có tính `self.smoothed_motion` nhưng **không dùng** ở dòng 103. | `backend/app/ml/preprocessing.py:253`; `backend/app/ml/endpoint_fsm.py:90-106`; `backend/app/services/isolated_session.py:420`; `config/runtime.yaml:44` | Test B5 & B9: **4/8 video kẹt không chốt** (`Trường học`, `Bố`, `Em`, `Giúp đỡ`) dù `live_tokens` đã đúng. Forensic 1: trên 60 frame tĩnh 100%, `Em` có `60/60` frame `>0.008` (`p95=0.0561`), `Bố` có `41/60` (`p95=0.0489`), `Trường học 640px` có `60/60` (`me_3d=0.0325`, trong đó `dz_1d=0.0231`). | **Critical** | **Đã xác minh nguyên nhân** |
| **ERR-04** | Kích hoạt giả khi ngồi yên và phát sinh từ ảo **`Cái quần`**, **`Ca sĩ`**, **`Đẹp (người)`** | **Lỗi logic chuyển `STARTING → SIGNING`:** Trong `EndpointFSM:122-128`, `min_sign_duration_sec = 0.3s` **nhỏ hơn** thời gian hủy (`0.6s/1.0s`), nên chỉ cần **1 frame nhiễu duy nhất** `>= 0.008` là *chắc chắn* tự chuyển sang `SIGNING` sau `0.3s` dù 100% frame sau đó đứng yên. Kèm việc không cắt đoạn tĩnh ở đầu/đuôi buffer. | `backend/app/ml/endpoint_fsm.py:122-128`; `backend/app/services/isolated_session.py:152-156`; `backend/app/services/continuous_session.py:388-405` | Test D4 (4 video tĩnh 100%): **3/4 video** lọt vào `SIGNING` và **2/4** chốt giả **`Cái quần`**. Test B7: **6/8 video** vào `STARTING` ngay tại frame tĩnh `-14/-13`. Test B3: **3/8 video** phát sinh token thừa ở đuôi tĩnh (`Ca sĩ`, `Đẹp (người)`). | **High** | **Đã xác minh nguyên nhân** |
| **ERR-05** | Tay hạ nghỉ ngang hông/đùi vẫn bị tính `in_signing_space = True` (100% khung hình), làm mất tác dụng chống nhiễu và chốt nhanh khi hạ tay | **Ngưỡng `check_signing_space` cố định theo chiều cao ảnh (`shoulder_y + 0.35`):** Không chuẩn hóa theo `shoulder_scale` và tỷ lệ ảnh. Với `shoulder_y = 0.48–0.51` và `scale = 0.18–0.22`, ngưỡng `0.83–0.86` bao trùm xuống tận `1.44–1.60×` bề rộng vai (ngang đùi/hông). | `backend/app/ml/preprocessing.py:146-176` (`check_signing_space`) | Forensic 1 & phân rã tọa độ: 7/8 video có `post_in_space_ratio = 1.00` dù cổ tay nghỉ ở `y = 0.796–0.803` (`1.44–1.60×` bề rộng vai dưới vai). | **High** | **Đã xác minh nguyên nhân** |
| **ERR-06** | Bộ đệm chuỗi quá dài (`125–160` frames cho ký hiệu `47–77` frames), độ trễ chốt cao (`2.3–2.7s`) và tăng nguy cơ từ ảo | **Thiếu cắt biên động (Active Motion Trimming):** Mode A `_get_trimmed_segment()` chỉ cắt đoạn mất tay hoàn toàn, không cắt đoạn tay đứng yên đầu/cuối. Mode B cập nhật `_last_active_buf_len` suốt `1.2s` tĩnh trước `POSSIBLE_END` rồi cộng thêm 10 frame đuôi. | `backend/app/services/isolated_session.py:152-230`; `backend/app/services/continuous_session.py:159-168, 403-404` | Test B3: `buf = 122–152f` (`+8 edge hold = 130–160f`) làm Greedy Exact giảm còn 5/8. Test B7: `Bố` (`47f` thật) bị kéo thành `125f`, `Mưa` (`66f`) thành `145f`, độ trễ chốt TB `2.30s`. | **Medium** | **Đã xác minh nguyên nhân** |
| **ERR-07** | Rớt 4–30% khung hình ở chế độ WebSocket 1-in-flight trên CPU; tụt độ chính xác nếu FPS thực tế xuống `< 15 FPS` | MediaPipe CPU mất `25–37 ms/frame` (đỉnh `58–68 ms`), vượt chu kỳ `33.3 ms` của 30 FPS nên cơ chế 1-in-flight (`useRecognitionSocket.ts`) + single-slot (`realtime.py`) bỏ qua frame trong lúc xử lý. | `frontend/src/hooks/useRecognitionSocket.ts:363-406`; `backend/app/api/realtime.py:60-85` | Test B11: rớt `7–54` frame/clip (`Trường học` rớt 29.7%). Forensic 3: `15–30 FPS` đạt **8/8**, nhưng `12 FPS` giảm còn **6/8**. | **Medium** | **Đã xác minh nguyên nhân** |
| **ERR-08** | Độ nhạy cao hơn của ký hiệu hai tay chồng chéo/phức tạp (`Nhà trọ`, `Trường học`) giữa MediaPipe Legacy (`mp.solutions`) và MediaPipe Tasks lúc huấn luyện | `extractor_config.json` dùng `MediaPipe Tasks` khi huấn luyện, trong khi runtime dùng `mp.solutions` (`0.10.14`). Trên video gốc, `Trường học` có `vL=0.61` vs ref `0.88`; `Nhà trọ` vẫn sai (`Cháu`/`Nhà`) trong C9/C10 (7/8) sau khi sửa tỷ lệ khung hình. | `backend/app/ml/landmark_extractor.py:23-81`; `extractor_config.json` | Probe V8 (`static_image_mode=True`) nâng `Trường học vL` từ `0.61 → 0.83`; V9 (`min_det=0.3`) nâng `vL/vR`; `Nhà trọ` là 1/8 mẫu duy nhất còn sai sau IsoFix ở 640px/854px. | **Medium** | **Có bằng chứng nghi ngờ nhưng cần kiểm tra thêm** |
| **ERR-09** | Khoảng nghỉ giữa 2 từ kéo dài (`>= 1.5s`) hoặc ký quá chậm (`0.5×`) / quá nhanh (`2.0×`) làm tăng WER và chèn từ thừa | Mô hình huấn luyện trên chuỗi ghép khoảng chuyển tiếp ngắn ở 25 FPS (`34–99f/từ`). Khi khoảng nghỉ `1.5s` vẫn nằm chung utterance (vì `silence_ms = 2.5s`), mạng CTC phát sinh token chen (`Bác`). | `backend/app/services/continuous_session.py`; `config/runtime.yaml` (`silence_ms: 2500`) | Test D1: tốc độ `0.5×` giảm còn `5/8`, `2.0×` còn `6/8`. Test D3: `M1/M2` (nghỉ 0.2–0.6s) đạt WER `0.00`, nhưng `M3` (nghỉ 1.5s) bị chèn `Bác` (WER `0.50`). | **Medium** | **Đã xác minh nguyên nhân** |
| **ERR-10** | Không đồng bộ thông số camera/ngưỡng giữa `runtime.yaml`, frontend và `isolated_session.py` | `runtime.yaml` đặt `1280×720` (16:9), `useWebcam.ts` hard-code `640×480` (4:3), `App.tsx` hard-code `targetW=384`, `isolated_session.py` hard-code ngưỡng không đọc từ YAML. | `config/runtime.yaml`; `frontend/src/hooks/useWebcam.ts`; `frontend/src/App.tsx`; `backend/app/services/isolated_session.py` | Đối chiếu mã nguồn trực tiếp giữa 4 tệp cấu hình. | **Low** | **Đã xác minh nguyên nhân** |

### Tại sao Offline dự đoán đúng 8/8 nhưng Realtime lại sai hoặc bỏ sót?

1. **Trong môi trường Offline (Notebook 21 & Test A):**
   - Video đầu vào có độ phân giải chuẩn `640×640` (tỷ lệ vuông `1:1`), không bị nén JPEG chất lượng thấp → tọa độ `(x, y)` đẳng hướng đúng như phân phối huấn luyện và tỷ lệ phát hiện bàn tay cao (`vL=0.88`, `vR=0.83`).
   - Toàn bộ chuỗi đã được cắt biên sẵn đúng độ dài 1 ký hiệu (`47–77 frames`), không đi qua FSM/Auto-Stop nên không bị kẹt trạng thái vì nhiễu jitter tĩnh, cũng không bị chèn 50–80 frame tĩnh gây từ ảo.
2. **Trong môi trường Realtime Webcam (Test B5, B6, B9, B10):**
   - Khung hình từ webcam là `4:3` hoặc `16:9` (làm sai tỷ lệ hình học `Y` so với `X` tới **33–78%**), sau đó bị frontend nén về `384×288` hoặc `384×216` JPEG `0.6` (làm mất **27–42%** landmark bàn tay).
   - Cùng lúc đó, khi người ký dừng tay lại, dao động ROI của MediaPipe + nhiễu chiều sâu `dz` tạo `motion_energy = 0.03–0.056` lớn hơn hẳn ngưỡng `0.008`, trong khi `check_signing_space` vẫn báo `True` ở ngang hông → FSM/Auto-Stop kẹt vĩnh viễn ở `SIGNING`/`RECORDING` (bỏ sót kết quả) hoặc tự kích hoạt trên khung hình tĩnh (tạo từ ảo `Cái quần`, `Ca sĩ`).

---

## 7. Bằng chứng xác minh từng lỗi

### 7.1. Ma trận chứng minh `ERR-01` (Aspect Ratio) & `ERR-02` (384px JPEG q=0.6)

| Cấu hình thử nghiệm (Forensic 2 — Pure Offline, không chịu ảnh hưởng FSM) | Greedy Exact | Top-1 | Mean GT Rank | Valid Left | Valid Right | MediaPipe TB (ms/f) | Kích thước JPEG (KB) | Các lỗi xảy ra |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `1:1` Gốc `640×640` Raw (Baseline) | **8/8 (100%)** | 8/8 | 1.00 | 0.88 | 0.83 | 29.6 | — | Không có |
| `4:3` `w=384` `q=60` **(Frontend hiện tại)** | **4/8 (50%)** | 4/8 | 2.50 | 0.82 | **0.64** | **30.2** | 6.6 | Con bò→Con thỏ, Giúp đỡ→Vòng tay, Thời tiết→Gió, Nhà trọ→Ngân hàng |
| `4:3` `w=384` `q=60` + **IsoFix** | 5/8 (62.5%) | 5/8 | 2.00 | 0.82 | 0.64 | 28.9 | 6.6 | Giúp đỡ→Vòng tay, Thời tiết→Gió, Nhà trọ→Ngân hàng |
| `4:3` `w=480` `q=75` + **IsoFix** | **8/8 (100%)** | **8/8** | **1.00** | **0.89** | **0.76** | **26.2** | 11.4 | **Không có (0 lỗi!)** |
| `4:3` `w=640` `q=80` (Không sửa tỷ lệ) | 6/8 (75%) | 6/8 | 2.00 | 0.87 | 0.81 | 25.7 | 19.3 | Con bò→Con thỏ, Nhà trọ→Ngân hàng |
| `4:3` `w=640` `q=80` + **IsoFix** | **7/8 (87.5%)** | 7/8 | 1.25 | 0.87 | **0.81** | **25.8** | 19.3 | Nhà trọ→Cháu |
| `16:9` `w=384` `q=60` **(Frontend hiện tại trên HD cam)** | **0/8 (0%)** | 1/8 | 7.38 | 0.69 | **0.46** | **28.8** | 5.1 | **Sai/Blank toàn bộ 8/8 video** |
| `16:9` `w=640` `q=80` (Không sửa tỷ lệ) | 4/8 (50%) | 4/8 | 6.12 | 0.88 | 0.66 | 30.9 | 14.9 | Con bò→Con thỏ, Mưa→Gội đầu, Bố→Thơm, Nhà trọ→Công viên |
| `16:9` `w=640` `q=80` + **IsoFix** | **7/8 (87.5%)** | 7/8 | 7.75 | 0.88 | 0.66 | 31.0 | 14.9 | Nhà trọ→Nhà |
| `16:9` `w=854` `q=80` + **IsoFix** | **7/8 (87.5%)** | 7/8 | **2.12** | 0.88 | **0.70** | **26.2** | 22.7 | Nhà trọ→Cháu |

![Ảnh hưởng của độ phân giải, tỷ lệ khung hình và Isotropic Fix lên độ chính xác và độ trễ MediaPipe](audit_diagnostics/figures/fig2_aspect_resolution_latency.png)

### 7.2. Bằng chứng xác minh `ERR-03` (Jitter tĩnh gây kẹt FSM), `ERR-04` (False Trigger) & `ERR-05` (Signing Space)

- **Phân rã nhiễu trên 60 khung hình tĩnh hoàn toàn sau ký hiệu (Forensic 1):**
  - `Trường học (010504)` ở gốc `640×640`: `me_3d = 0.03254` (trong đó `dxy_2d = 0.02039`, `dz_1d = 0.02314`); **60/60 frame > `0.008`**, `maxQuietRun = 0/30` (cần 30 frame liên tiếp < `0.008` mới vào `POSSIBLE_END`!). Cổ tay đang nằm tại `y = 0.803`, dưới vai (`0.485`) tới `1.44×` bề rộng vai (`scale = 0.220`), nhưng `postInSpace = 1.00`.
  - `Em (000004)` ở `4:3 + 384px`: **60/60 frame > `0.008`**, `p95 = 0.0561`, `maxQuietRun = 0/30`, `postInSpace = 1.00`.
  - `Bố (000715)` ở `4:3 + 384px`: **41/60 frame > `0.008`**, `p95 = 0.0489`, `maxQuietRun = 1/30`, `postInSpace = 1.00`.
  - `Giúp đỡ (024410)` ở `4:3 + 384px`: **39/60 frame > `0.008`**, `p95 = 0.0244`, `maxQuietRun = 4/30`, `postInSpace = 1.00`.
- **Kiểm tra tay đứng yên không ký hiệu (Test D4 — 100 frame tĩnh + nhiễu cảm biến ±3px):**
  - Video `010504` và `010556`: Tự kích hoạt `IDLE → STARTING → SIGNING → POSSIBLE_END → END_OF_UTTERANCE` và chốt giả **`['Cái quần']`**.
  - Video `000715`: Tự kích hoạt cả Isolated Auto (`iso_trig=True`) lẫn Continuous (`live=['Cái quần']`).

![Nhiễu motion_energy trên khung hình tĩnh vượt ngưỡng 0.008 gây kẹt FSM](audit_diagnostics/figures/fig3_static_jitter_fsm_stall.png)

### 7.3. Bằng chứng các phép cô lập khác (Test C vòng 1 & Test D Robustness)

- **Đã loại trừ các giả thuyết sai (Bằng chứng loại trừ):**
  - Thứ tự slot tay Trái/Phải (`0..20` Left, `21..41` Right): **ĐÚNG** (đảo slot làm rớt từ `8/8` xuống **`0/8`**, Mean Rank `127.2`).
  - Handedness đảo chiều cho ảnh không lật (`V5`): **ĐÚNG** (`8/8` vs `0/8` nếu dùng trực tiếp `V4`).
  - Tầm quan trọng của chiều sâu `z`: **CÓ SỬ DỤNG** (xóa `z=0` làm rớt từ `8/8` xuống **`5/8`** — do đó KHÔNG được xóa `z` ở đầu vào mô hình; chỉ loại `dz` ra khỏi `motion_energy` cho FSM!).
  - `pose_stride = 2` vs `1`: Không gây tụt độ chính xác trên video đơn (`8/8` như nhau), và KHÔNG phải nguyên nhân gây jitter (Forensic 1 chứng minh `stride=1` vẫn có `60/60` frame jitter trên `Em` và `41/60` trên `Bố`).
  - Độ nhạy FPS (Forensic 3): Hệ thống đạt **8/8** ổn định trong dải **15–30 FPS**, chỉ tụt xuống **6/8** khi rơi xuống **12 FPS**.

![Tổng hợp kiểm thử độ bền vững Test D](audit_diagnostics/figures/fig4_robustness_summary.png)

---

## 8. Danh sách giải pháp đề xuất theo mức độ ưu tiên

*(Chi tiết kỹ thuật từng bước, thông số hiện tại vs đề xuất, kiểm thử lại và phương án rollback xem đầy đủ tại [`audit_diagnostics/proposed_fix_plan.md`](audit_diagnostics/proposed_fix_plan.md).)*

| Mã giải pháp | Vấn đề cần xử lý | Phương án đề xuất | File cần sửa | Ảnh hưởng dự kiến | Rủi ro | Ưu tiên |
|---|---|---|---|---|---|---|
| **FIX-01** | `ERR-01` (Biến dạng tỷ lệ khung hình 4:3 & 16:9) | Áp dụng **Isotropic Aspect-Ratio Normalization** trong `VSLPreprocessor` (truyền `aspect_ratio = H / W` từ `frame_bgr.shape` vào chuẩn hóa tọa độ `y' = y * (H/W)` trước khi tính tâm vai, tỷ lệ vai `scale` và chuẩn hóa tay, đảm bảo trục X–Y–Z cùng hệ tỷ lệ pixel như video `1:1`, **không làm thay đổi đầu ra trên video `1:1` vì `H/W = 1.0`**). | `backend/app/ml/preprocessing.py`; `backend/app/services/isolated_session.py`; `backend/app/services/continuous_session.py` | Phục hồi độ chính xác trên webcam `4:3` (`6/8 → 7-8/8`) và `16:9` (`4/8 → 7/8`), tăng ổn định hình học với mọi loại webcam. | Rất thấp (tương đương phép đồng nhất trên ảnh `1:1`; không chạm kiến trúc mô hình). | **P0 — Critical** |
| **FIX-02** | `ERR-02` (Ép 384px + JPEG q=0.6 mất bàn tay & chậm MediaPipe) | Nâng độ phân giải gửi ở Frontend từ `targetW = 384` lên **`targetW = 512–640`** (mặc định đề xuất **`640`** cho 4:3 hoặc **`min(640, videoWidth)`** kèm chất lượng JPEG **`0.80`**; nếu tỷ lệ rộng `16:9` thì ưu tiên `targetH >= 360`, tức `w=640`). | `frontend/src/App.tsx`; `frontend/src/hooks/useRecognitionSocket.ts` | Giảm mất landmark tay phải từ `36–54%` xuống `19–30%`, nâng độ chính xác `4:3` lên **8/8** và `16:9` lên **7/8**, đồng thời **giảm latency MediaPipe CPU 4 ms/frame** (do ít chạy lại Palm Detector). | Rất thấp (payload JPEG tăng từ ~6.6 KB lên ~15–19 KB/frame, hoàn toàn không đáng kể trên localhost/LAN). | **P0 — Critical** |
| **FIX-03** | `ERR-03` (Kẹt vĩnh viễn `SIGNING`/`RECORDING` vì nhiễu `motion_energy` 3D) | Tách riêng `motion_energy` (chỉ dùng cho FSM/Auto-Stop, **GIỮ NGUYÊN 100% `dxyz` đầu vào mô hình**): tính `motion_energy` bằng **vận tốc 2D của tọa độ đã lọc EMA/cửa sổ trượt 3–4 frame** (triệt tiêu hoàn toàn dao động ROI ±0.0006 và loại nhiễu `dz`), đồng thời sửa `EndpointFSM` để sử dụng `self.smoothed_motion` thay vì giá trị thô. | `backend/app/ml/preprocessing.py`; `backend/app/ml/endpoint_fsm.py`; `backend/app/services/isolated_session.py` | Loại bỏ 100% hiện tượng kẹt trạng thái trên khung hình tĩnh (`4/8` video kẹt → `0/8`), giúp Mode A Auto và Mode B chốt kết quả tin cậy. | Thấp (cần tinh chỉnh hysteresis ngưỡng start/continue trên tập mẫu + test hồi quy để không bỏ sót cử chỉ nhẹ như `Bố`, `Em`). | **P0 — Critical** |
| **FIX-04** | `ERR-04` (False trigger `STARTING→SIGNING` tạo từ ảo `Cái quần`) | Sửa điều kiện chuyển `STARTING → SIGNING` trong `EndpointFSM`: yêu cầu **duy trì hoạt động liên tục** (ví dụ ≥ 60–70% số frame trong `min_sign_duration_sec` có `smoothed_motion >= threshold`, hoặc đặt ngưỡng hủy `no_motion_abort_sec = 0.20s < min_sign_duration_sec = 0.30s`), tránh 1 frame nhiễu đơn lẻ lọt vào `SIGNING`. | `backend/app/ml/endpoint_fsm.py` | Triệt tiêu hoàn toàn kích hoạt giả và từ ảo `Cái quần` trên khung hình đứng yên (`D4`: 3/4 false trigger → 0/4). | Thấp (có thể tăng độ trễ bắt đầu nhận diện thêm ~100 ms nếu đặt quá chặt; giữ `min_sign_duration_ms = 250–300ms`). | **P1 — High** |
| **FIX-05** | `ERR-05` (`check_signing_space` luôn `True` ở ngang hông/đùi) | Thay ngưỡng cố định `shoulder_y + 0.35` bằng ngưỡng **tương đối theo tỷ lệ vai** sau chuẩn hóa đẳng hướng: `wrist_y_norm <= 1.20` (hoặc `tip_y_norm <= 1.15`, tương đương từ ngực/bụng trên trở lên, loại vùng hông/đùi `>= 1.40 × scale`). | `backend/app/ml/preprocessing.py` (`check_signing_space`) | Phân biệt chính xác tay đang ký hiệu vs tay hạ nghỉ ở mọi khoảng cách camera và mọi tỷ lệ khung hình; kích hoạt cơ chế chốt nhanh khi hạ tay. | Trung bình (cần kiểm tra lại với các ký hiệu nằm thấp như `Cái quần`, `Con chó` để không cắt nhầm vùng ký hiệu hợp lệ). | **P1 — High** |
| **FIX-06** | `ERR-06` (Đuôi/đầu tĩnh quá dài gây từ ảo `Ca sĩ`, `Đẹp (người)` và tăng độ trễ chốt) | Bổ sung **Active Motion Boundary Trimming** cho cả Mode A (`_get_trimmed_segment`) và Mode B (`_get_trimmed_utterance_array`): cắt bỏ các frame tĩnh thừa ở đầu và đuôi vượt quá `4–6` frame hold dựa trên chuỗi `motion_energy` đã lưu song song trong buffer. | `backend/app/services/isolated_session.py`; `backend/app/services/continuous_session.py` | Loại bỏ 3/8 trường hợp phát sinh từ ảo ở Mode A Manual (`B3`), giảm độ dài tensor suy luận gần đúng độ dài ký hiệu thật (`~60–80f` thay vì `130–160f`), giảm latency forward & tăng độ sạch CTC. | Thấp (giữ đệm an toàn `6` frame ở đầu và `6` frame ở cuối để không cắt mất phần bắt đầu/kết thúc ký hiệu). | **P1 — High** |
| **FIX-07** | `ERR-08` & `ERR-07` (Cải thiện tracking MediaPipe & giảm rớt frame) | Tinh chỉnh `min_tracking_confidence = 0.4`, `min_detection_confidence = 0.4` cho `mp.solutions.hands` (giữ `model_complexity = 1`), và tối ưu `useWebcam.ts` chọn tỷ lệ gần vuông/4:3 (`640×480 @ 30fps`) đồng bộ với `runtime.yaml`. Tùy chọn bổ sung chế độ **MediaPipe Tasks** nếu muốn đạt parity 100% với `extractor_config.json`. | `backend/app/ml/landmark_extractor.py`; `config/runtime.yaml`; `frontend/src/hooks/useWebcam.ts` | Cải thiện tỷ lệ giữ dấu tay trên ký hiệu che khuất một phần (`Trường học`, `Nhà trọ`) và giữ FPS ổn định trên 25–30 FPS. | Thấp (không thay đổi định dạng vector 294 chiều). | **P2 — Medium** |
| **FIX-08** | `ERR-09` & `ERR-10` (Đồng bộ cấu hình & cải thiện khoảng nghỉ Continuous) | Đọc thống nhất các tham số Mode A/B từ `config/runtime.yaml`; giảm `silence_ms` mặc định ở mức `medium` từ `2500 ms` xuống `1500–1800 ms` và `possible_end_delay_ms` từ `1200 ms` xuống `800–900 ms` sau khi FIX-03 loại bỏ nhiễu jitter. | `config/runtime.yaml`; `backend/app/config/settings.py`; `backend/app/services/isolated_session.py` | Giảm độ trễ xác nhận câu từ `~2.5s` xuống `~1.5–1.8s`, giảm hiện tượng chèn từ thừa khi nghỉ giữa 2 từ `1.5s`. | Thấp (người dùng vẫn có thể chỉnh mức độ nhạy `low/medium/high` trên giao diện Settings). | **P2 — Medium** |

---

## 9. Phạm vi mã nguồn dự kiến cần chỉnh sửa (chỉ thực hiện sau khi được phê duyệt)

| Nhóm | Tệp mã nguồn | Mức độ thay đổi |
|---|---|---|
| Backend — Tiền xử lý & Landmark | `backend/app/ml/preprocessing.py` | Thêm tham số `aspect_ratio` (mặc định `1.0`), tách `motion_energy` 2D-smoothed khỏi `dxyz` mô hình, chuẩn hóa `check_signing_space` theo `shoulder_scale`. |
| Backend — FSM | `backend/app/ml/endpoint_fsm.py` | Dùng `self.smoothed_motion`, sửa điều kiện `STARTING → SIGNING` chống false trigger 1-frame. |
| Backend — Session Mode A & B | `backend/app/services/isolated_session.py`, `backend/app/services/continuous_session.py` | Truyền `aspect_ratio = H / W`, lưu chuỗi motion theo frame để cắt biên tĩnh thừa (giữ biên hold 6f). |
| Backend — Landmark Extractor & Config | `backend/app/ml/landmark_extractor.py`, `config/runtime.yaml`, `backend/app/config/settings.py` | Điều chỉnh confidence tracking (`0.4`), đồng bộ cấu hình YAML. |
| Frontend — Capture & Stream | `frontend/src/App.tsx`, `frontend/src/hooks/useRecognitionSocket.ts` | Nâng `targetW` từ `384` lên `640` (hoặc `512–640` theo aspect ratio) và JPEG quality từ `0.6` lên `0.80`. |
| Unit Tests | `backend/tests/test_modes_and_parity.py`, `backend/tests/test_holistic_and_signing_space.py`, `backend/tests/test_endpoint_fsm.py` | Bổ sung test hồi quy cho aspect ratio 4:3/16:9, jitter tĩnh không gây kẹt FSM, idle không false-trigger. |

**Cam kết KHÔNG thay đổi:** `backend/app/ml/model.py`, `backend/app/ml/model_loader.py`, checkpoint `best_seed_20260827.pt`, `label_map.json`, thiết kế giao diện UI.

---

## 10. Kế hoạch kiểm thử hồi quy sau khi sửa

Sau khi áp dụng lần lượt các bản sửa lỗi (theo đúng thứ tự P0 → P1 → P2), thực hiện kiểm thử nghiệm thu bắt buộc:
1. **Unit Tests hệ thống:** `pytest backend/tests -v` đạt **100% PASS** (30 test cũ + bộ test hồi quy mới cho FIX-01..FIX-06).
2. **Frontend Build:** `cd frontend && npm run build` đạt **0 lỗi TypeScript / Vite**.
3. **Kiểm thử hồi quy Test A (Offline Parity):** Đảm bảo 8/8 video gốc `640×640` vẫn giữ nguyên **8/8 (100%) Greedy Exact Match** (không ảnh hưởng bất kỳ ký hiệu nào đang nhận diện tốt).
4. **Kiểm thử hồi quy Test B (Realtime Replay):**
   - `B5` (Mode A Auto, Webcam 4:3) & `B9` (Mode B Continuous, Webcam 4:3): đạt **>= 7/8 (87.5%)** Exact Match, **0/8** trường hợp kẹt `<no_stop>`/`<none>`.
   - `B6` (Mode A Auto, Webcam 16:9) & `B10` (Mode B Continuous, Webcam 16:9): đạt **>= 6/8–7/8 (75–87.5%)**, **0/8** trường hợp kẹt.
   - `B3` (Mode A Manual, tay vẫn trong khung hình): **0/8** trường hợp phát sinh từ ảo ở đuôi tĩnh.
5. **Kiểm thử hồi quy Test D (Robustness):**
   - `D4` (Idle hands): **0/4** false trigger và **0/4** từ ảo `Cái quần`.
   - `D3` (Multi-sign & Repeated trên Webcam 4:3): WER giảm từ **35.18%** xuống **<= 15%**; nhận diện đúng các chuỗi lặp (`Con bò Con bò`, `Mưa Mưa`, `Bố Bố`).
   - **Hiệu năng:** MediaPipe CPU trung bình `<= 30 ms/frame`, FPS thực tế `>= 25 FPS`, độ trễ xác nhận kết quả `<= 1.8s` (Continuous) và `<= 1.0s` (Isolated Auto).
6. **Kiểm thử webcam trực tiếp với người dùng thật:** Thực hiện bảng ghi nhận 3 lượt/gloss theo mẫu `webcam_test_blank_template.csv` trong Notebook 21.

---

## 11. Những vấn đề chưa thể xác minh

1. **Tác động của webcam vật lý trực tiếp & điều kiện môi trường thực tế:** Môi trường kiểm thử hiện tại không có thiết bị webcam vật lý và người ký trực tiếp nên toàn bộ Test B/D sử dụng phương pháp **video replay** từ 8 video mẫu gốc. Các yếu tố thực tế như ánh sáng yếu (gây motion blur và ép webcam tụt FPS xuống 15 FPS), góc nghiêng camera, phông nền phức tạp có người phía sau chưa được đo trực tiếp.
2. **Domain Shift với người ký mới (Unseen Signer ngoài 26 người ký VSL400):** 8 video mẫu trong notebook thuộc tập Test (3 người ký `001`, `014`, `028` chưa thấy khi train, nhưng cùng phòng thu và quy chuẩn quay `front_view` của bộ VSL400). Khả năng tổng quát hóa khi người dùng mới tự thực hiện ký hiệu có sai khác nhỏ về tốc độ, biên độ tay hoặc góc đặt webcam cần được đo tiếp sau khi sửa pipeline.
3. **So sánh trực tiếp tensor đặc trưng giữa `MediaPipe Tasks` (lúc tạo tập NPZ huấn luyện) và `mp.solutions` (runtime):** Do bộ tệp NPZ keypoint gốc (`VSL400_SHARED_KEYPOINTS_V1`) nằm trên Kaggle và không có sẵn trong workspace cục bộ, chúng tôi chỉ đối chiếu được thông qua các thống kê QA trong `CATALOG_CSV` (`valid_left`, `valid_right`) và đầu ra suy luận. Cần xác minh thêm liệu việc chuyển sang MediaPipe Tasks có giải quyết nốt 1/8 trường hợp biên (`Nhà trọ` trên khung hình rộng) hay không.

---

## 12. Kết luận và phương án được khuyến nghị

- **Kết luận khoa học:** Mô hình `HandsTCNTransformerCTC` (`best_seed_20260827.pt`) **KHÔNG bị lỗi** và **KHÔNG cần huấn luyện lại hay thay đổi checkpoint**. Bằng chứng thực nghiệm khẳng định hệ thống đạt **8/8 (100%)** khi chạy Offline trên video gốc `640×640`, nhưng sụp đổ xuống **1/8 (12.5%)** ở webcam `4:3` và **0/8 (0%)** ở webcam `16:9` do **chuỗi lỗi tích lũy thuần túy trên pipeline triển khai realtime**: (1) méo tỷ lệ khung hình 4:3/16:9 vs 1:1, (2) frontend ép nén `384px JPEG 0.6`, (3) kẹt FSM/Auto-Stop vì nhiễu 3D `motion_energy` chưa lọc, (4) lỗi logic `STARTING → SIGNING` tạo từ ảo `Cái quần`, và (5) ngưỡng `check_signing_space` quá rộng.
- **Phương án khuyến nghị:** Triển khai ngay nhóm giải pháp **P0 (`FIX-01`, `FIX-02`, `FIX-03`)** kết hợp **P1 (`FIX-04`, `FIX-05`, `FIX-06`)** trên pipeline triển khai hiện tại, sau đó chạy lại toàn bộ bộ kiểm thử hồi quy `audit_diagnostics/` và nghiệm thu trên webcam trực tiếp theo bảng mẫu Notebook 21.
