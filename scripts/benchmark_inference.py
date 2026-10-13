"""Component-wise and End-to-End Latency Benchmark for VSL Recognition System.

Measures independently:
1. Model forward latency (PyTorch inference on target device)
2. Preprocessing latency (Shoulder anchor + Hand selection + Delta velocity + Valid mask)
3. MediaPipe landmark extraction latency
4. CTC Greedy Decoder latency
5. End-to-end pipeline latency & effective FPS
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch

# Ensure workspace root in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.config.settings import settings
from backend.app.ml.model_loader import ModelLoader
from backend.app.ml.preprocessing import VSLPreprocessor
from backend.app.ml.ctc_decoder import CTCDecoder
from backend.app.ml.landmark_extractor import LandmarkExtractor


def run_benchmark(n_runs: int = 100):
    print("=" * 75)
    print("VSL REALTIME SYSTEM LATENCY BENCHMARK")
    print("=" * 75)

    # 1. Model Latency Benchmark
    print("\n[1/4] Benchmarking Locked Model (HandsTCNTransformerCTC)...")
    model, device = ModelLoader.get_loaded_model()
    print(f"Device: {device}")

    for seq_len in [30, 45, 60, 90]:
        dummy_input = torch.randn(1, seq_len, settings.INPUT_DIM, device=device)

        # Warmup
        for _ in range(15):
            with torch.inference_mode():
                _ = model(dummy_input)

        timings = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            with torch.inference_mode():
                _ = model(dummy_input)
            timings.append((time.perf_counter() - t0) * 1000.0)

        mean_ms = np.mean(timings)
        med_ms = np.median(timings)
        p95_ms = np.percentile(timings, 95)
        print(f"  Seq T={seq_len:2d} frames: Mean={mean_ms:5.2f} ms | Median={med_ms:5.2f} ms | P95={p95_ms:5.2f} ms | Model FPS={1000.0/mean_ms:6.1f}")

    # 2. Preprocessing Latency Benchmark
    print("\n[2/4] Benchmarking Preprocessing Parity Pipeline...")
    prep = VSLPreprocessor()
    pose_lms = [{"x": 0.5, "y": 0.5, "z": 0.0, "visibility": 0.9, "presence": 0.9} for _ in range(33)]
    left_hand = [{"x": 0.4, "y": 0.6, "z": 0.0} for _ in range(21)]
    right_hand = [{"x": 0.6, "y": 0.6, "z": 0.0} for _ in range(21)]

    prep_timings = []
    for _ in range(n_runs * 2):
        t0 = time.perf_counter()
        _ = prep.process_frame(left_hand, right_hand, pose_lms)
        prep_timings.append((time.perf_counter() - t0) * 1000.0)

    prep_mean = np.mean(prep_timings)
    prep_med = np.median(prep_timings)
    prep_p95 = np.percentile(prep_timings, 95)
    print(f"  Preprocessing: Mean={prep_mean:5.3f} ms | Median={prep_med:5.3f} ms | P95={prep_p95:5.3f} ms")

    # 3. CTC Greedy Decoder Latency
    print("\n[3/4] Benchmarking CTC Greedy Decoder...")
    decoder = CTCDecoder(label_map_path=settings.LABEL_MAP_PATH, blank_id=settings.BLANK_ID)
    dummy_logits = torch.randn(60, settings.NUM_CLASSES)

    decode_timings = []
    for _ in range(n_runs * 2):
        t0 = time.perf_counter()
        _ = decoder.decode_greedy(dummy_logits)
        decode_timings.append((time.perf_counter() - t0) * 1000.0)

    dec_mean = np.mean(decode_timings)
    dec_med = np.median(decode_timings)
    dec_p95 = np.percentile(decode_timings, 95)
    print(f"  CTC Decoder:   Mean={dec_mean:5.3f} ms | Median={dec_med:5.3f} ms | P95={dec_p95:5.3f} ms")

    # 4. MediaPipe Landmark Extraction Latency
    print("\n[4/4] Benchmarking MediaPipe Landmark Extractor...")
    extractor = LandmarkExtractor()
    dummy_frame = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)

    mp_timings = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = extractor.extract(dummy_frame, draw_landmarks=False)
        mp_timings.append((time.perf_counter() - t0) * 1000.0)

    mp_mean = np.mean(mp_timings)
    mp_med = np.median(mp_timings)
    mp_p95 = np.percentile(mp_timings, 95)
    print(f"  MediaPipe:     Mean={mp_mean:5.2f} ms | Median={mp_med:5.2f} ms | P95={mp_p95:5.2f} ms")

    # Summary
    print("\n" + "=" * 75)
    print("PIPELINE LATENCY SUMMARY")
    print("=" * 75)
    print(f"MediaPipe Extraction:       {mp_mean:6.2f} ms")
    print(f"Preprocessing & Deltas:     {prep_mean:6.3f} ms")
    print(f"Model Forward (T=60):       {np.mean(timings):6.2f} ms")
    print(f"CTC Greedy Decoding:        {dec_mean:6.3f} ms")
    total_est = mp_mean + prep_mean + np.mean(timings) + dec_mean
    print(f"Total Per-Frame Latency:    {total_est:6.2f} ms (~{1000.0/total_est:4.1f} Realtime FPS)")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmark(50)

