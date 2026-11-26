"""End-to-End Webcam Evaluation Summary Tool for Thesis Section 5.6.1 (Table 5.4).

Reads lightweight JSONL telemetry files emitted automatically by ContinuousRecognitionSession
under logs/eval_sessions/eval_*.jsonl and computes:
  1. Gloss-level Word Error Rate (WER = (S + D + I) / N) vs scripted ground truth
  2. Exact Sequence Match (ESM %)
  3. MediaPipe landmark extraction latency per frame (Median / P95, ms)
  4. Model inference + CTC decode latency per utterance (Median / P95, ms)
  5. Gemini API translation latency (Median / P95, ms)
  6. End-to-end latency from end of signing to displayed sentence (Median / P95, ms)
  7. Effective processing frame rate (FPS: mean & median)

Also exports a CSV of (reference_gloss, predicted_gloss, gemini_sentence) to reuse in Section 5.6.2!

Usage examples:
  1) Inspect the latest webcam session (latencies, FPS, and captured trials):
     .venv\\Scripts\\python.exe scripts\\evaluate_webcam_5_6_1.py --latest

  2) Compute full Table 5.4 metrics against a test script JSON:
     .venv\\Scripts\\python.exe scripts\\evaluate_webcam_5_6_1.py --script scripts\\webcam_script_template.json --latest

  3) Aggregate across multiple participant sessions (e.g., 2 or 3 people):
     .venv\\Scripts\\python.exe scripts\\evaluate_webcam_5_6_1.py --script scripts\\webcam_script_template.json --logs logs\\eval_sessions\\eval_2026*.jsonl
"""

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def levenshtein_sdi(ref: List[str], hyp: List[str]) -> Tuple[int, int, int]:
    """Computes exact token-level (substitutions, deletions, insertions) via DP."""
    n, m = len(ref), len(hyp)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)  # cost, S, D, I
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1].strip().lower() == hyp[j - 1].strip().lower():
                dp[i][j] = dp[i - 1][j - 1]
            else:
                c_sub, s_sub, d_sub, i_sub = dp[i - 1][j - 1]
                c_del, s_del, d_del, i_del = dp[i - 1][j]
                c_ins, s_ins, d_ins, i_ins = dp[i][j - 1]
                cand_sub = (c_sub + 1, s_sub + 1, d_sub, i_sub)
                cand_del = (c_del + 1, s_del, d_del + 1, i_del)
                cand_ins = (c_ins + 1, s_ins, d_ins, i_ins + 1)
                dp[i][j] = min(cand_sub, cand_del, cand_ins, key=lambda x: x[0])

    _, s, d, ins = dp[n][m]
    return s, d, ins


def compute_wer_esm(
    refs: List[List[str]], hyps: List[List[str]]
) -> Dict[str, Any]:
    assert len(refs) == len(hyps), f"Trial count mismatch: {len(refs)} refs vs {len(hyps)} hyps"
    total_s = total_d = total_i = total_n = exact = 0
    per_trial = []
    for idx, (r, h) in enumerate(zip(refs, hyps), start=1):
        s, d, ins = levenshtein_sdi(r, h)
        total_s += s
        total_d += d
        total_i += ins
        total_n += len(r)
        is_exact = [x.strip().lower() for x in r] == [x.strip().lower() for x in h]
        if is_exact:
            exact += 1
        per_trial.append(
            {
                "trial": idx,
                "ref": r,
                "hyp": h,
                "S": s,
                "D": d,
                "I": ins,
                "exact": is_exact,
            }
        )
    wer = ((total_s + total_d + total_i) / total_n * 100.0) if total_n > 0 else 0.0
    esm = (exact / len(refs) * 100.0) if refs else 0.0
    return {
        "num_trials": len(refs),
        "total_ref_glosses": total_n,
        "S": total_s,
        "D": total_d,
        "I": total_i,
        "wer_pct": round(wer, 2),
        "esm_pct": round(esm, 2),
        "exact_count": exact,
        "per_trial": per_trial,
    }


def pct_stats(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=np.float64)
    return (
        round(float(np.median(arr)), 2),
        round(float(np.percentile(arr, 95)), 2),
        round(float(np.mean(arr)), 2),
    )


def parse_session_log(path: Path) -> Dict[str, Any]:
    """Parses one eval_*.jsonl session log into frame latencies, commits, and reset-delimited trials."""
    mediapipe_ms: List[float] = []
    preprocessing_ms: List[float] = []
    fps_vals: List[float] = []
    model_ms: List[float] = []
    gemini_ms: List[float] = []
    e2e_ms: List[float] = []
    trials: List[Dict[str, Any]] = []

    current_trial_commits: List[Dict[str, Any]] = []

    def finalize_trial():
        nonlocal current_trial_commits
        if not current_trial_commits:
            return
        last_c = current_trial_commits[-1]
        trials.append(
            {
                "source_log": path.name,
                "predicted_gloss": last_c.get("final_tokens", []),
                "predicted_sentence": last_c.get("final_sentence", ""),
                "translation_status": last_c.get("translation_status", ""),
                "num_sub_commits": len(current_trial_commits),
                "final_e2e_ms": last_c.get("e2e_from_sign_end_ms", 0.0),
                "final_llm_ms": last_c.get("llm_ms", 0.0),
                "final_model_ms": last_c.get("model_total_ms", 0.0),
            }
        )
        current_trial_commits = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            kind = ev.get("event")
            if kind == "frame":
                mp_v = float(ev.get("mediapipe_ms", 0.0))
                if mp_v > 0:
                    mediapipe_ms.append(mp_v)
                pr_v = float(ev.get("preprocessing_ms", 0.0))
                if pr_v >= 0:
                    preprocessing_ms.append(pr_v)
                fps_v = float(ev.get("fps", 0.0))
                if fps_v > 1.0:
                    fps_vals.append(fps_v)
            elif kind == "commit":
                current_trial_commits.append(ev)
                m_v = float(ev.get("model_total_ms", 0.0))
                if m_v > 0:
                    model_ms.append(m_v)
                if ev.get("translation_status") == "gemini":
                    g_v = float(ev.get("llm_ms", 0.0))
                    if g_v > 0:
                        gemini_ms.append(g_v)
                e_v = float(ev.get("e2e_from_sign_end_ms", 0.0))
                if e_v > 0:
                    e2e_ms.append(e_v)
            elif kind == "reset":
                finalize_trial()

    finalize_trial()
    return {
        "mediapipe_ms": mediapipe_ms,
        "preprocessing_ms": preprocessing_ms,
        "fps": fps_vals,
        "model_ms": model_ms,
        "gemini_ms": gemini_ms,
        "e2e_ms": e2e_ms,
        "trials": trials,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute Section 5.6.1 (Table 5.4) webcam evaluation metrics.")
    parser.add_argument("--logs", nargs="*", help="One or more eval_*.jsonl files (supports glob patterns)")
    parser.add_argument("--latest", action="store_true", help="Automatically use the most recent eval_*.jsonl log with commits")
    parser.add_argument("--script", type=str, default=None, help="Path to ground-truth script JSON (e.g. scripts/webcam_script_template.json)")
    parser.add_argument("--export-csv", type=str, default="logs/eval_translations_for_5_6_2.csv", help="CSV export path for Section 5.6.2")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    eval_dir = project_root / "logs" / "eval_sessions"

    log_paths: List[Path] = []
    if args.logs:
        for pat in args.logs:
            matched = [Path(p) for p in sorted(glob.glob(pat))]
            if matched:
                log_paths.extend(matched)
            elif Path(pat).is_file():
                log_paths.append(Path(pat))
    elif args.latest or not args.logs:
        candidates = sorted(eval_dir.glob("eval_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        for c in candidates:
            parsed_chk = parse_session_log(c)
            if parsed_chk["trials"]:
                log_paths = [c]
                break
        if not log_paths and candidates:
            log_paths = [candidates[0]]

    if not log_paths:
        print("Không tìm thấy tệp log nào trong logs/eval_sessions/. Hãy chạy ứng dụng Chế độ B và ký ít nhất 1 chuỗi trước.")
        sys.exit(1)

    all_mp, all_prep, all_fps, all_model, all_gemini, all_e2e = [], [], [], [], [], []
    all_trials: List[Dict[str, Any]] = []

    print(f"Đang đọc {len(log_paths)} tệp log kiểm thử:")
    for lp in log_paths:
        print(f"  - {lp}")
        res = parse_session_log(lp)
        all_mp.extend(res["mediapipe_ms"])
        all_prep.extend(res["preprocessing_ms"])
        all_fps.extend(res["fps"])
        all_model.extend(res["model_ms"])
        all_gemini.extend(res["gemini_ms"])
        all_e2e.extend(res["e2e_ms"])
        all_trials.extend(res["trials"])

    mp_p50, mp_p95, _ = pct_stats(all_mp)
    mod_p50, mod_p95, _ = pct_stats(all_model)
    gem_p50, gem_p95, _ = pct_stats(all_gemini)
    e2e_p50, e2e_p95, _ = pct_stats(all_e2e)
    fps_p50, _, fps_mean = pct_stats(all_fps)

    print(f"\n=== DANH SÁCH {len(all_trials)} CHUỖI ĐÃ GHI NHẬN ===")
    for i, tr in enumerate(all_trials, start=1):
        print(
            f"  #{i:02d} [{tr['translation_status']:14s}] "
            f"Gloss: {' -> '.join(tr['predicted_gloss'])}  |  Câu: \"{tr['predicted_sentence']}\""
        )

    wer_summary = None
    ref_trials = []
    if args.script:
        with open(args.script, "r", encoding="utf-8") as sf:
            sdata = json.load(sf)
        ref_trials = sdata.get("trials", [])
        n_ref = len(ref_trials)
        if n_ref > 0 and len(all_trials) > 0:
            if len(all_trials) % n_ref == 0:
                reps = len(all_trials) // n_ref
                expanded_refs = [t["reference_gloss"] for _ in range(reps) for t in ref_trials]
                expanded_ref_sents = [t.get("reference_sentence", "") for _ in range(reps) for t in ref_trials]
                hyps = [t["predicted_gloss"] for t in all_trials]
                wer_summary = compute_wer_esm(expanded_refs, hyps)
                print(f"\nKhớp tự động {reps} lượt người ký × {n_ref} chuỗi kịch bản = {len(all_trials)} chuỗi.")
            else:
                n_cmp = min(len(all_trials), n_ref)
                print(
                    f"\n[LƯU Ý] Số chuỗi ghi nhận ({len(all_trials)}) khác số chuỗi kịch bản ({n_ref}). "
                    f"Tạm so khớp {n_cmp} chuỗi đầu tiên:"
                )
                expanded_refs = [ref_trials[i]["reference_gloss"] for i in range(n_cmp)]
                expanded_ref_sents = [ref_trials[i].get("reference_sentence", "") for i in range(n_cmp)]
                hyps = [all_trials[i]["predicted_gloss"] for i in range(n_cmp)]
                wer_summary = compute_wer_esm(expanded_refs, hyps)

            # Export CSV for Section 5.6.2 (Table 5.5)
            csv_path = project_root / args.export_csv
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(csv_path, "w", encoding="utf-8-sig", newline="") as cf:
                writer = csv.writer(cf)
                writer.writerow(["trial", "reference_gloss", "reference_sentence", "predicted_gloss", "predicted_sentence", "translation_status", "exact_match"])
                for idx, row in enumerate(wer_summary["per_trial"]):
                    tr = all_trials[idx]
                    writer.writerow([
                        idx + 1,
                        " -> ".join(row["ref"]),
                        expanded_ref_sents[idx],
                        " -> ".join(row["hyp"]),
                        tr["predicted_sentence"],
                        tr["translation_status"],
                        row["exact"],
                    ])
            print(f"Đã xuất bảng đối chiếu chuỗi + câu dịch phục vụ Mục 5.6.2 tại: {csv_path}")

    print("\n" + "=" * 72)
    print("BẢNG 5.4 — KẾT QUẢ KIỂM THỬ ĐẦU CUỐI HỆ THỐNG TRÊN LUỒNG WEBCAM")
    print("=" * 72)
    if wer_summary:
        print(f"1. WER trên webcam (S={wer_summary['S']}, D={wer_summary['D']}, I={wer_summary['I']} / N={wer_summary['total_ref_glosses']}): {wer_summary['wer_pct']:.2f}%")
        print(f"2. Exact Sequence Match ({wer_summary['exact_count']}/{wer_summary['num_trials']} chuỗi):       {wer_summary['esm_pct']:.2f}%")
    else:
        print("1. WER trên webcam:                     (chưa truyền --script kịch bản)")
        print("2. Exact Sequence Match:                (chưa truyền --script kịch bản)")
    print(f"3. Độ trễ trích xuất điểm mốc / khung (P50 / P95, n={len(all_mp)}): {mp_p50:.2f} / {mp_p95:.2f} ms")
    print(f"4. Độ trễ mô hình mỗi đoạn (P50 / P95, n={len(all_model)}):          {mod_p50:.2f} / {mod_p95:.2f} ms")
    print(f"5. Độ trễ Gemini API (P50 / P95, n={len(all_gemini)}):               {gem_p50:.2f} / {gem_p95:.2f} ms")
    print(f"6. Độ trễ kết thúc ký -> hiện câu (P50 / P95, n={len(all_e2e)}):     {e2e_p50:.2f} / {e2e_p95:.2f} ms")
    print(f"7. Tốc độ khung hình xử lý được (TB / Trung vị):         {fps_mean:.1f} / {fps_p50:.1f} khung/giây")
    print("=" * 72)


if __name__ == "__main__":
    main()
