"""Section 5.6.2 (Table 5.5) Language Translation Evaluation Script.

Evaluates the production LanguageService (Gemini 3.5 Flash Lite, temperature=0.2) on 50 VSL400
gloss sequences (lengths 1..5, 10 sequences per length, total 150 glosses) across two input groups:
  (1) Gold gloss sequences (exact ground truth)
  (2) Noisy / predicted gloss sequences (combining the 8 real webcam predictions from Section 5.6.1
      + reproducible recognition errors matching the locked model's ~30% WER on the remaining 42 sequences)

Computes exact Popovic (2015) chrF (char n-gram 1..6, beta=2, whitespace ignored as in sacreBLEU)
and exports an Excel-ready CSV (`logs/eval_5_6_2_rating.csv`) for independent 2-rater scoring
(adequacy 1-5, fluency 1-5, hallucination 0/1).

Usage:
  1) Run live Gemini translations & compute chrF + export rating CSV:
     .venv\\Scripts\\python.exe scripts\\evaluate_translation_5_6_2.py --run

  2) Summarize Table 5.5 after the 2 raters fill in logs/eval_5_6_2_rating.csv:
     .venv\\Scripts\\python.exe scripts\\evaluate_translation_5_6_2.py --summarize logs\\eval_5_6_2_rating.csv
"""

import argparse
import asyncio
from collections import Counter
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.app.config.settings import settings
from backend.app.services.language_service import LanguageService, SYSTEM_PROMPT, FEW_SHOT_EXAMPLES
from scripts.evaluate_webcam_5_6_1 import levenshtein_sdi
import httpx


# 50 VSL400 gloss sequences (10 each of length 1, 2, 3, 4, 5 -> 150 glosses total).
# Every gloss strictly exists in VSL400 label_map.json.
# Trials 1..8 incorporate the exact 8 real webcam test trials from Section 5.6.1;
# Trials 9..50 cover realistic VSL word orders (S-O-V, Time-S-O-V, Place-fronting) and
# simulated recognition errors (substitutions/deletions/insertions) calibrated to ~30% WER.
DATASET_50: List[Dict[str, Any]] = [
    # --- Length 1 (10 sequences) ---
    {"id": 1,  "gold": ["Bố"], "noisy": ["Bố"], "ref": "Bố.", "source": "webcam"},
    {"id": 2,  "gold": ["Chụp hình"], "noisy": ["Chụp hình"], "ref": "Chụp hình.", "source": "webcam"},
    {"id": 3,  "gold": ["Giúp đỡ"], "noisy": ["Giúp đỡ"], "ref": "Giúp đỡ.", "source": "webcam"},
    {"id": 4,  "gold": ["Cảm ơn"], "noisy": ["Cảm ơn"], "ref": "Cảm ơn.", "source": "sim"},
    {"id": 5,  "gold": ["Xin lỗi"], "noisy": ["Xin lỗi"], "ref": "Xin lỗi.", "source": "sim"},
    {"id": 6,  "gold": ["Bệnh viện"], "noisy": ["Bệnh viện"], "ref": "Bệnh viện.", "source": "sim"},
    {"id": 7,  "gold": ["Ô tô"], "noisy": ["Ô tô"], "ref": "Ô tô.", "source": "sim"},
    {"id": 8,  "gold": ["Ấm"], "noisy": ["Chật"], "ref": "Ấm.", "source": "sim"},
    {"id": 9,  "gold": ["Mưa"], "noisy": ["Gió"], "ref": "Trời mưa.", "source": "sim"},
    {"id": 10, "gold": ["Ngủ"], "noisy": ["Nghỉ ngơi"], "ref": "Ngủ.", "source": "sim"},

    # --- Length 2 (10 sequences) ---
    {"id": 11, "gold": ["Bố", "Chụp hình"], "noisy": ["Bố", "Chụp hình"], "ref": "Bố chụp hình.", "source": "webcam"},
    {"id": 12, "gold": ["Bố", "Giúp đỡ"], "noisy": ["Bố", "Giúp đỡ"], "ref": "Bố giúp đỡ.", "source": "webcam"},
    {"id": 13, "gold": ["Giúp đỡ", "Chụp hình"], "noisy": ["Giúp đỡ", "Chụp hình"], "ref": "Giúp đỡ chụp hình.", "source": "webcam"},
    {"id": 14, "gold": ["Anh", "Đọc"], "noisy": ["Anh", "Đọc"], "ref": "Anh đang đọc.", "source": "sim"},
    {"id": 15, "gold": ["Mẹ", "Ngủ"], "noisy": ["Mẹ", "Ngủ"], "ref": "Mẹ đang ngủ.", "source": "sim"},
    {"id": 16, "gold": ["Chợ", "Đi"], "noisy": ["Chợ", "Đi"], "ref": "Đi chợ.", "source": "sim"},
    {"id": 17, "gold": ["Phở", "Ăn"], "noisy": ["Bún", "Ăn"], "ref": "Ăn phở.", "source": "sim"},
    {"id": 18, "gold": ["Cà phê", "Uống"], "noisy": ["Trà", "Uống"], "ref": "Uống cà phê.", "source": "sim"},
    {"id": 19, "gold": ["Học sinh", "Học"], "noisy": ["Sinh viên", "Học"], "ref": "Học sinh học bài.", "source": "sim"},
    {"id": 20, "gold": ["Bác sĩ", "Khám bệnh"], "noisy": ["Bác sĩ"], "ref": "Bác sĩ khám bệnh.", "source": "sim"},

    # --- Length 3 (10 sequences) ---
    {"id": 21, "gold": ["Bố", "Giúp đỡ", "Chụp hình"], "noisy": ["Bóng chuyền"], "ref": "Bố giúp đỡ chụp hình.", "source": "webcam"},
    {"id": 22, "gold": ["Bố", "Chụp hình", "Giúp đỡ"], "noisy": ["Bố"], "ref": "Bố chụp hình và giúp đỡ.", "source": "webcam"},
    {"id": 23, "gold": ["Bố", "Chú", "Giúp đỡ"], "noisy": ["Bố", "Chú", "Giúp đỡ"], "ref": "Bố giúp đỡ chú.", "source": "sim"},
    {"id": 24, "gold": ["Mẹ", "Phở", "Nấu"], "noisy": ["Mẹ", "Phở", "Nấu"], "ref": "Mẹ nấu phở.", "source": "sim"},
    {"id": 25, "gold": ["Bố", "Chợ", "Đi"], "noisy": ["Bố", "Chợ", "Đi"], "ref": "Bố đi chợ.", "source": "sim"},
    {"id": 26, "gold": ["Anh", "Sách", "Đọc"], "noisy": ["Anh", "Vở", "Đọc"], "ref": "Anh đọc sách.", "source": "sim"},
    {"id": 27, "gold": ["Chị", "Cà phê", "Uống"], "noisy": ["Em", "Cà phê", "Uống"], "ref": "Chị uống cà phê.", "source": "sim"},
    {"id": 28, "gold": ["Chú", "Công ty", "Làm việc"], "noisy": ["Chú", "Làm việc"], "ref": "Chú làm việc ở công ty.", "source": "sim"},
    {"id": 29, "gold": ["Em", "Bóng đá", "Thích"], "noisy": ["Em", "Bóng rổ", "Thích"], "ref": "Em thích bóng đá.", "source": "sim"},
    {"id": 30, "gold": ["Bác sĩ", "Bệnh viện", "Làm việc"], "noisy": ["Y tá", "Bệnh viện", "Làm việc"], "ref": "Bác sĩ làm việc ở bệnh viện.", "source": "sim"},

    # --- Length 4 (10 sequences) ---
    {"id": 31, "gold": ["Buổi sáng", "Bố", "Chú", "Giúp đỡ"], "noisy": ["Buổi sáng", "Bố", "Chú", "Giúp đỡ"], "ref": "Buổi sáng bố giúp đỡ chú.", "source": "sim"},
    {"id": 32, "gold": ["Mẹ", "Chợ", "Đi", "Mua bán"], "noisy": ["Mẹ", "Chợ", "Đi", "Mua bán"], "ref": "Mẹ đi chợ mua bán.", "source": "sim"},
    {"id": 33, "gold": ["Chú", "Bệnh viện", "Đi", "Khám bệnh"], "noisy": ["Chú", "Bệnh viện", "Đi", "Khám bệnh"], "ref": "Chú đi khám bệnh ở bệnh viện.", "source": "sim"},
    {"id": 34, "gold": ["Buổi tối", "Anh", "Sách", "Đọc"], "noisy": ["Buổi chiều", "Anh", "Sách", "Đọc"], "ref": "Buổi tối anh đọc sách.", "source": "sim"},
    {"id": 35, "gold": ["Chủ nhật", "Bố", "Công viên", "Đi"], "noisy": ["Thứ bảy", "Bố", "Công viên", "Đi"], "ref": "Chủ nhật bố đi công viên.", "source": "sim"},
    {"id": 36, "gold": ["Mẹ", "Nhà", "Phở", "Nấu"], "noisy": ["Mẹ", "Nhà", "Nấu"], "ref": "Mẹ nấu phở ở nhà.", "source": "sim"},
    {"id": 37, "gold": ["Sinh viên", "Trường Đại học", "Đi", "Học"], "noisy": ["Sinh viên", "Trường học", "Đi", "Học"], "ref": "Sinh viên đi học ở trường đại học.", "source": "sim"},
    {"id": 38, "gold": ["Buổi trưa", "Chị", "Nhà hàng", "Ăn"], "noisy": ["Buổi trưa", "Nhà hàng", "Ăn"], "ref": "Buổi trưa chị ăn ở nhà hàng.", "source": "sim"},
    {"id": 39, "gold": ["Bố", "Xe máy", "Công ty", "Đi"], "noisy": ["Bố", "Xe đạp", "Công ty", "Đi"], "ref": "Bố đi xe máy đến công ty.", "source": "sim"},
    {"id": 40, "gold": ["Em", "Buổi tối", "Nhà", "Làm bài tập"], "noisy": ["Em", "Buổi tối", "Làm bài tập", "Đọc"], "ref": "Buổi tối em làm bài tập ở nhà.", "source": "sim"},

    # --- Length 5 (10 sequences) ---
    {"id": 41, "gold": ["Buổi sáng", "Mẹ", "Chợ", "Mua bán", "Cà phê"], "noisy": ["Buổi sáng", "Mẹ", "Chợ", "Mua bán", "Cà phê"], "ref": "Buổi sáng mẹ đi chợ mua cà phê.", "source": "sim"},
    {"id": 42, "gold": ["Chủ nhật", "Bố", "Chú", "Công viên", "Chơi cờ"], "noisy": ["Chủ nhật", "Bố", "Chú", "Công viên", "Chơi cờ"], "ref": "Chủ nhật bố và chú chơi cờ ở công viên.", "source": "sim"},
    {"id": 43, "gold": ["Buổi tối", "Bố", "Mẹ", "Nhà", "Ăn"], "noisy": ["Buổi tối", "Bố", "Nhà", "Ăn"], "ref": "Buổi tối bố mẹ ăn ở nhà.", "source": "sim"},
    {"id": 44, "gold": ["Thứ hai", "Anh", "Xe buýt", "Trường học", "Đi"], "noisy": ["Thứ ba", "Anh", "Xe buýt", "Trường học", "Đi"], "ref": "Thứ hai anh đi xe buýt đến trường học.", "source": "sim"},
    {"id": 45, "gold": ["Buổi chiều", "Chị", "Siêu thị", "Đi", "Mua bán"], "noisy": ["Buổi chiều", "Chị", "Chợ", "Mua bán"], "ref": "Buổi chiều chị đi siêu thị mua bán.", "source": "sim"},
    {"id": 46, "gold": ["Mùa hè", "Anh", "Chị", "Bơi lội", "Thích"], "noisy": ["Mùa xuân", "Anh", "Chị", "Bơi lội"], "ref": "Mùa hè anh và chị thích bơi lội.", "source": "sim"},
    {"id": 47, "gold": ["Buổi sáng", "Chú", "Quán cà phê", "Cà phê", "Uống"], "noisy": ["Buổi sáng", "Quán cà phê", "Trà", "Uống"], "ref": "Buổi sáng chú uống cà phê ở quán cà phê.", "source": "sim"},
    {"id": 48, "gold": ["Thứ bảy", "Em", "Nhà sách", "Sách", "Đọc"], "noisy": ["Thứ bảy", "Em", "Nhà sách", "Vở", "Viết"], "ref": "Thứ bảy em đọc sách ở nhà sách.", "source": "sim"},
    {"id": 49, "gold": ["Buổi tối", "Mẹ", "Bố", "Giúp đỡ", "Rửa chén"], "noisy": ["Buổi tối", "Mẹ", "Giúp đỡ", "Giặt đồ"], "ref": "Buổi tối mẹ giúp đỡ bố rửa chén.", "source": "sim"},
    {"id": 50, "gold": ["Tết Âm lịch", "Ông nội", "Bà nội", "Nhà", "Bánh chưng"], "noisy": ["Tết Âm lịch", "Ông ngoại", "Nhà", "Bánh tét"], "ref": "Tết Âm lịch ông nội và bà nội ở nhà có bánh chưng.", "source": "sim"},
]


def _char_ngrams(text: str, n: int) -> Counter:
    s = "".join(text.lower().split())
    if len(s) < n:
        return Counter()
    return Counter(s[i : i + n] for i in range(len(s) - n + 1))


def compute_chrf(hyps: List[str], refs: List[str], char_order: int = 6, beta: float = 2.0) -> float:
    """Exact Popovic (2015) corpus-level chrF (0..100 scale, whitespace ignored, char n-gram 1..6, beta=2)."""
    total_f = 0.0
    valid_orders = 0
    beta2 = beta * beta
    for n in range(1, char_order + 1):
        matches = 0
        hyp_total = 0
        ref_total = 0
        for h, r in zip(hyps, refs):
            hc = _char_ngrams(h, n)
            rc = _char_ngrams(r, n)
            hyp_total += sum(hc.values())
            ref_total += sum(rc.values())
            for k, v in hc.items():
                if k in rc:
                    matches += min(v, rc[k])
        if hyp_total == 0 or ref_total == 0:
            continue
        p = matches / hyp_total
        rec = matches / ref_total
        f = ((1.0 + beta2) * p * rec / (beta2 * p + rec)) if (p + rec) > 0 else 0.0
        total_f += f
        valid_orders += 1
    return round((total_f / valid_orders) * 100.0, 2) if valid_orders > 0 else 0.0


def sentence_chrf(hyp: str, ref: str) -> float:
    return compute_chrf([hyp], [ref])


def verify_dataset_against_label_map():
    with open(settings.LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)["label_to_id"]
    total_s = total_d = total_i = total_n = 0
    for item in DATASET_50:
        for g in item["gold"] + item["noisy"]:
            assert g in label_to_id, f"Gloss {g!r} in item #{item['id']} not found in label_map.json!"
        s, d, ins = levenshtein_sdi(item["gold"], item["noisy"])
        total_s += s
        total_d += d
        total_i += ins
        total_n += len(item["gold"])
    wer = (total_s + total_d + total_i) / total_n * 100.0
    print(f"Verified 50 sequences ({total_n} gold glosses). Group (2) input WER = {wer:.2f}% (S={total_s}, D={total_d}, I={total_i})")
    return wer, total_s, total_d, total_i, total_n


async def translate_with_retry(
    client: httpx.AsyncClient,
    model_name: str,
    api_key: str,
    glosses: List[str],
    min_interval_sec: float = 4.25,
    last_call_ts: List[float] = None,
    retries: int = 5,
) -> Tuple[str, str, float]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    user_prompt = f"{FEW_SHOT_EXAMPLES}- Input: {glosses} -> Output:"
    gen_cfg: Dict[str, Any] = {"temperature": 0.2, "maxOutputTokens": 100}
    if "flash-lite" in model_name or "gemini-3-flash" in model_name:
        gen_cfg["thinkingConfig"] = {"thinkingLevel": "minimal"}
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_prompt}]}],
        "generationConfig": gen_cfg,
    }

    last_err = ""
    for attempt in range(retries):
        if last_call_ts is not None:
            wait_s = min_interval_sec - (time.perf_counter() - last_call_ts[0])
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            last_call_ts[0] = time.perf_counter()

        t0 = time.perf_counter()
        try:
            resp = await client.post(url, json=payload)
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            if resp.status_code == 200:
                data = resp.json()
                parts = data.get("candidates", [])[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                if text:
                    return text, "gemini", ms
                last_err = "Empty response"
            else:
                last_err = f"HTTP {resp.status_code}"
        except Exception as e:
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            last_err = type(e).__name__

        backoff = 60.0 if "429" in last_err else (5.0 * (attempt + 1))
        print(f"  [retry {attempt+1}/{retries}] {last_err} -> waiting {backoff:.0f}s...", flush=True)
        await asyncio.sleep(backoff)
    return "", "fallback_error", 0.0


async def run_evaluation():
    wer_in, s_in, d_in, i_in, n_in = verify_dataset_against_label_map()
    assert settings.LLM_API_KEY, "GEMINI_API_KEY not found in .env!"

    svc = LanguageService(
        enabled=True,
        provider="gemini",
        api_key=settings.LLM_API_KEY,
        model_name=settings.LLM_MODEL,
    )

    out_dir = PROJECT_ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "eval_5_6_2_cache.json"
    cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as cf:
            cache = json.load(cf)

    # Deduplicate identical gloss lists (saves ~22 API calls when gold == noisy)
    unique_inputs: List[List[str]] = []
    seen = set()
    for it in DATASET_50:
        for grp in (it["gold"], it["noisy"]):
            key = " | ".join(grp)
            if key not in seen:
                seen.add(key)
                unique_inputs.append(grp)

    todo = [g for g in unique_inputs if " | ".join(g) not in cache]
    print(
        f"Model={svc.model_name} (T=0.2, eval timeout=15s) | Total unique: {len(unique_inputs)} "
        f"| Cached: {len(unique_inputs) - len(todo)} | Remaining API calls: {len(todo)}",
        flush=True,
    )

    t_all = time.perf_counter()
    last_call_ts = [0.0]
    async with httpx.AsyncClient(timeout=15.0) as client:
        for idx, glosses in enumerate(todo, start=1):
            key = " | ".join(glosses)
            out, st, ms = await translate_with_retry(
                client,
                svc.model_name,
                svc.api_key,
                glosses,
                min_interval_sec=4.25,
                last_call_ts=last_call_ts,
            )
            if st == "gemini":
                cache[key] = {"out": out, "status": st, "ms": ms}
                with open(cache_path, "w", encoding="utf-8") as cf:
                    json.dump(cache, cf, ensure_ascii=False, indent=2)
            if idx % 10 == 0 or idx == len(todo):
                print(f"  Progress: {idx}/{len(todo)} done (last: {glosses} -> {out!r}, {st}, {ms:.0f}ms)", flush=True)

    elapsed_s = time.perf_counter() - t_all
    results: List[Dict[str, Any]] = []
    for item in DATASET_50:
        gk = " | ".join(item["gold"])
        nk = " | ".join(item["noisy"])
        g_rec = cache.get(gk, {"out": svc._fallback_format(item["gold"]), "status": "fallback_error", "ms": 0.0})
        n_rec = cache.get(nk, {"out": svc._fallback_format(item["noisy"]), "status": "fallback_error", "ms": 0.0})
        results.append(
            {
                "id": item["id"],
                "length": len(item["gold"]),
                "source": item["source"],
                "gold_gloss": item["gold"],
                "noisy_gloss": item["noisy"],
                "reference_sentence": item["ref"],
                "gold_out": g_rec["out"],
                "gold_status": g_rec["status"],
                "gold_ms": g_rec["ms"],
                "gold_chrf": sentence_chrf(g_rec["out"], item["ref"]),
                "noisy_out": n_rec["out"],
                "noisy_status": n_rec["status"],
                "noisy_ms": n_rec["ms"],
                "noisy_chrf": sentence_chrf(n_rec["out"], item["ref"]),
            }
        )

    refs = [r["reference_sentence"] for r in results]
    gold_hyps = [r["gold_out"] for r in results]
    noisy_hyps = [r["noisy_out"] for r in results]

    corpus_chrf_gold = compute_chrf(gold_hyps, refs)
    corpus_chrf_noisy = compute_chrf(noisy_hyps, refs)
    mean_chrf_gold = round(sum(r["gold_chrf"] for r in results) / len(results), 2)
    mean_chrf_noisy = round(sum(r["noisy_chrf"] for r in results) / len(results), 2)
    ok_gold = sum(1 for r in results if r["gold_status"] == "gemini")
    ok_noisy = sum(1 for r in results if r["noisy_status"] == "gemini")

    out_dir = PROJECT_ROOT / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_5_6_2_results.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(
            {
                "model": svc.model_name,
                "temperature": 0.2,
                "num_sequences": len(results),
                "total_gold_glosses": n_in,
                "group2_input_wer_pct": round(wer_in, 2),
                "group2_input_sdi": {"S": s_in, "D": d_in, "I": i_in},
                "gemini_success": {"gold": ok_gold, "noisy": ok_noisy},
                "corpus_chrf_gold": corpus_chrf_gold,
                "corpus_chrf_noisy": corpus_chrf_noisy,
                "mean_sentence_chrf_gold": mean_chrf_gold,
                "mean_sentence_chrf_noisy": mean_chrf_noisy,
                "items": results,
            },
            jf,
            ensure_ascii=False,
            indent=2,
        )

    csv_path = out_dir / "eval_5_6_2_rating.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as cf:
        w = csv.writer(cf)
        w.writerow([
            "id", "len", "gold_gloss", "reference_sentence",
            "gold_gemini_out", "gold_chrf",
            "noisy_gloss", "noisy_gemini_out", "noisy_chrf",
            "R1_gold_adequacy_1_5", "R1_gold_fluency_1_5", "R1_gold_halluc_0_1",
            "R2_gold_adequacy_1_5", "R2_gold_fluency_1_5", "R2_gold_halluc_0_1",
            "R1_noisy_adequacy_1_5", "R1_noisy_fluency_1_5", "R1_noisy_halluc_0_1",
            "R2_noisy_adequacy_1_5", "R2_noisy_fluency_1_5", "R2_noisy_halluc_0_1",
        ])
        for r in results:
            w.writerow([
                r["id"], r["length"], " -> ".join(r["gold_gloss"]), r["reference_sentence"],
                r["gold_out"], r["gold_chrf"],
                " -> ".join(r["noisy_gloss"]), r["noisy_out"], r["noisy_chrf"],
                "", "", "", "", "", "", "", "", "", "", "", "",
            ])

    print(f"\nHoàn tất 100 lượt gọi trong {elapsed_s:.1f}s (Gemini OK: Gold {ok_gold}/50, Noisy {ok_noisy}/50)")
    print("=" * 72)
    print("BẢNG 5.5 (PHẦN ĐỊNH LƯỢNG TỰ ĐỘNG — GEMINI 3.5 FLASH LITE, T=0.2)")
    print("=" * 72)
    print(f"Đầu vào nhóm (2) có lỗi nhận diện: WER = {wer_in:.2f}% (S={s_in}, D={d_in}, I={i_in} / N={n_in})")
    print(f"chrF (corpus, Popovic 2015):  Gloss đúng = {corpus_chrf_gold:.2f}  |  Gloss dự đoán = {corpus_chrf_noisy:.2f}")
    print(f"chrF (trung bình từng câu):   Gloss đúng = {mean_chrf_gold:.2f}  |  Gloss dự đoán = {mean_chrf_noisy:.2f}")
    print("=" * 72)
    print(f"Đã lưu chi tiết JSON: {json_path}")
    print(f"Đã xuất bảng chấm điểm 2 thành viên (mở bằng Excel): {csv_path}")


def summarize_ratings(csv_path: Path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    def avg_two(col1: str, col2: str) -> float:
        vals = []
        for r in rows:
            if r[col1].strip() and r[col2].strip():
                vals.append((float(r[col1]) + float(r[col2])) / 2.0)
            elif r[col1].strip():
                vals.append(float(r[col1]))
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def halluc_pct(col1: str, col2: str) -> float:
        flags = []
        for r in rows:
            v1 = int(r[col1]) if r[col1].strip() else 0
            v2 = int(r[col2]) if r[col2].strip() else v1
            flags.append((v1 + v2) / 2.0)
        return round((sum(flags) / len(flags)) * 100.0, 2) if flags else 0.0

    refs = [r["reference_sentence"] for r in rows]
    g_hyps = [r["gold_gemini_out"] for r in rows]
    n_hyps = [r["noisy_gemini_out"] for r in rows]

    print("=" * 72)
    print("BẢNG 5.5 — TỔNG HỢP ĐÁNH GIÁ CHẤT LƯỢNG DỊCH GLOSS SANG CÂU TIẾNG VIỆT")
    print("=" * 72)
    print(f"chrF so với câu tham chiếu:   Gloss đúng = {compute_chrf(g_hyps, refs):.2f}  |  Gloss dự đoán = {compute_chrf(n_hyps, refs):.2f}")
    print(f"Điểm đầy đủ ý (1-5):          Gloss đúng = {avg_two('R1_gold_adequacy_1_5', 'R2_gold_adequacy_1_5'):.2f}  |  Gloss dự đoán = {avg_two('R1_noisy_adequacy_1_5', 'R2_noisy_adequacy_1_5'):.2f}")
    print(f"Điểm tự nhiên (1-5):          Gloss đúng = {avg_two('R1_gold_fluency_1_5', 'R2_gold_fluency_1_5'):.2f}  |  Gloss dự đoán = {avg_two('R1_noisy_fluency_1_5', 'R2_noisy_fluency_1_5'):.2f}")
    print(f"Tỷ lệ ảo giác (%):            Gloss đúng = {halluc_pct('R1_gold_halluc_0_1', 'R2_gold_halluc_0_1'):.2f}% |  Gloss dự đoán = {halluc_pct('R1_noisy_halluc_0_1', 'R2_noisy_halluc_0_1'):.2f}%")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run live 50x2 Gemini translations and compute chrF")
    parser.add_argument("--summarize", type=str, default=None, help="Summarize human ratings from CSV")
    args = parser.parse_args()
    if args.summarize:
        summarize_ratings(Path(args.summarize))
    else:
        asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
