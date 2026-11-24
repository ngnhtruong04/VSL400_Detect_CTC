"""LanguageService: Optional post-processing layer to convert Vietnamese Gloss sequences into natural Vietnamese sentences."""

import logging
import re
from typing import List, Optional
import httpx

from backend.app.config.settings import settings

logger = logging.getLogger("vsl.language_service")

_SINGLE_WORD_PROPER_NOUNS = {"Mỹ", "Tết"}

_TIME_GLOSSES = {
    "buổi sáng", "buổi trưa", "buổi chiều", "buổi tối", "bây giờ", "bình minh", "hoàng hôn",
    "hôm qua", "hôm nay", "ngày mai", "chủ nhật",
    "thứ hai", "thứ ba", "thứ tư", "thứ năm", "thứ sáu", "thứ bảy",
    "tháng một", "tháng hai", "tháng ba", "tháng tư", "tháng năm", "tháng sáu",
    "tháng bảy", "tháng tám", "tháng chín", "tháng mười", "tháng mười một", "tháng mười hai",
    "mùa xuân", "mùa hè", "mùa thu", "mùa đông", "mùa mưa", "mùa khô",
    "giáng sinh", "trung thu", "tết âm lịch",
    "ngày nhà giáo việt nam", "ngày quốc tế lao động", "ngày quốc tế phụ nữ", "ngày quốc tế thiếu nhi",
}

_PERSON_GLOSSES = {
    "tôi", "anh", "chị", "em", "bố", "mẹ", "chú", "cô", "bác", "cậu", "dì",
    "ông nội", "ông ngoại", "bà nội", "bà ngoại", "cháu", "con trai", "con gái",
    "chồng", "vợ", "họ hàng", "học sinh", "sinh viên", "bác sĩ", "y tá",
    "giám đốc", "hiệu trưởng", "chủ tịch", "ca sĩ", "diễn viên", "đầu bếp",
    "nông dân", "luật sư", "kế toán", "lễ tân", "thư ký",
    "nhân viên phục vụ", "nhân viên văn phòng", "bảo vệ",
}

_PLACE_GLOSSES = {
    "bệnh viện", "chợ", "công ty", "công viên", "ngân hàng", "nhà", "nhà hàng",
    "nhà sách", "nhà trọ", "quán cà phê", "rạp chiếu phim", "siêu thị",
    "thành phố", "trường học", "trường cao đẳng", "trường đại học",
    "việt nam", "hàn quốc", "nhật bản", "thái lan", "trung quốc", "mỹ",
}

_TRANSITIVE_VERBS = {
    "giúp đỡ", "yêu thương", "ghét", "thích", "gọi", "tìm", "xem", "đọc", "viết",
    "ăn", "uống", "nấu", "nướng", "nếm", "mua bán", "mua", "cung cấp", "giới thiệu",
    "học", "khám bệnh", "khám", "bắt chước", "quan sát", "nghe", "ngửi", "cho", "xin",
    "mách", "hứa", "từ chối", "đồng ý", "cảm ơn", "xin lỗi", "đi", "nói", "nói chuyện",
    "làm bài tập", "làm việc", "chụp hình", "bắt buộc",
    "không cho", "không cần", "không nghe lời", "không quen",
    "muốn", "cần", "có", "quen", "thèm", "thử", "vâng lời",
    "thay đổi", "báo cáo", "chú ý", "hy vọng", "cảm thấy", "tiếp tục",
}

_PLACE_LOCATION_VERBS = {
    "làm việc", "học", "khám bệnh", "khám", "ăn", "uống", "ngủ", "nghỉ ngơi",
    "mua bán", "chơi cờ", "cắm trại", "bơi lội", "hát", "múa", "xem",
}


def _strip_gloss_meta(token: str) -> str:
    """Strips parenthetical VSL400 disambiguators (e.g. 'Cao (người)') and stray trailing punctuation."""
    cleaned = re.sub(r"\s*\([^)]*\)", "", token).strip()
    return cleaned.rstrip(".!?").strip()


def _reorder_vsl_to_vietnamese(tokens: List[str]) -> List[str]:
    """Deterministic VSL (S-O-V / Time-S-O-V) to Vietnamese (S-V-O) grammar reordering for offline fallback."""
    if len(tokens) < 2:
        return list(tokens)

    head: List[str] = []
    core = list(tokens)
    if core[0].lower() in _TIME_GLOSSES and len(core) >= 3:
        head = [core[0]]
        core = core[1:]

    low = [t.lower() for t in core]

    # Pattern A: [S, Place, 'Đi', V] -> [S, 'Đi', Place, V] (e.g. Mẹ Chợ Đi Mua bán -> Mẹ đi chợ mua bán)
    if (
        len(core) == 4
        and low[1] in _PLACE_GLOSSES
        and low[2] == "đi"
        and low[3] in _TRANSITIVE_VERBS
    ):
        core = [core[0], core[2], core[1], core[3]]

    # Pattern B: [S, O, V] -> [S, V, O] (e.g. Bố Chú Giúp đỡ -> Bố giúp đỡ chú; Chú Bệnh viện Khám bệnh -> Chú khám bệnh ở bệnh viện)
    elif (
        len(core) == 3
        and low[2] in _TRANSITIVE_VERBS
        and low[1] not in _TRANSITIVE_VERBS
        and low[0] not in _TRANSITIVE_VERBS
    ):
        if low[1] in _PLACE_GLOSSES and low[2] in _PLACE_LOCATION_VERBS and low[2] != "đi":
            core = [core[0], core[2], "ở", core[1]]
        else:
            core = [core[0], core[2], core[1]]

    # Pattern C: [O/Place, V] -> [V, O/Place] when first token is NOT a person (e.g. Phở Ăn -> Ăn phở; Chợ Đi -> Đi chợ)
    elif (
        len(core) == 2
        and low[1] in _TRANSITIVE_VERBS
        and low[0] not in _PERSON_GLOSSES
        and low[0] not in _TRANSITIVE_VERBS
    ):
        core = [core[1], core[0]]

    return head + core


def _clean_gloss_for_rule_sentence(token: str, is_first: bool) -> str:
    """Normalizes capitalization of a single VSL gloss token for natural Vietnamese sentence joining."""
    cleaned = _strip_gloss_meta(token)
    if not cleaned:
        return ""
    if is_first:
        return cleaned[0].upper() + cleaned[1:]
    # Preserve proper nouns (single-word proper nouns or multi-word names with internal capitals like 'Việt Nam')
    if cleaned in _SINGLE_WORD_PROPER_NOUNS or any(ch.isupper() for ch in cleaned[1:]):
        return cleaned
    return cleaned[0].lower() + cleaned[1:]


SYSTEM_PROMPT = (
    "Bạn là mô-đun biên dịch ngôn ngữ ký hiệu Việt Nam sang câu tiếng Việt tự nhiên. "
    "Hãy chuyển đổi danh sách các từ Gloss ký hiệu sau thành một câu tiếng Việt hoàn chỉnh, đúng ngữ pháp và dễ hiểu. "
    "Lưu ý trật tự từ trong ngôn ngữ ký hiệu có thể khác câu tiếng Việt, và danh sách đầu vào có thể chứa từ bị nhận diện nhầm do lỗi mô hình. "
    "Hãy khôi phục hư từ cần thiết và chọn nghĩa hợp lý nhất dựa trên các từ còn lại, tuyệt đối không thêm thông tin ngoài nội dung các từ gốc. "
    "Chỉ trả về duy nhất câu tiếng Việt kết quả, không kèm giải thích."
)

FEW_SHOT_EXAMPLES = (
    "Ví dụ mẫu:\n"
    "- Input: ['TÔI', 'BỆNH_VIỆN', 'ĐI', 'KHÁM'] -> Output: Tôi đi khám ở bệnh viện.\n"
    "- Input: ['HÔM_QUA', 'MẸ', 'CHỢ', 'MUA', 'CÁ'] -> Output: Hôm qua mẹ đi chợ mua cá.\n"
    "- Input: ['Bố', 'Chú', 'Giúp đỡ'] -> Output: Bố giúp đỡ chú.\n"
)


class LanguageService:
    def __init__(
        self,
        enabled: Optional[bool] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.enabled = enabled if enabled is not None else settings.LLM_ENABLED
        self.provider = provider or settings.LLM_PROVIDER
        self.api_key = api_key or settings.LLM_API_KEY
        self.model_name = model_name or settings.LLM_MODEL
        self.endpoint_url = settings.LLM_ENDPOINT
        # Status telemetry for UI fallback notification ('rule_based' | 'gemini' | 'ollama' | 'fallback_error')
        self.last_status: str = "rule_based"
        self.last_error: Optional[str] = None

    async def to_natural_sentence(self, gloss_list: List[str]) -> str:
        """Converts raw gloss list to a natural Vietnamese sentence.
        Always falls back gracefully to formatted gloss string if LLM fails or is disabled.
        """
        if not gloss_list:
            self.last_status = "rule_based"
            self.last_error = None
            return ""

        fallback_sentence = self._fallback_format(gloss_list)

        if not self.enabled or self.provider == "none":
            self.last_status = "rule_based"
            self.last_error = None
            return fallback_sentence

        try:
            if self.provider == "gemini":
                return await self._call_gemini(gloss_list, fallback_sentence)
            elif self.provider == "ollama":
                return await self._call_ollama(gloss_list, fallback_sentence)
            else:
                self.last_status = "rule_based"
                self.last_error = None
                return fallback_sentence
        except Exception as e:
            logger.warning(f"LLM translation failed ({e}); falling back to raw gloss format.")
            self.last_status = "fallback_error"
            self.last_error = f"Không thể kết nối dịch vụ {self.provider.upper()} ({type(e).__name__})"
            return fallback_sentence

    def _fallback_format(self, gloss_list: List[str]) -> str:
        """Rule-based natural sentence formatting with deterministic VSL (S-O-V) -> Vietnamese (S-V-O) reordering."""
        if not gloss_list:
            return ""
        raw_clean = [_strip_gloss_meta(str(raw)) for raw in gloss_list if raw and str(raw).strip()]
        raw_clean = [t for t in raw_clean if t]
        if not raw_clean:
            return ""

        reordered = _reorder_vsl_to_vietnamese(raw_clean)
        parts: List[str] = []
        for token in reordered:
            formatted = _clean_gloss_for_rule_sentence(token, is_first=(len(parts) == 0))
            if formatted:
                parts.append(formatted)
        if not parts:
            return ""
        text = " ".join(parts).strip()
        if text and not text.endswith((".", "!", "?")):
            text += "."
        return text

    async def _call_gemini(self, gloss_list: List[str], fallback: str) -> str:
        if not self.api_key:
            self.last_status = "fallback_error"
            self.last_error = "Chưa cấu hình GEMINI_API_KEY (đang giữ nguyên chuỗi Gloss cục bộ)"
            return fallback

        user_prompt = (
            f"{FEW_SHOT_EXAMPLES}"
            f"- Input: {gloss_list} -> Output:"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        gen_cfg = {"temperature": 0.2, "maxOutputTokens": 100}
        if "flash-lite" in self.model_name or "gemini-3-flash" in self.model_name:
            gen_cfg["thinkingConfig"] = {"thinkingLevel": "minimal"}
        payload = {
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": gen_cfg,
        }

        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                cand = data.get("candidates", [])[0]
                parts = cand.get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
                if text:
                    self.last_status = "gemini"
                    self.last_error = None
                    return text
                self.last_status = "fallback_error"
                self.last_error = "Dịch vụ Gemini API trả về phản hồi trống"
                return fallback
            else:
                logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text}")
                self.last_status = "fallback_error"
                self.last_error = f"Lỗi dịch vụ Gemini API (HTTP {resp.status_code})"
                return fallback

    async def _call_ollama(self, gloss_list: List[str], fallback: str) -> str:
        endpoint = self.endpoint_url or "http://localhost:11434/api/generate"
        prompt = (
            f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLES}"
            f"- Input: {gloss_list} -> Output:"
        )
        payload = {
            "model": self.model_name or "llama3",
            "prompt": prompt,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(endpoint, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                text = data.get("response", "").strip()
                if text:
                    self.last_status = "ollama"
                    self.last_error = None
                    return text
            self.last_status = "fallback_error"
            self.last_error = f"Lỗi dịch vụ Ollama cục bộ (HTTP {resp.status_code})"
            return fallback

