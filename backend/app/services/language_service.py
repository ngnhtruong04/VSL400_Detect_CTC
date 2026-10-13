"""LanguageService: Optional post-processing layer to convert Vietnamese Gloss sequences into natural Vietnamese sentences."""

import logging
from typing import List, Optional
import httpx

from backend.app.config.settings import settings

logger = logging.getLogger("vsl.language_service")


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

    async def to_natural_sentence(self, gloss_list: List[str]) -> str:
        """Converts raw gloss list to a natural Vietnamese sentence.
        Always falls back gracefully to formatted gloss string if LLM fails or is disabled.
        """
        if not gloss_list:
            return ""

        fallback_sentence = self._fallback_format(gloss_list)

        if not self.enabled or self.provider == "none":
            return fallback_sentence

        try:
            if self.provider == "gemini":
                return await self._call_gemini(gloss_list, fallback_sentence)
            elif self.provider == "ollama":
                return await self._call_ollama(gloss_list, fallback_sentence)
            else:
                return fallback_sentence
        except Exception as e:
            logger.warning(f"LLM translation failed ({e}); falling back to raw gloss format.")
            return fallback_sentence

    def _fallback_format(self, gloss_list: List[str]) -> str:
        """Rule-based natural sentence formatting."""
        if not gloss_list:
            return ""
        text = " ".join(gloss_list).strip()
        # Capitalize first letter and add period
        if len(text) > 0:
            text = text[0].upper() + text[1:]
        if not text.endswith((".", "!", "?")):
            text += "."
        return text

    async def _call_gemini(self, gloss_list: List[str], fallback: str) -> str:
        if not self.api_key:
            return fallback

        prompt = (
            "Bạn là trợ lý dịch thuật ngôn ngữ ký hiệu Việt Nam (VSL) sang tiếng Việt tự nhiên.\n"
            f"Chuỗi Gloss: {', '.join(gloss_list)}\n"
            "Hãy chuyển chuỗi Gloss này thành một câu tiếng Việt hoàn chỉnh, tự nhiên, đúng ngữ pháp.\n"
            "Chỉ trả về DUY NHẤT một câu kết quả, không thêm giải thích hay ghi chú."
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 100},
        }

        async with httpx.AsyncClient(timeout=4.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                cand = data.get("candidates", [])[0]
                text = cand["content"]["parts"][0]["text"].strip()
                return text
            else:
                logger.warning(f"Gemini API returned status {resp.status_code}: {resp.text}")
                return fallback

    async def _call_ollama(self, gloss_list: List[str], fallback: str) -> str:
        endpoint = self.endpoint_url or "http://localhost:11434/api/generate"
        prompt = (
            f"Dịch chuỗi ký hiệu VSL sau thành câu tiếng Việt tự nhiên: {', '.join(gloss_list)}. "
            "Chỉ trả về 1 câu kết quả duy nhất."
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
                return data.get("response", "").strip() or fallback
            return fallback

