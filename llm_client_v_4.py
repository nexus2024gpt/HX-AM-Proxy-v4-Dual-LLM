import os
import requests
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self):
        # ── Groq (основной генератор) ──────────────────────────
        self.gen_api_key  = os.getenv("GENERATOR_API_KEY")
        self.gen_base     = os.getenv("GENERATOR_API_BASE", "https://api.groq.com/openai/v1")
        self.gen_model    = os.getenv("GENERATOR_MODEL", "llama-3.3-70b-versatile")

        # ── Gemini (основной верификатор) ──────────────────────
        self.ver_api_key  = os.getenv("VERIFIER_API_KEY")
        self.ver_base     = os.getenv("VERIFIER_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        self.ver_model    = os.getenv("VERIFIER_MODEL", "gemini-2.5-flash")

        # ── OpenRouter (резервный для обоих) ───────────────────
        self.or_api_key   = os.getenv("OPENROUTER_API_KEY")
        self.or_base      = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        self.or_gen_model = os.getenv("OPENROUTER_GEN_MODEL", "anthropic/claude-3-haiku")
        self.or_ver_model = os.getenv("OPENROUTER_VER_MODEL", "anthropic/claude-3-haiku")

    # ══════════════════════════════════════════════
    # ГЕНЕРАТОР
    # ══════════════════════════════════════════════

    def generate(self, prompt: str) -> tuple[str, str]:
        """
        Возвращает (текст_ответа, название_модели).
        Сначала пробует Groq, при ошибке — OpenRouter.
        """
        text, model = self._call_openai_compat(
            base=self.gen_base,
            api_key=self.gen_api_key,
            model=self.gen_model,
            prompt=prompt,
            label="Groq",
        )
        if text:
            return text, f"groq/{self.gen_model}"

        # Фолбэк на OpenRouter
        text, model = self._call_openai_compat(
            base=self.or_base,
            api_key=self.or_api_key,
            model=self.or_gen_model,
            prompt=prompt,
            label="OpenRouter[gen]",
        )
        if text:
            return text, f"openrouter/{self.or_gen_model}"

        return "[Generator error] all providers failed", "none"

    # ══════════════════════════════════════════════
    # ВЕРИФИКАТОР
    # ══════════════════════════════════════════════

    def verify(self, statement: str, context: str = "") -> tuple[str, str]:
        """
        Возвращает (текст_ответа, название_модели).
        Сначала пробует Gemini, при ошибке — OpenRouter.
        """
        text = self._call_gemini(statement, context)
        if text:
            return text, f"gemini/{self.ver_model}"

        # Фолбэк на OpenRouter
        full_prompt = f"Context: {context}\n\n{statement}" if context else statement
        text, _ = self._call_openai_compat(
            base=self.or_base,
            api_key=self.or_api_key,
            model=self.or_ver_model,
            prompt=full_prompt,
            label="OpenRouter[ver]",
        )
        if text:
            return text, f"openrouter/{self.or_ver_model}"

        return "[Verifier error] all providers failed", "none"

    # ══════════════════════════════════════════════
    # ВНУТРЕННИЕ МЕТОДЫ
    # ══════════════════════════════════════════════

    def _call_openai_compat(
        self,
        base: str,
        api_key: str,
        model: str,
        prompt: str,
        label: str = "",
    ) -> tuple[str, str]:
        """OpenAI-совместимый вызов (Groq, OpenRouter, etc.)"""
        if not api_key:
            return "", ""
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # OpenRouter требует дополнительные заголовки
        if "openrouter" in base:
            headers["HTTP-Referer"] = "https://hxam.local"
            headers["X-Title"] = "HX-AM v4"

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        try:
            resp = requests.post(url, json=data, headers=headers, timeout=45)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"], model
        except Exception as e:
            print(f"[{label}] error: {e}")
            return "", ""

    def _call_gemini(self, statement: str, context: str = "") -> str:
        """Прямой вызов Gemini REST API."""
        if not self.ver_api_key:
            return ""
        url = (
            f"{self.ver_base}/models/{self.ver_model}"
            f":generateContent?key={self.ver_api_key}"
        )
        full_prompt = (
            f"Context: {context}\n\nStatement to verify:\n{statement}"
            if context else statement
        )
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        try:
            resp = requests.post(url, json=payload, timeout=45)
            resp.raise_for_status()
            result = resp.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"[Gemini] error: {e}")
            return ""