import os
import requests
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    def __init__(self):
        # ── Groq (генератор, уровни 1-2) ──────────────────────
        self.gen_keys = [
            k for k in [
                os.getenv("GENERATOR_API_KEY"),
                os.getenv("GENERATOR_API_KEY_2"),
            ] if k
        ]
        self.gen_base  = os.getenv("GENERATOR_API_BASE", "https://api.groq.com/openai/v1")
        self.gen_model = os.getenv("GENERATOR_MODEL", "llama-3.3-70b-versatile")

        # ── Gemini (верификатор, уровни 1-5) ──────────────────
        self.ver_keys = [
            k for k in [
                os.getenv("VERIFIER_API_KEY"),
                os.getenv("VERIFIER_API_KEY_2"),
                os.getenv("VERIFIER_API_KEY_3"),
                os.getenv("VERIFIER_API_KEY_4"),
                os.getenv("VERIFIER_API_KEY_5"),
            ] if k
        ]
        self.ver_base  = os.getenv("VERIFIER_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        self.ver_model = os.getenv("VERIFIER_MODEL", "gemini-2.5-flash")

        # ── OpenRouter (резервный 1) ───────────────────────────
        self.or_api_key   = os.getenv("OPENROUTER_API_KEY")
        self.or_base      = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
        self.or_gen_model = os.getenv("OPENROUTER_GEN_MODEL", "anthropic/claude-3-haiku")
        self.or_ver_model = os.getenv("OPENROUTER_VER_MODEL", "anthropic/claude-3-haiku")

        # ── HuggingFace (резервный 2, уровни HF1-HF2) ─────────
        self.hf_keys = [
            k for k in [
                os.getenv("HF_API_KEY"),
                os.getenv("HF_API_KEY_2"),
            ] if k
        ]
        self.hf_base  = "https://api-inference.huggingface.co/v1"
        self.hf_model = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

    # ══════════════════════════════════════════════
    # ГЕНЕРАТОР
    # Цепочка: Groq#1 → Groq#2 → OpenRouter → HF#1 → HF#2
    # ══════════════════════════════════════════════

    def generate(self, prompt: str) -> tuple[str, str]:
        # Уровни 1-2: все Groq-ключи по очереди
        for i, key in enumerate(self.gen_keys, 1):
            text, _ = self._call_openai_compat(
                base=self.gen_base, api_key=key,
                model=self.gen_model, prompt=prompt,
                label=f"Groq#{i}",
            )
            if text:
                return text, f"groq/{self.gen_model}"

        # Уровень 3: OpenRouter
        text, _ = self._call_openai_compat(
            base=self.or_base, api_key=self.or_api_key,
            model=self.or_gen_model, prompt=prompt,
            label="OpenRouter[gen]",
        )
        if text:
            return text, f"openrouter/{self.or_gen_model}"

        # Уровни 4-5: все HF-ключи по очереди
        for i, key in enumerate(self.hf_keys, 1):
            text, _ = self._call_openai_compat(
                base=self.hf_base, api_key=key,
                model=self.hf_model, prompt=prompt,
                label=f"HuggingFace#{i}[gen]",
                temperature=0.5,
            )
            if text:
                return text, f"huggingface/{self.hf_model}"

        return "[Generator error] all providers failed", "none"

    # ══════════════════════════════════════════════
    # ВЕРИФИКАТОР
    # Цепочка: Gemini#1→#5 → OpenRouter → HF#1 → HF#2
    # ══════════════════════════════════════════════

    def verify(self, statement: str, context: str = "") -> tuple[str, str]:
        # Уровни 1-5: все Gemini-ключи по очереди
        for i, key in enumerate(self.ver_keys, 1):
            text = self._call_gemini(statement, context, api_key=key)
            if text:
                return text, f"gemini/{self.ver_model}"

        # Уровень 6: OpenRouter
        full_prompt = f"Context: {context}\n\n{statement}" if context else statement
        text, _ = self._call_openai_compat(
            base=self.or_base, api_key=self.or_api_key,
            model=self.or_ver_model, prompt=full_prompt,
            label="OpenRouter[ver]",
        )
        if text:
            return text, f"openrouter/{self.or_ver_model}"

        # Уровни 7-8: все HF-ключи по очереди
        for i, key in enumerate(self.hf_keys, 1):
            text, _ = self._call_openai_compat(
                base=self.hf_base, api_key=key,
                model=self.hf_model, prompt=full_prompt,
                label=f"HuggingFace#{i}[ver]",
                temperature=0.3,
            )
            if text:
                return text, f"huggingface/{self.hf_model}"

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
        temperature: float = 0.7,
    ) -> tuple[str, str]:
        if not api_key:
            return "", ""
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter" in base:
            headers["HTTP-Referer"] = "https://hxam.local"
            headers["X-Title"] = "HX-AM v4"

        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
        }
        try:
            resp = requests.post(url, json=data, headers=headers, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if content and content.strip():
                return content, model
            print(f"[{label}] empty content")
            return "", ""
        except Exception as e:
            print(f"[{label}] error: {e}")
            return "", ""

    def _call_gemini(self, statement: str, context: str = "", api_key: str = "") -> str:
        if not api_key:
            return ""
        url = (
            f"{self.ver_base}/models/{self.ver_model}"
            f":generateContent?key={api_key}"
        )
        full_prompt = (
            f"Context: {context}\n\nStatement to verify:\n{statement}"
            if context else statement
        )
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        try:
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            return text if text and text.strip() else ""
        except Exception as e:
            print(f"[Gemini] error: {e}")
            return ""