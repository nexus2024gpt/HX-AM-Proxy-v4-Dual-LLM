# llm_client_v_4.py — HX-AM v4 LLM Client
"""
Полностью перестроен на APIUsageTracker.

Токен-лимиты (v4.2 fix):
  Gemini verifier: 4096 — новый prompt schema (operationalization) генерирует ~1500-2000 токенов
  Groq / HF:       1024 — генератор, компактный JSON
  Gemini generator: 1024 — если используется как gen (fallback)

Цепочки вызовов:
  generate(): role="generator"  → Groq Nexus → Groq Roman → HF Nexus → HF Roman
  verify():   role="verifier"   → Gemini Nexus (×3) → Gemini Roman (×2) → HF Nexus → HF Roman
"""

import logging
import requests
from api_usage_tracker import tracker, ProviderConfig

logger = logging.getLogger("HXAM.llm")


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMClient:

    def generate(self, prompt: str) -> tuple[str, str]:
        providers = tracker.get_providers_for_role("generator")
        if not providers:
            return "[Generator error] no providers configured", "none"

        for p in providers:
            text, tokens_in, tokens_out, err_msg = self._call(p, prompt, role="generator")
            if text:
                tracker.record_call(p.id, tokens_in=tokens_in, tokens_out=tokens_out)
                logger.info(f"LLMClient.generate ✓ {p.label} | in={tokens_in} out={tokens_out}")
                return text, f"{p.provider}/{p.model}"
            else:
                tracker.record_call(p.id, error=True, error_msg=err_msg)
                logger.warning(f"LLMClient.generate ✗ {p.label}: {err_msg[:80]}")

        return "[Generator error] all providers failed", "none"

    def verify(self, statement: str, context: str = "") -> tuple[str, str]:
        full_prompt = (
            f"Context: {context}\n\n{statement}" if context else statement
        )
        providers = tracker.get_providers_for_role("verifier")
        if not providers:
            return "[Verifier error] no providers configured", "none"

        for p in providers:
            text, tokens_in, tokens_out, err_msg = self._call(p, full_prompt, role="verifier")
            if text:
                tracker.record_call(p.id, tokens_in=tokens_in, tokens_out=tokens_out)
                logger.info(f"LLMClient.verify ✓ {p.label} | in={tokens_in} out={tokens_out}")
                return text, f"{p.provider}/{p.model}"
            else:
                tracker.record_call(p.id, error=True, error_msg=err_msg)
                logger.warning(f"LLMClient.verify ✗ {p.label}: {err_msg[:80]}")

        return "[Verifier error] all providers failed", "none"

    def _call(self, p: ProviderConfig, prompt: str, role: str = "generator") -> tuple[str, int, int, str]:
        if not p.api_key:
            return "", 0, 0, "api_key not set"
        try:
            if p.provider == "gemini":
                return self._call_gemini(p, prompt, role=role)
            else:
                return self._call_openai_compat(p, prompt, role=role)
        except Exception as e:
            return "", 0, 0, str(e)[:200]

    def _call_openai_compat(self, p: ProviderConfig, prompt: str, role: str = "generator") -> tuple[str, int, int, str]:
        url = f"{p.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {p.api_key}",
            "Content-Type": "application/json",
        }
        temperature = 0.5 if p.provider == "huggingface" else (0.3 if role == "verifier" else 0.7)
        # Verifier ответ длиннее из-за operationalization
        max_tokens = 2048 if role == "verifier" else 1024

        payload = {
            "model": p.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=60)
            resp.raise_for_status()
            rj = resp.json()
            content = rj["choices"][0]["message"]["content"]
            usage = rj.get("usage", {})
            tokens_in = usage.get("prompt_tokens", _estimate_tokens(prompt))
            tokens_out = usage.get("completion_tokens", _estimate_tokens(content or ""))
            if content and content.strip():
                return content, tokens_in, tokens_out, ""
            return "", tokens_in, 0, "empty content in response"
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            return "", 0, 0, f"HTTP {status}: {str(e)[:120]}"
        except Exception as e:
            return "", 0, 0, str(e)[:200]

    def _call_gemini(self, p: ProviderConfig, prompt: str, role: str = "generator") -> tuple[str, int, int, str]:
        url = (
            f"{p.api_base}/models/{p.model}"
            f":generateContent?key={p.api_key}"
        )
        # Верификатор генерирует 1500-2000 токенов с новым schema (operationalization).
        # 1024 гарантированно обрезало JSON до verdict — поднимаем до 4096.
        max_output = 4096 if role == "verifier" else 1024

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": max_output},
        }
        try:
            resp = requests.post(url, json=payload, timeout=90)
            resp.raise_for_status()
            rj = resp.json()
            text = rj["candidates"][0]["content"]["parts"][0]["text"]
            usage = rj.get("usageMetadata", {})
            tokens_in = usage.get("promptTokenCount", _estimate_tokens(prompt))
            tokens_out = usage.get("candidatesTokenCount", _estimate_tokens(text or ""))
            if text and text.strip():
                return text, tokens_in, tokens_out, ""
            return "", tokens_in, 0, "empty content in response"
        except requests.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            return "", 0, 0, f"HTTP {status}: {str(e)[:120]}"
        except Exception as e:
            return "", 0, 0, str(e)[:200]
