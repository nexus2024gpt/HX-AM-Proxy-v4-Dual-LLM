# llm_client_v_4.py — HX-AM v4 LLM Client
"""
Полностью перестроен на APIUsageTracker.
Жёстко закодированные ключи убраны — всё из config/providers.json.

Цепочки вызовов:
  generate(): role="generator"  → Groq Nexus → Groq Roman → HF Nexus → HF Roman
  verify():   role="verifier"   → Gemini Nexus (×3) → Gemini Roman (×2) → HF Nexus → HF Roman

Порядок определяется tracker.get_providers_for_role() на основе
account-level нагрузки сегодня — автоматически балансирует между Nexus и Roman.

Токены:
  Groq / HF:   response.usage.prompt_tokens / completion_tokens
  Gemini:       response.usageMetadata.promptTokenCount / candidatesTokenCount
  Фоллбэк:     len(text) // 4  (grубая оценка если API не вернул usage)
"""

import logging
import requests
from api_usage_tracker import tracker, ProviderConfig

logger = logging.getLogger("HXAM.llm")


def _estimate_tokens(text: str) -> int:
    """Грубая оценка: ~4 символа на токен для смешанного EN/RU текста."""
    return max(1, len(text) // 4)


class LLMClient:

    # ── Публичные методы ─────────────────────────────────────

    def generate(self, prompt: str) -> tuple[str, str]:
        """
        Генератор гипотез.
        Перебирает провайдеров с role="generator" в порядке ротации трекера.
        Возвращает (text, model_label).
        """
        providers = tracker.get_providers_for_role("generator")
        if not providers:
            logger.error("LLMClient.generate: нет доступных провайдеров для роли generator")
            return "[Generator error] no providers configured", "none"

        for p in providers:
            logger.debug(f"LLMClient.generate → trying {p.label}")
            text, tokens_in, tokens_out, err_msg = self._call(p, prompt)

            if text:
                tracker.record_call(p.id, tokens_in=tokens_in, tokens_out=tokens_out)
                logger.info(f"LLMClient.generate ✓ {p.label} | in={tokens_in} out={tokens_out}")
                return text, f"{p.provider}/{p.model}"
            else:
                tracker.record_call(p.id, error=True, error_msg=err_msg)
                logger.warning(f"LLMClient.generate ✗ {p.label}: {err_msg[:80]}")

        return "[Generator error] all providers failed", "none"

    def verify(self, statement: str, context: str = "") -> tuple[str, str]:
        """
        Верификатор гипотез.
        Перебирает провайдеров с role="verifier" в порядке ротации трекера.
        Возвращает (text, model_label).
        """
        full_prompt = (
            f"Context: {context}\n\n{statement}" if context else statement
        )
        providers = tracker.get_providers_for_role("verifier")
        if not providers:
            logger.error("LLMClient.verify: нет доступных провайдеров для роли verifier")
            return "[Verifier error] no providers configured", "none"

        for p in providers:
            logger.debug(f"LLMClient.verify → trying {p.label}")
            text, tokens_in, tokens_out, err_msg = self._call(p, full_prompt)

            if text:
                tracker.record_call(p.id, tokens_in=tokens_in, tokens_out=tokens_out)
                logger.info(f"LLMClient.verify ✓ {p.label} | in={tokens_in} out={tokens_out}")
                return text, f"{p.provider}/{p.model}"
            else:
                tracker.record_call(p.id, error=True, error_msg=err_msg)
                logger.warning(f"LLMClient.verify ✗ {p.label}: {err_msg[:80]}")

        return "[Verifier error] all providers failed", "none"

    # ── Диспетчер ────────────────────────────────────────────

    def _call(
        self, p: ProviderConfig, prompt: str
    ) -> tuple[str, int, int, str]:
        """
        Возвращает (text, tokens_in, tokens_out, error_msg).
        error_msg пустой при успехе.
        """
        if not p.api_key:
            return "", 0, 0, "api_key not set"
        try:
            if p.provider == "gemini":
                return self._call_gemini(p, prompt)
            else:
                return self._call_openai_compat(p, prompt)
        except Exception as e:
            return "", 0, 0, str(e)[:200]

    # ── OpenAI-совместимый вызов (Groq, HuggingFace) ─────────

    def _call_openai_compat(
        self, p: ProviderConfig, prompt: str
    ) -> tuple[str, int, int, str]:
        url = f"{p.api_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {p.api_key}",
            "Content-Type": "application/json",
        }

        # HF чуть стабильнее на 0.5, для чистого верификатора — 0.3
        if p.provider == "huggingface":
            temperature = 0.5
        elif p.roles == ["verifier"]:
            temperature = 0.3
        else:
            temperature = 0.7

        payload = {
            "model": p.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 1024,
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

    # ── Gemini REST API ───────────────────────────────────────

    def _call_gemini(
        self, p: ProviderConfig, prompt: str
    ) -> tuple[str, int, int, str]:
        url = (
            f"{p.api_base}/models/{p.model}"
            f":generateContent?key={p.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 4096},
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
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
