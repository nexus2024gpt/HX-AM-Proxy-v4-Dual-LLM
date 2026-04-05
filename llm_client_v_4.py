import os
import requests
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    def __init__(self):
        # Groq (Generator)
        self.gen_api_key = os.getenv("GENERATOR_API_KEY")
        self.gen_base = os.getenv("GENERATOR_API_BASE", "https://api.groq.com/openai/v1")
        self.gen_model = os.getenv("GENERATOR_MODEL", "llama3-70b-8192")

        # Gemini (Verifier)
        self.ver_api_key = os.getenv("VERIFIER_API_KEY")
        self.ver_base = os.getenv("VERIFIER_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        self.ver_model = os.getenv("VERIFIER_MODEL", "gemini-1.5-flash")

    def generate(self, prompt: str) -> str:
        """Вызов генератора (Groq) через OpenAI-совместимый API"""
        url = f"{self.gen_base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.gen_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.gen_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        try:
            resp = requests.post(url, json=data, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Generator error] {e}"

    def verify(self, statement: str, context: str = "") -> str:
        """Вызов верификатора (Gemini)"""
        # Правильный URL для Gemini
        url = f"{self.ver_base}/models/{self.ver_model}:generateContent?key={self.ver_api_key}"
        # Формируем запрос с контекстом и утверждением
        full_prompt = f"Context: {context}\n\nStatement to verify: {statement}\n\nIs this statement true, false, or uncertain? Explain briefly."
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }]
        }
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            # Извлекаем текст из ответа Gemini
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"[Verifier error] {e}"