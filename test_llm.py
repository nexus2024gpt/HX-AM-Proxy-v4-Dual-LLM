from llm_client_v_4 import LLMClient
import os
from dotenv import load_dotenv

load_dotenv()
client = LLMClient()

# Тест генератора
gen_response = client.generate("Напиши слово 'привет' на русском")
print("Generator:", gen_response)

# Тест верификатора (логическая проверка)
ver_response = client.verify("Солнце встаёт на востоке. Это правда?", "Проверь истинность")
print("Verifier:", ver_response)