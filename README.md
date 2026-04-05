# HX-AM Proxy v4 (Dual-LLM)

**Генератор + Верификатор** на основе Groq (генерация) и Gemini (верификация).
Сервер на FastAPI, интерфейс, история и артефакты.

## Запуск

1. Установите зависимости: `pip install -r requirements.txt`
2. Настройте `.env` (см. `.env.example`)
3. Запустите `start_server.bat` или `python hxam_v_4_server.py`
4. Откройте `http://127.0.0.1:8000`

## Структура

- `hxam_v_4_server.py` – основной сервер
- `llm_client_v_4.py` – клиент для Groq и Gemini
- `system_prompts/` – промпты генератора и верификатора
- `chat_history/` – история диалогов (JSONL)
- `artifacts/` – сохранённые артефакты (JSON)

## Управление

- `start_server.bat` – запуск
- `stop_server.bat` – остановка (по порту 8000)
