
# 🔮 HX-AM Proxy v4

**Dual-LLM система генерации и верификации гипотез с инвариантным движком.**

Groq генерирует структурные гипотезы → Gemini верифицирует → Invariant Engine строит семантический граф и обнаруживает устойчивые паттерны между доменами.

---

## Архитектура

```
Запрос пользователя
       │
       ▼
 ┌─────────────┐     ┌──────────────┐
 │  Generator  │────▶│   Verifier   │
 │   (Groq)    │     │   (Gemini)   │
 └─────────────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                ▼
       ┌─────────────────┐
       │ Invariant Engine│
       │  SemanticSpace  │  ← sentence-transformers (локально)
       │  InvariantGraph │  ← networkx
       │  PhaseDetector  │  ← scipy
       └────────┬────────┘
                ▼
         artifacts/ + graph
```

### Три слоя Invariant Engine

| Слой                                            | Класс         | Что делает                                                                     |
| --------------------------------------------------- | ------------------ | --------------------------------------------------------------------------------------- |
| Семантическое пространство | `SemanticSpace`  | Эмбеддинги гипотез, косинусное сходство              |
| Структурный граф                     | `InvariantGraph` | Узлы-артефакты, рёбра с весом `similarity × domain_distance` |
| Детектор фазовых переходов  | `PhaseDetector`  | Обнаружение `stable_cluster`, σ-примитивов                      |

### Типы артефактов

| Тип                        | Условие                                | Значение                                                  |
| ----------------------------- | --------------------------------------------- | ----------------------------------------------------------------- |
| `noise`                     | `isolated`+ низкий b_sync             | Случайный паттерн                                 |
| `weak_pattern`              | Один сосед с sim > 0.8              | Один прецедент, ждёт подтверждения  |
| `hyx-artifact`              | `stable_cluster`                            | Устойчивая структура                           |
| `hyx-portal`                | Узел-мост в графе               | Связывает разные кластеры                  |
| `sigma_primitive_candidate` | Фазовый переход (density > 0.6) | Кандидат в универсальный инвариант |

---

## Стек

```
FastAPI + Uvicorn       — HTTP сервер
Groq API                — генератор гипотез (llama3-70b-8192)
Gemini API              — верификатор (gemini-1.5-flash)
sentence-transformers   — локальные эмбеддинги (all-MiniLM-L6-v2, ~90 МБ)
networkx                — граф инвариантов в памяти
numpy / scipy           — матричные операции, кластеризация
D3.js                   — интерактивная визуализация графа (браузер)
```

Никаких баз данных. Персистентность — три файла:

* `artifacts/semantic_index.jsonl` — эмбеддинги
* `artifacts/invariant_graph.json` — граф
* `chat_history/history.jsonl` — история запросов

---

## Установка

### Требования

* Python 3.10+
* API ключи: Groq и Google Gemini

### Быстрый старт (PowerShell)

```powershell
# 1. Клонировать репозиторий
git clone <repo-url>
cd "HX-AM Proxy v4"

# 2. Разрешить выполнение скриптов (один раз)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Установить зависимости и запустить
.\install_and_run.ps1
```

### Ручная установка

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python hxam_v_4_server.py
```

### Переменные окружения

Создай файл `.env` в корне проекта:

```env
GENERATOR_API_KEY=your_groq_api_key
GENERATOR_API_BASE=https://api.groq.com/openai/v1
GENERATOR_MODEL=llama3-70b-8192

VERIFIER_API_KEY=your_gemini_api_key
VERIFIER_API_BASE=https://generativelanguage.googleapis.com/v1beta
VERIFIER_MODEL=gemini-1.5-flash
```

---

## Использование

После запуска открой `http://localhost:8000`

### Вкладки интерфейса

**💬 Запрос** — отправить гипотезу. В ответе видны три панели: генератор, верификатор, структурный анализ (тип артефакта, stability, centrality, bridge).

**🕸️ Граф** — интерактивная D3.js визуализация инвариантного графа. Zoom/pan, перетаскивание узлов, hover-тултипы, клик по узлу открывает детали. Цвет узла = тип стабильности.

**📜 История** — последние 20 запросов с метаданными.

**📦 Артефакты** — сохранённые гипотезы (VALID с confidence > 0.6 или WEAK с b_sync > 0.7).

### API эндпоинты

| Метод | Путь             | Описание                                                         |
| ---------- | -------------------- | ------------------------------------------------------------------------ |
| `POST`   | `/query`           | Отправить запрос                                          |
| `GET`    | `/graph`           | Статистика графа (узлы, рёбра, кластеры) |
| `GET`    | `/graph/data`      | Полные данные графа для D3                           |
| `GET`    | `/phase`           | Текущий фазовый сигнал                               |
| `GET`    | `/history`         | Последние 20 записей истории                      |
| `GET`    | `/artifacts`       | Список артефактов                                        |
| `GET`    | `/artifact/{name}` | Конкретный артефакт                                    |

---

## Структура проекта

```
├── hxam_v_4_server.py        — FastAPI сервер, оркестрация пайплайна
├── invariant_engine.py       — SemanticSpace, InvariantGraph, PhaseDetector
├── llm_client_v_4.py         — HTTP клиент для Groq и Gemini
├── index_v_4.html            — UI (D3.js граф + вкладки)
├── install_and_run.ps1       — скрипт установки и запуска
├── requirements.txt
├── .env                      — API ключи (не коммитить)
├── prompts/
│   ├── generator_prompt.txt  — промпт генератора (с domain и b_sync)
│   └── verifier_prompt.txt   — промпт верификатора
├── artifacts/
│   ├── semantic_index.jsonl  — индекс эмбеддингов (авто)
│   ├── invariant_graph.json  — граф (авто)
│   └── *.json                — сохранённые артефакты
└── chat_history/
    └── history.jsonl         — история запросов (авто)
```

---

## Как работает кластеризация

1. Каждая гипотеза кодируется в вектор через `all-MiniLM-L6-v2`
2. Косинусное сходство > 0.65 → потенциальное ребро в граф
3. Вес ребра = `similarity × domain_distance` — высокий вес означает семантически близкие гипотезы из **разных** доменов, что и является сигналом инварианта
4. Компоненты связности → кластеры; мосты → порталы
5. Если плотность последних 10 артефактов > 0.6 → фазовый переход → `sigma_primitive_candidate`

Для появления `stable_cluster` нужно 3–5 запросов на близкую тему из разных доменов. Домен определяется **генератором автоматически** и влияет на вес рёбер графа.
