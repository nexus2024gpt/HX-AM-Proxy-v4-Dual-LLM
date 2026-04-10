
# 🔮 HX-AM Proxy v4

**Dual-LLM система генерации и верификации структурных гипотез с инвариантным движком.**

Groq генерирует гипотезы → Gemini верифицирует через семантическую трансляцию → Invariant Engine строит граф и обнаруживает устойчивые паттерны между доменами → Archivist оценивает новизну.

---

## Архитектура

```
[QuestionGenerator]  ← Mode A (новый вопрос) | Mode B (уточнение) | ручной ввод
        │
[PipelineGuard]      ← валидация до изменения состояния
        │
  Generator (Groq)          Verifier (Gemini)
  Groq → OpenRouter → HF    Gemini → OpenRouter → HF
        │                         │
        └──────────┬──────────────┘
                   ▼
          [Invariant Engine]
          SemanticSpace  ← эмбеддинги, cosine similarity
          InvariantGraph ← граф с анти-весами
          PhaseDetector  ← фазовые переходы
                   │
          [Archivist]    ← PHENOMENAL | NOVEL | KNOWN | REPHRASING
                   │
          artifacts/ + graph
```

### Формула веса ребра (анти-шум)

```
weight = similarity × (1 + domain_distance) × specificity
```

* `domain_distance` — расстояние между доменами в пространстве эмбеддингов
* `specificity` — косинусное расстояние от центроида своего домена
* Физика↔поэзия с уникальной структурой = **максимальный вес**
* Банальное в одном домене = **минимальный вес**

### Step 0 — обязательная семантическая трансляция

Верификатор перед любой критикой **обязан** перевести механизм на язык чужого домена. Если механизм выживает — он `STRUCTURAL` (настоящий инвариант). Если рассыпается — `TERMINOLOGICAL` (псевдоаналогия).

### Типы артефактов

| Тип                        | Условие                           | Значение                               |
| ----------------------------- | ---------------------------------------- | ---------------------------------------------- |
| `noise`                     | isolated + низкий b_sync           | Случайный паттерн              |
| `weak_pattern`              | Один сосед sim > 0.8            | Ждёт подтверждения            |
| `hyx-artifact`              | stable_cluster                           | Устойчивый инвариант        |
| `hyx-portal`                | Мост между кластерами | Потенциальный Σ-примитив |
| `sigma_primitive_candidate` | Phase density > 0.6                      | Фазовый переход                  |

### Archivist — правила новизны

```
RULE 1 PHENOMENAL: sim > 0.72 + domain_distance > 0.6 + STRUCTURAL
RULE 2 REPHRASING: sim > 0.92 + domain_distance < 0.15 + same domain
RULE 3 NOVEL:      sim > 0.65 + новый домен или комбинация
RULE 4 KNOWN:      всё остальное
```

---

## Стек

```
FastAPI + Uvicorn          — HTTP сервер
Groq API                   — генератор (llama-3.3-70b-versatile)
Gemini API                 — верификатор (gemini-2.5-flash)
OpenRouter                 — резервный 1 для обоих
HuggingFace Inference API  — резервный 2 (Mistral-7B-Instruct-v0.3)
sentence-transformers      — локальные эмбеддинги (all-MiniLM-L6-v2, ~90 МБ)
networkx                   — граф инвариантов
numpy / scipy              — матричные операции, кластеризация
3d-force-graph             — интерактивная 3D визуализация (WebGL)
```

Никаких баз данных. Персистентность — flat files:

* `artifacts/semantic_index.jsonl` — эмбеддинги
* `artifacts/invariant_graph.json` — граф
* `chat_history/history.jsonl` — успешные запросы
* `chat_history/quarantine.jsonl` — отклонённые с кодами причин

---

## Установка

### Требования

* Python 3.10+
* API ключи: Groq, Google Gemini (+ опционально OpenRouter, HuggingFace)

### Быстрый старт (PowerShell)

```powershell
git clone https://github.com/nexus2024gpt/HX-AM-Proxy-v4-Dual-LLM
cd "HX-AM Proxy v4"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\install_and_run.ps1
```

### Переменные окружения (.env)

```env
# Основные
GENERATOR_API_KEY=your_groq_key
GENERATOR_MODEL=llama-3.3-70b-versatile
VERIFIER_API_KEY=your_gemini_key
VERIFIER_MODEL=gemini-2.5-flash

# Резервный 1 (OpenRouter)
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_GEN_MODEL=anthropic/claude-3-haiku
OPENROUTER_VER_MODEL=anthropic/claude-3-haiku

# Резервный 2 (HuggingFace)
HF_API_KEY=your_hf_token
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

---

## Использование

После запуска открой `http://localhost:8000`

### Вкладки интерфейса

**💬 Запрос** — три режима ввода:

* ✏️ **Ручной ввод** — как обычно
* 🎲 **Новый вопрос** — генерирует вопрос избегая доминирующих доменов (1 LLM-запрос)
* 🔧 **Уточнить артефакт** — выбор из списка WEAK-артефактов, генерирует уточняющий вопрос с [REF:id]

Панель результатов: Генератор · Верификатор · Структура · Archivist.
Бейджи моделей: зелёный = основная, оранжевый = фолбэк.

**🕸️ Граф 3D** — WebGL force-directed граф. Orbit-контроль, hover-тултипы, клик по узлу.

**📜 История** — последние 20 успешных запросов (кликабельно).

**📦 Артефакты** — сохранённые гипотезы (🌀 = hyx-portal).

**🚫 Карантин** — отклонённые запросы с кодами причин.

### API эндпоинты

| Метод | Путь                   | Описание                       |
| ---------- | -------------------------- | -------------------------------------- |
| `POST`   | `/query`                 | Основной пайплайн      |
| `GET`    | `/rag/context?text=`     | RAG lookup                             |
| `GET`    | `/graph/data`            | Данные графа для D3      |
| `GET`    | `/phase`                 | Фазовый сигнал            |
| `GET`    | `/history`               | Последние 20 запросов |
| `GET`    | `/artifacts`             | Список артефактов      |
| `GET`    | `/quarantine`            | Карантинный лог          |
| `GET`    | `/question/suggest`      | Mode A — новый вопрос      |
| `GET`    | `/question/clarify/{id}` | Mode B — уточнение           |
| `GET`    | `/question/candidates`   | Кандидаты для Mode B       |

---

## Структура проекта

```
├── hxam_v_4_server.py
├── invariant_engine.py
├── llm_client_v_4.py
├── archivist.py
├── pipeline_guard.py
├── question_generator.py
├── index_v_4.html
├── install_and_run.ps1
├── requirements.txt
├── .env
├── prompts/
│   ├── generator_prompt.txt
│   ├── verifier_prompt.txt
│   ├── archivist_prompt.txt
│   └── question_generator_prompt.txt
├── tools/
│   └── run_archivist.ps1
└── docs/
    └── SESSION_CONTEXT.md
```

---

## Batch-обработка архива

```powershell
.\tools\run_archivist.ps1 --all --dry-run   # preview
.\tools\run_archivist.ps1 --all             # запуск
.\tools\run_archivist.ps1 406f650a08aa      # один артефакт
```
