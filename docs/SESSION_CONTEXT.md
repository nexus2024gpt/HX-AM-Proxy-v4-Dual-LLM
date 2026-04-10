# HX-AM Proxy v4 — Контекст сессии разработки

# Формат: справочный документ для передачи между сессиями

# Дата: апрель 2026

# Репозиторий: https://github.com/nexus2024gpt/HX-AM-Proxy-v4-Dual-LLM

---

## Что такое проект

HX-AM Proxy v4 — система генерации и верификации структурных гипотез через dual-LLM пайплайн.
Основная цель: обнаружение кросс-доменных инвариантов — паттернов, которые сохраняются в разных
научных областях одновременно (физика↔биология, математика↔лингвистика и т.д.).

Философия: не поиск аналогий, а обнаружение структурного изоморфизма. Верификатор обязан перевести
механизм на язык чужого домена (Step 0) — если механизм выживает перевод, он структурный, не
терминологический.

---

## Архитектура (финальная)

```
Запрос пользователя
       │
  [QuestionGenerator]  ← Mode A (novel) | Mode B (clarify) | ручной ввод
       │
       ▼
[PipelineGuard] ← валидация RAW ответов ДО любого изменения состояния
       │
  [LLMClient.generate()]
  Groq → OpenRouter → HuggingFace
       │
  validate_gen_raw() → validate_gen()
  провал → QuarantineLog → _rejected_response()
       │
  [LLMClient.verify()]
  Gemini → OpenRouter → HuggingFace
       │
  validate_ver_raw() → validate_ver()
  провал → QuarantineLog → _rejected_response()
       │
  [RollbackManager.snapshot()]  ← точка отката
       │
  [process_with_invariants()]   ← SemanticSpace + InvariantGraph + PhaseDetector
       │
  save artifact (если VALID conf>0.6 или WEAK b_sync>0.7)
       │
  [Archivist.process()]         ← novelty scoring: PHENOMENAL|NOVEL|KNOWN|REPHRASING
       │
  log_history()                 ← только при полном успехе
```

---

## Файловая структура проекта

```
├── hxam_v_4_server.py          — FastAPI + оркестрация всего пайплайна
├── invariant_engine.py         — SemanticSpace, InvariantGraph, PhaseDetector
├── llm_client_v_4.py           — 3-уровневый фолбэк: Groq/Gemini → OpenRouter → HuggingFace
├── archivist.py                — оценка новизны артефактов (PHENOMENAL/NOVEL/KNOWN/REPHRASING)
├── pipeline_guard.py           — PipelineGuard + RollbackManager + QuarantineLog
├── question_generator.py       — Mode A (новый вопрос) + Mode B (уточнение артефакта)
├── index_v_4.html              — UI: 3D граф + RAG-блок + режимы запросов + карантин
├── install_and_run.ps1
├── requirements.txt
├── .env                        — ключи API (не в репо)
├── prompts/
│   ├── generator_prompt.txt         — генератор гипотез (с domain diversity constraint)
│   ├── verifier_prompt.txt          — верификатор (Step 0 обязателен)
│   ├── archivist_prompt.txt         — Archivist (4 правила, PHENOMENAL первым)
│   └── question_generator_prompt.txt — Mode A + Mode B
├── tools/
│   └── run_archivist.ps1            — batch обработка артефактов без archivist-оценки
├── artifacts/
│   ├── semantic_index.jsonl    — эмбеддинги (авто)
│   ├── invariant_graph.json    — граф (авто)
│   ├── TRUST_*.json            — 10 seed-артефактов (фундамент)
│   └── *.json / *.hyx-portal.json
└── chat_history/
    ├── history.jsonl           — только успешные запросы
    └── quarantine.jsonl        — отклонённые с кодами причин
```

---

## Ключевые архитектурные решения и почему они именно такие

### InvariantEngine — формула веса ребра

```
weight = similarity × (1 + domain_distance) × specificity
```

* `domain_distance` — расстояние между доменами в пространстве эмбеддингов (НЕ жёсткая матрица)
* `specificity` — косинусное расстояние гипотезы от центроида своего домена
* Вычисляется ДО добавления вектора в space (иначе centroid смещается самой гипотезой)
* Цель: физика↔поэзия с уникальной структурой = максимальный вес; банальное в math↔math = минимальный

### Verifier — Step 0 (семантическая трансляция)

Верификатор ОБЯЗАН сначала объяснить механизм терминами чужого домена, затем критиковать.
Поле `translation.survival`: STRUCTURAL (механизм выживает) или TERMINOLOGICAL (рассыпается).
Это влияет на stability_type и b_sync через оркестратор.

### Archivist — порядок правил (критично)

```
RULE 1: PHENOMENAL — сначала (sim>0.72 + domain_distance>0.6 + STRUCTURAL)
RULE 2: REPHRASING — высокий порог (sim>0.92 + domain_distance<0.15 + same domain)
RULE 3: NOVEL
RULE 4: KNOWN
```

До исправления: Rule 2 стояло первым с порогом 0.88 → 43/44 артефактов получали REPHRASING.
Артефакт никогда не сравнивается сам с собой (self-exclusion в _get_neighbors_excluding_self).

### PipelineGuard — схема отказа

```
GEN_ALL_PROVIDERS_FAILED / GEN_EMPTY_JSON / GEN_NO_HYPOTHESIS / GEN_INVALID_B_SYNC
VER_ALL_PROVIDERS_FAILED / VER_EMPTY_JSON / VER_NO_VERDICT / VER_NO_TRANSLATION
PIPELINE_EXCEPTION
```

Граф и semantic_space НЕ изменяются до успешного прохождения обоих валидаторов.
RollbackManager снимает снапшот перед engine и откатывает при exception.
log_history() вызывается ТОЛЬКО при полном успехе — никаких битых записей в истории.

### QuestionGenerator — принцип работы

ВСЕГДА ручной запуск, никогда автоматический. Возвращает только строку-вопрос.
Mode A: статистика графа вычисляется ЛОКАЛЬНО (без LLM-запроса). Один вызов LLM = один вопрос.
Mode B: анализирует issues верификатора, добавляет [REF:artifact_id] в конец вопроса.
Пользователь ОБЯЗАН нажать "Отправить" вручную после генерации вопроса.

### LLMClient — цепочки фолбэка

```
generate(): Groq → OpenRouter → HuggingFace
verify():   Gemini → OpenRouter → HuggingFace
```

Возвращает tuple (text, model_name) — имя модели отображается в UI как цветной бейдж.
HuggingFace Inference API v1 (OpenAI-compatible). temperature=0.5 для gen, 0.3 для ver.

### TRUST-узлы — seed архива

10 артефактов с именами TRUST_{hash} — верифицированные инварианты из предыдущей сессии.
Загружены в artifacts/ как фундамент перед сбросом накопленного архива.
8 из 9 успешно интегрировались в граф. TRUST_4ef9731c изолирован (нет соседей выше 0.65).
Отслеживаются через grep "TRUST_" по артефактам и через граф.

---

## Текущее состояние архива (на момент перехода сессии)

* 53 узла, 220 рёбер, 1 связная компонента
* 100% STRUCTURAL survival, 100% stable_cluster
* Доменный перекос: mathematics 62% (33/53) — требует внимания
* Специфичность низкая: 29/53 узлов < 0.15 — генератор воспроизводит шаблоны
* Фазовая плотность: 0.806 — sigma_primitive_candidate активен устойчиво
* TRUST_b2bc031a [economics] и TRUST_33045ca7 [ecology] — лучшие якорные узлы

---

## Переменные окружения (.env)

```env
GENERATOR_API_KEY=      # Groq
GENERATOR_API_BASE=https://api.groq.com/openai/v1
GENERATOR_MODEL=llama-3.3-70b-versatile

VERIFIER_API_KEY=       # Gemini
VERIFIER_API_BASE=https://generativelanguage.googleapis.com/v1beta
VERIFIER_MODEL=gemini-2.5-flash

OPENROUTER_API_KEY=     # резервный 1
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
OPENROUTER_GEN_MODEL=anthropic/claude-3-haiku
OPENROUTER_VER_MODEL=anthropic/claude-3-haiku

HF_API_KEY=             # резервный 2 (HuggingFace)
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.3
```

---

## Roadmap — что обсуждалось, что не реализовано

### Реализовано в этой сессии (полностью)

* [X] Invariant Engine (SemanticSpace + InvariantGraph + PhaseDetector)
* [X] Анти-веса с domain_distance и specificity
* [X] Verifier Step 0 (семантическая трансляция)
* [X] 3D граф (3d-force-graph, orbit control)
* [X] RAG контекст — явный блок в UI + инъекция в промпт генератора
* [X] Archivist v2 (PHENOMENAL/NOVEL/KNOWN/REPHRASING с правильным порядком)
* [X] PipelineGuard + RollbackManager + QuarantineLog
* [X] OpenRouter + HuggingFace фолбэк с именами моделей в UI
* [X] QuestionGenerator Mode A + Mode B
* [X] TRUST seed-артефакты (10 штук, 7 доменов)
* [X] hyx-portal.json автосоздание для bridge-узлов

### Обсуждалось, не реализовано

* [ ] Phase density dashboard — мини-дашборд с историей плотности (sparkline)
* [ ] Hybrid-X экспорт — формат для NodeNet/Σ-узлов (ждёт фиксации формата NodeNet)
* [ ] RAG по научной литературе в верификаторе — отличать "умная аналогия" от "новое знание"
* [ ] Мониторинг узла 51732daee238 — центральный узел старого архива с UNKNOWN survival
* [ ] Снижение порога specificity для stable_cluster с 0.3 до 0.2 (подавление math-шаблонов)

---

## Известные проблемы и решения

| Проблема                            | Причина                                                         | Статус                                                            |
| ------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Mathematics 62% доменов              | Генератор нашёл шаблон, воспроизводит | Частично: domain diversity в промпте                    |
| specificity < 0.15 у 29/53                 | Домен перенасыщен, центроид плотный     | Промпт исправлен, нужен порог в PhaseDetector |
| TRUST_4ef9731c изолирован         | Нет соседей выше порога similarity=0.65            | Ожидаем новых релевантных запросов       |
| survival=UNKNOWN у старых узлов | Созданы до введения Step 0                            | Batch через run_archivist.ps1 --all                                |

---

## Принципы разработки (договорённости)

1. **Полные файлы всегда** — никогда не давать "добавь строку X в место Y" без полного файла
2. **Синтаксическая проверка перед выдачей** — ast.parse() на каждый Python-файл
3. **Deployment** : PowerShell + Git. Ветка main, structured commit messages
4. **Среда** : i5-6300U, 8GB RAM, Windows 10, Python 3.10+, PowerShell 7.6
5. **Persistence** : только flat files (JSONL + JSON), никаких БД
6. **QuestionGenerator** : ТОЛЬКО ручной запуск, никогда автоматический
7. **История** : только при полном успехе пайплайна (guard обеспечивает)

---

## Терминология проекта

* **b_sync** — мера кросс-доменной устойчивости гипотезы (0.0–1.0)
* **specificity** — косинусное расстояние от центроида домена (низкое = банальная)
* **survival** — STRUCTURAL | TERMINOLOGICAL | UNKNOWN (из Step 0 верификатора)
* **hyx-artifact** — устойчивый инвариант (stable_cluster)
* **hyx-portal** — узел-мост между кластерами
* **sigma_primitive_candidate** — фазовый переход (density > 0.6 в окне 10)
* **TRUST_{hash}** — seed-артефакт, верифицированный фундамент архива
* **[REF:id]** — тег в вопросе Mode B, связывает новый запрос с источником
