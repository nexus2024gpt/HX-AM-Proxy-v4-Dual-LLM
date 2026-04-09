# pipeline_guard.py — HX-AM Proxy v4
"""
Защитный слой пайплайна.

Принцип: никакие данные не попадают в граф, space или artifacts
до успешного прохождения всех этапов валидации.

Три компонента:
  PipelineGuard   — валидация gen/ver на каждом этапе
  RollbackManager — откат всех изменений при сбое
  QuarantineLog   — запись отклонённых запросов для анализа
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("HXAM.guard")


# ══════════════════════════════════════════════
# КОДЫ ОТКАЗА
# ══════════════════════════════════════════════

class FailureCode:
    GEN_ALL_PROVIDERS_FAILED  = "GEN_ALL_PROVIDERS_FAILED"
    GEN_EMPTY_JSON            = "GEN_EMPTY_JSON"
    GEN_NO_HYPOTHESIS         = "GEN_NO_HYPOTHESIS"
    GEN_NO_DOMAIN             = "GEN_NO_DOMAIN"
    GEN_INVALID_B_SYNC        = "GEN_INVALID_B_SYNC"

    VER_ALL_PROVIDERS_FAILED  = "VER_ALL_PROVIDERS_FAILED"
    VER_EMPTY_JSON            = "VER_EMPTY_JSON"
    VER_NO_VERDICT            = "VER_NO_VERDICT"
    VER_NO_TRANSLATION        = "VER_NO_TRANSLATION"   # Step 0 не выполнен

    PIPELINE_EXCEPTION        = "PIPELINE_EXCEPTION"


# ══════════════════════════════════════════════
# РЕЗУЛЬТАТ ВАЛИДАЦИИ
# ══════════════════════════════════════════════

class ValidationResult:
    def __init__(self, ok: bool, code: str = "", reason: str = ""):
        self.ok = ok
        self.code = code
        self.reason = reason

    def __bool__(self):
        return self.ok

    def to_dict(self) -> dict:
        return {"ok": self.ok, "code": self.code, "reason": self.reason}


# ══════════════════════════════════════════════
# GUARD
# ══════════════════════════════════════════════

class PipelineGuard:
    """
    Валидирует каждый этап пайплайна.
    При провале возвращает ValidationResult с кодом и причиной.
    """

    # ── Генератор ────────────────────────────

    def validate_gen_raw(self, raw: str, model: str) -> ValidationResult:
        """Проверяет сырой ответ генератора до парсинга JSON."""
        if not raw or not raw.strip():
            return ValidationResult(False, FailureCode.GEN_EMPTY_JSON,
                                    "Generator returned empty response")

        if raw.strip().startswith("[Generator error]"):
            return ValidationResult(False, FailureCode.GEN_ALL_PROVIDERS_FAILED,
                                    f"All generator providers failed (last: {model}). "
                                    f"Response: {raw[:120]}")
        return ValidationResult(True)

    def validate_gen(self, gen: dict, model: str) -> ValidationResult:
        """Проверяет распарсенный объект генератора."""
        if not gen:
            return ValidationResult(False, FailureCode.GEN_EMPTY_JSON,
                                    f"Generator JSON parse failed (model: {model})")

        hypothesis = gen.get("hypothesis", "").strip()
        if not hypothesis or len(hypothesis) < 20:
            return ValidationResult(False, FailureCode.GEN_NO_HYPOTHESIS,
                                    f"Hypothesis missing or too short: '{hypothesis[:50]}'")

        domain = gen.get("domain", "").strip().lower()
        if not domain or domain == "general":
            # Предупреждение, не блокировка — сервер подберёт домен из запроса
            logger.warning(f"Generator did not set domain (model: {model})")

        b_sync = gen.get("b_sync")
        if b_sync is None:
            return ValidationResult(False, FailureCode.GEN_INVALID_B_SYNC,
                                    "Generator did not return b_sync field")
        try:
            b = float(b_sync)
            if not (0.0 <= b <= 1.0):
                return ValidationResult(False, FailureCode.GEN_INVALID_B_SYNC,
                                        f"b_sync out of range: {b}")
        except (TypeError, ValueError):
            return ValidationResult(False, FailureCode.GEN_INVALID_B_SYNC,
                                    f"b_sync is not numeric: {b_sync}")

        return ValidationResult(True)

    # ── Верификатор ──────────────────────────

    def validate_ver_raw(self, raw: str, model: str) -> ValidationResult:
        """Проверяет сырой ответ верификатора до парсинга JSON."""
        if not raw or not raw.strip():
            return ValidationResult(False, FailureCode.VER_EMPTY_JSON,
                                    "Verifier returned empty response")

        if raw.strip().startswith("[Verifier error]"):
            return ValidationResult(False, FailureCode.VER_ALL_PROVIDERS_FAILED,
                                    f"All verifier providers failed (last: {model}). "
                                    f"Response: {raw[:120]}")
        return ValidationResult(True)

    def validate_ver(self, ver: dict, model: str) -> ValidationResult:
        """Проверяет распарсенный объект верификатора."""
        if not ver:
            return ValidationResult(False, FailureCode.VER_EMPTY_JSON,
                                    f"Verifier JSON parse failed (model: {model})")

        verdict = ver.get("verdict", "").strip().upper()
        if verdict not in ("VALID", "WEAK", "FALSE"):
            return ValidationResult(False, FailureCode.VER_NO_VERDICT,
                                    f"Verifier returned invalid verdict: '{verdict}'")

        # Step 0 — translation обязателен. Предупреждение, не блокировка:
        # модель могла выдать строку вместо объекта
        translation = ver.get("translation")
        if not translation:
            logger.warning(
                f"Verifier Step 0 missing: no 'translation' field (model: {model})"
            )
        elif isinstance(translation, dict):
            survival = translation.get("survival", "")
            if survival not in ("STRUCTURAL", "TERMINOLOGICAL"):
                logger.warning(
                    f"Verifier Step 0 incomplete: survival='{survival}' (model: {model})"
                )

        return ValidationResult(True)


# ══════════════════════════════════════════════
# ROLLBACK MANAGER
# ══════════════════════════════════════════════

class RollbackManager:
    """
    Откатывает все изменения при сбое пайплайна.

    Регистрирует:
      - добавления в SemanticSpace (по индексу)
      - добавления нод и рёбер в InvariantGraph
      - созданные файлы (artifact, portal)
    """

    def __init__(self):
        self._space_snapshot: Optional[int] = None     # len(space.vectors) до операции
        self._graph_node: Optional[str] = None          # id добавленной ноды
        self._files: List[Path] = []                    # созданные файлы для удаления

    def snapshot_space(self, space_len: int):
        """Запоминаем размер space до добавления."""
        self._space_snapshot = space_len

    def register_graph_node(self, node_id: str):
        self._graph_node = node_id

    def register_file(self, path: Path):
        self._files.append(path)

    def rollback(self, space, graph) -> List[str]:
        """
        Выполняет откат. Возвращает список выполненных действий.
        """
        actions = []

        # 1. Откат SemanticSpace
        if self._space_snapshot is not None:
            current_len = len(space.vectors)
            removed = current_len - self._space_snapshot
            if removed > 0:
                space.vectors = space.vectors[:self._space_snapshot]
                space.meta = space.meta[:self._space_snapshot]
                actions.append(f"space: removed {removed} vector(s)")
            self._space_snapshot = None

        # 2. Откат InvariantGraph
        if self._graph_node and self._graph_node in graph.G:
            # Удаляем ноду и все её рёбра
            graph.G.remove_node(self._graph_node)
            graph._save()
            actions.append(f"graph: removed node {self._graph_node}")
            self._graph_node = None

        # 3. Удаление файлов
        for path in self._files:
            if path.exists():
                path.unlink()
                actions.append(f"file: deleted {path.name}")
        self._files.clear()

        return actions

    def clear(self):
        """Очищает регистры после успешного завершения (без отката)."""
        self._space_snapshot = None
        self._graph_node = None
        self._files.clear()


# ══════════════════════════════════════════════
# QUARANTINE LOG
# ══════════════════════════════════════════════

class QuarantineLog:
    """
    Записывает отклонённые запросы в chat_history/quarantine.jsonl.
    Не попадают в history.jsonl и не отображаются в UI.
    """

    def __init__(self, path: str = "chat_history/quarantine.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(exist_ok=True)

    def record(
        self,
        job_id: str,
        query: str,
        failure_code: str,
        reason: str,
        stage: str,
        gen_model: str = "unknown",
        ver_model: str = "unknown",
        rollback_actions: Optional[List[str]] = None,
    ):
        entry = {
            "time": time.time(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "query": query[:300],
            "stage": stage,                # "generation" | "verification" | "engine"
            "failure_code": failure_code,
            "reason": reason,
            "gen_model": gen_model,
            "ver_model": ver_model,
            "rollback_actions": rollback_actions or [],
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.warning(
            f"[QUARANTINE] job={job_id} stage={stage} code={failure_code} | {reason[:80]}"
        )

    def recent(self, n: int = 20) -> List[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()[-n:]
        result = []
        for line in lines:
            try:
                result.append(json.loads(line))
            except Exception:
                continue
        return result