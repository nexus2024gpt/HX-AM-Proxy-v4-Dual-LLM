# pipeline_guard.py — HX-AM Proxy v4
"""
Защитный слой пайплайна.

v4.3 — адаптирован для работы ПОСЛЕ response_normalizer.py:
  - Удалён circular import hxam_v_4_server.extract_json
  - validate_gen / validate_ver работают с уже нормализованными данными
  - Порог для hypothesis снижен с 20 до 10 символов (нормализатор уже отфильтровал)
  - Добавлен код GEN_UNRECOVERABLE / VER_UNRECOVERABLE (от нормализатора)
  - validate_ver больше не принимает raw= (нормализатор обработал до вызова)
  - translation warning остался, но НЕ блокирует пайплайн
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("HXAM.guard")


class FailureCode:
    # Generator
    GEN_ALL_PROVIDERS_FAILED = "GEN_ALL_PROVIDERS_FAILED"
    GEN_EMPTY_JSON           = "GEN_EMPTY_JSON"
    GEN_NO_HYPOTHESIS        = "GEN_NO_HYPOTHESIS"
    GEN_INVALID_B_SYNC       = "GEN_INVALID_B_SYNC"
    GEN_UNRECOVERABLE        = "GEN_UNRECOVERABLE"   # от нормализатора

    # Verifier
    VER_ALL_PROVIDERS_FAILED = "VER_ALL_PROVIDERS_FAILED"
    VER_EMPTY_JSON           = "VER_EMPTY_JSON"
    VER_NO_VERDICT           = "VER_NO_VERDICT"
    VER_UNRECOVERABLE        = "VER_UNRECOVERABLE"   # от нормализатора

    # Pipeline
    PIPELINE_EXCEPTION       = "PIPELINE_EXCEPTION"


class ValidationResult:
    def __init__(self, ok: bool, code: str = "", reason: str = ""):
        self.ok = ok
        self.code = code
        self.reason = reason

    def __bool__(self):
        return self.ok

    def to_dict(self) -> dict:
        return {"ok": self.ok, "code": self.code, "reason": self.reason}


class PipelineGuard:

    # ── Генератор (RAW) ─────────────────────────────────────

    def validate_gen_raw(self, raw: str, model: str) -> ValidationResult:
        """Проверяет сырой ответ провайдера до нормализации."""
        if not raw or not raw.strip():
            return ValidationResult(
                False, FailureCode.GEN_EMPTY_JSON,
                "Generator returned empty response"
            )
        if raw.strip().startswith("[Generator error]"):
            return ValidationResult(
                False, FailureCode.GEN_ALL_PROVIDERS_FAILED,
                f"All generator providers failed (last: {model}). Response: {raw[:120]}"
            )
        return ValidationResult(True)

    def validate_gen(self, gen: dict, model: str) -> ValidationResult:
        """
        Финальная проверка нормализованного gen-объекта.
        К этому моменту нормализатор уже:
          - извлёк JSON
          - привёл b_sync к float
          - установил домен
          - выставил default b_sync=0.55 если отсутствовал

        Здесь блокируем только если hypothesis совсем нет.
        """
        if not gen:
            return ValidationResult(
                False, FailureCode.GEN_EMPTY_JSON,
                f"Generator produced empty dict after normalization (model: {model})"
            )

        hypothesis = str(gen.get("hypothesis", "")).strip()
        if len(hypothesis) < 10:
            return ValidationResult(
                False, FailureCode.GEN_NO_HYPOTHESIS,
                f"Hypothesis missing or critically short after normalization: '{hypothesis[:60]}'"
            )

        # b_sync должен уже быть float после нормализатора
        b_sync = gen.get("b_sync")
        try:
            b = float(b_sync)
            if not (0.0 <= b <= 1.0):
                # Нормализатор должен был clamped это — защитный код
                gen["b_sync"] = max(0.0, min(1.0, b))
        except (TypeError, ValueError):
            return ValidationResult(
                False, FailureCode.GEN_INVALID_B_SYNC,
                f"b_sync not numeric after normalization: '{b_sync}' (model: {model})"
            )

        return ValidationResult(True)

    # ── Верификатор (RAW) ────────────────────────────────────

    def validate_ver_raw(self, raw: str, model: str) -> ValidationResult:
        """Проверяет сырой ответ провайдера до нормализации."""
        if not raw or not raw.strip():
            return ValidationResult(
                False, FailureCode.VER_EMPTY_JSON,
                "Verifier returned empty response"
            )
        if raw.strip().startswith("[Verifier error]"):
            return ValidationResult(
                False, FailureCode.VER_ALL_PROVIDERS_FAILED,
                f"All verifier providers failed (last: {model}). Response: {raw[:120]}"
            )
        return ValidationResult(True)

    def validate_ver(self, ver: dict, model: str) -> ValidationResult:
        """
        Финальная проверка нормализованного ver-объекта.
        К этому моменту нормализатор уже:
          - нормализовал verdict (ru/en → VALID/WEAK/FALSE)
          - добавил default WEAK если verdict отсутствовал
          - нормализовал translation.survival
          - привёл issues к list[str]

        Здесь блокируем только если verdict невалиден (что после нормализатора не должно случаться).
        """
        if not ver:
            return ValidationResult(
                False, FailureCode.VER_EMPTY_JSON,
                f"Verifier produced empty dict after normalization (model: {model})"
            )

        verdict = str(ver.get("verdict", "")).strip().upper()
        if verdict not in ("VALID", "WEAK", "FALSE"):
            return ValidationResult(
                False, FailureCode.VER_NO_VERDICT,
                f"Verdict invalid after normalization: '{verdict}' (model: {model})"
            )

        # Translation — предупреждение, не блокировка
        translation = ver.get("translation", {})
        if not translation or not translation.get("survival"):
            logger.warning(
                f"PipelineGuard: Step 0 translation incomplete after normalization "
                f"(model: {model})"
            )
        else:
            survival = translation.get("survival", "")
            if survival not in ("STRUCTURAL", "TERMINOLOGICAL", "UNKNOWN"):
                logger.warning(
                    f"PipelineGuard: survival='{survival}' unrecognized (model: {model})"
                )

        # Логируем наличие новых полей
        if ver.get("operationalization"):
            logger.debug(
                f"Guard: operationalization present "
                f"model={ver['operationalization'].get('model', '?')} ({model})"
            )
        if ver.get("refined_hypothesis"):
            logger.debug(f"Guard: refined_hypothesis present ({model})")

        return ValidationResult(True)


# ════════════════════════════════════════════════════════════════
# ROLLBACK MANAGER
# ════════════════════════════════════════════════════════════════

class RollbackManager:

    def __init__(self):
        self._space_snapshot: Optional[int] = None
        self._space_ids: Optional[set] = None
        self._graph_node: Optional[str] = None
        self._files: List[Path] = []

    def snapshot_space(self, space: "SemanticSpace"):  # type: ignore
        self._space_snapshot = len(space.vectors)
        self._space_ids = set(space._id_to_idx.keys())

    def register_graph_node(self, node_id: str):
        self._graph_node = node_id

    def register_file(self, path: Path):
        self._files.append(path)

    def rollback(self, space, graph) -> List[str]:
        actions: List[str] = []

        if self._space_snapshot is not None:
            current_len = len(space.vectors)
            removed = current_len - self._space_snapshot
            if removed > 0:
                space.vectors = space.vectors[:self._space_snapshot]
                space.meta = space.meta[:self._space_snapshot]
                # Rollback _id_to_idx too
                if self._space_ids is not None:
                    to_remove = [k for k in space._id_to_idx if k not in self._space_ids]
                    for k in to_remove:
                        del space._id_to_idx[k]
                actions.append(f"space: removed {removed} vector(s)")
            self._space_snapshot = None
            self._space_ids = None

        if self._graph_node and self._graph_node in graph.G:
            graph.G.remove_node(self._graph_node)
            graph._save()
            actions.append(f"graph: removed node {self._graph_node}")
            self._graph_node = None

        for path in self._files:
            if path.exists():
                path.unlink()
                actions.append(f"file: deleted {path.name}")
        self._files.clear()

        return actions

    def clear(self):
        self._space_snapshot = None
        self._space_ids = None
        self._graph_node = None
        self._files.clear()


# ════════════════════════════════════════════════════════════════
# QUARANTINE LOG
# ════════════════════════════════════════════════════════════════

class QuarantineLog:

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
        gen_repairs: Optional[List[str]] = None,
        ver_repairs: Optional[List[str]] = None,
    ):
        entry = {
            "time": time.time(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "query": query[:300],
            "stage": stage,
            "failure_code": failure_code,
            "reason": reason,
            "gen_model": gen_model,
            "ver_model": ver_model,
            "rollback_actions": rollback_actions or [],
            "gen_repairs": gen_repairs or [],
            "ver_repairs": ver_repairs or [],
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.warning(
            f"[QUARANTINE] job={job_id} stage={stage} code={failure_code} | {reason[:80]}"
        )

    def recent(self, n: int = 20) -> List[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        result = []
        for line in lines:
            try:
                result.append(json.loads(line))
            except Exception:
                continue
        return result