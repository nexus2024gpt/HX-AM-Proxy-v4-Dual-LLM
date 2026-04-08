# archivist.py — HX-AM Proxy v4 Native Archivist
"""
Оценивает новизну и структурную устойчивость артефакта относительно архива.

Исправления относительно черновика:
  - artifact["data"]["gen"] — правильная структура (не artifact["hypothesis"])
  - llm_client возвращает tuple (text, model) — распаковываем
  - get_similar_nodes принимает (embedding, space) — не только embedding
  - PhaseDetector.log_phenomenal вместо signal_phase_transition (метод существует)
  - JSON-парсинг с защитой от markdown-оберток (```json ... ```)
  - CLI использует существующий SemanticSpace/InvariantGraph с правильными путями
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from invariant_engine import InvariantGraph, PhaseDetector, SemanticSpace
from llm_client_v_4 import LLMClient

logger = logging.getLogger("HXAM.archivist")


class Archivist:
    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        space: Optional[SemanticSpace] = None,
        graph: Optional[InvariantGraph] = None,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        # Принимаем уже инициализированные объекты из сервера (переиспользование RAM)
        # или создаём новые для CLI-режима
        self.space = space or SemanticSpace()
        self.graph = graph or InvariantGraph()
        self.detector = PhaseDetector()
        self.llm = LLMClient()
        self.prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = Path("prompts/archivist_prompt.txt")
        if not path.exists():
            raise FileNotFoundError("prompts/archivist_prompt.txt not found")
        return path.read_text(encoding="utf-8")

    # ──────────────────────────────────────────────
    # ПУБЛИЧНЫЙ МЕТОД
    # ──────────────────────────────────────────────

    def process(self, artifact_id: str) -> Dict[str, Any]:
        """
        Основной метод. Вызывается из сервера после сохранения артефакта.
        Возвращает результат archivist-оценки и записывает его в артефакт.
        """
        artifact_path = self.artifacts_dir / f"{artifact_id}.json"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact {artifact_id}.json not found")

        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

        # Извлекаем данные из правильной структуры artifact["data"]
        data = artifact.get("data", {})
        gen = data.get("gen", {})
        ver = data.get("ver", {})
        structural = data.get("structural", {})
        domain = data.get("domain", "general")

        hypothesis = gen.get("hypothesis", "")
        mechanism = gen.get("mechanism", "")
        translation_obj = ver.get("translation", {})
        translated_mech = (
            translation_obj.get("translated_mechanism", "")
            if isinstance(translation_obj, dict) else ""
        )

        # Строим эмбеддинг из всего доступного текста
        full_text = " ".join(filter(None, [hypothesis, mechanism, translated_mech]))
        if not full_text.strip():
            logger.warning(f"Archivist: artifact {artifact_id} has no text to embed")
            return self._fallback_result("empty text")

        embedding = self.space.encode(full_text)

        # Топ-8 ближайших из существующего пространства
        neighbors = self.graph.get_similar_nodes(embedding, self.space, top_k=8)

        # Subgraph вокруг узла (может не существовать если новый)
        subgraph = self.graph.get_subgraph(artifact_id, depth=2)

        # Собираем контекст для LLM
        context = {
            "new_artifact": {
                "id": artifact_id,
                "domain": domain,
                "hypothesis": hypothesis,
                "mechanism": mechanism,
                "translation": translation_obj,
                "survival": structural.get("translation", {}).get("survival", "UNKNOWN")
                             if isinstance(structural.get("translation"), dict)
                             else structural.get("survival", "UNKNOWN"),
                "specificity": structural.get("specificity"),
                "b_sync": gen.get("b_sync"),
                "similar_invariants": structural.get("similar_invariants", [])[:5],
            },
            "neighbors": neighbors,
            "subgraph": {
                "cluster_count": len(subgraph["clusters"]),
                "bridge_count": len(subgraph["bridges"]),
                "local_nodes": len(subgraph["nodes"]),
                "bridges": subgraph["bridges"],
            },
        }

        full_prompt = (
            self.prompt
            + "\n\nContext:\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )

        # Вызов LLM — используем generate() который возвращает (text, model)
        raw_text, model_used = self.llm.generate(full_prompt)
        logger.info(f"Archivist LLM response via {model_used}")

        result = self._parse_result(raw_text)

        # Записываем в артефакт
        artifact["archivist"] = result
        artifact["archivist_model"] = model_used
        artifact["last_archivist_update"] = datetime.now(timezone.utc).isoformat()
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))

        # Обновляем граф
        self.graph.update_with_archivist(artifact_id, result)

        # Логируем феноменальные связи
        if (
            result.get("novelty") == "PHENOMENAL"
            and result.get("mathematical_verification") == "STRUCTURAL"
        ):
            self.detector.log_phenomenal(
                artifact_id,
                f"cross_domain={result.get('cross_domain_links')}",
            )

        logger.info(
            f"Archivist done: {artifact_id} → novelty={result.get('novelty')} "
            f"score={result.get('novelty_score')} confidence={result.get('confidence')}"
        )
        return result

    # ──────────────────────────────────────────────
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ──────────────────────────────────────────────

    def _parse_result(self, raw: str) -> Dict[str, Any]:
        """
        Парсит JSON из ответа LLM.
        Защита от markdown-оберток (```json ... ```) и мусора вне JSON.
        """
        # Убираем markdown-обёртки
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

        # Ищем JSON-блок
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.warning(f"Archivist JSON parse error: {e}")

        return self._fallback_result("json_parse_error")

    @staticmethod
    def _fallback_result(reason: str) -> Dict[str, Any]:
        return {
            "novelty": "KNOWN",
            "is_rephrasing_of": None,
            "cross_domain_links": [],
            "mathematical_verification": "TERMINOLOGICAL",
            "novelty_score": 0.45,
            "suggested_tags": [],
            "confidence": 0.3,
            "linked_to": [],
            "reasoning_summary": f"Fallback result: {reason}",
        }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python archivist.py <artifact_id>")
        print("Example: python archivist.py 406f650a08aa")
        sys.exit(1)

    artifact_id = sys.argv[1]
    arch = Archivist()

    try:
        result = arch.process(artifact_id)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)