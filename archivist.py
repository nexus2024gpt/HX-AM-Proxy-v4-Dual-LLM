# archivist.py — HX-AM Proxy v4 Native Archivist v2
"""
Оценивает новизну и структурную устойчивость артефакта.

Ключевые исправления v2:
  - Артефакт исключается из собственных соседей (self-comparison bug)
  - domain_distance вычисляется явно для каждого соседа перед отправкой в LLM
  - Правила проверяются локально до LLM-вызова для детерминированных случаев
  - Порог REPHRASING поднят до 0.92 + требует одинакового домена
  - PHENOMENAL проверяется первым
"""

import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.distance import cosine

from invariant_engine import InvariantGraph, PhaseDetector, SemanticSpace, _embedder
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
        artifact_path = self.artifacts_dir / f"{artifact_id}.json"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact {artifact_id}.json not found")

        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
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
        survival = (
            translation_obj.get("survival", "UNKNOWN")
            if isinstance(translation_obj, dict) else "UNKNOWN"
        )

        full_text = " ".join(filter(None, [hypothesis, mechanism, translated_mech]))
        if not full_text.strip():
            logger.warning(f"Archivist: {artifact_id} has no text")
            return self._fallback_result("empty text")

        embedding = self.space.encode(full_text)

        # Соседи — исключаем сам артефакт
        neighbors = self._get_neighbors_excluding_self(
            artifact_id, embedding, domain, top_k=8
        )

        # Детерминированная пре-проверка (до LLM)
        fast_result = self._fast_classify(
            artifact_id, domain, survival, neighbors
        )
        if fast_result:
            logger.info(
                f"Archivist fast-path: {artifact_id} → {fast_result['novelty']} "
                f"(no LLM needed)"
            )
            result = fast_result
        else:
            # Серая зона — отправляем в LLM
            subgraph = self.graph.get_subgraph(artifact_id, depth=2)
            context = self._build_context(
                artifact_id, domain, hypothesis, mechanism,
                translation_obj, survival, structural, gen,
                neighbors, subgraph,
            )
            full_prompt = (
                self.prompt
                + "\n\nContext:\n"
                + json.dumps(context, ensure_ascii=False, indent=2)
            )
            raw_text, model_used = self.llm.generate(full_prompt)
            logger.info(f"Archivist LLM via {model_used}: {artifact_id}")
            result = self._parse_result(raw_text)
            result["_model"] = model_used

        # Постобработка: никогда не ссылаемся на себя
        if result.get("is_rephrasing_of") == artifact_id:
            logger.warning(
                f"Archivist self-reference bug: {artifact_id} → clearing"
            )
            result["novelty"] = "KNOWN"
            result["is_rephrasing_of"] = None

        # Запись в артефакт
        artifact["archivist"] = result
        artifact["last_archivist_update"] = datetime.now(timezone.utc).isoformat()
        artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2))

        # Защита от self-ссылки в linked_to
        links = result.get("linked_to") or []
        if artifact_id in links:
            result["linked_to"] = [link for link in links if link != artifact_id]

        # Обновление графа
        self.graph.update_with_archivist(artifact_id, result)

        # Логируем феноменальные
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
            f"score={result.get('novelty_score')} conf={result.get('confidence')}"
        )
        return result

    # ──────────────────────────────────────────────
    # ДЕТЕРМИНИРОВАННАЯ КЛАССИФИКАЦИЯ (без LLM)
    # ──────────────────────────────────────────────

    def _fast_classify(
        self,
        artifact_id: str,
        domain: str,
        survival: str,
        neighbors: list,
    ) -> Optional[Dict[str, Any]]:
        """
        Детерминированно определяет очевидные случаи без LLM.

        PHENOMENAL: кросс-доменный структурный сигнал явно сильный
        REPHRASING: sim > 0.95 и одинаковый домен (очевидный дубликат)

        Возвращает None если случай неоднозначный → идёт в LLM.
        """
        if not neighbors:
            return None

        # Проверяем PHENOMENAL — берём приоритет
        for n in neighbors:
            sim = n["similarity"]
            dist = n["domain_distance"]
            n_survival = n.get("survival", "UNKNOWN")

            if sim > 0.72 and dist > 0.6 and survival == "STRUCTURAL":
                cross = [f"{domain}→{n['domain']}"]
                return {
                    "novelty": "PHENOMENAL",
                    "is_rephrasing_of": None,
                    "cross_domain_links": cross,
                    "mathematical_verification": "STRUCTURAL",
                    "novelty_score": round(0.7 + dist * 0.3, 3),
                    "suggested_tags": ["cross_domain_invariant"],
                    "confidence": 0.85,
                    "linked_to": [n["id"]],
                    "reasoning_summary": (
                        f"Кросс-доменная структурная связь: {domain}→{n['domain']} "
                        f"(sim={sim}, dist={dist}, survival=STRUCTURAL)"
                    ),
                }

        # Очевидный дубликат — очень высокий порог
        best = neighbors[0]
        if (
            best["similarity"] > 0.95
            and best["domain_distance"] < 0.15
            and best["id"] != artifact_id
        ):
            return {
                "novelty": f"REPHRASING_OF:{best['id']}",
                "is_rephrasing_of": best["id"],
                "cross_domain_links": [],
                "mathematical_verification": best.get("survival", "UNKNOWN"),
                "novelty_score": round(1 - best["similarity"], 3),
                "suggested_tags": ["duplicate"],
                "confidence": 0.9,
                "linked_to": [best["id"]],
                "reasoning_summary": (
                    f"Очевидный дубликат: sim={best['similarity']} "
                    f"в том же домене {best['domain']}"
                ),
            }

        return None  # серая зона → LLM

    # ──────────────────────────────────────────────
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ──────────────────────────────────────────────

    def _get_neighbors_excluding_self(
        self,
        artifact_id: str,
        embedding: np.ndarray,
        domain: str,
        top_k: int = 8,
    ) -> list:
        """
        Возвращает ближайших соседей БЕЗ самого артефакта.
        Обогащает каждого соседа domain_distance относительно текущего домена.
        """
        filtered = self.graph.get_similar_nodes(
            embedding,
            self.space,
            top_k=top_k,
            exclude_id=artifact_id,
        )

        # Обогащаем domain_distance
        try:
            domain_vec = _embedder.encode(domain)
        except Exception:
            domain_vec = None

        result = []
        for r in filtered:
            dist = 0.0
            if domain_vec is not None:
                try:
                    neighbor_domain_vec = _embedder.encode(r.get("domain", "general"))
                    dist = round(float(cosine(domain_vec, neighbor_domain_vec)), 3)
                except Exception:
                    dist = 0.0

            node_attrs = self.graph.G.nodes.get(r["id"], {})
            result.append({
                **r,
                "domain_distance": dist,
                "stability": node_attrs.get("stability", "unknown"),
                "survival": node_attrs.get("survival", r.get("survival", "UNKNOWN")),
            })

        return result

    def _build_context(
        self,
        artifact_id, domain, hypothesis, mechanism,
        translation_obj, survival, structural, gen,
        neighbors, subgraph,
    ) -> dict:
        return {
            "new_artifact": {
                "id": artifact_id,
                "domain": domain,
                "hypothesis": hypothesis,
                "mechanism": mechanism,
                "translation": translation_obj,
                "survival": survival,
                "specificity": structural.get("specificity"),
                "b_sync": gen.get("b_sync"),
                "similar_invariants": structural.get("similar_invariants", [])[:5],
            },
            "neighbors": neighbors,
            "subgraph": {
                "cluster_count": len(subgraph["clusters"]),
                "bridge_count": len(subgraph["bridges"]),
                "local_nodes": len(subgraph["nodes"]),
            },
        }

    def _parse_result(self, raw: str) -> Dict[str, Any]:
        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
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
            "reasoning_summary": f"Fallback: {reason}",
        }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python archivist.py <artifact_id>")
        print("       python archivist.py 406f650a08aa")
        sys.exit(1)

    arch = Archivist()
    try:
        result = arch.process(sys.argv[1])
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)