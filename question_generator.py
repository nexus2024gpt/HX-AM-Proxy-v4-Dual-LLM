# question_generator.py — HX-AM Proxy v4
"""
Генератор вопросов — два режима:

  Mode A (NOVEL):     генерирует вопрос для нового инварианта,
                      избегая доминирующих доменов и повторов

  Mode B (CLARIFY):   генерирует уточняющий вопрос для конкретного артефакта
                      на основе замечаний верификатора

Принципы:
  - Всегда ручной запуск, никогда автоматический
  - Возвращает только строку-вопрос (~50 токенов)
  - Статистика графа вычисляется локально, без LLM
  - Не сохраняет ничего сам — только передаёт вопрос в UI
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from invariant_engine import InvariantGraph, SemanticSpace
from llm_client_v_4 import LLMClient

logger = logging.getLogger("HXAM.qgen")

# Домены которых нет в архиве — источник разнообразия
_MISSING_DOMAINS = [
    "chemistry", "psychology", "linguistics",
    "geology", "astronomy", "medicine", "sociology",
]

# Порог доминирования домена (доля от общего числа артефактов)
_OVERREPRESENTED_THRESHOLD = 0.30


class QuestionGenerator:

    def __init__(
        self,
        space: Optional[SemanticSpace] = None,
        graph: Optional[InvariantGraph] = None,
    ):
        self.space = space or SemanticSpace()
        self.graph = graph or InvariantGraph()
        self.llm = LLMClient()
        self.prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        path = Path("prompts/question_generator_prompt.txt")
        if not path.exists():
            raise FileNotFoundError("prompts/question_generator_prompt.txt not found")
        return path.read_text(encoding="utf-8")

    # ──────────────────────────────────────────────
    # ПУБЛИЧНЫЕ МЕТОДЫ
    # ──────────────────────────────────────────────

    def suggest_novel(self) -> Dict[str, Any]:
        """
        Mode A: генерирует вопрос для нового инварианта.
        Возвращает {question, stats, model}.
        """
        stats = self._build_stats()
        recent = self._recent_hypotheses(n=5)

        context = {
            "mode": "A",
            "archive_stats": stats,
            "recent_hypotheses": recent,
        }

        prompt = (
            self.prompt
            + "\n\nCONTEXT:\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )

        raw, model = self.llm.generate(prompt)
        question = self._clean_question(raw)

        logger.info(f"QuestionGenerator Mode A → model={model} q={question[:60]}")
        return {
            "mode": "novel",
            "question": question,
            "stats": stats,
            "model": model,
        }

    def suggest_clarification(self, artifact_id: str) -> Dict[str, Any]:
        """
        Mode B: генерирует уточняющий вопрос для артефакта с issues.
        Возвращает {question, artifact_summary, model}.
        """
        artifact = self._load_artifact(artifact_id)
        if not artifact:
            return {"error": f"Artifact {artifact_id} not found", "question": ""}

        data = artifact.get("data", {})
        gen = data.get("gen", {})
        ver = data.get("ver", {})
        structural = data.get("structural", {})

        issues = ver.get("issues", [])
        verdict = ver.get("verdict", "")
        confidence = ver.get("confidence", 0)
        translation = ver.get("translation", {})

        # Определяем что нужно уточнить
        improvement_hint = self._improvement_hint(verdict, confidence, issues, structural)

        context = {
            "mode": "B",
            "artifact": {
                "id": artifact_id,
                "domain": data.get("domain", "general"),
                "hypothesis": gen.get("hypothesis", ""),
                "mechanism": gen.get("mechanism", ""),
                "verdict": verdict,
                "confidence": confidence,
                "issues": issues,
                "translation": translation,
                "survival": translation.get("survival", "UNKNOWN")
                             if isinstance(translation, dict) else "UNKNOWN",
                "specificity": structural.get("specificity"),
                "b_sync": gen.get("b_sync"),
                "artifact_type": structural.get("artifact_type"),
            },
            "suggested_improvement": improvement_hint,
        }

        prompt = (
            self.prompt
            + "\n\nCONTEXT:\n"
            + json.dumps(context, ensure_ascii=False, indent=2)
        )

        raw, model = self.llm.generate(prompt)
        question = self._clean_question(raw)

        # Убедимся что REF тег есть
        if f"[REF:{artifact_id}]" not in question:
            question = question.rstrip() + f" [REF:{artifact_id}]"

        logger.info(f"QuestionGenerator Mode B → artifact={artifact_id} model={model}")
        return {
            "mode": "clarify",
            "question": question,
            "artifact_id": artifact_id,
            "artifact_summary": {
                "hypothesis": gen.get("hypothesis", "")[:120],
                "verdict": verdict,
                "confidence": confidence,
                "issues": issues,
                "improvement_hint": improvement_hint,
            },
            "model": model,
        }

    def list_clarification_candidates(self) -> List[Dict[str, Any]]:
        """
        Возвращает список артефактов требующих уточнения.
        Критерии: WEAK verdict ИЛИ низкая specificity ИЛИ issues непустые.
        Используется UI для выпадающего списка.
        """
        artifacts_dir = Path("artifacts")
        if not artifacts_dir.exists():
            return []

        candidates = []
        for f in sorted(artifacts_dir.glob("*.json"),
                        key=lambda x: x.stat().st_mtime, reverse=True):
            if f.stem in ("invariant_graph",):
                continue
            if ".hyx-portal" in f.name:
                continue
            try:
                a = json.loads(f.read_text(encoding="utf-8"))
                data = a.get("data", {})
                gen = data.get("gen", {})
                ver = data.get("ver", {})
                structural = data.get("structural", {})

                verdict = ver.get("verdict", "")
                issues = ver.get("issues", [])
                specificity = structural.get("specificity", 0.5)
                b_sync = gen.get("b_sync", 0)
                artifact_type = structural.get("artifact_type", "")

                # Критерии включения
                needs_work = (
                    verdict == "WEAK"
                    or len(issues) > 0
                    or specificity < 0.15
                    or artifact_type == "noise"
                )
                if not needs_work:
                    continue

                priority = 0
                if verdict == "WEAK": priority += 3
                if len(issues) > 1: priority += len(issues)
                if specificity < 0.12: priority += 2
                if artifact_type == "noise": priority += 1

                candidates.append({
                    "id": a.get("id", f.stem),
                    "domain": data.get("domain", "general"),
                    "hypothesis_short": gen.get("hypothesis", "")[:80],
                    "verdict": verdict,
                    "confidence": ver.get("confidence", 0),
                    "issues_count": len(issues),
                    "specificity": specificity,
                    "artifact_type": artifact_type,
                    "priority": priority,
                })
            except Exception:
                continue

        return sorted(candidates, key=lambda x: -x["priority"])[:20]

    # ──────────────────────────────────────────────
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ──────────────────────────────────────────────

    def _build_stats(self) -> dict:
        """Сжатая статистика графа — вычисляется локально."""
        G = self.graph.G
        total = G.number_of_nodes()
        if total == 0:
            return {
                "total_artifacts": 0,
                "domain_distribution": {},
                "overrepresented": [],
                "underrepresented": [],
                "missing_domains": _MISSING_DOMAINS,
                "sigma_active": False,
            }

        dom: Dict[str, int] = {}
        for n, a in G.nodes(data=True):
            d = a.get("domain", "general")
            dom[d] = dom.get(d, 0) + 1

        overrepresented = [d for d, c in dom.items()
                           if c / total > _OVERREPRESENTED_THRESHOLD]
        underrepresented = [d for d, c in dom.items()
                            if c / total < 0.05]

        # Фазовый переход — смотрим последние 10 векторов
        sigma_active = False
        if len(self.space.vectors) >= 10:
            import numpy as np
            recent = np.array(self.space.vectors[-10:])
            norms = np.linalg.norm(recent, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            normalized = recent / norms
            sim_matrix = normalized @ normalized.T
            np.fill_diagonal(sim_matrix, 0)
            density = sim_matrix.sum() / (10 * 9)
            sigma_active = float(density) > 0.60

        return {
            "total_artifacts": total,
            "domain_distribution": dom,
            "overrepresented": overrepresented,
            "underrepresented": underrepresented,
            "missing_domains": [d for d in _MISSING_DOMAINS if d not in dom],
            "sigma_active": sigma_active,
        }

    def _recent_hypotheses(self, n: int = 5) -> List[str]:
        """Последние N гипотез из semantic_index."""
        if not self.space.meta:
            return []
        recent = self.space.meta[-n:]
        return [m.get("invariant", "")[:100] for m in reversed(recent)]

    def _load_artifact(self, artifact_id: str) -> Optional[dict]:
        path = Path("artifacts") / f"{artifact_id}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _improvement_hint(
        self,
        verdict: str,
        confidence: float,
        issues: List[str],
        structural: dict,
    ) -> str:
        """Определяет какого типа уточнение нужно артефакту."""
        hints = []

        if verdict == "WEAK":
            hints.append("уточнить механизм — верификатор нашёл структурные пробелы")
        if confidence < 0.6:
            hints.append("повысить конкретность — гипотеза слишком абстрактна")
        if structural.get("specificity", 0.5) < 0.15:
            hints.append("добавить специфику — гипотеза семантически банальна для своего домена")
        if structural.get("artifact_type") == "noise":
            hints.append("найти похожие прецеденты — изолированный узел без связей")

        # Анализируем текст issues
        for issue in issues:
            issue_lower = issue.lower()
            if "механизм" in issue_lower or "mechanism" in issue_lower:
                hints.append("конкретизировать механизм изоморфизма")
                break
            if "математ" in issue_lower or "доказательств" in issue_lower:
                hints.append("добавить математическое обоснование")
                break
            if "масштаб" in issue_lower or "уровень" in issue_lower:
                hints.append("объяснить инвариантность по масштабу")
                break

        return "; ".join(hints) if hints else "общее уточнение и конкретизация"

    @staticmethod
    def _clean_question(raw: str) -> str:
        """Убирает мусор из ответа — оставляет только вопрос."""
        if not raw:
            return ""
        # Убираем markdown
        cleaned = re.sub(r"```[^`]*```", "", raw).strip()
        # Убираем JSON если вдруг вернулся
        if cleaned.startswith("{"):
            return ""
        # Берём первую непустую строку
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        if not lines:
            return ""
        # Убираем префиксы типа "Вопрос:", "Question:", "A:", цифры
        question = re.sub(r"^(вопрос|question|mode [ab]|[а-яa-z][\.:]\s*|\d+[\.:]\s*)",
                          "", lines[0], flags=re.IGNORECASE).strip()
        return question