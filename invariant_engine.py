# invariant_engine.py — HX-AM v4 Invariant Engine
# Три слоя: SemanticSpace | InvariantGraph | PhaseDetector
# Адаптирован под структуру данных hxam_v_4_server.py

from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import fclusterdata

# Модель загружается один раз при импорте (~90 МБ, скачается автоматически)
_embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ══════════════════════════════════════════════
# СЛОЙ 1 — Семантическое пространство
# ══════════════════════════════════════════════

class SemanticSpace:
    """
    Хранит эмбеддинги гипотез в памяти.
    Персистентность — artifacts/semantic_index.jsonl
    """

    def __init__(self, index_path: str = "artifacts/semantic_index.jsonl"):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(exist_ok=True)
        self.vectors: list = []
        self.meta: list = []
        self._load()

    def add(self, artifact_id: str, invariant: str, domain: str, b_sync: float):
        vec = _embedder.encode(invariant)
        self.vectors.append(vec)
        entry = {
            "id": artifact_id,
            "invariant": invariant,
            "domain": domain,
            "b_sync": b_sync,
        }
        self.meta.append(entry)
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _load(self):
        if not self.index_path.exists():
            return
        with open(self.index_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self.meta.append(entry)
                    self.vectors.append(_embedder.encode(entry["invariant"]))
                except Exception:
                    continue

    def nearest(self, invariant: str, top_k: int = 5, threshold: float = 0.65) -> list:
        """Косинусное сходство — находит семантически близкие инварианты."""
        if not self.vectors:
            return []
        query_vec = _embedder.encode(invariant)
        matrix = np.array(self.vectors)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        # Защита от деления на ноль
        norms = np.where(norms == 0, 1e-9, norms)
        similarities = matrix @ query_vec / norms
        results = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                results.append({**self.meta[i], "similarity": round(float(sim), 3)})
        return sorted(results, key=lambda x: -x["similarity"])[:top_k]


# ══════════════════════════════════════════════
# СЛОЙ 2 — Структурный граф
# ══════════════════════════════════════════════

class InvariantGraph:
    """
    Граф инвариантов. Узлы — артефакты.
    Рёбра — устойчивые связи (similarity > threshold).
    Персистентность — artifacts/invariant_graph.json
    """

    def __init__(self, graph_path: str = "artifacts/invariant_graph.json"):
        self.path = Path(graph_path)
        self.path.parent.mkdir(exist_ok=True)
        self.G = self._load()

    def add_node(self, artifact_id: str, **attrs):
        self.G.add_node(artifact_id, **attrs)

    def add_edge(self, id1: str, id2: str, similarity: float, domain_distance: float):
        weight = round(similarity * domain_distance, 3)
        self.G.add_edge(
            id1, id2,
            similarity=similarity,
            domain_distance=domain_distance,
            weight=weight,
        )
        self._save()

    def get_invariant_clusters(self) -> list:
        components = list(nx.connected_components(self.G))
        return [c for c in components if len(c) >= 2]

    def get_bridges(self) -> list:
        try:
            return list(nx.bridges(self.G))
        except Exception:
            return []

    def node_centrality(self, artifact_id: str) -> float:
        centrality = nx.betweenness_centrality(self.G)
        return round(centrality.get(artifact_id, 0.0), 3)

    def _save(self):
        data = nx.node_link_data(self.G)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _load(self) -> nx.Graph:
        if not self.path.exists():
            return nx.Graph()
        try:
            data = json.loads(self.path.read_text())
            return nx.node_link_graph(data)
        except Exception:
            return nx.Graph()


# ══════════════════════════════════════════════
# СЛОЙ 3 — Детектор фазовых переходов
# ══════════════════════════════════════════════

class PhaseDetector:
    """
    Отличает случайный паттерн от устойчивой структуры.
    Фазовый переход = резкое изменение плотности кластера.
    """

    def is_stable_invariant(
        self,
        invariant: str,
        similar_artifacts: list,
        graph: InvariantGraph,
    ) -> tuple:
        if len(similar_artifacts) == 0:
            return False, "isolated"

        if len(similar_artifacts) == 1:
            sim = similar_artifacts[0]["similarity"]
            return sim > 0.80, "weak_pattern"

        vectors = np.array([
            _embedder.encode(a["invariant"]) for a in similar_artifacts
        ])
        try:
            labels = fclusterdata(vectors, t=0.35, criterion="distance", metric="cosine")
            unique_clusters = len(set(labels))
        except Exception:
            return False, "mixed_patterns"

        if unique_clusters == 1:
            return True, "stable_cluster"
        return False, "mixed_patterns"

    def detect_phase_transition(self, space: SemanticSpace, window: int = 10) -> dict:
        if len(space.vectors) < window:
            return {"transition": False, "density": 0.0, "window": window, "signal": "noise"}

        recent = np.array(space.vectors[-window:])
        norms = np.linalg.norm(recent, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normalized = recent / norms
        sim_matrix = normalized @ normalized.T
        np.fill_diagonal(sim_matrix, 0)
        density = sim_matrix.sum() / (window * (window - 1))

        return {
            "transition": bool(density > 0.60),
            "density": round(float(density), 3),
            "window": window,
            "signal": "sigma_primitive_candidate" if density > 0.60 else "noise",
        }


# ══════════════════════════════════════════════
# ОРКЕСТРАТОР
# ══════════════════════════════════════════════

def process_with_invariants(
    result: dict,
    job_id: str,
    space: SemanticSpace,
    graph: InvariantGraph,
    detector: PhaseDetector,
) -> dict:
    """
    Вызывается после генерации/верификации в process_query().
    Принимает result со структурой {generation, verification, domain, ...}
    Возвращает result с добавленным полем 'structural'.
    """
    gen = result.get("generation", {})

    # В нашем проекте инвариант = гипотеза из генератора
    invariant = gen.get("hypothesis")
    if not invariant:
        result["structural"] = {"error": "no hypothesis in generation"}
        return result

    b_sync = float(gen.get("b_sync", 0.0))
    domain = result.get("domain", "general")

    # 1. Найти похожие инварианты в семантическом пространстве
    similar = space.nearest(invariant, threshold=0.65)

    # 2. Проверить устойчивость паттерна
    is_stable, stability_type = detector.is_stable_invariant(invariant, similar, graph)

    # 3. Добавить текущий узел в граф и пространство
    space.add(job_id, invariant, domain, b_sync)
    graph.add_node(job_id, domain=domain, b_sync=b_sync, stability=stability_type)

    # 4. Провести рёбра к похожим узлам
    for s in similar:
        try:
            domain_vec = _embedder.encode(domain)
            neighbor_domain_vec = _embedder.encode(s["domain"])
            dist = round(float(cosine(domain_vec, neighbor_domain_vec)), 3)
        except Exception:
            dist = 0.0
        graph.add_edge(
            job_id,
            s["id"],
            similarity=s["similarity"],
            domain_distance=dist,
        )

    # 5. Проверить фазовый переход
    phase = detector.detect_phase_transition(space)

    # 6. Определить тип артефакта по таблице критериев из База.txt
    bridge_nodes = {n for e in graph.get_bridges() for n in e}
    centrality = graph.node_centrality(job_id)

    if phase["signal"] == "sigma_primitive_candidate":
        artifact_type = "sigma_primitive_candidate"
    elif job_id in bridge_nodes:
        artifact_type = "hyx-portal"
    elif stability_type == "stable_cluster":
        artifact_type = "hyx-artifact"
    elif stability_type == "weak_pattern":
        artifact_type = "weak_pattern"
    else:
        artifact_type = "noise"

    # 7. Корректировка b_sync при нестабильном паттерне
    if not is_stable and b_sync > 0.50:
        adjusted_b_sync = round(b_sync * 0.75, 2)
        result["generation"] = {**gen, "b_sync": adjusted_b_sync}

    # 8. Обогащаем результат структурными данными
    result["structural"] = {
        "similar_invariants": similar,
        "stability": stability_type,
        "is_stable": is_stable,
        "centrality": centrality,
        "is_bridge": job_id in bridge_nodes,
        "artifact_type": artifact_type,
        "phase_signal": phase,
    }

    return result
