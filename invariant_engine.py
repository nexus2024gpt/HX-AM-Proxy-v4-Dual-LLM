# invariant_engine.py — HX-AM v4 Invariant Engine
# Три слоя: SemanticSpace | InvariantGraph | PhaseDetector
#
# АНТИ-ВЕСА (v4.1):
#   weight = similarity × (1 + domain_distance) × specificity
#
#   specificity — насколько гипотеза отклоняется от центроида своего домена.
#   Терминологически банальная гипотеза сидит в центре кластера → specificity низкая.
#   Структурно уникальная — на периферии → specificity высокая.

from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import json
import logging
import networkx as nx
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import fclusterdata

_embedder = SentenceTransformer("all-MiniLM-L6-v2")
_logger = logging.getLogger("HXAM.engine")


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

    def encode(self, text: str) -> np.ndarray:
        """Публичный метод кодирования — используется Archivist."""
        return _embedder.encode(text)

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
        return self._nearest_by_vec(query_vec, top_k=top_k, threshold=threshold)

    def nearest_by_vec(self, vec: np.ndarray, top_k: int = 8, threshold: float = 0.0) -> list:
        """Поиск ближайших по готовому вектору — используется Archivist."""
        return self._nearest_by_vec(vec, top_k=top_k, threshold=threshold)

    def _nearest_by_vec(self, query_vec: np.ndarray, top_k: int = 5, threshold: float = 0.65) -> list:
        if not self.vectors:
            return []
        matrix = np.array(self.vectors)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        norms = np.where(norms == 0, 1e-9, norms)
        similarities = matrix @ query_vec / norms
        results = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                results.append({**self.meta[i], "similarity": round(float(sim), 3)})
        return sorted(results, key=lambda x: -x["similarity"])[:top_k]

    def domain_centroid(self, domain: str):
        """Центроид всех векторов домена (нормализованный). None если < 2 гипотез."""
        domain_vecs = [
            self.vectors[i]
            for i, m in enumerate(self.meta)
            if m.get("domain") == domain
        ]
        if len(domain_vecs) < 2:
            return None
        centroid = np.mean(domain_vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        return centroid

    def specificity(self, vec: np.ndarray, domain: str) -> float:
        """Специфичность = косинусное расстояние от центроида домена."""
        centroid = self.domain_centroid(domain)
        if centroid is None:
            return 0.5
        try:
            vec_norm = vec / np.linalg.norm(vec)
            dist = float(cosine(vec_norm, centroid))
            return round(min(dist, 1.0), 3)
        except Exception:
            return 0.5


# ══════════════════════════════════════════════
# СЛОЙ 2 — Структурный граф
# ══════════════════════════════════════════════

class InvariantGraph:
    """
    Граф инвариантов. Узлы — артефакты.

    Формула веса ребра (анти-шум):
        weight = similarity × (1 + domain_distance) × specificity
    """

    def __init__(self, graph_path: str = "artifacts/invariant_graph.json"):
        self.path = Path(graph_path)
        self.path.parent.mkdir(exist_ok=True)
        self.G = self._load()

    def add_node(self, artifact_id: str, **attrs):
        self.G.add_node(artifact_id, **attrs)

    def add_edge(
        self,
        id1: str,
        id2: str,
        similarity: float,
        domain_distance: float,
        specificity: float = 0.5,
    ):
        weight = round(similarity * (1 + domain_distance) * specificity, 3)
        self.G.add_edge(
            id1, id2,
            similarity=similarity,
            domain_distance=domain_distance,
            specificity=specificity,
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

    # ── Методы для Archivist ──────────────────────────────

    def get_similar_nodes(self, embedding: np.ndarray, space: "SemanticSpace", top_k: int = 8, exclude_id: str = None) -> list:
        """
        Возвращает топ-K ближайших узлов графа с их атрибутами и метриками ребра.
        Использует SemanticSpace для косинусного поиска, затем обогащает
        данными из графа (domain_distance, weight).
        """
        # Ищем ближайших по семантике (без порога — берём топ)
        candidates = space.nearest_by_vec(embedding, top_k=top_k * 2, threshold=0.0)

        results = []
        for c in candidates:
            if exclude_id and c["id"] == exclude_id:
                continue
            node_id = c["id"]
            node_attrs = self.G.nodes.get(node_id, {})
            results.append({
                "id": node_id,
                "domain": c.get("domain", node_attrs.get("domain", "general")),
                "similarity": c["similarity"],
                "domain_distance": 0.0,   # заполнится в archivist по домену
                "weight": 0.0,
                "b_sync": c.get("b_sync", node_attrs.get("b_sync", 0.0)),
                "stability": node_attrs.get("stability", "unknown"),
                "survival": node_attrs.get("survival", "UNKNOWN"),
                "invariant": c.get("invariant", ""),
            })
            if len(results) >= top_k:
                break
        return results

    def get_subgraph(self, node_id: str, depth: int = 2) -> dict:
        """
        Возвращает ego-граф вокруг узла глубиной depth.
        Для Archivist: контекст кластеров и мостов вокруг нового артефакта.
        """
        if node_id not in self.G:
            return {"nodes": [], "edges": [], "clusters": [], "bridges": []}

        ego = nx.ego_graph(self.G, node_id, radius=depth)
        clusters = [list(c) for c in nx.connected_components(ego) if len(c) >= 2]
        try:
            bridges = list(nx.bridges(ego))
        except Exception:
            bridges = []

        nodes = [
            {
                "id": n,
                **{k: v for k, v in self.G.nodes[n].items()
                   if k in ("domain", "b_sync", "stability", "survival", "specificity")}
            }
            for n in ego.nodes
        ]
        edges = [
            {
                "source": u,
                "target": v,
                "weight": d.get("weight", 0),
                "similarity": d.get("similarity", 0),
            }
            for u, v, d in ego.edges(data=True)
        ]
        return {
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
            "bridges": [list(b) for b in bridges],
        }

    def update_with_archivist(self, node_id: str, archivist_result: dict):
        """
        Обновляет атрибуты узла и novelty_weight на рёбрах.
        Вызывается после получения результата от Archivist.
        """
        if node_id not in self.G:
            return

        novelty_score = archivist_result.get("novelty_score", 0.6)
        novelty = archivist_result.get("novelty", "KNOWN")
        math_ver = archivist_result.get("mathematical_verification", "TERMINOLOGICAL")

        # Обновляем атрибуты узла
        self.G.nodes[node_id]["novelty"] = novelty
        self.G.nodes[node_id]["novelty_score"] = novelty_score
        self.G.nodes[node_id]["math_verification"] = math_ver

        tags = archivist_result.get("suggested_tags") or []
        normalized_tags = []
        for tag in tags:
            if tag == "hyx_portal":
                normalized_tags.append("hyx-portal")
            else:
                normalized_tags.append(tag)
        self.G.nodes[node_id]["suggested_tags"] = normalized_tags

        linked = archivist_result.get("linked_to") or []
        if node_id in linked:
            linked = [link for link in linked if link != node_id]
        self.G.nodes[node_id]["linked_to"] = linked

        # Обновляем novelty_weight на всех рёбрах узла
        for neighbor in list(self.G.neighbors(node_id)):
            edge_data = self.G[node_id][neighbor]
            base_weight = edge_data.get("weight", 0.5)
            novelty_category = archivist_result.get("novelty", "KNOWN")
            if novelty_category.startswith("REPHRASING_OF"):
                multiplier = 0.1
            elif novelty_category == "PHENOMENAL":
                multiplier = 0.9
            elif novelty_category == "NOVEL":
                multiplier = 0.6
            else:  # KNOWN
                multiplier = 0.3
            edge_data["novelty_weight"] = round(base_weight * multiplier, 3)

        self._save()
        _logger.info(f"Archivist updated node {node_id}: novelty={novelty} score={novelty_score}")

    def save_graph(self):
        """Публичный алиас для _save() — используется Archivist."""
        self._save()

    # ── Внутренние методы ────────────────────────────────

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

    def is_stable_invariant(
        self,
        invariant: str,
        similar_artifacts: list,
        graph: InvariantGraph,
        specificity: float = 0.5,
    ) -> tuple:
        """
        Определяет устойчивость инварианта.
        Кластер из банальных гипотез (avg_specificity < 0.3) не считается устойчивым.
        """
        if len(similar_artifacts) == 0:
            return False, "isolated"

        if len(similar_artifacts) == 1:
            sim = similar_artifacts[0]["similarity"]
            return sim > 0.80, "weak_pattern"

        similar_specs = [a.get("specificity", 0.5) for a in similar_artifacts]
        avg_specificity = (specificity + sum(similar_specs)) / (len(similar_artifacts) + 1)

        if avg_specificity < 0.3:
            return False, "low_specificity_cluster"

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

    def log_phenomenal(self, node_id: str, reason: str):
        """Логирует феноменальную кросс-доменную связь для мониторинга."""
        _logger.info(f"PHENOMENAL link detected: node={node_id} reason={reason}")


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
    gen = result.get("generation", {})
    ver = result.get("verification", {})

    invariant = gen.get("hypothesis")
    if not invariant:
        result["structural"] = {"error": "no hypothesis in generation"}
        return result

    b_sync = float(gen.get("b_sync", 0.0))
    domain = result.get("domain", "general")

    # Кодируем вектор ДО добавления в space
    current_vec = _embedder.encode(invariant)
    spec = space.specificity(current_vec, domain)

    # Извлечение survival из перевода верификатора
    translation = ver.get("translation", {})
    survival = translation.get("survival", "UNKNOWN") if isinstance(translation, dict) else "UNKNOWN"
    if survival == "UNKNOWN":
        _logger.warning(f"Job {job_id}: verifier did not return translation — Step 0 skipped")

    # 1. Найти похожие инварианты
    similar = space.nearest(invariant, threshold=0.65)

    # 2. Проверить устойчивость
    is_stable, stability_type = detector.is_stable_invariant(
        invariant, similar, graph, specificity=spec
    )

    if survival == "TERMINOLOGICAL":
        is_stable = False
        stability_type = "terminological"

    # 3. Добавить в пространство и граф
    space.add(job_id, invariant, domain, b_sync)
    graph.add_node(
        job_id,
        domain=domain,
        b_sync=b_sync,
        stability=stability_type,
        specificity=spec,
        survival=survival,
        translation=translation.get("translated_mechanism", "") if isinstance(translation, dict) else "",
    )

    # 4. Рёбра с анти-весами
    for s in similar:
        try:
            domain_vec = _embedder.encode(domain)
            neighbor_domain_vec = _embedder.encode(s["domain"])
            dist = round(float(cosine(domain_vec, neighbor_domain_vec)), 3)
        except Exception:
            dist = 0.0

        neighbor_spec = graph.G.nodes.get(s["id"], {}).get("specificity", 0.5)
        edge_spec = round((spec + neighbor_spec) / 2, 3)

        graph.add_edge(
            job_id, s["id"],
            similarity=s["similarity"],
            domain_distance=dist,
            specificity=edge_spec,
        )

    # 5. Фазовый переход
    phase = detector.detect_phase_transition(space)

    # 6. Тип артефакта
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

    # 7. Корректировка b_sync
    if (not is_stable or survival == "TERMINOLOGICAL") and b_sync > 0.50:
        result["generation"] = {**gen, "b_sync": round(b_sync * 0.75, 2)}

    # 8. Результат
    result["structural"] = {
        "similar_invariants": similar,
        "stability": stability_type,
        "is_stable": is_stable,
        "specificity": spec,
        "centrality": centrality,
        "is_bridge": job_id in bridge_nodes,
        "artifact_type": artifact_type,
        "phase_signal": phase,
        "translation": translation if isinstance(translation, dict) else {},
    }

    return result
