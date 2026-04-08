# HX-AM v4 Full Server (Dual LLM + Invariant Engine + UI + Storage)
import os, json, time, hashlib, logging, re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from llm_client_v_4 import LLMClient
from archivist import Archivist
from invariant_engine import SemanticSpace, InvariantGraph, PhaseDetector, process_with_invariants

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HXAM.v4")

app = FastAPI(title="HX-AM v4")

Path("artifacts").mkdir(exist_ok=True)
logger.info("Загрузка семантического индекса...")
semantic_space = SemanticSpace()
logger.info("Загрузка графа инвариантов...")
invariant_graph = InvariantGraph()
phase_detector = PhaseDetector()
logger.info("Invariant Engine готов.")
archivist = Archivist(space=semantic_space, graph=invariant_graph)


class QueryRequest(BaseModel):
    text: str
    domain: str = "general"
    x_coordinate: float = 500.0


def load_prompt(name: str) -> str:
    path = Path("prompts") / name
    if not path.exists():
        logger.warning(f"Prompt file not found: {path}")
        return ""
    return path.read_text(encoding="utf-8")


GEN_PROMPT = load_prompt("generator_prompt.txt")
VER_PROMPT = load_prompt("verifier_prompt.txt")


def extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def resolve_domain(gen: dict, req_domain: str) -> str:
    gen_domain = gen.get("domain", "").strip().lower()
    if gen_domain and gen_domain != "general":
        return gen_domain
    if req_domain and req_domain != "general":
        return req_domain
    return "general"


def save_artifact(job_id: str, data: Dict[str, Any]) -> str:
    path = Path("artifacts")
    path.mkdir(exist_ok=True)
    obj = {"id": job_id, "created_at": datetime.now(timezone.utc).isoformat(), "data": data}
    file = path / f"{job_id}.json"
    file.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return str(file)


def log_history(entry: Dict[str, Any]):
    path = Path("chat_history")
    path.mkdir(exist_ok=True)
    file = path / "history.jsonl"
    with open(file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def process_query(req: QueryRequest):
    client = LLMClient()
    job_id = hashlib.md5(f"{req.text}{time.time()}".encode()).hexdigest()[:12]

    # RAG: найти похожие инварианты из базы знаний
    rag_similar = semantic_space.nearest(req.text, top_k=3, threshold=0.55)
    rag_block = ""
    if rag_similar:
        rag_block = "\n\nРелевантные инварианты из базы знаний системы:\n"
        for s in rag_similar:
            rag_block += f"- [{s['domain']}] {s['invariant']} (similarity: {s['similarity']})\n"

    gen_input = f"""{GEN_PROMPT}{rag_block}

X: {req.x_coordinate}

User input:
{req.text}
"""
    logger.info(f"Job {job_id}: generating... rag_hits={len(rag_similar)}")
    gen_raw, gen_model = client.generate(gen_input)
    gen = extract_json(gen_raw)

    domain = resolve_domain(gen, req.domain)
    logger.info(f"Job {job_id}: generation done → b_sync={gen.get('b_sync')} domain={domain} model={gen_model}")

    ver_input = f"""{VER_PROMPT}

Hypothesis:
{json.dumps(gen, ensure_ascii=False)}
"""
    logger.info(f"Job {job_id}: verifying...")
    ver_raw, ver_model = client.verify(ver_input, context=req.text)
    ver = extract_json(ver_raw)
    logger.info(f"Job {job_id}: verification done → verdict={ver.get('verdict')} confidence={ver.get('confidence')} model={ver_model}")

    verdict = ver.get("verdict", "FALSE")
    confidence = ver.get("confidence", 0)

    save = False
    if verdict == "VALID" and confidence > 0.6:
        save = True
    elif verdict == "WEAK" and gen.get("b_sync", 0) > 0.7:
        save = True

    result = {
        "job_id": job_id,
        "generation": gen,
        "verification": ver,
        "saved": save,
        "artifact": None,
        "domain": domain,
        "rag_context": rag_similar,
        "gen_model": gen_model,
        "ver_model": ver_model,
    }

    logger.info(f"Job {job_id}: running invariant engine...")
    result = process_with_invariants(
        result=result, job_id=job_id,
        space=semantic_space, graph=invariant_graph, detector=phase_detector,
    )
    logger.info(
        f"Job {job_id}: structural={result.get('structural', {}).get('artifact_type')} "
        f"stability={result.get('structural', {}).get('stability')} domain={domain}"
    )

    # Сохранение .hyx-portal.json для bridge-артефактов
    structural = result.get("structural", {})
    if structural.get("is_bridge"):
        portal_path = Path("artifacts") / f"{job_id}.hyx-portal.json"
        portal_data = {
            "id": job_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "type": "hyx-portal",
            "domain": domain,
            "hypothesis": gen.get("hypothesis", ""),
            "centrality": structural.get("centrality", 0),
            "similar_invariants": structural.get("similar_invariants", []),
            "phase_signal": structural.get("phase_signal", {}),
        }
        portal_path.write_text(json.dumps(portal_data, indent=2, ensure_ascii=False))
        logger.info(f"Job {job_id}: hyx-portal.json saved")

    if save:
        artifact_path = save_artifact(job_id, {
            "gen": result["generation"], "ver": ver,
            "domain": domain, "structural": structural,
        })
        result["artifact"] = artifact_path
        # Запускаем Archivist — оценка новизны после сохранения
        try:
            archivist_result = archivist.process(job_id)
            result["archivist"] = archivist_result
            logger.info(
                f"Job {job_id}: archivist → "
                f"novelty={archivist_result.get('novelty')} "
                f"score={archivist_result.get('novelty_score')}"
            )
        except Exception as e:
            logger.warning(f"Job {job_id}: archivist failed — {e}")
            result["archivist"] = None

    log_history({
        "time": time.time(), "query": req.text, "domain": domain,
        "gen": result["generation"], "ver": ver,
        "saved": save, "structural": structural,
        "rag_context": rag_similar,
    })

    return result


@app.post("/query")
def query(req: QueryRequest):
    return process_query(req)


@app.get("/rag/context")
def rag_context(text: str, top_k: int = 3):
    """Возвращает похожие инварианты для текста — используется UI для явного RAG-блока."""
    similar = semantic_space.nearest(text, top_k=top_k, threshold=0.55)
    return {"similar": similar, "count": len(similar)}


@app.get("/history")
def history():
    file = Path("chat_history/history.jsonl")
    if not file.exists():
        return {"history": []}
    lines = file.read_text().splitlines()[-20:]
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except Exception:
            continue
    return {"history": result}


@app.get("/artifacts")
def artifacts():
    path = Path("artifacts")
    if not path.exists():
        return {"artifacts": []}
    files = sorted(
        [f for f in path.glob("*.json") if f.stem != "invariant_graph"],
        key=lambda f: f.stat().st_mtime, reverse=True,
    )
    return {"artifacts": [{"file": f.name} for f in files[:20]]}


@app.get("/artifact/{name}")
def artifact(name: str):
    file = Path("artifacts") / name
    if not file.exists():
        raise HTTPException(404)
    return json.loads(file.read_text())


@app.get("/graph/data")
def graph_data():
    G = invariant_graph.G
    nodes = []
    for node_id, attrs in G.nodes(data=True):
        nodes.append({
            "id": node_id,
            "domain": attrs.get("domain", "general"),
            "b_sync": attrs.get("b_sync", 0.0),
            "stability": attrs.get("stability", "unknown"),
        })
    links = []
    for u, v, attrs in G.edges(data=True):
        links.append({
            "source": u, "target": v,
            "weight": attrs.get("weight", 0.0),
            "similarity": attrs.get("similarity", 0.0),
            "domain_distance": attrs.get("domain_distance", 0.0),
        })
    clusters = [list(c) for c in invariant_graph.get_invariant_clusters()]
    bridge_nodes = {n for e in invariant_graph.get_bridges() for n in e}
    cluster_map = {}
    for i, cluster in enumerate(clusters):
        for nid in cluster:
            cluster_map[nid] = i
    for node in nodes:
        node["cluster"] = cluster_map.get(node["id"], -1)
        node["is_bridge"] = node["id"] in bridge_nodes
    return {
        "nodes": nodes, "links": links,
        "meta": {
            "total_nodes": len(nodes), "total_edges": len(links),
            "cluster_count": len(clusters), "bridge_count": len(bridge_nodes),
        }
    }


@app.get("/graph")
def graph():
    clusters = [list(c) for c in invariant_graph.get_invariant_clusters()]
    bridges = invariant_graph.get_bridges()
    return {
        "nodes": len(invariant_graph.G.nodes), "edges": len(invariant_graph.G.edges),
        "clusters": clusters, "bridges": bridges, "cluster_count": len(clusters),
    }


@app.get("/phase")
def phase():
    return phase_detector.detect_phase_transition(semantic_space)


@app.get("/")
def ui():
    html_path = Path("index_v_4.html")
    if not html_path.exists():
        html_path = Path("index.html")
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)