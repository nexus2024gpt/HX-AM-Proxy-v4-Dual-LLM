# HX-AM v4 Full Server — v4.2 (new prompt schema: operationalization + refined_hypothesis)
import os, json, time, hashlib, logging, re
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from llm_client_v_4 import LLMClient
from archivist import Archivist
from invariant_engine import SemanticSpace, InvariantGraph, PhaseDetector, process_with_invariants
from pipeline_guard import PipelineGuard, RollbackManager, QuarantineLog
from question_generator import QuestionGenerator
from api_usage_tracker import tracker

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

guard = PipelineGuard()
quarantine = QuarantineLog()
question_gen = QuestionGenerator(space=semantic_space, graph=invariant_graph)


class QueryRequest(BaseModel):
    text: str
    domain: str = "general"
    x_coordinate: float = 500.0

class ProvidersUpdateRequest(BaseModel):
    providers: List[Dict[str, Any]]

class ProviderAddRequest(BaseModel):
    id: str
    provider: str
    account: str
    label: str
    api_key: str
    api_base: str
    model: str
    roles: List[str]
    enabled: bool = True
    priority: int = 99

class ResetRequest(BaseModel):
    scope: str = "today"


def load_prompt(name: str) -> str:
    path = Path("prompts") / name
    if not path.exists():
        logger.warning(f"Prompt file not found: {path}")
        return ""
    return path.read_text(encoding="utf-8")

GEN_PROMPT = load_prompt("generator_prompt.txt")
VER_PROMPT = load_prompt("verifier_prompt.txt")


def _close_brackets(s: str) -> str:
    """Закрывает незакрытые { и [ в обрезанном JSON (не трогает строки)."""
    depth_curly = 0
    depth_square = 0
    in_string = False
    escape = False
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{': depth_curly += 1
            elif ch == '}': depth_curly = max(0, depth_curly - 1)
            elif ch == '[': depth_square += 1
            elif ch == ']': depth_square = max(0, depth_square - 1)
    s = s.rstrip().rstrip(',')
    s += ']' * depth_square
    s += '}' * depth_curly
    return s


def extract_json(text: str) -> Dict[str, Any]:
    """
    Извлекает JSON из ответа LLM.

    Четыре стратегии в порядке убывания надёжности:
      1. Прямой парсинг полного JSON
      2. Скользящее окно — обрезаем по последней , или : и закрываем скобки
      3. Regex-извлечение критичных полей (verdict, confidence, survival, translation)
      4. Пустой dict — пайплайн уйдёт в quarantine с кодом VER_EMPTY_JSON

    Случай "обрезан до verdict" (maxOutputTokens слишком мал) → стратегия 3
    возвращает частичный dict; validate_ver выдаст VER_NO_VERDICT.
    Основной fix: maxOutputTokens=4096 для Gemini-верификатора (llm_client_v_4.py).
    """
    if not text:
        return {}
    text = re.sub(r"```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    start = text.find('{')
    if start == -1:
        return {}
    candidate = text[start:]

    # 1. Прямой парсинг
    end = candidate.rfind('}')
    if end != -1:
        try:
            return json.loads(candidate[:end+1])
        except Exception:
            pass

    # 2. Скользящее окно по разделителям
    s = candidate.rstrip()
    for trim_char in [',', ':']:
        last = s.rfind(trim_char)
        if last > 0:
            try:
                closed = _close_brackets(s[:last].rstrip())
                result = json.loads(closed)
                if isinstance(result, dict) and result:
                    logger.info("extract_json: recovered via bracket-closing")
                    return result
            except Exception:
                pass

    # 3. Regex-извлечение критичных полей из обрезанного текста
    partial: Dict[str, Any] = {}
    for key in ("verdict", "confidence"):
        m = re.search(rf'"{key}"\s*:\s*"([^"]+)"', candidate)
        if m:
            partial[key] = m.group(1)
        else:
            m = re.search(rf'"{key}"\s*:\s*([0-9.]+)', candidate)
            if m:
                try:
                    partial[key] = float(m.group(1))
                except Exception:
                    pass

    # Восстановить translation
    trans_m = re.search(r'"translation"\s*:\s*(\{[^}]+\})', candidate, re.DOTALL)
    if trans_m:
        try:
            partial["translation"] = json.loads(trans_m.group(1))
        except Exception:
            td = re.search(r'"target_domain"\s*:\s*"([^"]+)"', candidate)
            sv = re.search(r'"survival"\s*:\s*"([^"]+)"', candidate)
            if td or sv:
                partial["translation"] = {}
                if td: partial["translation"]["target_domain"] = td.group(1)
                if sv: partial["translation"]["survival"] = sv.group(1)

    if partial:
        logger.warning(
            f"extract_json: partial recovery — fields={list(partial.keys())} "
            f"(response likely truncated by token limit)"
        )
        return partial

    logger.warning(f"extract_json: failed to parse from: {candidate[:200]}")
    return {}


def resolve_domain(gen: dict, req_domain: str) -> str:
    gen_domain = gen.get("domain", "").strip().lower()
    if gen_domain and gen_domain != "general":
        return gen_domain
    if req_domain and req_domain != "general":
        return req_domain
    return "general"


def save_artifact(job_id: str, data: Dict[str, Any]) -> Path:
    path = Path("artifacts")
    path.mkdir(exist_ok=True)
    obj = {"id": job_id, "created_at": datetime.now(timezone.utc).isoformat(), "data": data}
    file = path / f"{job_id}.json"
    file.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return file


def log_history(entry: Dict[str, Any]):
    path = Path("chat_history")
    path.mkdir(exist_ok=True)
    file = path / "history.jsonl"
    with open(file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _rejected_response(job_id: str, code: str, reason: str, stage: str) -> dict:
    return {
        "job_id": job_id, "rejected": True, "stage": stage,
        "failure_code": code, "reason": reason,
        "generation": None, "verification": None,
        "saved": False, "artifact": None, "domain": None,
    }


def filter_rag_diversity(similar: List[dict], max_per_domain: int = 1, sim_cap: float = 0.88) -> List[dict]:
    """
    RAG anti-amplification filter (v4.2):
    - Отсекает инварианты с sim >= sim_cap (генератор скопирует терминологию)
    - Ограничивает 1 инвариант на домен (не усиливать доминирующий кластер)
    """
    seen_domains: dict = {}
    result = []
    for s in similar:
        sim = s.get("similarity", 0.0)
        domain = s.get("domain", "general")
        if sim >= sim_cap:
            continue
        if seen_domains.get(domain, 0) >= max_per_domain:
            continue
        seen_domains[domain] = seen_domains.get(domain, 0) + 1
        result.append(s)
    if len(similar) != len(result):
        logger.info(f"RAG filter: {len(similar)} -> {len(result)} (dropped {len(similar)-len(result)})")
    return result


def process_query(req: QueryRequest):
    client = LLMClient()
    job_id = hashlib.md5(f"{req.text}{time.time()}".encode()).hexdigest()[:12]
    rollback = RollbackManager()
    gen_model = "unknown"
    ver_model = "unknown"

    try:
        # ══ ЭТАП 1 — ГЕНЕРАЦИЯ ══
        rag_raw = semantic_space.nearest(req.text, top_k=5, threshold=0.55)
        rag_similar = filter_rag_diversity(rag_raw, max_per_domain=1, sim_cap=0.88)

        rag_block = ""
        if rag_similar:
            rag_block = "\n\nRAG context (structural inspiration only — do NOT copy phrases):\n"
            for s in rag_similar:
                rag_block += f"- [{s['domain']}] {s['invariant']} (sim:{s['similarity']})\n"

        gen_input = f"{GEN_PROMPT}{rag_block}\n\nX: {req.x_coordinate}\n\nUser input:\n{req.text}"

        logger.info(f"Job {job_id}: generating... rag_raw={len(rag_raw)} rag_filtered={len(rag_similar)}")
        gen_raw, gen_model = client.generate(gen_input)

        vr = guard.validate_gen_raw(gen_raw, gen_model)
        if not vr:
            quarantine.record(job_id, req.text, vr.code, vr.reason, "generation", gen_model=gen_model)
            return _rejected_response(job_id, vr.code, vr.reason, "generation")

        gen = extract_json(gen_raw)
        vr = guard.validate_gen(gen, gen_model)
        if not vr:
            quarantine.record(job_id, req.text, vr.code, vr.reason, "generation", gen_model=gen_model)
            return _rejected_response(job_id, vr.code, vr.reason, "generation")

        domain = resolve_domain(gen, req.domain)
        logger.info(f"Job {job_id}: gen OK -> b_sync={gen.get('b_sync')} domain={domain} model={gen_model}")

        # ══ ЭТАП 2 — ВЕРИФИКАЦИЯ ══
        ver_input = f"{VER_PROMPT}\n\nHypothesis:\n{json.dumps(gen, ensure_ascii=False)}"

        logger.info(f"Job {job_id}: verifying...")
        ver_raw, ver_model = client.verify(ver_input, context=req.text)

        vr = guard.validate_ver_raw(ver_raw, ver_model)
        if not vr:
            quarantine.record(job_id, req.text, vr.code, vr.reason, "verification", gen_model=gen_model, ver_model=ver_model)
            return _rejected_response(job_id, vr.code, vr.reason, "verification")

        ver = extract_json(ver_raw)
        vr = guard.validate_ver(ver, ver_model, raw=ver_raw)
        if not vr:
            quarantine.record(job_id, req.text, vr.code, vr.reason, "verification", gen_model=gen_model, ver_model=ver_model)
            return _rejected_response(job_id, vr.code, vr.reason, "verification")

        verdict = ver.get("verdict", "FALSE")
        confidence = ver.get("confidence", 0)
        logger.info(f"Job {job_id}: ver OK -> verdict={verdict} conf={confidence} model={ver_model}")

        # Если верификатор предложил уточнённую гипотезу — логируем
        if ver.get("refined_hypothesis"):
            logger.info(f"Job {job_id}: refined_hypothesis present → available in result")

        # ══ ЭТАП 3 — РЕШЕНИЕ О СОХРАНЕНИИ ══
        save = False
        if verdict == "VALID" and confidence > 0.6:
            save = True
        elif verdict == "WEAK" and float(gen.get("b_sync", 0)) > 0.7:
            save = True

        result = {
            "job_id": job_id, "generation": gen, "verification": ver,
            "saved": save, "artifact": None, "domain": domain,
            "rag_context": rag_similar, "rag_dropped": len(rag_raw) - len(rag_similar),
            "gen_model": gen_model, "ver_model": ver_model,
        }

        # ══ ЭТАП 4 — INVARIANT ENGINE ══
        rollback.snapshot_space(len(semantic_space.vectors))
        rollback.register_graph_node(job_id)

        logger.info(f"Job {job_id}: running invariant engine...")
        result = process_with_invariants(
            result=result, job_id=job_id,
            space=semantic_space, graph=invariant_graph, detector=phase_detector,
        )
        structural = result.get("structural", {})
        phase_signal = structural.get("phase_signal", {})
        logger.info(
            f"Job {job_id}: engine OK -> type={structural.get('artifact_type')} "
            f"phase={phase_signal.get('signal')} unique_domains={phase_signal.get('unique_domains')}"
        )

        # ══ ЭТАП 5 — СОХРАНЕНИЕ ══
        if structural.get("is_bridge"):
            portal_path = Path("artifacts") / f"{job_id}.hyx-portal.json"
            portal_data = {
                "id": job_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "type": "hyx-portal", "domain": domain,
                "hypothesis": gen.get("hypothesis", ""),
                "centrality": structural.get("centrality", 0),
                "similar_invariants": structural.get("similar_invariants", []),
                "phase_signal": phase_signal,
            }
            portal_path.write_text(json.dumps(portal_data, indent=2, ensure_ascii=False))
            rollback.register_file(portal_path)

        if save:
            artifact_file = save_artifact(job_id, {
                "gen": result["generation"], "ver": ver,
                "domain": domain, "structural": structural,
            })
            rollback.register_file(artifact_file)
            result["artifact"] = str(artifact_file)
            try:
                archivist_result = archivist.process(job_id)
                result["archivist"] = archivist_result
            except Exception as e:
                logger.warning(f"Job {job_id}: archivist failed - {e}")
                result["archivist"] = None

        log_history({
            "time": time.time(), "query": req.text, "domain": domain,
            "gen": result["generation"], "ver": ver, "saved": save,
            "structural": structural, "rag_context": rag_similar,
            "rag_dropped": len(rag_raw) - len(rag_similar),
        })

        rollback.clear()
        return result

    except Exception as exc:
        logger.error(f"Job {job_id}: unexpected exception - {exc}", exc_info=True)
        actions = rollback.rollback(semantic_space, invariant_graph)
        quarantine.record(job_id, req.text, "PIPELINE_EXCEPTION", str(exc), "unknown",
                          gen_model=gen_model, ver_model=ver_model, rollback_actions=actions)
        return _rejected_response(job_id, "PIPELINE_EXCEPTION", str(exc), "unknown")


# ══════════════════════════════════════
# ЭНДПОИНТЫ
# ══════════════════════════════════════

@app.post("/query")
def query(req: QueryRequest):
    return process_query(req)

@app.get("/quarantine")
def get_quarantine(limit: int = 20):
    return {"quarantine": quarantine.recent(limit)}

@app.get("/rag/context")
def rag_context(text: str, top_k: int = 3):
    similar = semantic_space.nearest(text, top_k=top_k, threshold=0.55)
    filtered = filter_rag_diversity(similar, max_per_domain=1, sim_cap=0.88)
    return {"similar": filtered, "similar_raw": similar, "count": len(filtered), "dropped": len(similar)-len(filtered)}

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
    files = sorted([f for f in path.glob("*.json") if f.stem != "invariant_graph"],
                   key=lambda f: f.stat().st_mtime, reverse=True)
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
        nodes.append({"id": node_id, "domain": attrs.get("domain","general"),
                       "b_sync": attrs.get("b_sync",0.0), "stability": attrs.get("stability","unknown")})
    links = []
    for u, v, attrs in G.edges(data=True):
        links.append({"source": u, "target": v, "weight": attrs.get("weight",0.0),
                       "similarity": attrs.get("similarity",0.0), "domain_distance": attrs.get("domain_distance",0.0)})
    clusters = [list(c) for c in invariant_graph.get_invariant_clusters()]
    bridge_nodes = {n for e in invariant_graph.get_bridges() for n in e}
    cluster_map = {}
    for i, cluster in enumerate(clusters):
        for nid in cluster:
            cluster_map[nid] = i
    for node in nodes:
        node["cluster"] = cluster_map.get(node["id"], -1)
        node["is_bridge"] = node["id"] in bridge_nodes
    return {"nodes": nodes, "links": links, "meta": {"total_nodes": len(nodes), "total_edges": len(links),
            "cluster_count": len(clusters), "bridge_count": len(bridge_nodes)}}

@app.get("/graph")
def graph():
    clusters = [list(c) for c in invariant_graph.get_invariant_clusters()]
    bridges = invariant_graph.get_bridges()
    return {"nodes": len(invariant_graph.G.nodes), "edges": len(invariant_graph.G.edges),
            "clusters": clusters, "bridges": bridges, "cluster_count": len(clusters)}

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

@app.get("/question/suggest")
def suggest_question():
    try:
        return question_gen.suggest_novel()
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/question/clarify/{artifact_id}")
def clarify_artifact(artifact_id: str):
    try:
        return question_gen.suggest_clarification(artifact_id)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/question/candidates")
def clarification_candidates():
    return {"candidates": question_gen.list_clarification_candidates()}


# ══ TRACKER ══

@app.get("/tracker/stats")
def tracker_stats():
    return tracker.get_stats()

@app.get("/tracker/providers")
def tracker_providers_get():
    return {"providers": tracker.get_providers(), "known_models": tracker.get_known_models()}

@app.post("/tracker/providers")
def tracker_providers_update(req: ProvidersUpdateRequest):
    ok = tracker.update_providers(req.providers)
    if not ok:
        raise HTTPException(400, "Ошибка сохранения конфига")
    return {"ok": True, "count": len(req.providers)}

@app.post("/tracker/providers/add")
def tracker_provider_add(req: ProviderAddRequest):
    ok = tracker.add_provider(req.dict())
    if not ok:
        raise HTTPException(400, "Ошибка добавления провайдера")
    return {"ok": True}

@app.delete("/tracker/providers/{provider_id}")
def tracker_provider_delete(provider_id: str):
    ok = tracker.delete_provider(provider_id)
    if not ok:
        raise HTTPException(404, f"Провайдер {provider_id} не найден")
    return {"ok": True}

@app.post("/tracker/reset")
def tracker_reset_stats(req: ResetRequest):
    if req.scope == "all":
        tracker.reset_all()
    else:
        tracker.reset_today()
    return {"ok": True, "scope": req.scope}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
