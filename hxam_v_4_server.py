# HX-AM v4 Full Server (Dual LLM + UI + Storage)
# Использует LLMClient с методами generate() и verify()

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from llm_client_v_4 import LLMClient

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HXAM.v4")

app = FastAPI(title="HX-AM v4")

# =====================
# MODELS
# =====================

class QueryRequest(BaseModel):
    text: str
    domain: str = "general"
    x_coordinate: float = 500.0

# =====================
# HELPERS
# =====================

def load_prompt(name: str) -> str:
    path = Path("prompts") / name
    if not path.exists():
        logger.warning(f"Prompt file not found: {path}")
        return ""
    return path.read_text(encoding="utf-8")

GEN_PROMPT = load_prompt("generator_prompt.txt")
VER_PROMPT = load_prompt("verifier_prompt.txt")

def extract_json(text: str) -> Dict[str, Any]:
    import re
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except:
        return {}

def save_artifact(job_id: str, data: Dict[str, Any]):
    path = Path("artifacts")
    path.mkdir(exist_ok=True)
    obj = {
        "id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data": data
    }
    file = path / f"{job_id}.json"
    file.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    return str(file)

def log_history(entry: Dict[str, Any]):
    path = Path("chat_history")
    path.mkdir(exist_ok=True)
    file = path / "history.jsonl"
    with open(file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# =====================
# CORE
# =====================

def process_query(req: QueryRequest):
    # Создаём один экземпляр клиента (он умеет и generate, и verify)
    client = LLMClient()

    job_id = hashlib.md5(f"{req.text}{time.time()}".encode()).hexdigest()[:12]

    # ---- Генерация ----
    gen_input = f"""{GEN_PROMPT}

Domain: {req.domain}
X: {req.x_coordinate}

User input:
{req.text}
"""
    logger.info(f"Job {job_id}: generating...")
    gen_raw = client.generate(gen_input)
    gen = extract_json(gen_raw)
    logger.info(f"Job {job_id}: generation done")

    # ---- Верификация ----
    ver_input = f"""{VER_PROMPT}

Hypothesis:
{json.dumps(gen, ensure_ascii=False)}
"""
    logger.info(f"Job {job_id}: verifying...")
    ver_raw = client.verify(ver_input, context=req.text)  # передаём исходный запрос как контекст
    ver = extract_json(ver_raw)
    logger.info(f"Job {job_id}: verification done")

    verdict = ver.get("verdict", "FALSE")
    confidence = ver.get("confidence", 0)

    save = False
    if verdict == "VALID" and confidence > 0.6:
        save = True
    elif verdict == "WEAK" and gen.get("b_sync", 0) > 0.7:
        save = True

    artifact = None
    if save:
        artifact = save_artifact(job_id, {"gen": gen, "ver": ver})

    log_history({
        "time": time.time(),
        "query": req.text,
        "gen": gen,
        "ver": ver,
        "saved": save
    })

    return {
        "job_id": job_id,
        "generation": gen,
        "verification": ver,
        "saved": save,
        "artifact": artifact
    }

# =====================
# API
# =====================

@app.post("/query")
def query(req: QueryRequest):
    return process_query(req)

@app.get("/history")
def history():
    file = Path("chat_history/history.jsonl")
    if not file.exists():
        return {"history": []}
    lines = file.read_text().splitlines()[-20:]
    return {"history": [json.loads(l) for l in lines]}

@app.get("/artifacts")
def artifacts():
    path = Path("artifacts")
    if not path.exists():
        return {"artifacts": []}
    files = sorted(path.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    result = [{"file": f.name} for f in files[:20]]
    return {"artifacts": result}

@app.get("/artifact/{name}")
def artifact(name: str):
    file = Path("artifacts") / name
    if not file.exists():
        raise HTTPException(404)
    return json.loads(file.read_text())

@app.get("/")
def ui():
    # Убедитесь, что файл называется index.html или index_v_4.html – подставьте нужное
    html_path = Path("index_v_4.html")
    if not html_path.exists():
        html_path = Path("index.html")
    if not html_path.exists():
        return HTMLResponse("<h1>index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

# =====================
# RUN
# =====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)