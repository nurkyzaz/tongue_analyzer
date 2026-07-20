"""TongueInsight API + demo server.

    uvicorn deployment.api.app:app --host 0.0.0.0 --port 7860
Env: TIH_SEG_CKPT, TIH_MT_CKPT select checkpoints (defaults to the combined seg + multitask_v2).
Set TIH_LLM_BACKEND=openai (+ TIH_LLM_BASE_URL/API_KEY/MODEL) to enable LLM-polished reports.
"""
import os
import json
import base64
import numpy as np
import cv2
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from deployment.api.service import TongueService

SEG = os.getenv("TIH_SEG_CKPT", "checkpoints/seg_combined/best.pt")
MT = os.getenv("TIH_MT_CKPT", "checkpoints/multitask_v5/best.pt")  # v5 = +severity +WB-aug (best benchmark)
API_KEY = os.getenv("TIH_API_KEY", "")          # if set, /analyze requires X-API-Key header
CORS_ORIGINS = os.getenv("TIH_CORS_ORIGINS", "*").split(",")
HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="TongueInsight Hybrid")
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")
_service = None


def service():
    global _service
    if _service is None:
        _service = TongueService(SEG, MT)
    return _service


class AnalyzeReq(BaseModel):
    image: str                 # data URL or bare base64
    metadata: dict | None = None


def _decode(data_url: str) -> np.ndarray:
    b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
    arr = np.frombuffer(base64.b64decode(b64), np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@app.get("/health")
def health():
    return {"status": "ok", "seg": SEG, "mt": MT}


@app.post("/analyze")
def analyze(req: AnalyzeReq, x_api_key: str = Header(default="")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    img = _decode(req.image)
    return service().analyze(img, metadata=req.metadata)


class RefineReq(BaseModel):
    base_confidence: float = 0.0
    answers: list                              # [{"weight": float, "answer": bool, "target_pattern"?: id}]
    patterns: list | None = None               # WS-B pass-2: [{"id","confidence",...}] to re-rank


@app.post("/refine")
def refine_endpoint(req: RefineReq):
    """Follow-up flow (WS-B pass-2). If `patterns` is given, the yes/no answers enter as evidence and
    RE-RANK the whole candidate set over the KG's answer->pattern edges (a strong 'yes' on the runner-up
    can overtake the lead). Otherwise falls back to the single-pattern log-odds update on
    `base_confidence`. Exploratory refinement of a traditional framework's perspective, not a diagnosis."""
    service()                       # ensures stage2_interpretation is on sys.path
    if req.patterns:
        from kg.refine_engine import rescore
        reranked, deltas = rescore(req.patterns, req.answers)
        return {"patterns": reranked, "deltas": deltas,
                "confidence": reranked[0]["confidence"] if reranked else req.base_confidence}
    from interpret import refine
    return {"confidence": refine(req.base_confidence, req.answers)}


@app.get("/examples")
def examples():
    p = os.path.join(HERE, "static", "examples", "examples.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"examples": []}


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(HERE, "static", "index.html"), encoding="utf-8") as f:
        return f.read()
