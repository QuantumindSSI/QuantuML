#!/usr/bin/env python3
"""QuantuML unified model-serving API for AKS.

One container image serves any of the three QuantuML models, selected by the
TASK environment variable (pqc | crypto | ddi). The base model is baked into
the image at /models/base; the LoRA adapter is pulled from the HuggingFace Hub
at startup, pinned to an exact revision (model-registry pattern: HF Hub is the
registry, ADAPTER_REVISION is the promoted version).

Environment variables:
  TASK                  pqc | crypto | ddi (required)
  BASE_MODEL_PATH       local path to base model (default /models/base)
  ADAPTER_REPO          HF repo overriding the per-task default
  ADAPTER_REVISION      git revision of the adapter to serve (default main)
  HF_TOKEN              only needed if the adapter repo is private
  APPLICATIONINSIGHTS_CONNECTION_STRING
                        enables Azure Monitor OpenTelemetry export when set
  MAX_CONCURRENT_GENERATIONS  concurrent generate calls (default 1; CPU-bound)
  PORT                  listen port (default 8000)

Endpoints:
  POST /generate   inference (validated request/response schema)
  GET  /healthz    liveness  - process is up
  GET  /readyz     readiness - model loaded and able to serve
  GET  /metrics    Prometheus exposition
  GET  /           service metadata
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("quantuml.serve")

TASK_CONFIG = {
    "pqc": {
        "title": "Post-Quantum Key Migration Advisor",
        "adapter_repo": "QuantumindSSI/quantuml-pqc-0.5b-poc",
        "system_prompt": (
            "You are a Post-Quantum Cryptography Migration Advisor. "
            "Provide detailed, actionable migration guidance for organizations transitioning "
            "from classical to quantum-safe cryptography. Be specific about algorithms, "
            "tools, timelines, and compliance requirements."
        ),
        "response_marker": "Advisor:",
    },
    "crypto": {
        "title": "Quantum-Resistant Crypto Analyzer",
        "adapter_repo": "QuantumindSSI/01-quantum-resistant-crypto-analyzer",
        "system_prompt": (
            "You are a Quantum-Resistant Cryptographic Protocol Analyzer. "
            "Your task is to analyze cryptographic protocol implementations, identify "
            "quantum-vulnerable patterns, classify the vulnerability type, explain the "
            "quantum attack vector, and recommend specific mitigations aligned with "
            "NIST PQC standards."
        ),
        "response_marker": "Analyzer:",
    },
    "ddi": {
        "title": "Drug Interaction Predictor",
        "adapter_repo": "QuantumindSSI/09-drug-interaction-predictor",
        "system_prompt": (
            "You are a Drug Interaction Predictor. Analyze medication lists for potential "
            "drug-drug interactions. Report interaction type, severity, mechanism, "
            "predicted outcomes, clinical recommendations, and evidence."
        ),
        "response_marker": "Analysis:",
    },
}

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
REQUESTS_TOTAL = Counter(
    "quantuml_requests_total", "Generate requests", ["task", "status"]
)
LATENCY_SECONDS = Histogram(
    "quantuml_generate_latency_seconds", "End-to-end generate latency", ["task"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 60, 120),
)
TOKENS_GENERATED = Counter(
    "quantuml_tokens_generated_total", "Completion tokens generated", ["task"]
)
INFLIGHT = Gauge("quantuml_inflight_requests", "Requests currently generating", ["task"])
MODEL_INFO = Gauge(
    "quantuml_model_info", "Static model metadata (value always 1)",
    ["task", "adapter_repo", "adapter_revision"],
)

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
class State:
    model = None
    tokenizer = None
    task = None
    cfg = None
    adapter_repo = None
    adapter_revision = None
    ready = False
    semaphore = None


STATE = State()


def _configure_telemetry() -> None:
    """Enable Azure Monitor OpenTelemetry when a connection string is present."""
    conn = os.environ.get("APPLICATIONINSIGHTS_CONNECTION_STRING")
    if not conn:
        logger.info("App Insights connection string not set; OTel export disabled")
        return
    from azure.monitor.opentelemetry import configure_azure_monitor

    configure_azure_monitor(connection_string=conn)
    logger.info("Azure Monitor OpenTelemetry configured")


def _load_model() -> None:
    """Load base model + pinned adapter revision. Raises on any failure."""
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    task = os.environ.get("TASK")
    if task not in TASK_CONFIG:
        raise ValueError(f"TASK must be one of {sorted(TASK_CONFIG)}, got {task!r}")
    cfg = TASK_CONFIG[task]

    base_path = os.environ.get("BASE_MODEL_PATH", "/models/base")
    adapter_repo = os.environ.get("ADAPTER_REPO", cfg["adapter_repo"])
    adapter_revision = os.environ.get("ADAPTER_REVISION", "main")

    logger.info("Loading base model from %s", base_path)
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.float32)

    logger.info("Fetching adapter %s @ %s", adapter_repo, adapter_revision)
    adapter_dir = snapshot_download(
        adapter_repo,
        revision=adapter_revision,
        allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
    )
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    model.eval()

    STATE.model = model
    STATE.tokenizer = tokenizer
    STATE.task = task
    STATE.cfg = cfg
    STATE.adapter_repo = adapter_repo
    STATE.adapter_revision = adapter_revision
    STATE.semaphore = asyncio.Semaphore(int(os.environ.get("MAX_CONCURRENT_GENERATIONS", "1")))
    MODEL_INFO.labels(task=task, adapter_repo=adapter_repo, adapter_revision=adapter_revision).set(1)
    STATE.ready = True
    logger.info("Model ready: task=%s adapter=%s@%s", task, adapter_repo, adapter_revision)


@asynccontextmanager
async def lifespan(app: FastAPI):
    _configure_telemetry()
    _load_model()
    yield
    STATE.ready = False
    STATE.model = None
    STATE.tokenizer = None


app = FastAPI(title="QuantuML Model Server", version="2.0.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=4000)
    max_tokens: int = Field(default=512, ge=16, le=1024)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)
    top_k: int = Field(default=40, ge=1, le=100)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    result: str
    task: str
    adapter_repo: str
    adapter_revision: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


def _generate_sync(req: GenerateRequest) -> GenerateResponse:
    """Blocking generation; executed in a worker thread."""
    cfg = STATE.cfg
    prompt = f"{cfg['system_prompt']}\n\nUser: {req.prompt}\n{cfg['response_marker']}"
    inputs = STATE.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    prompt_tokens = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        outputs = STATE.model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            do_sample=True,
            pad_token_id=STATE.tokenizer.pad_token_id,
        )
    latency = time.perf_counter() - start

    new_tokens = outputs[0][prompt_tokens:]
    text = STATE.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    LATENCY_SECONDS.labels(task=STATE.task).observe(latency)
    TOKENS_GENERATED.labels(task=STATE.task).inc(len(new_tokens))
    return GenerateResponse(
        result=text,
        task=STATE.task,
        adapter_repo=STATE.adapter_repo,
        adapter_revision=STATE.adapter_revision,
        prompt_tokens=prompt_tokens,
        completion_tokens=len(new_tokens),
        latency_ms=round(latency * 1000, 2),
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not STATE.ready:
        REQUESTS_TOTAL.labels(task=str(STATE.task), status="unavailable").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    async with STATE.semaphore:
        INFLIGHT.labels(task=STATE.task).inc()
        try:
            result = await asyncio.to_thread(_generate_sync, req)
            REQUESTS_TOTAL.labels(task=STATE.task, status="success").inc()
            return result
        except Exception:
            REQUESTS_TOTAL.labels(task=STATE.task, status="error").inc()
            logger.exception("generation failed")
            raise HTTPException(status_code=500, detail="Generation failed")
        finally:
            INFLIGHT.labels(task=STATE.task).dec()


@app.get("/healthz")
async def healthz():
    return {"status": "alive"}


@app.get("/readyz")
async def readyz():
    if not STATE.ready:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready", "task": STATE.task}


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    return {
        "service": STATE.cfg["title"] if STATE.cfg else "QuantuML Model Server",
        "task": STATE.task,
        "adapter_repo": STATE.adapter_repo,
        "adapter_revision": STATE.adapter_revision,
        "endpoints": ["/generate", "/healthz", "/readyz", "/metrics"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
