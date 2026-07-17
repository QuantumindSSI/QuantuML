#!/usr/bin/env python3
"""
FastAPI Server for Drug Interaction Predictor
Production-ready inference endpoint with request validation and logging.
"""

import os
import yaml
import argparse
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from uvicorn import run as uvicorn_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL = None
TOKENIZER = None
CONFIG = None


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_once(model_path: str, adapter_path: Optional[str], cfg: dict):
    global MODEL, TOKENIZER
    logger.info(f"Loading model from {model_path}")
    TOKENIZER = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], trust_remote_code=cfg["model"].get("trust_remote_code", False))
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
        TOKENIZER.pad_token_id = TOKENIZER.eos_token_id
    MODEL = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]), device_map=cfg["model"]["device_map"], trust_remote_code=cfg["model"].get("trust_remote_code", False))
    if adapter_path:
        MODEL = PeftModel.from_pretrained(MODEL, adapter_path)
        MODEL = MODEL.merge_and_unload()
        logger.info("LoRA adapter merged")
    MODEL.eval()
    logger.info("Model loaded and ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CONFIG
    CONFIG = load_config(os.environ.get("CONFIG_PATH", "./configs/config_ddi.yaml"))
    model_path = os.environ.get("MODEL_PATH", "./outputs/models_ddi_quick/checkpoint-final")
    adapter_path = os.environ.get("ADAPTER_PATH", None)
    load_model_once(model_path, adapter_path, CONFIG)
    yield
    if MODEL is not None:
        del MODEL
    if TOKENIZER is not None:
        del TOKENIZER
    torch.cuda.empty_cache()


app = FastAPI(
    title="Drug Interaction Predictor API",
    description="Enterprise API for detecting adverse drug-drug interactions from medication lists",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictRequest(BaseModel):
    medications: str = Field(..., min_length=5, max_length=2000, description="Comma-separated medication list")
    patient_context: Optional[str] = Field(default=None, max_length=1000, description="Optional patient demographics and conditions")
    max_tokens: int = Field(default=512, ge=50, le=2048)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.1, le=1.0)


class PredictResponse(BaseModel):
    result: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if MODEL is None or TOKENIZER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    system = (
        "You are a Drug Interaction Predictor. Analyze the medication list for potential drug-drug interactions. "
        "Report interaction type, severity, mechanism, predicted outcomes, clinical recommendations, and evidence."
    )
    user_msg = f"## Medication List\n{request.medications}\n\n"
    if request.patient_context:
        user_msg += f"## Patient Context\n{request.patient_context}\n\n"
    prompt = f"{system}\n\n{user_msg}Predictor:"
    inputs = TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(MODEL.device) for k, v in inputs.items()}
    prompt_tokens = inputs["input_ids"].shape[1]
    start = time.time()
    with torch.no_grad():
        outputs = MODEL.generate(**inputs, max_new_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p, do_sample=True, pad_token_id=TOKENIZER.pad_token_id)
    latency_ms = (time.time() - start) * 1000
    generated = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    completion_tokens = outputs.shape[1] - prompt_tokens
    return PredictResponse(result=generated, model=CONFIG["project"]["name"], prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens, latency_ms=round(latency_ms, 2))


@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": MODEL is not None, "device": str(MODEL.device) if MODEL else "none"}


@app.get("/")
async def root():
    return {"service": "Drug Interaction Predictor", "version": "1.0.0", "endpoints": ["/predict", "/health"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", type=str, default="./configs/config_ddi.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    os.environ["CONFIG_PATH"] = args.config
    os.environ["MODEL_PATH"] = args.model_path
    if args.adapter_path:
        os.environ["ADAPTER_PATH"] = args.adapter_path
    uvicorn_run("api_server_ddi:app", host=args.host, port=args.port, workers=args.workers, log_level="info")


if __name__ == "__main__":
    main()
