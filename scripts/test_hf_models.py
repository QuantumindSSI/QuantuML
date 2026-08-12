#!/usr/bin/env python3
"""Smoke-test the QuantumindSSI HuggingFace models downloaded locally.

Loads the shared base model (Qwen2-0.5B-Instruct) once per task, applies each
LoRA adapter, runs one domain-specific prompt, and reports output + latency.

Authentication: all model files are local; no HF token is required at runtime.
If Hub access were needed, huggingface_hub reads the token from its credential
store (~/.cache/huggingface/token) automatically - the token must never be
passed on the command line.

Usage:
    ./venv/bin/python scripts/test_hf_models.py [--max-new-tokens N] [--task pqc|crypto|ddi]
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL_DIR = PROJECT_ROOT / "base_model"

# Prompt formats mirror src/inference_{pqc,crypto,ddi}.py exactly.
TASKS = {
    "pqc": {
        "name": "PQC Migration Advisor (quantuml-pqc-0.5b-poc)",
        "adapter_dir": PROJECT_ROOT / "outputs" / "hf_models" / "pqc",
        "prompt": (
            "You are a Post-Quantum Cryptography Migration Advisor. "
            "Provide detailed, actionable migration guidance for organizations "
            "transitioning from classical to quantum-safe cryptography. "
            "Be specific about algorithms, tools, timelines, and compliance requirements.\n\n"
            "User: Our organization uses RSA-2048 for TLS and code signing. "
            "What is the recommended migration path to post-quantum cryptography?\nAdvisor:"
        ),
    },
    "crypto": {
        "name": "Quantum-Resistant Crypto Analyzer (01-quantum-resistant-crypto-analyzer)",
        "adapter_dir": PROJECT_ROOT / "outputs" / "hf_models" / "crypto",
        "prompt": (
            "You are a Quantum-Resistant Cryptographic Protocol Analyzer. "
            "Your task is to analyze cryptographic protocol implementations, "
            "identify quantum-vulnerable patterns, classify the vulnerability type, "
            "explain the quantum attack vector, and recommend specific mitigations "
            "aligned with NIST PQC standards.\n\n"
            "User: Analyze this TLS configuration: key exchange uses ECDHE with P-256, "
            "authentication uses RSA-2048 certificates, symmetric cipher is AES-128-GCM.\nAnalyzer:"
        ),
    },
    "ddi": {
        "name": "Drug Interaction Predictor (09-drug-interaction-predictor)",
        "adapter_dir": PROJECT_ROOT / "outputs" / "hf_models" / "ddi",
        "prompt": (
            "You are a Drug Interaction Predictor. Analyze medication lists for "
            "potential drug-drug interactions. Report interaction type, severity, "
            "mechanism, predicted outcomes, clinical recommendations, and evidence.\n\n"
            "Medications: warfarin, aspirin\nAnalysis:"
        ),
    },
}


def pick_device() -> str:
    """Return the fastest available torch device on this machine."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_task(task_key: str, device: str, max_new_tokens: int) -> dict:
    """Load base + adapter for one task, generate once, and return results.

    Returns a dict with keys: task, load_seconds, generate_seconds,
    tokens_generated, output. Raises FileNotFoundError if model files are
    missing and RuntimeError on generation failure.
    """
    task = TASKS[task_key]
    adapter_dir = task["adapter_dir"]
    if not BASE_MODEL_DIR.joinpath("config.json").is_file():
        raise FileNotFoundError(f"Base model not found at {BASE_MODEL_DIR}")
    if not adapter_dir.joinpath("adapter_config.json").is_file():
        raise FileNotFoundError(f"Adapter not found at {adapter_dir}")

    dtype = torch.float16 if device != "cpu" else torch.float32
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL_DIR, dtype=dtype)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - t0

    prompt = task["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    t1 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    generate_seconds = time.perf_counter() - t1

    new_tokens = outputs[0][prompt_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not text:
        raise RuntimeError(f"Task '{task_key}' produced empty output")

    # Free memory before the next model loads.
    del model, base
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

    return {
        "task": task["name"],
        "load_seconds": load_seconds,
        "generate_seconds": generate_seconds,
        "tokens_generated": len(new_tokens),
        "output": text,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-test downloaded QuantumindSSI models")
    parser.add_argument("--task", choices=sorted(TASKS), default=None,
                        help="Test a single task (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    device = pick_device()
    print(f"Device: {device} | torch {torch.__version__}")
    task_keys = [args.task] if args.task else list(TASKS)

    failures = 0
    for key in task_keys:
        print("\n" + "=" * 70)
        print(f"Testing: {TASKS[key]['name']}")
        print("=" * 70)
        try:
            result = run_task(key, device, args.max_new_tokens)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            failures += 1
            print(f"FAILED: {exc}")
            continue
        tps = result["tokens_generated"] / result["generate_seconds"]
        print(f"Load: {result['load_seconds']:.1f}s | "
              f"Generate: {result['generate_seconds']:.1f}s "
              f"({result['tokens_generated']} tokens, {tps:.1f} tok/s)")
        print("-" * 70)
        print(result["output"])

    print("\n" + "=" * 70)
    total = len(task_keys)
    print(f"Result: {total - failures}/{total} tasks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
