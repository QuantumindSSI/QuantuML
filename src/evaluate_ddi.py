#!/usr/bin/env python3
"""
Evaluation Script for Drug Interaction Predictor
Computes perplexity, latency, memory, and task-specific DDI metrics.
"""

import os
import yaml
import argparse
import json
import time
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk, load_dataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_perplexity(model, tokenizer, texts, max_length=1024, batch_size=2):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length, padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            n_tokens = inputs["attention_mask"].sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    return torch.exp(torch.tensor(avg_loss)).item()


def compute_latency(model, tokenizer, prompt, num_runs=5, max_new_tokens=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    for _ in range(2):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
        latencies.append(time.time() - start)
    return {
        "mean_latency_s": sum(latencies) / len(latencies),
        "min_latency_s": min(latencies),
        "max_latency_s": max(latencies),
        "p95_latency_s": sorted(latencies)[int(len(latencies) * 0.95)],
    }


def compute_memory_usage():
    if not torch.cuda.is_available():
        import psutil
        mem = psutil.virtual_memory()
        return {"gpu_allocated_mb": 0, "gpu_reserved_mb": 0, "cpu_ram_used_gb": mem.used / 1024**3, "cpu_ram_total_gb": mem.total / 1024**3}
    return {
        "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
    }


def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()
    return decoded


def evaluate_ddi_quality(generated: str, reference: dict) -> dict:
    gen_lower = generated.lower()
    score = 0
    checks = {}
    checks["mentions_severity"] = any(s.lower() in gen_lower for s in ["critical", "high", "medium", "low"])
    checks["mentions_mechanism"] = "mechanism" in gen_lower or "cyp" in gen_lower or "pharmacodynamic" in gen_lower or "pharmacokinetic" in gen_lower
    checks["has_recommendations"] = len(re.findall(r"\d+\.", generated)) >= 2 or "recommend" in gen_lower
    checks["mentions_evidence"] = "fda" in gen_lower or "guideline" in gen_lower or "trial" in gen_lower or "evidence" in gen_lower or "label" in gen_lower
    checks["mentions_drug_pair"] = any(d.lower() in gen_lower for d in reference.get("drug_pair", []))
    checks["has_headers"] = "##" in generated or "**" in generated
    checks["mentions_outcomes"] = "outcome" in gen_lower or "risk" in gen_lower or "toxicity" in gen_lower or "bleeding" in gen_lower
    for check, passed in checks.items():
        if passed:
            score += 1
    return {"quality_score": score, "max_score": len(checks), "checks": checks, "quality_percentage": (score / len(checks)) * 100}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Drug Interaction Predictor")
    parser.add_argument("--config", type=str, default="./configs/config_ddi.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--raw_test_data", type=str, default="./data/raw_ddi/test.jsonl")
    parser.add_argument("--output_file", type=str, default="./outputs/evaluations/eval_ddi_results.json")
    parser.add_argument("--num_samples", type=int, default=20)
    args = parser.parse_args()

    cfg = load_config(args.config)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], trust_remote_code=cfg["model"].get("trust_remote_code", False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]), device_map=cfg["model"].get("device_map", "auto"), trust_remote_code=cfg["model"].get("trust_remote_code", False))
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    results = {}

    print("Computing perplexity...")
    if args.test_data:
        ds = load_dataset("json", data_files=args.test_data, split="train")
        texts = [ex["text"] for ex in ds.select(range(min(args.num_samples, len(ds))))]
    else:
        datasets = load_from_disk(cfg["data"]["processed_dir"])
        texts = [tokenizer.decode(ex["input_ids"], skip_special_tokens=True) for ex in datasets["test"].select(range(min(args.num_samples, len(datasets["test"]))))]
    perplexity = compute_perplexity(model, tokenizer, texts)
    results["perplexity"] = perplexity
    print(f"Perplexity: {perplexity:.2f}")

    print("Measuring latency...")
    sample_prompt = "Analyze medication list: Warfarin, Aspirin, Omeprazole. Report interactions, severity, mechanism, and recommendations."
    latency = compute_latency(model, tokenizer, sample_prompt)
    results["latency"] = latency
    print(f"Mean latency: {latency['mean_latency_s']:.3f}s")

    print("Checking memory usage...")
    memory = compute_memory_usage()
    results["memory"] = memory
    if torch.cuda.is_available():
        print(f"GPU allocated: {memory.get('gpu_allocated_gb', 0):.2f} GB")
    else:
        print(f"CPU RAM used: {memory.get('cpu_ram_used_gb', 0):.2f} GB")

    print("Evaluating DDI quality...")
    quality_scores = []
    raw_tests = []
    if Path(args.raw_test_data).exists():
        with open(args.raw_test_data, "r") as f:
            for line in f:
                raw_tests.append(json.loads(line))

    test_prompts = raw_tests[:min(args.num_samples, len(raw_tests))]
    for i, item in enumerate(test_prompts):
        prompt = item.get("instruction", "") + "\n" + item.get("input", "")
        reference = {"drug_pair": item.get("drug_pair", [])}
        response = generate_response(model, tokenizer, prompt, max_new_tokens=400)
        quality = evaluate_ddi_quality(response, reference)
        quality_scores.append(quality)
        print(f"  Sample {i+1}: Quality {quality['quality_score']}/{quality['max_score']} ({quality['quality_percentage']:.1f}%)")

    if quality_scores:
        avg_quality = sum(q["quality_percentage"] for q in quality_scores) / len(quality_scores)
        results["task_quality"] = {"average_quality_percentage": avg_quality, "samples": quality_scores}
        print(f"Average task quality: {avg_quality:.1f}%")
    else:
        results["task_quality"] = {"average_quality_percentage": 0, "samples": []}

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

    edge_cfg = cfg.get("edge", {})
    max_mem = edge_cfg.get("max_memory_mb", 4096)
    max_lat = edge_cfg.get("max_latency_ms", 1000)
    mem_mb = memory.get("gpu_allocated_mb", 0) or (memory.get("cpu_ram_used_gb", 0) * 1024)
    compatible = mem_mb <= max_mem and (latency["mean_latency_s"] * 1000 <= max_lat)
    print(f"\nEdge compatibility ({max_mem}MB RAM, {max_lat}ms): {'PASS' if compatible else 'FAIL'}")


if __name__ == "__main__":
    main()
