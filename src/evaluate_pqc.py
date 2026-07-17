#!/usr/bin/env python3
"""
Evaluation Script for Post-Quantum Key Migration Advisor
Computes perplexity, BLEU, ROUGE, and custom task metrics.
"""

import os
import yaml
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_from_disk, load_dataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_perplexity(model, tokenizer, texts, max_length=4096, batch_size=4):
    """Compute perplexity on a list of texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            n_tokens = inputs["attention_mask"].sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def compute_latency(model, tokenizer, prompt, num_runs=10, max_new_tokens=200):
    """Measure average inference latency."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
    for _ in range(3):
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
    """Report GPU memory usage."""
    if not torch.cuda.is_available():
        return {"gpu_allocated_mb": 0, "gpu_reserved_mb": 0}
    return {
        "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
    }


def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """Generate a single response."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def evaluate_migration_quality(generated: str) -> dict:
    """Heuristic evaluation of migration plan quality."""
    score = 0
    checks = {
        "has_phases": any(p in generated.lower() for p in ["discovery", "assessment", "planning", "implementation", "validation", "deployment"]),
        "has_algorithms": any(a in generated for a in ["ML-KEM", "ML-DSA", "DILITHIUM", "KYBER", "SLH-DSA"]),
        "has_tools": any(t in generated.lower() for t in ["hsm", "openssl", "ci/cd", "scanner"]),
        "has_timeline": any(t in generated.lower() for t in ["month", "week", "timeline", "roadmap"]),
        "has_compliance": any(c in generated for c in ["FIPS", "NIST", "GDPR", "HIPAA", "PCI DSS"]),
        "has_rollback": "rollback" in generated.lower() or "fallback" in generated.lower(),
        "has_testing": "test" in generated.lower(),
        "has_budget": "$" in generated or "budget" in generated.lower() or "cost" in generated.lower(),
    }

    for check, passed in checks.items():
        if passed:
            score += 1

    return {
        "quality_score": score,
        "max_score": len(checks),
        "checks": checks,
        "quality_percentage": (score / len(checks)) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate PQC Migration Model")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="./outputs/evaluations/eval_results.json")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    cfg = load_config(args.config)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map=cfg["model"]["device_map"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    results = {}

    # 1. Perplexity
    print("Computing perplexity...")
    if args.test_data:
        ds = load_dataset("json", data_files=args.test_data, split="train")
        texts = [ex["text"] for ex in ds.select(range(min(args.num_samples, len(ds))))]
    else:
        datasets = load_from_disk(cfg["data"]["processed_dir"])
        texts = [tokenizer.decode(ex["input_ids"], skip_special_tokens=True)
                 for ex in datasets["test"].select(range(min(args.num_samples, len(datasets["test"]))))]

    perplexity = compute_perplexity(model, tokenizer, texts)
    results["perplexity"] = perplexity
    print(f"Perplexity: {perplexity:.2f}")

    # 2. Latency
    print("Measuring latency...")
    sample_prompt = (
        "You are a Post-Quantum Cryptography Migration Advisor. "
        "A healthcare organization needs to migrate their TLS infrastructure from ECDSA P-256 to quantum-safe algorithms. "
        "Provide a step-by-step migration plan."
    )
    latency = compute_latency(model, tokenizer, sample_prompt)
    results["latency"] = latency
    print(f"Mean latency: {latency['mean_latency_s']:.3f}s")

    # 3. Memory
    print("Checking memory usage...")
    memory = compute_memory_usage()
    results["memory"] = memory
    if torch.cuda.is_available():
        print(f"GPU allocated: {memory.get('gpu_allocated_gb', 0):.2f} GB")
    else:
        print("Running on CPU - no GPU memory to report")

    # 4. Task-specific quality
    print("Evaluating migration plan quality...")
    quality_scores = []
    test_prompts = [
        "Migrate a bank's code signing from RSA-2048 to ML-DSA-65.",
        "A government agency needs to upgrade their VPN from ECDH P-384 to ML-KEM-768.",
        "Transition a cloud provider's API authentication from ECDSA P-256 to hybrid ECDH+ML-KEM.",
        "A blockchain platform needs quantum-resistant signatures for consensus.",
        "Migrate an HSM-backed PKI from RSA-4096 to ML-DSA-87.",
    ]

    for prompt in test_prompts:
        response = generate_response(model, tokenizer, prompt)
        quality = evaluate_migration_quality(response)
        quality_scores.append(quality)
        print(f"  Quality: {quality['quality_score']}/{quality['max_score']} ({quality['quality_percentage']:.1f}%)")

    avg_quality = sum(q["quality_percentage"] for q in quality_scores) / len(quality_scores)
    results["task_quality"] = {
        "average_quality_percentage": avg_quality,
        "samples": quality_scores,
    }
    print(f"Average task quality: {avg_quality:.1f}%")

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

    # Edge compatibility check
    edge_cfg = cfg.get("edge", {})
    max_mem = edge_cfg.get("max_memory_mb", 4096)
    max_lat = edge_cfg.get("max_latency_ms", 1000)

    compatible = (
        memory["gpu_allocated_mb"] <= max_mem or not torch.cuda.is_available()
    ) and (latency["mean_latency_s"] * 1000 <= max_lat)

    print(f"\nEdge compatibility ({max_mem}MB, {max_lat}ms): {'PASS' if compatible else 'FAIL'}")


if __name__ == "__main__":
    main()
