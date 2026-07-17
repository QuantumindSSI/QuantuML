#!/usr/bin/env python3
"""
Evaluation Script for Quantum-Resistant Cryptographic Protocol Analyzer
Computes perplexity, latency, memory, and task-specific vulnerability detection metrics.
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

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


def compute_latency(model, tokenizer, prompt, num_runs=5, max_new_tokens=200):
    """Measure average inference latency."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup
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
    """Report GPU/CPU memory usage."""
    if not torch.cuda.is_available():
        import psutil
        mem = psutil.virtual_memory()
        return {
            "gpu_allocated_mb": 0,
            "gpu_reserved_mb": 0,
            "cpu_ram_used_gb": mem.used / 1024**3,
            "cpu_ram_total_gb": mem.total / 1024**3,
        }
    return {
        "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
        "gpu_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
    }


def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """Generate a single response."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Strip prompt if echoed
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()
    return decoded


def evaluate_vulnerability_detection(generated: str, reference: dict) -> dict:
    """Heuristic evaluation of vulnerability detection quality."""
    gen_lower = generated.lower()
    score = 0
    checks = {}

    # Check if vulnerability type is mentioned
    vuln_type_lower = reference.get("vulnerability_type", "").lower()
    checks["mentions_vulnerability_type"] = vuln_type_lower in gen_lower or any(
        keyword in gen_lower for keyword in vuln_type_lower.split()
    )

    # Check if attack vector is mentioned
    attack_vector_lower = reference.get("quantum_attack_vector", "").lower()
    checks["mentions_attack_vector"] = attack_vector_lower in gen_lower or any(
        keyword in gen_lower for keyword in attack_vector_lower.split()
    )

    # Check if severity is mentioned
    severity = reference.get("severity", "").lower()
    checks["mentions_severity"] = severity in gen_lower

    # Check if mitigations are present
    checks["has_mitigations"] = len(re.findall(r"\d+\.", generated)) >= 2 or "mitigation" in gen_lower

    # Check for NIST references
    checks["mentions_nist"] = "nist" in gen_lower or "fips" in gen_lower or "cnsa" in gen_lower

    # Check for specific algorithm mentions
    algo = reference.get("algorithm_used", "").lower()
    checks["mentions_algorithm"] = algo in gen_lower

    # Check for protocol name
    protocol = reference.get("protocol_name", "").lower()
    checks["mentions_protocol"] = protocol in gen_lower

    # Structure quality
    checks["has_headers"] = "##" in generated or "**" in generated

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
    parser = argparse.ArgumentParser(description="Evaluate Quantum-Resistant Crypto Protocol Analyzer")
    parser.add_argument("--config", type=str, default="./configs/config_crypto_analyzer.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--raw_test_data", type=str, default="./data/raw_vuln/test.jsonl",
                        help="Raw test data with reference labels for task metrics")
    parser.add_argument("--output_file", type=str, default="./outputs/evaluations/eval_crypto_results.json")
    parser.add_argument("--num_samples", type=int, default=50)
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
        device_map=cfg["model"].get("device_map", "auto"),
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
        "Analyze the following TLS 1.2 implementation for quantum-vulnerable patterns:\n"
        "```python\n"
        "context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)\n"
        "context.set_ciphers('RSA-AES256-GCM-SHA384')\n"
        "```\n"
        "Identify the vulnerability type, quantum attack vector, and recommend mitigations."
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
        print(f"CPU RAM used: {memory.get('cpu_ram_used_gb', 0):.2f} GB")

    # 4. Task-specific quality
    print("Evaluating vulnerability detection quality...")
    quality_scores = []

    # Load raw test data for labels
    raw_tests = []
    if Path(args.raw_test_data).exists():
        with open(args.raw_test_data, "r") as f:
            for line in f:
                raw_tests.append(json.loads(line))
    else:
        print(f"Warning: raw test data not found at {args.raw_test_data}")

    test_prompts = raw_tests[:min(args.num_samples, len(raw_tests))]
    for i, item in enumerate(test_prompts):
        prompt = item.get("instruction", "") + "\n" + item.get("input", "")
        reference = {
            "vulnerability_type": item.get("vulnerability_type", ""),
            "quantum_attack_vector": item.get("quantum_attack_vector", ""),
            "severity": item.get("severity", ""),
            "algorithm_used": item.get("algorithm_used", ""),
            "protocol_name": item.get("protocol_name", ""),
        }
        response = generate_response(model, tokenizer, prompt, max_new_tokens=400)
        quality = evaluate_vulnerability_detection(response, reference)
        quality_scores.append(quality)
        print(f"  Sample {i+1}: Quality {quality['quality_score']}/{quality['max_score']} ({quality['quality_percentage']:.1f}%)")

    if quality_scores:
        avg_quality = sum(q["quality_percentage"] for q in quality_scores) / len(quality_scores)
        results["task_quality"] = {
            "average_quality_percentage": avg_quality,
            "samples": quality_scores,
        }
        print(f"Average task quality: {avg_quality:.1f}%")
    else:
        results["task_quality"] = {"average_quality_percentage": 0, "samples": []}

    # Save results
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_file}")

    # Edge compatibility check
    edge_cfg = cfg.get("edge", {})
    max_mem = edge_cfg.get("max_memory_mb", 4096)
    max_lat = edge_cfg.get("max_latency_ms", 1000)
    mem_mb = memory.get("gpu_allocated_mb", 0) or (memory.get("cpu_ram_used_gb", 0) * 1024)

    compatible = mem_mb <= max_mem and (latency["mean_latency_s"] * 1000 <= max_lat)
    print(f"\nEdge compatibility ({max_mem}MB RAM, {max_lat}ms): {'PASS' if compatible else 'FAIL'}")


if __name__ == "__main__":
    main()
