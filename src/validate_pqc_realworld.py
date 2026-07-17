#!/usr/bin/env python3
"""
Real-World Validation Script for Post-Quantum Key Migration Advisor
Compares model migration plans against NIST guidelines, NSA CNSA 2.0, and vendor documentation.
"""

import json
import yaml
import argparse
import torch
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_ground_truth(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_path: str, adapter_path: str = None, cfg: dict = None):
    cfg = cfg or {}
    tokenizer = AutoTokenizer.from_pretrained(
        "./base_model", trust_remote_code=cfg.get("trust_remote_code", False)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, cfg.get("torch_dtype", "float32"), torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        "./base_model",
        torch_dtype=dtype,
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    elif Path(model_path).exists() and (Path(model_path) / "adapter_config.json").exists():
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        adapter_cfg = json.load(open(Path(model_path) / "adapter_config.json"))
        valid_keys = set(LoraConfig.__dataclass_fields__.keys())
        cleaned_cfg = {k: v for k, v in adapter_cfg.items() if k in valid_keys}
        lora_config = LoraConfig(**cleaned_cfg)
        model = get_peft_model(model, lora_config)
        adapter_weights_path = Path(model_path) / "adapter_model.safetensors"
        if adapter_weights_path.exists():
            state_dict = load_file(str(adapter_weights_path))
            model.load_state_dict(state_dict, strict=False)
        model = model.merge_and_unload()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 600):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=0.7, top_p=0.9,
            do_sample=True, pad_token_id=tokenizer.pad_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()
    return decoded


def build_prompt(industry: str, system: str, current: str) -> str:
    return (
        f"You are a Post-Quantum Cryptography Migration Advisor. "
        f"An organization in the {industry} sector needs to migrate their "
        f"{system} from classical cryptography to quantum-safe algorithms. "
        f"Current stack: {current}.\n\n"
        f"Provide a step-by-step migration plan including tools, deliverables, effort estimates, and responsible teams."
    )


def check_keywords(text: str, keywords: List[str]) -> bool:
    return any(kw.lower() in text.lower() for kw in keywords)


def evaluate_migration_case(model, tokenizer, case: dict) -> dict:
    prompt = build_prompt(case["industry"], case["system"], ", ".join(case["current_stack"].split(", ")))
    generated = generate(model, tokenizer, prompt, max_new_tokens=600)

    checks = {}
    checks["has_phases"] = all(p.lower() in generated.lower() for p in case["expected_phases"])
    checks["has_algorithms"] = check_keywords(generated, case["expected_algorithms"])
    checks["has_tools"] = check_keywords(generated, case["expected_tools"])
    checks["has_compliance"] = check_keywords(generated, case["expected_compliance"])
    checks["has_timeline"] = any(t in generated.lower() for t in ["month", "week", "quarter", "year"])
    checks["has_rollback"] = "rollback" in generated.lower() or "fallback" in generated.lower()
    checks["has_testing"] = "test" in generated.lower()
    checks["has_budget"] = "$" in generated or "budget" in generated.lower() or "cost" in generated.lower()

    score = sum(checks.values())
    return {
        "industry": case["industry"],
        "system": case["system"],
        "checks": checks,
        "score": score,
        "max_score": len(checks),
        "accuracy_pct": (score / len(checks)) * 100,
        "source": case["source"],
        "generated_text": generated,
    }


def print_report(results: dict):
    print("\n" + "=" * 70)
    print("REAL-WORLD PQC MIGRATION VALIDATION REPORT")
    print("=" * 70)

    cases = results["migration_cases"]
    total = len(cases)

    for key in ["has_phases", "has_algorithms", "has_tools", "has_compliance", "has_timeline", "has_rollback", "has_testing", "has_budget"]:
        passed = sum(r["checks"][key] for r in cases)
        print(f"{key:25s}: {passed}/{total} ({passed/total*100:.1f}%)")

    avg = sum(r["accuracy_pct"] for r in cases) / total if total else 0
    print(f"\nAverage Accuracy: {avg:.1f}%")

    print("\n--- DETAILED ---")
    for r in cases:
        status = "PASS" if r["accuracy_pct"] >= 70 else "FAIL"
        print(f"  [{status}] {r['industry']} / {r['system']} | {r['score']}/{r['max_score']} ({r['accuracy_pct']:.0f}%)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Real-world PQC migration validation")
    parser.add_argument("--config", type=str, default="./configs/config_pqc_migration.yaml")
    parser.add_argument("--model_path", type=str, default="./outputs/models_poc/checkpoint-final")
    parser.add_argument("--ground_truth", type=str, default="./data/pqc_migration_ground_truth.json")
    parser.add_argument("--output_file", type=str, default="./outputs/evaluations/pqc_realworld_validation.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth(args.ground_truth)

    model_cfg = cfg.get("model", {})
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path, None, model_cfg)

    results = []
    print(f"\nTesting {len(gt['migration_cases'])} migration scenarios...")
    for i, case in enumerate(gt["migration_cases"]):
        print(f"  [{i+1}] {case['industry']} / {case['system']}...")
        results.append(evaluate_migration_case(model, tokenizer, case))

    output = {
        "model_path": args.model_path,
        "migration_cases": results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output_file}")
    print_report(output)


if __name__ == "__main__":
    main()
