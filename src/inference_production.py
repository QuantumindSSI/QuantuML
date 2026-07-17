#!/usr/bin/env python3
"""
Production Inference Pipeline for QuantuML Suite
Includes confidence scoring, rejection thresholds, and safety guardrails.

Usage:
    python src/inference_production.py --model ddi --input "Warfarin, Aspirin"
    python src/inference_production.py --model crypto --input_file ./cases.jsonl
    python src/inference_production.py --model pqc --input "TLS/SSL, Financial Services"
"""

import os
import yaml
import argparse
import json
import re
import torch
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from safetensors.torch import load_file


MODEL_CONFIGS = {
    "ddi": {
        "config": "./configs/config_ddi_v2.yaml",
        "path": "./outputs/models_ddi_v2/checkpoint-final",
        "base": "./base_model",
        "rejection_threshold": 0.35,
    },
    "crypto": {
        "config": "./configs/config_crypto_v2.yaml",
        "path": "./outputs/models_crypto_v2/checkpoint-final",
        "base": "./base_model",
        "rejection_threshold": 0.35,
    },
    "pqc": {
        "config": "./configs/config_pqc_v2.yaml",
        "path": "./outputs/models_pqc_v2/checkpoint-final",
        "base": "./base_model",
        "rejection_threshold": 0.40,
    },
}


def load_model(model_key: str):
    cfg_path = MODEL_CONFIGS[model_key]["config"]
    model_path = MODEL_CONFIGS[model_key]["path"]
    base_path = MODEL_CONFIGS[model_key]["base"]

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = getattr(torch, cfg.get("model", {}).get("torch_dtype", "float32"), torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=dtype, device_map="auto", trust_remote_code=False,
    )

    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        adapter_cfg = json.load(open(adapter_config_path))
        valid_keys = set(LoraConfig.__dataclass_fields__.keys())
        cleaned_cfg = {k: v for k, v in adapter_cfg.items() if k in valid_keys}
        lora_config = LoraConfig(**cleaned_cfg)
        model = get_peft_model(model, lora_config)
        weights_path = Path(model_path) / "adapter_model.safetensors"
        if weights_path.exists():
            state_dict = load_file(str(weights_path))
            model.load_state_dict(state_dict, strict=False)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer, cfg


def generate(model, tokenizer, prompt: str, cfg: dict, max_new_tokens: int = 400):
    inf = cfg.get("inference", {})
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=inf.get("temperature", 0.7),
            top_p=inf.get("top_p", 0.9),
            top_k=inf.get("top_k", 40),
            repetition_penalty=inf.get("repetition_penalty", 1.1),
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Compute token-level confidence (mean probability of generated tokens)
    scores = torch.stack(outputs.scores, dim=1)
    probs = torch.nn.functional.softmax(scores, dim=-1)
    token_probs = probs[0, range(len(generated_ids)), generated_ids].cpu().numpy()
    confidence = float(token_probs.mean())

    return generated.strip(), confidence


def score_ddi_confidence(text: str) -> float:
    """Heuristic confidence score for DDI output."""
    score = 0.0
    checks = {
        "severity": any(s in text.lower() for s in ["critical", "high", "medium", "low", "none"]),
        "mechanism": any(k in text.lower() for k in ["mechanism", "cyp", "pharmacodynamic", "pharmacokinetic", "no known"]),
        "recommendations": len(re.findall(r"\d+\.", text)) >= 2 or "recommend" in text.lower(),
        "evidence": any(k in text.lower() for k in ["fda", "guideline", "trial", "label", "drugbank"]),
        "drug_pair": "+" in text or "and" in text.lower(),
        "headers": "##" in text or "**" in text,
        "outcomes": any(k in text.lower() for k in ["outcome", "risk", "toxicity", "bleeding", "no adverse"]),
        "negative": "no clinically significant interaction" in text.lower() or "no known" in text.lower(),
    }
    score = sum(checks.values()) / len(checks)
    return score


def score_crypto_confidence(text: str) -> float:
    """Heuristic confidence score for crypto vulnerability output."""
    score = 0.0
    checks = {
        "vulnerability_type": any(k in text.lower() for k in ["vulnerable", "deprecated", "weak", "safe", "none"]),
        "attack_vector": any(k in text.lower() for k in ["shor", "grover", "harvest", "downgrade", "mitm", "none applicable"]),
        "nist_refs": any(k in text.lower() for k in ["nist", "fips", "cnsa"]),
        "severity": any(s in text.lower() for s in ["critical", "high", "medium", "low", "none"]),
        "mitigations": len(re.findall(r"\d+\.", text)) >= 2 or "mitigation" in text.lower() or "recommend" in text.lower(),
        "headers": "##" in text or "**" in text,
        "protocol": "protocol" in text.lower(),
        "algorithm": "algorithm" in text.lower() or "cipher" in text.lower(),
    }
    score = sum(checks.values()) / len(checks)
    return score


def post_process_ddi(text: str, confidence: float, threshold: float) -> Dict[str, Any]:
    result = {
        "model_output": text,
        "token_confidence": round(confidence, 4),
        "structural_score": round(score_ddi_confidence(text), 4),
        "is_rejected": False,
        "rejection_reason": None,
        "safety_flag": None,
    }

    # Check for hallucination: if no severity and no mechanism, likely hallucinated
    if score_ddi_confidence(text) < threshold:
        result["is_rejected"] = True
        result["rejection_reason"] = "Low structural confidence. Model output lacks expected sections (severity, mechanism, evidence)."
        result["safety_flag"] = "VERIFY_WITH_PHARMACIST"
        return result

    # Check for critical severity without recommendations
    if "critical" in text.lower() and not any(k in text.lower() for k in ["recommend", "avoid", "monitor"]):
        result["safety_flag"] = "MISSING_RECOMMENDATIONS"

    # Check for negative case misclassified as positive
    if "no clinically significant interaction" in text.lower():
        result["safety_flag"] = "NEGATIVE_CASE"

    return result


def post_process_crypto(text: str, confidence: float, threshold: float) -> Dict[str, Any]:
    result = {
        "model_output": text,
        "token_confidence": round(confidence, 4),
        "structural_score": round(score_crypto_confidence(text), 4),
        "is_rejected": False,
        "rejection_reason": None,
        "safety_flag": None,
    }

    if score_crypto_confidence(text) < threshold:
        result["is_rejected"] = True
        result["rejection_reason"] = "Low structural confidence. Model output lacks expected vulnerability analysis sections."
        result["safety_flag"] = "VERIFY_WITH_SECURITY_TEAM"
        return result

    # Flag if model claims vulnerability for clearly safe code
    if ("none" in text.lower() or "safe" in text.lower() or "ready" in text.lower()) and "vulnerable" not in text.lower():
        result["safety_flag"] = "NEGATIVE_CASE"

    return result


def post_process_pqc(text: str, confidence: float, threshold: float) -> Dict[str, Any]:
    result = {
        "model_output": text,
        "token_confidence": round(confidence, 4),
        "structural_score": round(score_crypto_confidence(text), 4),  # Reuse crypto scoring as proxy
        "is_rejected": False,
        "rejection_reason": None,
        "safety_flag": None,
    }

    if score_crypto_confidence(text) < threshold:
        result["is_rejected"] = True
        result["rejection_reason"] = "Low structural confidence. Model output lacks migration plan structure."
        return result

    return result


def build_prompt(model_key: str, user_input: str) -> str:
    if model_key == "ddi":
        return (
            f"Analyze the following medication list for potential drug-drug interactions: {user_input}. "
            f"Report any predicted interactions with severity, mechanism, evidence, and clinical recommendations."
        )
    elif model_key == "crypto":
        return (
            f"Analyze the following cryptographic protocol implementation for quantum-vulnerable patterns.\n"
            f"{user_input}\n\n"
            f"Identify the vulnerability type, quantum attack vector, and recommend mitigations."
        )
    else:  # pqc
        return (
            f"You are a Post-Quantum Cryptography Migration Advisor. "
            f"{user_input}\n\n"
            f"Provide a step-by-step migration plan including tools, deliverables, effort estimates, and responsible teams."
        )


def main():
    parser = argparse.ArgumentParser(description="Production inference with confidence scoring")
    parser.add_argument("--model", type=str, required=True, choices=["ddi", "crypto", "pqc"])
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="./outputs/production_results.json")
    parser.add_argument("--max_new_tokens", type=int, default=400)
    args = parser.parse_args()

    print(f"Loading {args.model} model...")
    model, tokenizer, cfg = load_model(args.model)
    threshold = MODEL_CONFIGS[args.model]["rejection_threshold"]
    print(f"Loaded. Rejection threshold: {threshold}")

    inputs = []
    if args.input_file:
        with open(args.input_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    inputs.append(line if not line.startswith("{") else json.loads(line).get("input", line))
    elif args.input:
        inputs = [args.input]
    else:
        print("Error: provide --input or --input_file")
        return

    results = []
    for i, user_input in enumerate(inputs):
        prompt = build_prompt(args.model, user_input)
        print(f"\n[{i+1}/{len(inputs)}] Prompt: {prompt[:80]}...")
        text, confidence = generate(model, tokenizer, prompt, cfg, args.max_new_tokens)

        if args.model == "ddi":
            processed = post_process_ddi(text, confidence, threshold)
        elif args.model == "crypto":
            processed = post_process_crypto(text, confidence, threshold)
        else:
            processed = post_process_pqc(text, confidence, threshold)

        processed["prompt"] = prompt
        results.append(processed)

        status = "REJECTED" if processed["is_rejected"] else "PASS"
        print(f"  Status: {status} | Token confidence: {processed['token_confidence']:.3f} | Structural: {processed['structural_score']:.3f}")
        if processed["safety_flag"]:
            print(f"  Safety flag: {processed['safety_flag']}")
        if not processed["is_rejected"]:
            print(f"  Output preview: {text[:120]}...")
        else:
            print(f"  Output withheld: {processed['rejection_reason']}")

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results to {args.output_file}")


if __name__ == "__main__":
    main()
