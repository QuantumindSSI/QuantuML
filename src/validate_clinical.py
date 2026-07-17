#!/usr/bin/env python3
"""
Clinical Validation Script for Drug Interaction Predictor
Compares model predictions against FDA/DrugBank/Guideline ground truth.

Metrics computed:
- Drug Pair Detection Accuracy
- Severity Classification Accuracy
- Mechanism Presence (keyword match)
- Outcome Presence (keyword match)
- Recommendation Presence (keyword match)
- False Positive Rate (negative cases)
- False Negative Rate (positive cases)
- Overall Clinical Accuracy Score
"""

import json
import yaml
import argparse
import re
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
    # Try loading tokenizer from model_path first, fallback to base_model
    tokenizer_paths = [model_path, "./base_model", cfg.get("base_model", "Qwen/Qwen2-0.5B-Instruct")]
    tokenizer = None
    for tp in tokenizer_paths:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tp, trust_remote_code=cfg.get("trust_remote_code", False))
            print(f"Loaded tokenizer from {tp}")
            break
        except Exception as e:
            print(f"Failed to load tokenizer from {tp}: {e}")
            continue
    if tokenizer is None:
        raise RuntimeError("Could not load tokenizer from any known path")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, cfg.get("torch_dtype", "float32"), torch.float32)
    # Load base model first, then adapter
    base_model_path = cfg.get("base_model", "./base_model")
    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    print(f"Loading adapter from {model_path}...")
    # Some adapter configs may contain unsupported fields from newer PEFT versions
    adapter_config_path = Path(model_path) / "adapter_config.json"
    if adapter_config_path.exists():
        with open(adapter_config_path, "r") as f:
            adapter_cfg = json.load(f)
        # Filter out fields not supported by installed peft
        from peft import LoraConfig
        valid_keys = set(LoraConfig.__dataclass_fields__.keys())
        cleaned_cfg = {k: v for k, v in adapter_cfg.items() if k in valid_keys}
        lora_config = LoraConfig(**cleaned_cfg)
        from peft import get_peft_model
        model = get_peft_model(model, lora_config)
        # Load adapter weights
        from safetensors.torch import load_file
        adapter_weights_path = Path(model_path) / "adapter_model.safetensors"
        if adapter_weights_path.exists():
            state_dict = load_file(str(adapter_weights_path))
            model.load_state_dict(state_dict, strict=False)
        else:
            raise FileNotFoundError(f"Adapter weights not found at {adapter_weights_path}")
    else:
        model = PeftModel.from_pretrained(model, model_path)
    print("Merging adapter for inference...")
    model = model.merge_and_unload()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 400):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
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
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):].strip()
    return decoded


def build_prompt(drug_a: str, drug_b: str) -> str:
    return (
        f"Analyze the following medication list for potential drug-drug interactions: {drug_a}, {drug_b}. "
        f"Report any predicted interactions with severity, mechanism, evidence, and clinical recommendations."
    )


def extract_severity(text: str) -> str:
    text_lower = text.lower()
    severities = ["critical", "high", "medium", "low"]
    # Take the first severity mentioned that appears near an interaction description
    for sev in severities:
        if sev in text_lower:
            return sev.capitalize()
    return "Unknown"


def check_keywords(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def detect_interaction(text: str) -> bool:
    """Heuristic: does the model claim an interaction exists?"""
    text_lower = text.lower()
    positive_signals = [
        "interaction", "contraindicated", "avoid", "caution",
        "severe", "major", "significant", "additive", "increased risk"
    ]
    negative_signals = [
        "no interaction", "no known interaction", "no significant",
        "unlikely", "not expected", "minimal interaction"
    ]
    pos_score = sum(1 for s in positive_signals if s in text_lower)
    neg_score = sum(1 for s in negative_signals if s in text_lower)
    return pos_score > neg_score


def evaluate_positive_case(model, tokenizer, case: dict) -> dict:
    drug_a, drug_b = case["drug_pair"]
    prompt = build_prompt(drug_a, drug_b)
    generated = generate(model, tokenizer, prompt, max_new_tokens=400)

    detected = detect_interaction(generated)
    predicted_severity = extract_severity(generated)
    expected_severity = case["severity"]

    severity_correct = predicted_severity.lower() == expected_severity.lower()
    mechanism_present = check_keywords(generated, case["mechanism_keywords"])
    outcome_present = check_keywords(generated, case["outcome_keywords"])
    recommendation_present = check_keywords(generated, case["recommendation_keywords"])
    drug_pair_mentioned = drug_a.lower() in generated.lower() and drug_b.lower() in generated.lower()

    # Weighted clinical accuracy
    score = 0
    if detected:
        score += 2
    if severity_correct:
        score += 2
    if mechanism_present:
        score += 1
    if outcome_present:
        score += 1
    if recommendation_present:
        score += 1
    if drug_pair_mentioned:
        score += 1
    max_score = 8

    return {
        "drug_pair": case["drug_pair"],
        "expected_severity": expected_severity,
        "predicted_severity": predicted_severity,
        "severity_correct": severity_correct,
        "interaction_detected": detected,
        "drug_pair_mentioned": drug_pair_mentioned,
        "mechanism_present": mechanism_present,
        "outcome_present": outcome_present,
        "recommendation_present": recommendation_present,
        "clinical_score": score,
        "max_score": max_score,
        "clinical_accuracy_pct": (score / max_score) * 100,
        "source": case["source"],
        "generated_text": generated,
    }


def evaluate_negative_case(model, tokenizer, case: dict) -> dict:
    drug_a, drug_b = case["drug_pair"]
    prompt = build_prompt(drug_a, drug_b)
    generated = generate(model, tokenizer, prompt, max_new_tokens=300)
    detected = detect_interaction(generated)

    return {
        "drug_pair": case["drug_pair"],
        "expected_interaction": False,
        "predicted_interaction": detected,
        "false_positive": detected,
        "generated_text": generated,
    }


def print_report(results: dict):
    print("\n" + "=" * 70)
    print("CLINICAL VALIDATION REPORT - Drug Interaction Predictor")
    print("=" * 70)

    pos = results["positive_cases"]
    neg = results["negative_cases"]
    n_pos = len(pos)
    n_neg = len(neg)

    # Positive metrics
    detected_count = sum(1 for r in pos if r["interaction_detected"])
    severity_correct = sum(1 for r in pos if r["severity_correct"])
    mechanism_present = sum(1 for r in pos if r["mechanism_present"])
    outcome_present = sum(1 for r in pos if r["outcome_present"])
    rec_present = sum(1 for r in pos if r["recommendation_present"])
    drug_pair_mentioned = sum(1 for r in pos if r["drug_pair_mentioned"])
    avg_clinical_score = sum(r["clinical_accuracy_pct"] for r in pos) / n_pos if n_pos > 0 else 0

    print("\n--- POSITIVE CASES (Known Interactions) ---")
    print(f"Total tested:        {n_pos}")
    print(f"Interaction detected: {detected_count}/{n_pos} ({detected_count/n_pos*100:.1f}%)")
    print(f"Severity correct:     {severity_correct}/{n_pos} ({severity_correct/n_pos*100:.1f}%)")
    print(f"Drug pair mentioned:  {drug_pair_mentioned}/{n_pos} ({drug_pair_mentioned/n_pos*100:.1f}%)")
    print(f"Mechanism present:    {mechanism_present}/{n_pos} ({mechanism_present/n_pos*100:.1f}%)")
    print(f"Outcome present:      {outcome_present}/{n_pos} ({outcome_present/n_pos*100:.1f}%)")
    print(f"Recommendations:      {rec_present}/{n_pos} ({rec_present/n_pos*100:.1f}%)")
    print(f"Avg Clinical Score:   {avg_clinical_score:.1f}%")

    # Negative metrics
    fp_count = sum(1 for r in neg if r["false_positive"])
    print("\n--- NEGATIVE CASES (No Known Interaction) ---")
    print(f"Total tested:         {n_neg}")
    print(f"False positives:      {fp_count}/{n_neg} ({fp_count/n_neg*100:.1f}%)")

    # Overall
    fn_count = n_pos - detected_count
    print("\n--- OVERALL ---")
    print(f"True Positives:       {detected_count}")
    print(f"False Negatives:      {fn_count}")
    print(f"False Positives:      {fp_count}")
    print(f"True Negatives:       {n_neg - fp_count}")
    print(f"Sensitivity (Recall): {detected_count/n_pos*100:.1f}%" if n_pos > 0 else "N/A")
    print(f"Specificity:          {(n_neg-fp_count)/n_neg*100:.1f}%" if n_neg > 0 else "N/A")
    print(f"Overall Clinical Score: {avg_clinical_score:.1f}%")

    print("\n--- DETAILED BREAKDOWN ---")
    for r in pos:
        status = "PASS" if r["clinical_accuracy_pct"] >= 60 else "FAIL"
        print(f"  [{status}] {r['drug_pair'][0]} + {r['drug_pair'][1]} | "
              f"Sev: {r['predicted_severity']} (exp: {r['expected_severity']}) | "
              f"Score: {r['clinical_score']}/{r['max_score']} ({r['clinical_accuracy_pct']:.0f}%)")

    print("\n--- NEGATIVE CASES ---")
    for r in neg:
        status = "PASS" if not r["false_positive"] else "FAIL (FP)"
        print(f"  [{status}] {r['drug_pair'][0]} + {r['drug_pair'][1]}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Clinical validation of Drug Interaction Predictor")
    parser.add_argument("--config", type=str, default="./configs/config_ddi.yaml")
    parser.add_argument("--model_path", type=str, default="./outputs/models_ddi_quick/checkpoint-final")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--ground_truth", type=str, default="./data/clinical_ground_truth.json")
    parser.add_argument("--output_file", type=str, default="./outputs/evaluations/clinical_validation_results.json")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit number of positive cases to test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(args.ground_truth)
    positive_cases = gt.get("positive_interactions", [])
    negative_cases = gt.get("negative_interactions", [])

    if args.max_cases:
        positive_cases = positive_cases[:args.max_cases]

    print(f"Loading model from {args.model_path}...")
    model_cfg = cfg.get("model", {})
    model, tokenizer = load_model(args.model_path, args.adapter_path, model_cfg)

    print(f"\nValidating against {len(positive_cases)} positive and {len(negative_cases)} negative cases...")

    positive_results = []
    for i, case in enumerate(positive_cases):
        print(f"  [{i+1}/{len(positive_cases)}] Testing {case['drug_pair'][0]} + {case['drug_pair'][1]}...")
        positive_results.append(evaluate_positive_case(model, tokenizer, case))

    negative_results = []
    for i, case in enumerate(negative_cases):
        print(f"  [{i+1}/{len(negative_cases)}] Testing negative case {case['drug_pair'][0]} + {case['drug_pair'][1]}...")
        negative_results.append(evaluate_negative_case(model, tokenizer, case))

    results = {
        "model_path": args.model_path,
        "positive_cases": positive_results,
        "negative_cases": negative_results,
        "ground_truth_source": gt.get("description", ""),
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results saved to {args.output_file}")

    print_report(results)


if __name__ == "__main__":
    main()
