#!/usr/bin/env python3
"""
Real-World Validation Script for Quantum-Resistant Cryptographic Protocol Analyzer
Compares model predictions against known CVEs, NIST standards, and real attack histories.
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


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 400):
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


def build_prompt(protocol: str, code: str, algorithm: str) -> str:
    return (
        f"Analyze the following {protocol} implementation for quantum-vulnerable patterns.\n"
        f"```\n{code}\n```\n"
        f"Algorithm: {algorithm}\n\n"
        f"Identify the vulnerability type, quantum attack vector, and recommend mitigations."
    )


def check_keywords(text: str, keywords: List[str]) -> bool:
    return any(kw.lower() in text.lower() for kw in keywords)


def evaluate_vulnerable_case(model, tokenizer, case: dict) -> dict:
    prompt = build_prompt(case["protocol"], case["code_snippet"], case["algorithm"])
    generated = generate(model, tokenizer, prompt, max_new_tokens=400)

    vuln_detected = check_keywords(generated, [case["expected_vulnerability"].split()[0]]) or "vulnerable" in generated.lower() or "deprecated" in generated.lower()
    attack_vector = check_keywords(generated, case["expected_attack_vector"].split()[:3])
    nist_ref = check_keywords(generated, ["nist", "fips", "cnsa"])
    severity = check_keywords(generated, [case["expected_severity"].lower()])
    mitigation = "mitigation" in generated.lower() or "recommend" in generated.lower() or "1." in generated

    score = sum([vuln_detected, attack_vector, nist_ref, severity, mitigation])

    return {
        "protocol": case["protocol"],
        "algorithm": case["algorithm"],
        "vulnerability_detected": vuln_detected,
        "attack_vector_recognized": attack_vector,
        "nist_reference_present": nist_ref,
        "severity_mentioned": severity,
        "mitigations_present": mitigation,
        "score": score,
        "max_score": 5,
        "accuracy_pct": (score / 5) * 100,
        "source": case["source"],
        "generated_text": generated,
    }


def evaluate_non_vulnerable_case(model, tokenizer, case: dict) -> dict:
    prompt = build_prompt(case["protocol"], case["code_snippet"], case["algorithm"])
    generated = generate(model, tokenizer, prompt, max_new_tokens=300)
    false_positive = "vulnerable" in generated.lower() or "deprecated" in generated.lower() or "critical" in generated.lower()

    return {
        "protocol": case["protocol"],
        "algorithm": case["algorithm"],
        "false_positive": false_positive,
        "generated_text": generated,
    }


def print_report(results: dict):
    print("\n" + "=" * 70)
    print("REAL-WORLD CRYPTO VALIDATION REPORT")
    print("=" * 70)

    vuln = results["vulnerable_cases"]
    non_vuln = results["non_vulnerable_cases"]

    detected = sum(r["vulnerability_detected"] for r in vuln)
    attack_rec = sum(r["attack_vector_recognized"] for r in vuln)
    nist_pres = sum(r["nist_reference_present"] for r in vuln)
    sev_pres = sum(r["severity_mentioned"] for r in vuln)
    mit_pres = sum(r["mitigations_present"] for r in vuln)
    avg_score = sum(r["accuracy_pct"] for r in vuln) / len(vuln) if vuln else 0

    print("\n--- VULNERABLE CASES (Known CVEs / Standards Violations) ---")
    print(f"Total tested:          {len(vuln)}")
    print(f"Vulnerability detected: {detected}/{len(vuln)} ({detected/len(vuln)*100:.1f}%)")
    print(f"Attack vector found:    {attack_rec}/{len(vuln)} ({attack_rec/len(vuln)*100:.1f}%)")
    print(f"NIST reference present: {nist_pres}/{len(vuln)} ({nist_pres/len(vuln)*100:.1f}%)")
    print(f"Severity mentioned:     {sev_pres}/{len(vuln)} ({sev_pres/len(vuln)*100:.1f}%)")
    print(f"Mitigations present:    {mit_pres}/{len(vuln)} ({mit_pres/len(vuln)*100:.1f}%)")
    print(f"Average Accuracy:       {avg_score:.1f}%")

    fp = sum(r["false_positive"] for r in non_vuln) if non_vuln else 0
    print("\n--- NON-VULNERABLE CASES (Modern / PQC-Ready) ---")
    print(f"Total tested:           {len(non_vuln)}")
    print(f"False positives:        {fp}/{len(non_vuln)} ({fp/len(non_vuln)*100:.1f}%)" if non_vuln else "N/A")

    print("\n--- DETAILED ---")
    for r in vuln:
        status = "PASS" if r["accuracy_pct"] >= 60 else "FAIL"
        print(f"  [{status}] {r['protocol']} ({r['algorithm']}) | Score: {r['score']}/5 ({r['accuracy_pct']:.0f}%)")
    for r in non_vuln:
        status = "FAIL (FP)" if r["false_positive"] else "PASS"
        print(f"  [{status}] {r['protocol']} ({r['algorithm']})")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Real-world crypto validation")
    parser.add_argument("--config", type=str, default="./configs/config_crypto_analyzer_quick.yaml")
    parser.add_argument("--model_path", type=str, default="./outputs/models_crypto_quick/checkpoint-final")
    parser.add_argument("--ground_truth", type=str, default="./data/crypto_real_world_ground_truth.json")
    parser.add_argument("--output_file", type=str, default="./outputs/evaluations/crypto_realworld_validation.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    gt = load_ground_truth(args.ground_truth)

    model_cfg = cfg.get("model", {})
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model(args.model_path, None, model_cfg)

    vuln_results = []
    print(f"\nTesting {len(gt['vulnerable_cases'])} vulnerable cases...")
    for i, case in enumerate(gt["vulnerable_cases"]):
        print(f"  [{i+1}] {case['protocol']}...")
        vuln_results.append(evaluate_vulnerable_case(model, tokenizer, case))

    non_vuln_results = []
    if gt.get("non_vulnerable_cases"):
        print(f"\nTesting {len(gt['non_vulnerable_cases'])} non-vulnerable cases...")
        for case in gt["non_vulnerable_cases"]:
            non_vuln_results.append(evaluate_non_vulnerable_case(model, tokenizer, case))

    results = {
        "model_path": args.model_path,
        "vulnerable_cases": vuln_results,
        "non_vulnerable_cases": non_vuln_results,
    }

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output_file}")
    print_report(results)


if __name__ == "__main__":
    main()
