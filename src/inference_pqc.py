#!/usr/bin/env python3
"""
Local Inference Script for Post-Quantum Key Migration Advisor
Supports base model, LoRA adapters, and merged models.
"""

import os
import yaml
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model(model_path: str, adapter_path: str = None, cfg: dict = None):
    """Load model with optional LoRA adapter."""
    cfg = cfg or {}
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=getattr(torch, cfg.get("torch_dtype", "bfloat16")),
        device_map=cfg.get("device_map", "auto"),
        trust_remote_code=cfg.get("trust_remote_code", False),
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("Adapter merged for faster inference")

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repetition_penalty: float = 1.1,
):
    """Generate text from prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove prompt from output if it appears
    if generated.startswith(prompt):
        generated = generated[len(prompt):].strip()
    return generated


def interactive_mode(model, tokenizer, cfg: dict):
    """Interactive CLI for testing."""
    print("\n" + "=" * 60)
    print("Post-Quantum Key Migration Advisor - Interactive Mode")
    print("=" * 60)
    print("Type your migration question or 'exit' to quit.\n")

    inf_cfg = cfg.get("inference", {})

    system_prompt = (
        "You are a Post-Quantum Cryptography Migration Advisor. "
        "Provide detailed, actionable migration guidance for organizations transitioning from classical to quantum-safe cryptography. "
        "Be specific about algorithms, tools, timelines, and compliance requirements.\n\n"
    )

    while True:
        try:
            user_input = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("exit", "quit", "q"):
            break
        if not user_input:
            continue

        prompt = system_prompt + f"User: {user_input}\nAdvisor:"
        response = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=inf_cfg.get("max_new_tokens", 512),
            temperature=inf_cfg.get("temperature", 0.7),
            top_p=inf_cfg.get("top_p", 0.9),
            top_k=inf_cfg.get("top_k", 40),
            repetition_penalty=inf_cfg.get("repetition_penalty", 1.1),
        )

        print(f"Advisor: {response}\n")


def batch_evaluate(model, tokenizer, test_file: str, output_file: str, cfg: dict):
    """Batch evaluation on test prompts."""
    with open(test_file, "r") as f:
        if test_file.endswith(".jsonl"):
            test_data = [json.loads(line) for line in f]
        else:
            test_data = json.load(f)

    results = []
    inf_cfg = cfg.get("inference", {})

    for i, item in enumerate(test_data):
        prompt = item.get("instruction", item.get("prompt", ""))
        reference = item.get("output", item.get("response", ""))

        print(f"[{i+1}/{len(test_data)}] Processing: {prompt[:80]}...")
        generated = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=inf_cfg.get("max_new_tokens", 512),
            temperature=inf_cfg.get("temperature", 0.7),
        )

        results.append({
            "prompt": prompt,
            "reference": reference,
            "generated": generated,
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} evaluations to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Inference for PQC Migration Advisor")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model or checkpoint")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--mode", type=str, default="interactive", choices=["interactive", "batch"])
    parser.add_argument("--input_file", type=str, default=None, help="Input for batch mode")
    parser.add_argument("--output_file", type=str, default="./outputs/evaluations/generations.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})

    print(f"Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path, args.adapter_path, model_cfg)

    if args.mode == "interactive":
        interactive_mode(model, tokenizer, cfg)
    elif args.mode == "batch":
        if not args.input_file:
            raise ValueError("--input_file required for batch mode")
        batch_evaluate(model, tokenizer, args.input_file, args.output_file, cfg)


if __name__ == "__main__":
    main()
