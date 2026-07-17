#!/usr/bin/env python3
"""
DPO (Direct Preference Optimization) Fine-Tuning Script
Trains the model to prefer chosen over rejected migration outputs.
"""

import os
import yaml
import argparse
import random
import numpy as np
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def format_preference_dataset(examples, tokenizer):
    """Format dataset for DPO: prompt, chosen, rejected."""
    # Assumes examples have: instruction, chosen_output, rejected_output
    prompts = []
    chosens = []
    rejecteds = []

    for i in range(len(examples["instruction"])):
        prompt = examples["instruction"][i]
        chosen = examples["chosen"][i]
        rejected = examples["rejected"][i]

        prompts.append(prompt)
        chosens.append(chosen)
        rejecteds.append(rejected)

    return {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    }


def main():
    parser = argparse.ArgumentParser(description="DPO Fine-Tuning")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--model_path", type=str, required=True, help="Base or LoRA model path")
    parser.add_argument("--preference_data", type=str, required=True, help="JSONL with chosen/rejected")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    output_dir = args.output_dir or (cfg["training"]["output_dir"] + "_dpo")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map=cfg["model"]["device_map"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )

    # Reference model (frozen)
    print("Creating reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map=cfg["model"]["device_map"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )

    # Load preference dataset
    print(f"Loading preference data: {args.preference_data}")
    dataset = load_dataset("json", data_files={"train": args.preference_data}, split="train")

    # Format
    dataset = dataset.map(
        lambda x: format_preference_dataset(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    train_test = dataset.train_test_split(test_size=0.1, seed=cfg["training"]["seed"])

    # DPO config
    rl_cfg = cfg.get("rl", {})
    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=rl_cfg.get("dpo_beta", 0.1),
        learning_rate=rl_cfg.get("dpo_learning_rate", 1e-6),
        per_device_train_batch_size=cfg["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        num_train_epochs=cfg["training"]["num_train_epochs"],
        logging_steps=cfg["training"]["logging_steps"],
        eval_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_strategy="steps",
        save_steps=cfg["training"]["save_steps"],
        save_total_limit=2,
        fp16=cfg["training"]["fp16"],
        bf16=cfg["training"]["bf16"],
        report_to=cfg["training"].get("report_to", ["tensorboard"]),
        seed=cfg["training"]["seed"],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_test["train"],
        eval_dataset=train_test["test"],
        tokenizer=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    final_dir = Path(output_dir) / "checkpoint-final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved DPO model to {final_dir}")


if __name__ == "__main__":
    main()
