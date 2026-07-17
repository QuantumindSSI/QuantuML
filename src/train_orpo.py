#!/usr/bin/env python3
"""
ORPO (Odds Ratio Preference Optimization) Fine-Tuning Script
Single-stage alignment that trains on supervised data with preference pairs.
"""

import os
import yaml
import argparse
import random
import numpy as np
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import ORPOTrainer, ORPOConfig


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="ORPO Fine-Tuning")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--preference_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    output_dir = args.output_dir or (cfg["training"]["output_dir"] + "_orpo")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map=cfg["model"]["device_map"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )

    # Optional: apply LoRA for ORPO
    lora_cfg = cfg["lora"]
    if lora_cfg.get("use_lora", True):
        peft_config = LoraConfig(
            r=lora_cfg["r"],
            lora_alpha=lora_cfg["lora_alpha"],
            target_modules=lora_cfg["target_modules"],
            lora_dropout=lora_cfg["lora_dropout"],
            bias=lora_cfg["bias"],
            task_type=lora_cfg["task_type"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Load preference dataset
    print(f"Loading preference data: {args.preference_data}")
    dataset = load_dataset("json", data_files={"train": args.preference_data}, split="train")

    # Ensure columns: prompt, chosen, rejected
    if "prompt" not in dataset.column_names:
        dataset = dataset.map(lambda x: {
            "prompt": x["instruction"],
            "chosen": x["chosen"],
            "rejected": x["rejected"],
        })

    train_test = dataset.train_test_split(test_size=0.1, seed=cfg["training"]["seed"])

    rl_cfg = cfg.get("rl", {})
    orpo_config = ORPOConfig(
        output_dir=output_dir,
        beta=rl_cfg.get("orpo_beta", 0.1),
        learning_rate=8e-6,
        lr_scheduler_type="cosine",
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

    trainer = ORPOTrainer(
        model=model,
        args=orpo_config,
        train_dataset=train_test["train"],
        eval_dataset=train_test["test"],
        tokenizer=tokenizer,
    )

    print("Starting ORPO training...")
    trainer.train()

    final_dir = Path(output_dir) / "checkpoint-final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved ORPO model to {final_dir}")


if __name__ == "__main__":
    main()
