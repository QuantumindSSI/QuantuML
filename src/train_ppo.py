#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) Fine-Tuning Script
Requires a reward model for RLHF-style training.
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
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead


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
    parser = argparse.ArgumentParser(description="PPO Fine-Tuning")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--reward_model_path", type=str, required=True)
    parser.add_argument("--prompt_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    output_dir = args.output_dir or (cfg["training"]["output_dir"] + "_ppo")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load policy model with value head
    print(f"Loading policy model: {args.model_path}")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map=cfg["model"]["device_map"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )

    # Reference model (frozen)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map=cfg["model"]["device_map"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )

    # Reward model
    print(f"Loading reward model: {args.reward_model_path}")
    reward_model = AutoModelForCausalLM.from_pretrained(
        args.reward_model_path,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map=cfg["model"]["device_map"],
    )
    reward_model.eval()

    # Dataset
    dataset = load_dataset("json", data_files={"train": args.prompt_data}, split="train")
    dataset = dataset.map(lambda x: {"query": x["instruction"]})

    # PPO config
    rl_cfg = cfg.get("rl", {})
    ppo_config = PPOConfig(
        model_name=cfg["model"]["base_model"],
        learning_rate=rl_cfg.get("ppo_learning_rate", 1.41e-5),
        batch_size=128,
        mini_batch_size=4,
        gradient_accumulation_steps=1,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
    )

    print("Starting PPO training...")
    for epoch in range(3):
        for batch in trainer.dataloader:
            queries = batch["query"]
            response_tensors = trainer.generate(queries)
            rewards = []
            for response in response_tensors:
                # Simplified reward: use reward model's logit
                inputs = tokenizer(response, return_tensors="pt").to(reward_model.device)
                with torch.no_grad():
                    outputs = reward_model(**inputs)
                rewards.append(outputs.logits[0, -1, :].mean())

            stats = trainer.step(queries, response_tensors, rewards)
            print(f"Epoch {epoch}, KL: {stats['objective/kl']:.4f}")

    final_dir = Path(output_dir) / "checkpoint-final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved PPO model to {final_dir}")


if __name__ == "__main__":
    main()
