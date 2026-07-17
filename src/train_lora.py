#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for Post-Quantum Key Migration Advisor
Enterprise-grade training with checkpointing, evaluation, and logging.
"""

import os
import sys
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
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk


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
    parser = argparse.ArgumentParser(description="LoRA Fine-Tuning")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    output_dir = args.output_dir or cfg["training"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"GPUs available: {torch.cuda.device_count()}")

    # Load tokenizer
    print(f"Loading tokenizer: {cfg['model']['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    print(f"Loading model: {cfg['model']['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map=cfg["model"]["device_map"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        attn_implementation="flash_attention_2" if cfg["model"].get("use_flash_attention") else None,
    )

    # Prepare model for training (handles gradient checkpointing, etc.)
    if cfg["training"].get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # LoRA config
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        use_rslora=lora_cfg.get("use_rslora", False),
    )

    print("Applying LoRA adapters...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load datasets
    data_dir = cfg["data"]["processed_dir"]
    print(f"Loading datasets from {data_dir}")
    datasets = load_from_disk(data_dir)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["eval_strategy"],
        eval_steps=train_cfg["eval_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=train_cfg["greater_is_better"],
        fp16=train_cfg["fp16"],
        bf16=train_cfg["bf16"],
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        max_grad_norm=train_cfg["max_grad_norm"],
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        logging_dir=train_cfg.get("logging_dir", "./outputs/logs"),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 0),
        remove_unused_columns=train_cfg.get("remove_unused_columns", True),
        seed=train_cfg["seed"],
    )

    # Callbacks
    callbacks = []
    if train_cfg.get("early_stopping_patience"):
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=train_cfg["early_stopping_patience"])
        )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Train
    print("Starting training...")
    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save final model
    final_dir = Path(output_dir) / "checkpoint-final"
    print(f"Saving final model to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Evaluate on test set
    print("Evaluating on test set...")
    test_results = trainer.evaluate(datasets["test"])
    print(f"Test results: {test_results}")

    # Save test results
    import json
    with open(Path(output_dir) / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("Training complete!")


if __name__ == "__main__":
    main()
