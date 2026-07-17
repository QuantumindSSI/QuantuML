#!/usr/bin/env python3
"""
QLoRA Fine-Tuning Script for Post-Quantum Key Migration Advisor
4-bit quantization for training on consumer GPUs / edge hardware.
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
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
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
    parser = argparse.ArgumentParser(description="QLoRA Fine-Tuning")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["training"]["seed"])

    output_dir = args.output_dir or (cfg["training"]["output_dir"] + "_qlora")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Using QLoRA (4-bit) training")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Quantization config
    qlora_cfg = cfg["qlora"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=qlora_cfg["load_in_4bit"],
        bnb_4bit_quant_type=qlora_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, qlora_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=qlora_cfg["bnb_4bit_use_double_quant"],
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantized model
    print(f"Loading 4-bit quantized model: {cfg['model']['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["base_model"],
        quantization_config=bnb_config,
        device_map=cfg["model"]["device_map"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
        attn_implementation="flash_attention_2" if cfg["model"].get("use_flash_attention") else None,
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config
    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )

    print("Applying LoRA to quantized model...")
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Datasets
    datasets = load_from_disk(cfg["data"]["processed_dir"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
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
        fp16=False,
        bf16=False,  # Disabled for QLoRA compatibility
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        optim="paged_adamw_8bit",
        report_to=train_cfg.get("report_to", ["tensorboard"]),
        logging_dir=train_cfg.get("logging_dir", "./outputs/logs"),
        seed=train_cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["eval"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Starting QLoRA training...")
    trainer.train()

    final_dir = Path(output_dir) / "checkpoint-final"
    print(f"Saving to {final_dir}")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Test eval
    test_results = trainer.evaluate(datasets["test"])
    print(f"Test results: {test_results}")

    import json
    with open(Path(output_dir) / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    print("QLoRA training complete!")


if __name__ == "__main__":
    main()
