#!/usr/bin/env python3
"""
Data Preprocessing Pipeline for Post-Quantum Key Migration Advisor
Converts raw instruction data into tokenized HF datasets for training.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def load_instruction_data(filepath: Path) -> List[Dict[str, str]]:
    """Load instruction-response pairs from JSONL."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_chat_template(example: Dict[str, str], tokenizer) -> Dict[str, str]:
    """Format instruction-response into chat template compatible with Phi-3."""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    # Use the tokenizer's chat template if available
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def format_alpaca_style(example: Dict[str, str]) -> Dict[str, str]:
    """Fallback: format instruction in Alpaca style."""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text.strip():
        prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"

    text = prompt + output + tokenizer.eos_token
    return {"text": text}


def tokenize_function(examples: Dict[str, List], tokenizer, max_length: int):
    """Tokenize and prepare labels for causal LM training."""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    # Labels = input_ids for causal LM
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    parser = argparse.ArgumentParser(description="Preprocess training data")
    parser.add_argument("--input", type=str, default="./raw/instructions.jsonl", help="Raw instruction JSONL")
    parser.add_argument("--output_dir", type=str, default="./processed", help="Output directory")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-3-mini-4k-instruct", help="Tokenizer model")
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--test_size", type=float, default=0.05, help="Test split ratio")
    parser.add_argument("--eval_size", type=float, default=0.10, help="Eval split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit to N samples for quick POC")
    parser.add_argument("--push_to_hub", type=str, default=None, help="HF dataset repo to push to")
    args = parser.parse_args()

    print(f"Loading tokenizer: {args.model_name}")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading raw data from {args.input}")
    raw_data = load_instruction_data(Path(args.input))
    print(f"Loaded {len(raw_data)} examples")
    if args.max_samples and len(raw_data) > args.max_samples:
        raw_data = raw_data[:args.max_samples]
        print(f"Subsampled to {len(raw_data)} examples for POC")

    # Format
    print("Formatting examples...")
    formatted = []
    for ex in raw_data:
        try:
            formatted.append(format_chat_template(ex, tokenizer))
        except Exception:
            formatted.append(format_alpaca_style(ex))

    # Split
    train_eval, test = train_test_split(
        formatted, test_size=args.test_size, random_state=args.seed, shuffle=True
    )
    eval_ratio = args.eval_size / (1 - args.test_size)
    train, eval_data = train_test_split(
        train_eval, test_size=eval_ratio, random_state=args.seed, shuffle=True
    )

    print(f"Train: {len(train)} | Eval: {len(eval_data)} | Test: {len(test)}")

    # Create HF datasets
    ds_train = Dataset.from_list(train)
    ds_eval = Dataset.from_list(eval_data)
    ds_test = Dataset.from_list(test)

    # Tokenize
    print("Tokenizing datasets...")
    ds_train = ds_train.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )
    ds_eval = ds_eval.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )
    ds_test = ds_test.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        batched=True,
        num_proc=4,
        remove_columns=["text"],
    )

    dataset_dict = DatasetDict({
        "train": ds_train,
        "eval": ds_eval,
        "test": ds_test,
    })

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    print(f"Saved processed datasets to {output_dir}")

    # Also save as JSONL for inspection
    for split_name, ds in [("train", ds_train), ("eval", ds_eval), ("test", ds_test)]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in ds:
                # Decode for human readability
                item_decoded = {k: v for k, v in item.items()}
                f.write(json.dumps(item_decoded) + "\n")
        print(f"Saved {split_name} JSONL to {path}")

    if args.push_to_hub:
        print(f"Pushing to Hugging Face: {args.push_to_hub}")
        dataset_dict.push_to_hub(args.push_to_hub)

    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
