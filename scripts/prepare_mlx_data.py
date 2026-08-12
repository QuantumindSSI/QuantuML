#!/usr/bin/env python3
"""Convert QuantuML raw instruction JSONL into MLX-LM chat-format datasets.

Input:  raw instructions.jsonl produced by data/generate_*.py, where each line
        is {"instruction": str, "input": str (optional), "output": str, ...}.
Output: <output_dir>/train.jsonl, valid.jsonl, test.jsonl in the format
        mlx_lm.lora expects: {"messages": [{"role": "user", ...},
        {"role": "assistant", ...}]}.

Split ratios follow the project convention (AGENTS.md): 85% train,
10% valid, 5% test, shuffled with a fixed seed (default 42).

Usage:
    ./venv/bin/python scripts/prepare_mlx_data.py \
        --input ./data/raw_pqc_v2/instructions.jsonl \
        --output_dir ./data/mlx_pqc_v2
"""

import argparse
import json
import random
import sys
from pathlib import Path

TRAIN_RATIO = 0.85
VALID_RATIO = 0.10
# test ratio is the remainder (0.05)


def load_instruction_data(path: Path) -> list[dict]:
    """Load and validate instruction-response pairs from JSONL.

    Raises ValueError on malformed lines or missing required keys so bad
    data fails loudly instead of silently training on garbage.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not item.get("instruction") or not item.get("output"):
                raise ValueError(
                    f"{path}:{lineno}: missing required 'instruction' or 'output' field"
                )
            records.append(item)
    if not records:
        raise ValueError(f"{path}: no records found")
    return records


def to_chat(item: dict) -> dict:
    """Map one instruction record to the MLX-LM chat schema."""
    user_content = item["instruction"]
    extra_input = item.get("input", "").strip()
    if extra_input:
        user_content = f"{user_content}\n\n{extra_input}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item["output"]},
        ]
    }


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert raw instruction JSONL to MLX-LM chat format")
    parser.add_argument("--input", type=str, required=True, help="Raw instructions.jsonl path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for train/valid/test.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        return 1

    records = load_instruction_data(input_path)
    chat_records = [to_chat(r) for r in records]

    rng = random.Random(args.seed)
    rng.shuffle(chat_records)

    n = len(chat_records)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    train = chat_records[:n_train]
    valid = chat_records[n_train:n_train + n_valid]
    test = chat_records[n_train + n_valid:]
    assert len(train) + len(valid) + len(test) == n, "split sizes must sum to total"
    assert train and valid and test, "every split must be non-empty"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(valid, output_dir / "valid.jsonl")
    write_jsonl(test, output_dir / "test.jsonl")

    print(f"Total: {n} | train: {len(train)} | valid: {len(valid)} | test: {len(test)}")
    print(f"Written to {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
