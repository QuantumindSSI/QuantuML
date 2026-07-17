#!/usr/bin/env python3
"""
Hybrid Dataset Compiler for Post-Quantum Key Migration Advisor
Merges real-world curated data with synthetic data, removes exact duplicates,
and produces the final training corpus.
"""

import json
import argparse
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Set


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error at {filepath}:{line_num}: {e}")
    return data


def normalize_for_dedup(text: str) -> str:
    """Normalize text for deduplication."""
    return " ".join(text.lower().split())


def deduplicate_fast(data: List[Dict]) -> List[Dict]:
    """Remove exact and near-exact duplicates using normalized hash."""
    seen: Set[str] = set()
    unique = []
    removed = 0

    for item in data:
        inst = item.get("instruction", "")
        out = item.get("output", "")[:200]  # First 200 chars of output
        key = hashlib.sha256((normalize_for_dedup(inst) + normalize_for_dedup(out)).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(item)
        else:
            removed += 1

    return unique, removed


def add_metadata(data: List[Dict], source_tag: str) -> List[Dict]:
    """Add source metadata to each example."""
    for item in data:
        item["_source"] = source_tag
        item["_id"] = hashlib.sha256(
            (item.get("instruction", "") + item.get("output", "")[:100]).encode()
        ).hexdigest()[:16]
    return data


def split_dataset(data: List[Dict], eval_ratio: float = 0.10, test_ratio: float = 0.05, seed: int = 42):
    """Shuffle and split data into train/eval/test."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    test_size = int(n * test_ratio)
    eval_size = int(n * eval_ratio)

    test = shuffled[:test_size]
    eval_data = shuffled[test_size:test_size + eval_size]
    train = shuffled[test_size + eval_size:]

    return train, eval_data, test


def save_jsonl(data: List[Dict], filepath: Path):
    """Save data as JSONL."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved {len(data)} examples to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Compile hybrid real + synthetic dataset")
    parser.add_argument("--real_data", type=str, default="./real_sources/curated_real_instructions.jsonl")
    parser.add_argument("--synthetic_data", type=str, default="./raw/instructions.jsonl")
    parser.add_argument("--output_dir", type=str, default="./hybrid")
    parser.add_argument("--eval_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_total", type=int, default=5500, help="Target total examples")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load real data
    print(f"Loading real data from {args.real_data}")
    real_data = load_jsonl(Path(args.real_data))
    real_data = add_metadata(real_data, "real_curated")
    print(f"  Loaded {len(real_data)} real examples")

    # Load synthetic data
    synthetic_path = Path(args.synthetic_data)
    if synthetic_path.exists():
        print(f"Loading synthetic data from {args.synthetic_data}")
        synth_data = load_jsonl(synthetic_path)
        synth_data = add_metadata(synth_data, "synthetic")
        print(f"  Loaded {len(synth_data)} synthetic examples")
    else:
        print(f"Synthetic data not found at {args.synthetic_data}; using real data only")
        synth_data = []

    # If we have a target total and too much synthetic data, sample down
    if synth_data and args.target_total:
        max_synth = args.target_total - len(real_data)
        if max_synth > 0 and len(synth_data) > max_synth:
            random.seed(args.seed)
            synth_data = random.sample(synth_data, max_synth)
            print(f"  Sampled synthetic data down to {len(synth_data)} examples")

    # Merge
    merged = real_data + synth_data
    print(f"\nMerged dataset: {len(merged)} examples")

    # Deduplicate (fast hash-based)
    print("Deduplicating exact/near-exact duplicates...")
    deduped, removed = deduplicate_fast(merged)
    print(f"  After dedup: {len(deduped)} examples")
    print(f"  Removed {removed} exact duplicates")

    # Source statistics
    sources = {}
    for item in deduped:
        src = item.get("_source", "unknown")
        sources[src] = sources.get(src, 0) + 1
    print(f"\nSource breakdown:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")

    # Split
    print(f"\nSplitting: eval={args.eval_ratio}, test={args.test_ratio}")
    train, eval_data, test = split_dataset(deduped, args.eval_ratio, args.test_ratio, args.seed)
    print(f"  Train: {len(train)} | Eval: {len(eval_data)} | Test: {len(test)}")

    # Save
    save_jsonl(train, output_dir / "train.jsonl")
    save_jsonl(eval_data, output_dir / "eval.jsonl")
    save_jsonl(test, output_dir / "test.jsonl")

    # Save combined for inspection
    with open(output_dir / "all_hybrid.jsonl", "w", encoding="utf-8") as f:
        for item in deduped:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved combined dataset to {output_dir / 'all_hybrid.jsonl'}")

    # Summary
    print("\n" + "=" * 50)
    print("HYBRID DATASET COMPILATION COMPLETE")
    print("=" * 50)
    print(f"Total examples: {len(deduped)}")
    print(f"  Real curated: {sources.get('real_curated', 0)}")
    print(f"  Synthetic: {sources.get('synthetic', 0)}")
    print(f"Train: {len(train)} | Eval: {len(eval_data)} | Test: {len(test)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
