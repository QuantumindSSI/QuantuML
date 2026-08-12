#!/usr/bin/env python3
"""Build a mixed real-world + synthetic MLX-LM training dataset.

Combines a real-world instruction JSONL (from scripts/prepare_real_world_data.py)
with a synthetic instruction JSONL (from data/generate_*.py), converts to the
MLX-LM chat schema, splits 85/10/5 (train/valid/test, seed-fixed), and writes
a manifest.json that states the exact real-vs-synthetic composition of every
split. Provenance is tracked per record through the shuffle so the manifest
counts are measured, not assumed.

Usage:
    ./venv/bin/python scripts/build_mixed_dataset.py \
        --real ./data/real_world/real_pqc_instructions.jsonl \
        --synthetic ./data/raw_pqc_v2/instructions.jsonl \
        --output_dir ./data/mlx_pqc_v3 \
        --real_ratio 0.5
"""

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path

TRAIN_RATIO = 0.85
VALID_RATIO = 0.10
# test = remainder (0.05)


def load_jsonl(path: Path, data_kind: str) -> list[dict]:
    """Load instruction records, tagging each with its provenance kind."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not item.get("instruction") or not item.get("output"):
                raise ValueError(f"{path}:{lineno}: missing 'instruction' or 'output'")
            item["data_kind"] = item.get("data_kind", data_kind)
            records.append(item)
    if not records:
        raise ValueError(f"{path}: empty dataset")
    return records


def to_chat(item: dict) -> dict:
    """Convert one instruction record to MLX chat schema, keeping provenance."""
    user_content = item["instruction"]
    extra = item.get("input", "").strip()
    if extra:
        user_content = f"{user_content}\n\n{extra}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": item["output"]},
        ],
        "_kind": item["data_kind"],          # stripped before writing
        "_license": item.get("source_license", "project-synthetic"),
    }


def _normalized_key(record: dict) -> str:
    """Dedup key: whitespace/case-normalized user turn + assistant turn."""
    parts = [m["content"] for m in record["messages"]]
    return " ".join(" ".join(parts).lower().split())


def deduplicate(records: list[dict]) -> tuple[list[dict], int]:
    """Remove exact (normalized) duplicates, keeping first occurrence."""
    seen = set()
    unique = []
    for rec in records:
        key = _normalized_key(rec)
        if key not in seen:
            seen.add(key)
            unique.append(rec)
    return unique, len(records) - len(unique)


def mix(real: list[dict], synthetic: list[dict], real_ratio: float, rng: random.Random) -> list[dict]:
    """Subsample the larger pool so the final mix matches real_ratio exactly
    (subject to availability), then shuffle."""
    assert 0.0 < real_ratio < 1.0, "real_ratio must be in (0, 1)"
    rng.shuffle(real)
    rng.shuffle(synthetic)

    # Largest total achievable with the requested ratio given both pool sizes.
    total_by_real = int(len(real) / real_ratio)
    total_by_synth = int(len(synthetic) / (1.0 - real_ratio))
    total = min(total_by_real, total_by_synth)
    n_real = int(total * real_ratio)
    n_synth = total - n_real

    combined = real[:n_real] + synthetic[:n_synth]
    rng.shuffle(combined)
    return combined


def split_counts(records: list[dict]) -> dict:
    real = sum(1 for r in records if r["_kind"] == "real")
    synth = len(records) - real
    pct = (100.0 * real / len(records)) if records else 0.0
    return {"total": len(records), "real": real, "synthetic": synth,
            "real_pct": round(pct, 1)}


def write_jsonl(records: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps({"messages": rec["messages"]}, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mix real + synthetic data for MLX-LM training")
    parser.add_argument("--real", type=str, required=True)
    parser.add_argument("--synthetic", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--real_ratio", type=float, default=0.5,
                        help="Fraction of the final mix drawn from real data (default 0.5)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    real_path, synth_path = Path(args.real), Path(args.synthetic)
    for p in (real_path, synth_path):
        if not p.is_file():
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 1

    real = [to_chat(r) for r in load_jsonl(real_path, "real")]
    synthetic = [to_chat(r) for r in load_jsonl(synth_path, "synthetic")]

    real, dup_real = deduplicate(real)
    synthetic, dup_synth = deduplicate(synthetic)

    rng = random.Random(args.seed)
    combined = mix(real, synthetic, args.real_ratio, rng)
    combined, dup_cross = deduplicate(combined)  # cross-source decontamination

    n = len(combined)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    splits = {
        "train": combined[:n_train],
        "valid": combined[n_train:n_train + n_valid],
        "test": combined[n_train + n_valid:],
    }
    assert sum(len(s) for s in splits.values()) == n
    assert all(splits.values()), "every split must be non-empty"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created": date.today().isoformat(),
        "seed": args.seed,
        "requested_real_ratio": args.real_ratio,
        "sources": {
            "real": str(real_path),
            "synthetic": str(synth_path),
            "real_pool_size": len(real),
            "synthetic_pool_size": len(synthetic),
        },
        "deduplication": {
            "real_duplicates_removed": dup_real,
            "synthetic_duplicates_removed": dup_synth,
            "cross_source_duplicates_removed": dup_cross,
        },
        "licenses": sorted({r["_license"] for r in combined}),
        "splits": {},
    }
    for name, records in splits.items():
        write_jsonl(records, output_dir / f"{name}.jsonl")
        manifest["splits"][name] = split_counts(records)

    manifest["overall"] = split_counts(combined)
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"\nWritten to {output_dir}/ (train/valid/test.jsonl + manifest.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
