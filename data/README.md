# Data Generation & Preprocessing

This directory contains all data generation and preprocessing scripts.

## Scripts

| Script | Purpose | Default Output |
|--------|---------|----------------|
| `generate_drug_interaction_data.py` | Generate synthetic DDI scenarios | `./data/raw_ddi/` |
| `generate_crypto_vulnerability_data.py` | Generate crypto vulnerability analyses | `./data/raw_vuln/` |
| `generate_pqc_migration_data.py` | Generate PQC migration scenarios | `./data/raw/` |
| `compile_hybrid_dataset.py` | Merge real + synthetic data | `./data/hybrid/` |
| `preprocess.py` | Tokenize and split for training | `./data/processed_*/` |

## Quick Commands

### Drug Interaction Data

```bash
python data/generate_drug_interaction_data.py \
    --num_scenarios 102000 \
    --output_dir ./data/raw_ddi \
    --seed 42
```

### Crypto Vulnerability Data

```bash
python data/generate_crypto_vulnerability_data.py \
    --num_scenarios 10500 \
    --output_dir ./data/raw_vuln \
    --seed 42
```

### PQC Migration Data

```bash
python data/generate_pqc_migration_data.py \
    --num_scenarios 5500 \
    --output_dir ./data/raw \
    --seed 42
```

### Hybrid Dataset

```bash
python data/compile_hybrid_dataset.py \
    --real_data ./data/real_sources/curated_real_instructions.jsonl \
    --synthetic_data ./data/raw/instructions.jsonl \
    --output_dir ./data/hybrid \
    --target_total 5500 \
    --seed 42
```

### Preprocessing

```bash
python data/preprocess.py \
    --input ./data/raw_ddi/instructions.jsonl \
    --output_dir ./data/processed_ddi_quick \
    --model_name ./base_model \
    --max_length 1024 \
    --seed 42
```

## Output Format

All generators produce:
- `instructions.jsonl` - Full instruction dataset
- `scenarios.json` - Structured raw scenarios (where applicable)
- `train.jsonl` - Training split (~85%)
- `eval.jsonl` - Validation split (~10%)
- `test.jsonl` - Test split (~5%)
