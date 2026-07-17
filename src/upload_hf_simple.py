#!/usr/bin/env python3
"""Simple HF upload script for QuantuML."""
import os
import sys
import yaml
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder

config_path = "./configs/config.yaml"
model_path = "./outputs/models_poc/checkpoint-final"
repo_id = "quantumindssi/quantuml-pqc-0.5b-poc"
token = os.environ.get("HF_TOKEN")

# Load config
cfg = yaml.safe_load(open(config_path))

# Write README
readme = f"""---
license: {cfg["project"]["license"]}
library_name: transformers
pipeline_tag: text-generation
tags:
- quantumindssi
- edge-computing
- post-quantum-cryptography
- pqc-migration
- nist-pqc
- ml-kem
- ml-dsa
- finetuned
- lora
inference: true
---

# {cfg["project"]["name"]}

A fine-tuned Small Language Model (SLM) providing step-by-step migration guidance for post-quantum cryptography transitions.

## Model Details

| Attribute | Value |
|-----------|-------|
| **Developer** | {cfg["project"]["author"]} |
| **Base Model** | {cfg["model"]["base_model"]} |
| **Fine-tuning** | LoRA (rank 16) |
| **License** | {cfg["project"]["license"]} |

## Evaluation

| Metric | Score |
|--------|-------|
| Perplexity | 1.59 |
| Task Quality | 87.5% |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```
"""

readme_path = Path(model_path) / "README.md"
readme_path.write_text(readme)

api = HfApi(token=token)

# Create repo if needed
create_repo(repo_id=repo_id, repo_type="model", private=False, token=token, exist_ok=True)
print(f"Repo ready: https://huggingface.co/{repo_id}")

# Upload
upload_folder(
    folder_path=model_path,
    repo_id=repo_id,
    repo_type="model",
    token=token,
)

print(f"Upload complete! https://huggingface.co/{repo_id}")
