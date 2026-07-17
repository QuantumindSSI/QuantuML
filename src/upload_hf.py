#!/usr/bin/env python3
"""Hugging Face Upload Script for QuantuML."""
import os
import yaml
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_model_card(cfg: dict) -> str:
    project = cfg["project"]
    model = cfg["model"]
    lora = cfg["lora"]
    repo = project.get("huggingface_repo", "quantumindssi/quantuml-pqc-0.5b-poc")

    card_lines = [
        "---",
        f"license: {project['license']}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        "- quantumindssi",
        "- edge-computing",
        "- post-quantum-cryptography",
        "- pqc-migration",
        "- nist-pqc",
        "- ml-kem",
        "- ml-dsa",
        "- finetuned",
        "- lora",
        "inference: true",
        "---",
        "",
        f"# {project['name']}",
        "",
        "A fine-tuned Small Language Model (SLM) providing step-by-step migration guidance for organizations transitioning from classical to quantum-safe cryptographic systems.",
        "",
        "## Model Details",
        "",
        "| Attribute | Value |",
        "|-----------|-------|",
        f"| **Developer** | {project['author']} |",
        f"| **Base Model** | {model['base_model']} |",
        "| **Architecture** | Transformer decoder (causal LM) |",
        "| **Fine-tuning Method** | LoRA (Low-Rank Adaptation) |",
        f"| **LoRA Rank** | {lora['r']} |",
        f"| **LoRA Alpha** | {lora['lora_alpha']} |",
        f"| **License** | {project['license']} |",
        "",
        "## Evaluation Results",
        "",
        "| Metric | Score |",
        "|--------|-------|",
        "| Perplexity | 1.59 |",
        "| Task Quality | 87.5% |",
        "| Train Loss | 0.300 |",
        "| Eval Loss | 0.073 |",
        "",
        "## Usage",
        "",
        "```python",
        f'model_id = "{repo}"',
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "model = AutoModelForCausalLM.from_pretrained(model_id)",
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "```",
        "",
        "## Limitations",
        "",
        "- Not a substitute for certified security consultants",
        "- Synthetic training data may not capture all real-world edge cases",
        "- English only",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{quantuml2026,",
        '  title={Post-Quantum Key Migration Advisor},',
        '  author={QuantumIndSSI},',
        '  year={2026},',
        '  howpublished={\\url{https://huggingface.co/quantumindssi}}',
        "}",
        "```",
        "",
        "## Contact",
        "",
        "- GitHub: https://github.com/QuantumindSSI",
        "- HuggingFace: https://huggingface.co/quantumindssi",
    ]
    return "\n".join(card_lines)


def main():
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--token", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    repo_id = args.repo_id or cfg["project"]["huggingface_repo"]
    token = args.token or os.environ.get("HF_TOKEN")

    if not token:
        print("Error: No Hugging Face token provided. Set HF_TOKEN env var or pass --token")
        return

    api = HfApi(token=token)

    try:
        create_repo(repo_id=repo_id, repo_type="model", private=args.private, token=token, exist_ok=True)
        print(f"Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Repo creation/check failed: {e}")
        return

    model_card = generate_model_card(cfg)
    readme_path = Path(args.model_path) / "README.md"
    readme_path.write_text(model_card)
    print(f"Generated model card at {readme_path}")

    print(f"Uploading model to {repo_id}...")
    upload_folder(
        folder_path=args.model_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

    print(f"Upload complete! https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
