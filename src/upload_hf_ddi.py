#!/usr/bin/env python3
"""Hugging Face Upload Script for Drug Interaction Predictor."""
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
    repo = project.get("huggingface_repo", "QuantumindSSI/09_drug_interaction_predictor")

    card_lines = [
        "---",
        f"license: {project['license']}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        "- quantumindssi",
        "- sovereign-ai",
        "- edge-computing",
        "- healthcare-medical-ai",
        "- drug-interactions",
        "- pharmacovigilance",
        "- ddi",
        "- patient-safety",
        "- 09_drug_interaction_predictor",
        "- finetuned",
        "- lora",
        "inference: true",
        "---",
        "",
        f"# {project['name']}",
        "",
        "A fine-tuned Small Language Model (SLM) that predicts potential drug-drug interactions from medication lists, including severity, mechanism, evidence, and clinical recommendations.",
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
        "## Intended Use",
        "",
        "- Automated drug-drug interaction screening from medication lists",
        "- Clinical decision support for pharmacists and prescribers",
        "- Patient safety and pharmacovigilance workflows",
        "- Edge deployment on Victron and other constrained hardware",
        "",
        "## Training Data",
        "",
        "- 100,000+ synthetic drug-drug interaction examples",
        "- Drug classes: anticoagulants, statins, antibiotics, antidepressants, PPIs, NSAIDs, opioids, immunosuppressants, and more",
        "- Interaction types: pharmacokinetic (CYP inhibition/induction, P-gp), pharmacodynamic (additive toxicity, QT prolongation, bleeding)",
        "- Severity levels: Critical, High, Medium, Low",
        "",
        "## Usage",
        "",
        "```python",
        f'model_id = "{repo}"',
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "model = AutoModelForCausalLM.from_pretrained(model_id)",
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "",
        'prompt = """Medications: Warfarin, Aspirin, Omeprazole. Analyze for interactions."""',
        'inputs = tokenizer(prompt, return_tensors="pt")',
        "outputs = model.generate(**inputs, max_new_tokens=512)",
        'print(tokenizer.decode(outputs[0], skip_special_tokens=True))',
        "```",
        "",
        "## Limitations",
        "",
        "- Not a substitute for clinical pharmacologist review",
        "- Synthetic training data may not capture all real-world interactions",
        "- English only",
        "- Always verify with FDA labels, clinical guidelines, and drug databases",
        "",
        "## Hardware Requirements",
        "",
        "| Target | RAM | Notes |",
        "|--------|-----|-------|",
        "| Cloud GPU | 4GB | FP16 inference |",
        "| Workstation | 3GB | INT8 quantized |",
        "| Victron Edge | 2-3GB | INT8/INT4 quantized, CPU |",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{09_drug_interaction_predictor,",
        f'  title={{{project["name"]}}},',
        '  author={QuantumIndSSI Ltd},',
        '  year={2026},',
        '  publisher={Hugging Face},',
        '  howpublished={\\url{https://huggingface.co/QuantumindSSI/09_drug_interaction_predictor}}',
        "}",
        "```",
        "",
        "## Contact",
        "",
        "- GitHub: https://github.com/QuantumindSSI",
        "- HuggingFace: https://huggingface.co/QuantumindSSI",
    ]
    return "\n".join(card_lines)


def main():
    parser = argparse.ArgumentParser(description="Upload Drug Interaction Predictor to Hugging Face")
    parser.add_argument("--config", type=str, default="./configs/config_ddi.yaml")
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
    upload_folder(folder_path=args.model_path, repo_id=repo_id, repo_type="model", token=token)
    print(f"Upload complete! https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
