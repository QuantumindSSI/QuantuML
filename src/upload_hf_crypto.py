#!/usr/bin/env python3
"""Hugging Face Upload Script for Quantum-Resistant Cryptographic Protocol Analyzer."""
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
    repo = project.get("huggingface_repo", "quantumindssi/01_quantum_resistant_crypto_analyzer")

    card_lines = [
        "---",
        f"license: {project['license']}",
        "library_name: transformers",
        "pipeline_tag: text-generation",
        "tags:",
        "- quantumindssi",
        "- sovereign-ai",
        "- edge-computing",
        "- post-quantum-cryptography",
        "- quantum-cryptanalysis",
        "- nist-pqc",
        "- vulnerability-detection",
        "- ml-kem",
        "- ml-dsa",
        "- slh-dsa",
        "- 01_quantum_resistant_crypto_analyzer",
        "- finetuned",
        "- lora",
        "inference: true",
        "---",
        "",
        f"# {project['name']}",
        "",
        "A fine-tuned Small Language Model (SLM) that analyzes cryptographic protocol implementations and identifies quantum-vulnerable patterns, attack vectors, and NIST-aligned mitigations.",
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
        "- Automated quantum vulnerability scanning of protocol implementations (TLS, SSH, VPN, etc.)",
        "- Security audit assistance for classical-to-PQC migration planning",
        "- Developer education on quantum cryptanalysis risks",
        "- Edge deployment on Victron and other constrained hardware",
        "",
        "## Training Data",
        "",
        "- 10,500+ synthetic cryptographic protocol analyses",
        "- Protocols: TLS/SSL, SSH, IPsec, WireGuard, WPA, S/MIME, OpenPGP, DNSSEC, Kerberos, Bitcoin, Ethereum, gRPC, MQTT, Bluetooth",
        "- Vulnerability types: Shor-vulnerable, Grover-amplified, HNDL, downgrade attacks, weak randomness, deprecated protocols, transition gaps",
        "- Attack vectors: Shor factoring, Shor DLP, Grover search, quantum collision finding, HNDL passive collection, quantum MITM",
        "",
        "## Evaluation Results",
        "",
        "| Metric | Target | Score |",
        "|--------|--------|-------|",
        "| Perplexity | < 10.0 | TBD |",
        "| Vulnerability Detection Rate | > 85% | TBD |",
        "| Attack Vector Recognition | > 80% | TBD |",
        "| Edge Latency (CPU) | < 1000ms | TBD |",
        "| Memory Footprint | < 4GB | TBD |",
        "",
        "## Usage",
        "",
        "```python",
        f'model_id = "{repo}"',
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "model = AutoModelForCausalLM.from_pretrained(model_id)",
        "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "",
        'prompt = """Analyze the following TLS 1.2 implementation for quantum-vulnerable patterns:', 
        "```python",
        "context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)",
        "context.set_ciphers('RSA-AES256-GCM-SHA384')",
        "```",
        "Identify the vulnerability type, quantum attack vector, and recommend mitigations.\"\"\"",
        "",
        'inputs = tokenizer(prompt, return_tensors="pt")',
        "outputs = model.generate(**inputs, max_new_tokens=512)",
        'print(tokenizer.decode(outputs[0], skip_special_tokens=True))',
        "```",
        "",
        "## Limitations",
        "",
        "- Not a substitute for certified security consultants or formal verification",
        "- Synthetic training data may not capture all real-world edge cases",
        "- English only",
        "- Analysis is heuristic; false positives/negatives are possible",
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
        "@misc{01_quantum_resistant_crypto_analyzer,",
        f'  title={{{project["name"]}}},',
        '  author={QuantumIndSSI Ltd},',
        '  year={2026},',
        '  publisher={Hugging Face},',
        '  howpublished={\\url{https://huggingface.co/quantumindssi/01_quantum_resistant_crypto_analyzer}}',
        "}",
        "```",
        "",
        "## Contact",
        "",
        "- GitHub: https://github.com/QuantumindSSI",
        "- HuggingFace: https://huggingface.co/quantumindssi",
        "- Email: contact@quantumindssi.com",
    ]
    return "\n".join(card_lines)


def main():
    parser = argparse.ArgumentParser(description="Upload Quantum-Resistant Crypto Analyzer to Hugging Face")
    parser.add_argument("--config", type=str, default="./configs/config_crypto_analyzer.yaml")
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
