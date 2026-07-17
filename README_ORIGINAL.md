# QuantuML: Sovereign AI Edge Model Suite

Enterprise ML fine-tuning workflow for deploying specialized Small Language Models (SLMs) on edge hardware. This repo contains multiple task-specific fine-tuned models for sovereign, privacy-preserving AI applications.

## Models in this Suite

| # | Model | Task | HF Repo |
|---|-------|------|---------|
| 01 | **Quantum-Resistant Cryptographic Protocol Analyzer** | Detect quantum-vulnerable crypto patterns | [QuantumindSSI/01-quantum-resistant-crypto-analyzer](https://huggingface.co/QuantumindSSI/01-quantum-resistant-crypto-analyzer) |
| 09 | **Drug Interaction Predictor** | Predict adverse drug-drug interactions | [QuantumindSSI/09-drug-interaction-predictor](https://huggingface.co/QuantumindSSI/09-drug-interaction-predictor) |

## 01: Quantum-Resistant Cryptographic Protocol Analyzer

Enterprise ML fine-tuning workflow for deploying a quantum-safe cryptography vulnerability detector on edge hardware.

### Overview

This project fine-tunes a Small Language Model (SLM) to analyze cryptographic protocol implementations and identify quantum-vulnerable patterns, attack vectors, and NIST-aligned mitigations.

## Quick Start

```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Generate synthetic training data (10,500 scenarios)
python data/generate_vulnerability_data.py --num_scenarios 10500 --output_dir ./data/raw_vuln

# 3. Preprocess data
python data/preprocess.py --input ./data/raw_vuln/instructions.jsonl --output_dir ./data/processed_vuln --model_name ./base_model --max_length 1024

# 4. Download base model (if not present)
huggingface-cli download Qwen/Qwen2-0.5B-Instruct --local-dir ./base_model

# 5. Train with LoRA (CPU: ~2h for full 512-sample POC; GPU: <10 min)
python src/train_lora.py --config ./configs/config_crypto_analyzer_quick.yaml --output_dir ./outputs/models_crypto_quick

# 6. Evaluate
python src/evaluate_crypto.py --config ./configs/config_crypto_analyzer_quick.yaml --model_path ./outputs/models_crypto_quick/checkpoint-final --num_samples 20

# 7. Run inference
python src/inference_crypto.py --config ./configs/config_crypto_analyzer_quick.yaml --model_path ./outputs/models_crypto_quick/checkpoint-final --mode interactive
```

## Project Structure

```
QuantuML/
├── configs/
│   ├── config.yaml                          # Original PQC Migration Advisor config
│   ├── config_crypto_analyzer.yaml          # Full crypto analyzer config
│   └── config_crypto_analyzer_quick.yaml    # Quick POC config (512 samples)
├── data/
│   ├── generate_synthetic_data.py           # Original migration data generator
│   ├── generate_vulnerability_data.py       # Crypto vulnerability data generator
│   ├── preprocess.py                        # Tokenization pipeline
│   ├── raw_vuln/                            # Generated vulnerability data (10,500)
│   ├── processed_vuln/                      # Tokenized full dataset
│   └── processed_vuln_quick/                # Tokenized POC dataset (512)
├── src/
│   ├── train_lora.py                        # LoRA fine-tuning (reusable)
│   ├── evaluate_crypto.py                   # Crypto analyzer evaluation
│   ├── inference_crypto.py                  # Local inference CLI
│   └── upload_hf_crypto.py                  # Hugging Face upload
├── deployment/
│   ├── api/
│   │   ├── api_server.py                    # Original migration API
│   │   └── api_server_crypto.py             # Crypto analyzer API
│   └── victron/
│       ├── deploy_victron.py                # Original Victron deployment
│       └── deploy_victron_crypto.py         # Crypto analyzer Victron deployment
│       └── crypto/victron_package/          # Generated deployment package
├── notebooks/                               # Exploration notebooks
├── scripts/                                 # Automation scripts
├── outputs/
│   ├── models_crypto_quick/                 # Trained checkpoints
│   │   └── checkpoint-final/
│   ├── logs_crypto_quick/                   # Training logs
│   └── evaluations/
│       └── eval_crypto_results.json         # Evaluation metrics
├── base_model/                              # Qwen2-0.5B-Instruct
├── requirements.txt
└── README.md
```

## Training Methods

| Method | VRAM Required | Speed | Quality | Use Case |
|--------|--------------|-------|---------|----------|
| **LoRA** | ~16GB | Fast | High | Standard training |
| **QLoRA** | ~8GB | Medium | High | Consumer GPUs |
| **DPO** | ~20GB | Medium | Very High | Preference tuning |

## Base Model

**`Qwen/Qwen2-0.5B-Instruct`** (0.5B params, locally stored in `./base_model`)
- Excellent for constrained edge deployment (<2GB in float32)
- Instruction-tuned for following analysis prompts
- Apache 2.0 license

*(Alternative: `microsoft/phi-3-mini-4k-instruct` for higher quality at ~3.8B params)*

## Training Data

10,500 synthetic cryptographic vulnerability analyses covering:
- **Protocols**: TLS 1.0-1.3, SSH, IPsec/IKE, WireGuard, OpenVPN, WPA2/WPA3, S/MIME, OpenPGP, DNSSEC, Kerberos, Bitcoin, Ethereum, gRPC, MQTT, Bluetooth, Signal, Noise
- **Algorithms**: RSA (1024-4096), DH, ECDH, ECDSA, DSA, AES, 3DES, ChaCha20, SHA, MD5, PBKDF2
- **Vulnerability Types**: Shor's Algorithm Vulnerable, Grover's Algorithm Amplified, Harvest Now Decrypt Later (HNDL), Quantum-Enabled Downgrade, Insufficient Key Size, Deprecated Protocol, Weak Randomness, Small Subgroup, Post-Quantum Transition Gap, Side-Channel Leakage
- **Attack Vectors**: Shor factoring, Shor DLP, Grover search, quantum period finding, quantum collision finding (BHT), HNDL passive collection, quantum MITM, store-now-decrypt-later, quantum-enhanced brute force
- **Mitigations**: NIST FIPS 203/204/205, NSA CNSA 2.0, hybrid key exchange (X25519Kyber768), crypto-agility frameworks

### Data Format

```json
{
  "scenario_id": "QRCA-00001",
  "protocol_description": "Enterprise TLS 1.2 deployment using RSA-2048 for key exchange...",
  "protocol_name": "TLS 1.2",
  "implementation_code": "import ssl\ncontext = ssl.SSLContext(...)\n...",
  "algorithm_used": "RSA-2048",
  "vulnerability_type": "Shor's Algorithm Vulnerable",
  "vulnerability_category": "shor_vulnerable",
  "severity": "Critical",
  "severity_score": 9.5,
  "quantum_attack_vector": "Shor's Integer Factorization",
  "quantum_attack_details": "A CRQC with ~20M physical qubits could factor 2048-bit RSA in ~8 hours...",
  "secondary_attack_vector": "Harvest Now, Decrypt Later (HNDL)",
  "mitigations": ["Transition to hybrid PQC key exchange...", "Enable crypto-agility..."],
  "nist_references": ["FIPS 203 (ML-KEM)", "NIST IR 8547"]
}
```

## Evaluation Metrics

Results from the quick POC (512 samples, 55 steps, CPU):

| Metric | Score | Target |
|--------|-------|--------|
| Perplexity | **1.45** | < 10.0 |
| Task Quality | **83.3%** | > 80% |
| Vulnerability Detection | **7/8 checks** | > 85% |
| Attack Vector Recognition | **Present** | > 80% |
| Eval Loss | **0.094** | < 0.5 |

**Note**: Full training on 10,500 samples with a GPU achieves significantly higher quality. The POC run on CPU is for pipeline validation.

## Deployment Targets

| Target | Memory | Latency | Command |
|--------|--------|---------|---------|
| Cloud GPU (A100) | 2GB FP16 | ~0.3s | `src/inference_crypto.py` |
| Workstation GPU | 2GB INT8 | ~0.5s | `deployment/victron/deploy_victron_crypto.py --quantize int8` |
| Victron Edge (CPU) | 3GB INT8 | ~2-5s | `deployment/victron/deploy_victron_crypto.py --quantize int8` |
| Tiny Edge | 2GB INT4 | ~3-8s | `deployment/victron/deploy_victron_crypto.py --quantize int4` |

*(Edge latency targets assume ONNX Runtime + INT8 quantization. Raw PyTorch CPU inference is slower.)*

## API Server

```bash
python deployment/api/api_server_crypto.py \
  --model_path ./outputs/models_crypto_quick/checkpoint-final \
  --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "protocol_description": "Banking TLS 1.2 infrastructure using RSA-2048 key exchange",
    "implementation_code": "context.set_ciphers('RSA-AES256-GCM-SHA384')",
    "algorithm": "RSA-2048",
    "max_tokens": 512
  }'
```

## Victron Edge Deployment

```bash
# Build deployment package
python deployment/victron/deploy_victron_crypto.py \
  --model_path ./outputs/models_crypto_quick/checkpoint-final \
  --output_dir ./deployment/victron/crypto \
  --quantize int8 \
  --benchmark

# Deploy to device
scp -r ./deployment/victron/crypto/victron_package root@victron-device:/opt/crypto-analyzer
ssh root@victron-device "cd /opt/crypto-analyzer && sudo cp crypto-analyzer.service /etc/systemd/system/ && sudo systemctl enable --now crypto-analyzer"
```

## Upload to Hugging Face

```bash
export HF_TOKEN=hf_...
python src/upload_hf_crypto.py \
  --model_path ./outputs/models_crypto_quick/checkpoint-final \
  --repo_id quantumindssi/01_quantum_resistant_crypto_analyzer
```

## Model Card

The model card is automatically generated during upload and includes:
- License: Apache 2.0
- Tags: quantumindssi, sovereign-ai, edge-computing, post-quantum-cryptography, vulnerability-detection, nist-pqc, ml-kem, ml-dsa, slh-dsa
- Hardware requirements
- Usage examples
- Citation information

## 09: Drug Interaction Predictor

Local pharmacovigilance system for detecting adverse drug-drug interactions from medication lists.

### Overview

This model fine-tunes a Qwen2-0.5B-Instruct SLM to predict drug-drug interactions, including severity, mechanism, predicted outcomes, clinical recommendations, and evidence sources.

### Quick Start (DDI)

```bash
# 1. Generate synthetic DDI data (102,000 examples)
python data/generate_drug_interaction_data.py --num_scenarios 102000 --output_dir ./data/raw_ddi

# 2. Preprocess
python data/preprocess.py --input ./data/raw_ddi/instructions.jsonl --output_dir ./data/processed_ddi_quick --model_name ./base_model --max_length 1024

# 3. Train
python src/train_lora.py --config ./configs/config_ddi.yaml --output_dir ./outputs/models_ddi_quick

# 4. Evaluate
python src/evaluate_ddi.py --config ./configs/config_ddi.yaml --model_path ./outputs/models_ddi_quick/checkpoint-final --num_samples 20

# 5. Interactive inference
python src/inference_ddi.py --config ./configs/config_ddi.yaml --model_path ./outputs/models_ddi_quick/checkpoint-final --mode interactive
```

### Training Data

- **102,000** synthetic DDI scenarios
- **30 drug classes**: anticoagulants, statins, antibiotics, antidepressants, PPIs, NSAIDs, opioids, immunosuppressants, anticonvulsants, oral hypoglycemics, and more
- **Interaction types**: CYP inhibition/induction, P-gp effects, additive toxicity, QT prolongation, serotonin syndrome, bleeding risk
- **Severity levels**: Critical, High, Medium, Low

### Evaluation Results (POC)

| Metric | Score | Target |
|--------|-------|--------|
| Perplexity | **3.38** | < 10.0 |
| Task Quality | **71.4%** | > 70% |
| Eval Loss | **0.280** | < 0.5 |
| Test Loss | **0.308** | < 0.5 |

### API Server (DDI)

```bash
python deployment/api/api_server_ddi.py \
  --model_path ./outputs/models_ddi_quick/checkpoint-final \
  --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"medications": "Warfarin, Aspirin, Omeprazole", "patient_context": "72-year-old female with AF"}'
```

### Upload to Hugging Face (DDI)

```bash
export HF_TOKEN=hf_...
python src/upload_hf_ddi.py \
  --model_path ./outputs/models_ddi_quick/checkpoint-final \
  --repo_id QuantumindSSI/09-drug-interaction-predictor
```

---

## Community & Contributing

- **GitHub**: https://github.com/QuantumindSSI
- **Hugging Face**: https://huggingface.co/QuantumindSSI
- **Discord**: `#quantumindssi-models`

## License

Apache License 2.0

## Contact

- Website: https://quantumindssi.com
- Email: contact@quantumindssi.com
