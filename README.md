# QuantuML: Sovereign AI Edge Model Suite

> **Enterprise-grade fine-tuning pipeline for deploying specialized Small Language Models (SLMs) on sovereign, privacy-preserving edge hardware.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.39+-yellow.svg)](https://huggingface.co/docs/transformers/)

---

## Models in this Suite

| # | Model | Task | HuggingFace | Kaggle Dataset |
|---|-------|------|-------------|----------------|
| 00 | **QuantuML PQC POC** | Base model & pipeline validation | [quantumindssi/quantuml-pqc-0.5b-poc](https://huggingface.co/quantumindssi/quantuml-pqc-0.5b-poc) | [PQC Synthetic](https://www.kaggle.com/datasets/quantumind/post-quantum-key-migration-synthetic-dataset) |
| 01 | **Quantum-Resistant Cryptographic Protocol Analyzer** | Detect quantum-vulnerable crypto patterns | [quantumindssi/01_quantum_resistant_crypto_analyzer](https://huggingface.co/quantumindssi/01_quantum_resistant_crypto_analyzer) | [Crypto Vulnerability](https://www.kaggle.com/datasets/quantumind/quantum-vulnerability-crypto-protocol-dataset) |
| 02 | **Post-Quantum Key Migration Advisor** | Enterprise PQC migration planning | [quantumindssi/02_post_quantum_key_migration_advisor](https://huggingface.co/quantumindssi/02_post_quantum_key_migration_advisor) | [Hybrid Dataset](https://www.kaggle.com/datasets/quantumind/post-quantum-key-migration-hybrid-dataset) |
| 09 | **Drug Interaction Predictor** | Predict adverse drug-drug interactions | [QuantumindSSI/09_drug_interaction_predictor](https://huggingface.co/QuantumindSSI/09_drug_interaction_predictor) | [DDI Dataset](https://www.kaggle.com/datasets/quantumind/synthetic-drug-drug-interaction-dataset) |

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/QuantumindSSI/QuantuML.git
cd QuantuML
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Download base model
huggingface-cli download Qwen/Qwen2-0.5B-Instruct --local-dir ./base_model

# 3. Pick a model and follow its pipeline (see docs/MODELS.md)
```

---

## Repository Structure

```
QuantuML/
├── README.md                     # This file
├── LICENSE                       # Apache 2.0
├── requirements.txt              # Python dependencies
├── configs/                      # YAML configs per model
│   ├── config_pqc_migration.yaml
│   ├── config_crypto_analyzer.yaml
│   ├── config_crypto_analyzer_quick.yaml
│   └── config_ddi.yaml
├── data/                         # Data generation & preprocessing
│   ├── generate_pqc_migration_data.py
│   ├── generate_crypto_vulnerability_data.py
│   ├── generate_drug_interaction_data.py
│   ├── compile_hybrid_dataset.py
│   ├── preprocess.py
│   └── README.md
├── src/                          # Training, evaluation, inference
│   ├── train_lora.py
│   ├── train_qlora.py
│   ├── train_dpo.py
│   ├── train_ppo.py
│   ├── train_orpo.py
│   ├── evaluate_pqc.py
│   ├── evaluate_crypto.py
│   ├── evaluate_ddi.py
│   ├── inference_pqc.py
│   ├── inference_crypto.py
│   ├── inference_ddi.py
│   ├── export_onnx_int8.py
│   ├── upload_hf.py
│   ├── upload_hf_crypto.py
│   └── upload_hf_ddi.py
├── deployment/                   # Production deployment
│   ├── api/
│   │   ├── api_server_pqc.py
│   │   ├── api_server_crypto.py
│   │   └── api_server_ddi.py
│   └── victron/
│       ├── deploy_victron_pqc.py
│       ├── deploy_victron_crypto.py
│       └── deploy_victron_ddi.py
├── scripts/                      # Automation scripts
│   ├── setup_env.sh
│   ├── run_training_pqc.sh
│   └── run_deployment.sh
├── notebooks/                    # Jupyter notebooks (exploration)
├── benchmarks/                   # Evaluation results
│   ├── pqc_migration_results.json
│   ├── crypto_analyzer_results.json
│   ├── drug_interaction_results.json
│   └── pqc_migration_generations.json
└── docs/                         # Technical documentation
    ├── MODELS.md
    ├── BENCHMARKS.md
    ├── ARCHITECTURE.md
    └── DATASETS.md
```

---

## Training Methods

| Method | VRAM Required | Speed | Quality | Use Case |
|--------|--------------|-------|---------|----------|
| **LoRA** | ~16GB | Fast | High | Standard fine-tuning |
| **QLoRA** | ~8GB | Medium | High | Consumer GPUs / limited VRAM |
| **DPO** | ~20GB | Medium | Very High | Preference-based alignment |
| **PPO** | ~24GB | Slow | Very High | RLHF with reward model |
| **ORPO** | ~16GB | Fast | Very High | Single-stage preference optimization |

---

## Base Model

**`Qwen/Qwen2-0.5B-Instruct`** (0.5B parameters)

| Attribute | Value |
|-----------|-------|
| Architecture | Transformer decoder (causal LM) |
| Parameters | 0.5B |
| Context length | 32,768 tokens |
| License | Apache 2.0 |
| Size (FP32) | ~2.0 GB |
| Size (INT8) | ~1.0 GB |
| Size (INT4) | ~0.5 GB |

*Alternative: `microsoft/phi-3-mini-4k-instruct` for higher quality at ~3.8B params*

---

## Evaluation Summary

### 00: QuantuML PQC POC (Base)

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Perplexity | **1.59** | < 10.0 | Pass |
| Task Quality | **87.5%** | > 80% | Pass |
| Train Loss | 0.300 | < 0.5 | Pass |
| Eval Loss | 0.073 | < 0.5 | Pass |

### 01: Quantum-Resistant Crypto Analyzer

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Perplexity | **1.45** | < 10.0 | Pass |
| Task Quality | **83.3%** | > 80% | Pass |
| Vulnerability Detection | 7/8 checks | > 85% | Pass |
| Attack Vector Recognition | Present | > 80% | Pass |
| Eval Loss | 0.094 | < 0.5 | Pass |

### 02: Post-Quantum Key Migration Advisor

Uses same base architecture with hybrid dataset. See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for full results.

### 09: Drug Interaction Predictor

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Perplexity | **3.38** | < 10.0 | Pass |
| Task Quality | **71.4%** | > 70% | Pass |
| Eval Loss | 0.280 | < 0.5 | Pass |
| Test Loss | 0.308 | < 0.5 | Pass |

**Clinical Validation (POC)**: See [docs/CLINICAL_VALIDATION.md](docs/CLINICAL_VALIDATION.md)

| Metric | POC Score | Target | Status |
|--------|-----------|--------|--------|
| Sensitivity | **100%** | > 95% | Pass |
| Specificity | **0%** | > 85% | **FAIL** |
| Severity Accuracy | **26.7%** | > 70% | **FAIL** |
| Overall Clinical Score | **53.8%** | > 75% | **FAIL** |

> **Not clinically ready.** The model hallucinates interactions for every safe drug pair (100% false positive rate) because the original training data contained zero negative examples. Remediation is documented in `docs/CLINICAL_VALIDATION.md`.

---

## Deployment Targets

| Target | Memory | Latency | Quantization | Script |
|--------|--------|---------|--------------|--------|
| Cloud GPU (A100/H100) | 2GB | ~0.3s | FP16 | `src/inference_*.py` |
| Workstation GPU | 2GB | ~0.5s | INT8 | `deployment/victron/deploy_victron_*.py --quantize int8` |
| Victron Edge (CPU) | 3GB | ~2-5s | INT8 | `deployment/victron/deploy_victron_*.py --quantize int8` |
| Tiny Edge | 2GB | ~3-8s | INT4 | `deployment/victron/deploy_victron_*.py --quantize int4` |

---

## API Server

```bash
# Start FastAPI server for any model
python deployment/api/api_server_crypto.py \
  --model_path ./outputs/models_crypto_quick/checkpoint-final \
  --host 0.0.0.0 --port 8000
```

```bash
# Query example (Crypto Analyzer)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "protocol_description": "Banking TLS 1.2 using RSA-2048",
    "implementation_code": "context.set_ciphers('RSA-AES256-GCM-SHA384')",
    "algorithm": "RSA-2048",
    "max_tokens": 512
  }'

# Query example (Drug Interaction)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"medications": "Warfarin, Aspirin, Omeprazole", "patient_context": "72-year-old female with AF"}'
```

---

## Datasets on Kaggle

All training datasets are publicly available on Kaggle for reproducibility:

1. **[Synthetic Drug-Drug Interaction Dataset](https://www.kaggle.com/datasets/quantumind/synthetic-drug-drug-interaction-dataset)**
   - 102,000 synthetic DDI examples
   - Train: 86,700 | Eval: 10,200 | Test: 5,100

2. **[Quantum Vulnerability Cryptographic Protocol Dataset](https://www.kaggle.com/datasets/quantumind/quantum-vulnerability-crypto-protocol-dataset)**
   - 10,500 synthetic crypto vulnerability analyses
   - Train: 8,925 | Eval: 1,050 | Test: 525

3. **[Post-Quantum Key Migration Hybrid Dataset](https://www.kaggle.com/datasets/quantumind/post-quantum-key-migration-hybrid-dataset)**
   - 5,317 hybrid real + synthetic PQC migration scenarios
   - Train: 4,521 | Eval: 531 | Test: 265

4. **[Post-Quantum Key Migration Synthetic Dataset](https://www.kaggle.com/datasets/quantumind/post-quantum-key-migration-synthetic-dataset)**
   - 5,300 synthetic PQC migration scenarios

5. **[PQC Real Curated Instructions Dataset](https://www.kaggle.com/datasets/quantumind/pqc-real-curated-instructions-dataset)**
   - 18 high-quality real-world PQC instructions

---

## Hugging Face Upload

```bash
export HF_TOKEN=hf_...

# PQC Migration Advisor
python src/upload_hf.py \
  --model_path ./outputs/models_poc/checkpoint-final \
  --repo_id quantumindssi/02_post_quantum_key_migration_advisor

# Crypto Analyzer
python src/upload_hf_crypto.py \
  --model_path ./outputs/models_crypto_quick/checkpoint-final \
  --repo_id quantumindssi/01_quantum_resistant_crypto_analyzer

# Drug Interaction Predictor
python src/upload_hf_ddi.py \
  --model_path ./outputs/models_ddi_quick/checkpoint-final \
  --repo_id QuantumindSSI/09_drug_interaction_predictor
```

---

## ONNX + INT8 Export for Edge

```bash
python src/export_onnx_int8.py \
  --model_path ./outputs/models_crypto_quick/checkpoint-final \
  --output_dir ./deployment/onnx/crypto \
  --task_name "Quantum-Resistant Crypto Analyzer" \
  --prompt "Analyze TLS 1.2 with RSA-2048 for quantum vulnerabilities"
```

This exports:
1. ONNX model
2. INT8 quantized model
3. Benchmark report
4. Victron deployment package

---

## Technical Documentation

- **[docs/MODELS.md](docs/MODELS.md)** - Detailed model cards, training configs, and hyperparameters
- **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** - Full benchmark results, latency measurements, and memory profiles
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture, data flow, and deployment diagrams
- **[docs/DATASETS.md](docs/DATASETS.md)** - Dataset specifications, generation methodology, and schema documentation

---

## Contributing

We welcome contributions! Please see our [GitHub Issues](https://github.com/QuantumindSSI/QuantuML/issues) for open tasks.

## Citation

```bibtex
@misc{quantuml2026,
  title={QuantuML: Sovereign AI Edge Model Suite},
  author={QuantumIndSSI Ltd},
  year={2026},
  howpublished={\url{https://github.com/QuantumindSSI/QuantuML}}
}
```

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

## Contact

- **Website**: https://quantumindssi.com
- **GitHub**: https://github.com/QuantumindSSI
- **Hugging Face**: https://huggingface.co/QuantumindSSI
- **Kaggle**: https://www.kaggle.com/quantumind
- **Email**: contact@quantumindssi.com
