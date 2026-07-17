# Agent Instructions for QuantuML

## Project Overview

QuantuML is a suite of fine-tuned Small Language Models (SLMs) for specialized edge AI tasks:
- Post-quantum cryptography advisory
- Cryptographic vulnerability analysis
- Drug-drug interaction prediction

Base model: Qwen/Qwen2-0.5B-Instruct (0.5B params, Apache 2.0)

## Build & Run

### Environment Setup
```bash
./scripts/setup_env.sh
source venv/bin/activate
```

### Download Base Model
```bash
huggingface-cli download Qwen/Qwen2-0.5B-Instruct --local-dir ./base_model
```

### Training Pipeline (GPU Required for Production)
```bash
# All models at once (GPU, 2 epochs, balanced data)
./scripts/run_training_gpu_all.sh

# Individual models
python src/train_lora.py --config ./configs/config_ddi_v2.yaml --output_dir ./outputs/models_ddi_v2
python src/train_lora.py --config ./configs/config_crypto_v2.yaml --output_dir ./outputs/models_crypto_v2
python src/train_lora.py --config ./configs/config_pqc_v2.yaml --output_dir ./outputs/models_pqc_v2
```

### Evaluation
```bash
python src/evaluate_pqc.py --config ./configs/config_pqc_migration.yaml --model_path ./outputs/models_poc/checkpoint-final

python src/evaluate_crypto.py --config ./configs/config_crypto_analyzer_quick.yaml --model_path ./outputs/models_crypto_quick/checkpoint-final

python src/evaluate_ddi.py --config ./configs/config_ddi.yaml --model_path ./outputs/models_ddi_quick/checkpoint-final

# Clinical validation against FDA/DrugBank ground truth
python src/validate_clinical.py --config ./configs/config_ddi.yaml --model_path ./outputs/models_ddi_quick/checkpoint-final --ground_truth ./data/clinical_ground_truth.json
```

### Inference
```bash
python src/inference_pqc.py --config ./configs/config_pqc_migration.yaml --model_path ./outputs/models_poc/checkpoint-final --mode interactive

python src/inference_crypto.py --config ./configs/config_crypto_analyzer_quick.yaml --model_path ./outputs/models_crypto_quick/checkpoint-final --mode interactive

python src/inference_ddi.py --config ./configs/config_ddi.yaml --model_path ./outputs/models_ddi_quick/checkpoint-final --mode interactive
```

### Deployment
```bash
# ONNX + INT8 export
python src/export_onnx_int8.py --model_path ./outputs/models_crypto_quick/checkpoint-final --output_dir ./deployment/onnx/crypto --task_name "Quantum-Resistant Crypto Analyzer"

# HuggingFace upload
export HF_TOKEN=hf_...
python src/upload_hf_crypto.py --model_path ./outputs/models_crypto_quick/checkpoint-final
```

## Key Conventions

- All configs are YAML in `configs/`
- All training uses LoRA by default (trainable params ~0.3% of model)
- Seeds are fixed at 42 for reproducibility
- Max sequence length is typically 1024 for quick POCs
- Data splits: 85% train, 10% eval, 5% test
- Evaluation uses custom heuristic quality metrics (not BLEU/ROUGE)
- Edge target: <4GB RAM, <1000ms latency (requires GPU or INT8 quantization)

## File Locations

| Purpose | Path |
|---------|------|
| Configs | `configs/config_*.yaml` |
| Training | `src/train_*.py` |
| Evaluation | `src/evaluate_*.py` |
| Inference | `src/inference_*.py` |
| Data generation | `data/generate_*.py` |
| Preprocessing | `data/preprocess.py` |
| API servers | `deployment/api/api_server_*.py` |
| Edge deployment | `deployment/victron/deploy_victron_*.py` |
| Benchmarks | `benchmarks/*.json` |

## External Links

- HuggingFace org: https://huggingface.co/QuantumindSSI
- Kaggle datasets: https://www.kaggle.com/quantumind
- GitHub: https://github.com/QuantumindSSI
