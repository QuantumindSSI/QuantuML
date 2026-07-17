#!/bin/bash
# Full GPU training pipeline for ALL models with balanced datasets
# Requires: NVIDIA GPU with 16GB+ VRAM, CUDA 12.1+

set -e

echo "=========================================="
echo "QuantuML Production GPU Training Pipeline"
echo "=========================================="

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU required for production training."
    exit 1
fi

echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# ============================================================
# MODEL 1: Drug Interaction Predictor v2
# ============================================================
echo ""
echo "[1/3] Training Drug Interaction Predictor v2..."
echo "  Dataset: 25,000 examples (20% negative)"
echo "  Epochs: 2, Batch: 4, GradAccum: 4"
echo "  ~1.5 hours on RTX 4090 / ~30 min on A100"

python src/train_lora.py \
    --config ./configs/config_ddi_v2.yaml \
    --output_dir ./outputs/models_ddi_v2

python src/validate_clinical.py \
    --config ./configs/config_ddi_v2.yaml \
    --model_path ./outputs/models_ddi_v2/checkpoint-final \
    --ground_truth ./data/clinical_ground_truth.json \
    --output_file ./outputs/evaluations/clinical_validation_v2.json

echo "  DDI v2 trained and validated."

# ============================================================
# MODEL 2: Quantum-Resistant Crypto Analyzer v2
# ============================================================
echo ""
echo "[2/3] Training Quantum-Resistant Crypto Analyzer v2..."
echo "  Dataset: 12,500 examples (20% safe)"
echo "  Epochs: 2, Batch: 4, GradAccum: 4"
echo "  ~45 min on RTX 4090 / ~15 min on A100"

python src/train_lora.py \
    --config ./configs/config_crypto_v2.yaml \
    --output_dir ./outputs/models_crypto_v2

python src/validate_crypto_realworld.py \
    --config ./configs/config_crypto_v2.yaml \
    --model_path ./outputs/models_crypto_v2/checkpoint-final \
    --output_file ./outputs/evaluations/crypto_realworld_validation_v2.json

echo "  Crypto v2 trained and validated."

# ============================================================
# MODEL 3: Post-Quantum Key Migration Advisor v2
# ============================================================
echo ""
echo "[3/3] Training Post-Quantum Key Migration Advisor v2..."
echo "  Dataset: 6,500 examples (20% already-migrated)"
echo "  Epochs: 2, Batch: 4, GradAccum: 4"
echo "  ~25 min on RTX 4090 / ~8 min on A100"

python src/train_lora.py \
    --config ./configs/config_pqc_v2.yaml \
    --output_dir ./outputs/models_pqc_v2

python src/validate_pqc_realworld.py \
    --config ./configs/config_pqc_v2.yaml \
    --model_path ./outputs/models_pqc_v2/checkpoint-final \
    --output_file ./outputs/evaluations/pqc_realworld_validation_v2.json

echo "  PQC v2 trained and validated."

# ============================================================
# DONE
# ============================================================
echo ""
echo "=========================================="
echo "PRODUCTION TRAINING COMPLETE"
echo "=========================================="
echo "Models:"
echo "  DDI:    ./outputs/models_ddi_v2/checkpoint-final"
echo "  Crypto: ./outputs/models_crypto_v2/checkpoint-final"
echo "  PQC:    ./outputs/models_pqc_v2/checkpoint-final"
echo ""
echo "Validation Reports:"
echo "  DDI:    ./outputs/evaluations/clinical_validation_v2.json"
echo "  Crypto: ./outputs/evaluations/crypto_realworld_validation_v2.json"
echo "  PQC:    ./outputs/evaluations/pqc_realworld_validation_v2.json"
echo ""
echo "Next steps:"
echo "  1. Review validation results"
echo "  2. Export to ONNX: python src/export_onnx_int8.py --model_path ..."
echo "  3. Deploy: python src/inference_production.py --model ddi --input ..."
echo "=========================================="
