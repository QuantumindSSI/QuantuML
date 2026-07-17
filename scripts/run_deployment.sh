#!/bin/bash
# Deployment pipeline for Victron edge

set -e

MODEL_PATH="./outputs/models/checkpoint-final"
DEPLOY_DIR="./deployment/victron"

echo "=========================================="
echo "QuantuML Edge Deployment Pipeline"
echo "=========================================="

# Step 1: Quantize for edge
echo "[1/3] Quantizing model for edge deployment (INT8)..."
python deployment/victron/deploy_victron.py \
    --model_path $MODEL_PATH \
    --output_dir $DEPLOY_DIR \
    --quantize int8 \
    --benchmark

# Step 2: Package
echo "[2/3] Creating deployment package..."
# (deploy_victron.py handles packaging)

# Step 3: Upload to HF (optional)
if [ -n "$HF_TOKEN" ]; then
    echo "[3/3] Uploading to Hugging Face..."
    python src/upload_hf.py \
        --model_path $MODEL_PATH \
        --repo_id quantumindssi/02_post_quantum_key_migration_advisor
else
    echo "[3/3] Skipping HF upload (HF_TOKEN not set)"
fi

echo ""
echo "=========================================="
echo "Deployment package ready at:"
echo "  $DEPLOY_DIR/victron_package"
echo ""
echo "To deploy on Victron hardware:"
echo "  sudo cp $DEPLOY_DIR/victron_package /opt/pqc-advisor"
echo "  sudo cp /opt/pqc-advisor/pqc-advisor.service /etc/systemd/system/"
echo "  sudo systemctl enable --now pqc-advisor"
echo "=========================================="
