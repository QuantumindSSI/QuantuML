#!/bin/bash
# End-to-end training pipeline for QuantuML

set -e

CONFIG="./configs/config.yaml"
OUTPUT_DIR="./outputs/models"
NUM_SCENARIOS=5500

echo "=========================================="
echo "QuantuML Training Pipeline"
echo "=========================================="

# Step 1: Generate data
echo "[1/5] Generating synthetic training data..."
python data/generate_synthetic_data.py \
    --num_scenarios $NUM_SCENARIOS \
    --output_dir ./data/raw \
    --seed 42

# Step 2: Preprocess
echo "[2/5] Preprocessing data..."
python data/preprocess.py \
    --input ./data/raw/instructions.jsonl \
    --output_dir ./data/processed \
    --model_name microsoft/phi-3-mini-4k-instruct \
    --max_length 4096 \
    --seed 42

# Step 3: Train LoRA
echo "[3/5] Training with LoRA..."
python src/train_lora.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR

# Step 4: Evaluate
echo "[4/5] Evaluating model..."
python src/evaluate.py \
    --config $CONFIG \
    --model_path "$OUTPUT_DIR/checkpoint-final" \
    --output_file ./outputs/evaluations/eval_results.json

# Step 5: Test inference
echo "[5/5] Testing inference..."
python src/inference.py \
    --config $CONFIG \
    --model_path "$OUTPUT_DIR/checkpoint-final" \
    --mode batch \
    --input_file ./data/raw/instructions.jsonl \
    --output_file ./outputs/evaluations/sample_generations.json

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "  Model: $OUTPUT_DIR/checkpoint-final"
echo "  Eval:  ./outputs/evaluations/eval_results.json"
echo "=========================================="
