#!/bin/bash
# End-to-end training pipeline for Drug Interaction Predictor

set -e

CONFIG="./configs/config_ddi.yaml"
OUTPUT_DIR="./outputs/models_ddi_quick"
NUM_SCENARIOS=102000

echo "=========================================="
echo "Drug Interaction Predictor Training Pipeline"
echo "=========================================="

# Step 1: Generate data
echo "[1/5] Generating DDI data..."
python data/generate_drug_interaction_data.py \
    --num_scenarios $NUM_SCENARIOS \
    --output_dir ./data/raw_ddi \
    --seed 42

# Step 2: Preprocess
echo "[2/5] Preprocessing data..."
python data/preprocess.py \
    --input ./data/raw_ddi/instructions.jsonl \
    --output_dir ./data/processed_ddi_quick \
    --model_name ./base_model \
    --max_length 1024 \
    --seed 42

# Step 3: Train LoRA
echo "[3/5] Training with LoRA..."
python src/train_lora.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR

# Step 4: Evaluate
echo "[4/5] Evaluating model..."
python src/evaluate_ddi.py \
    --config $CONFIG \
    --model_path "$OUTPUT_DIR/checkpoint-final" \
    --output_file ./outputs/evaluations/eval_ddi_results.json

# Step 5: Test inference
echo "[5/5] Testing inference..."
python src/inference_ddi.py \
    --config $CONFIG \
    --model_path "$OUTPUT_DIR/checkpoint-final" \
    --mode batch \
    --input_file ./data/raw_ddi/test.jsonl \
    --output_file ./outputs/evaluations/generations_ddi.json

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "  Model: $OUTPUT_DIR/checkpoint-final"
echo "  Eval:  ./outputs/evaluations/eval_ddi_results.json"
echo "=========================================="
