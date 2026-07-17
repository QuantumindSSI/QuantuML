# Production Readiness Guide

This guide documents the complete pipeline from data generation to production deployment for all QuantuML models.

---

## Current Status

| Model | Ready for Training | Ready for Production |
|-------|-------------------|----------------------|
| DDI Predictor v2 | Yes (balanced data, GPU config) | Awaiting GPU retraining |
| Crypto Analyzer v2 | Yes (balanced data, GPU config) | Awaiting GPU retraining |
| PQC Migration Advisor v2 | Yes (balanced data, GPU config) | Awaiting GPU retraining |

**All models were validated on synthetic metrics but FAILED real-world validation on CPU-only POC runs (55 steps). Full GPU retraining is required.**

---

## Step 1: Generate Balanced Training Data

All three generators now support `--negative_ratio` (default 0.20):

```bash
# DDI: 25,000 total (20,000 positive + 5,000 negative)
python data/generate_drug_interaction_data.py \
    --num_scenarios 25000 --negative_ratio 0.20 \
    --output_dir ./data/raw_ddi_v2 --seed 42

# Crypto: 12,500 total (10,000 vulnerable + 2,500 safe)
python data/generate_vulnerability_data.py \
    --num_scenarios 12500 --negative_ratio 0.20 \
    --output_dir ./data/raw_crypto_v2 --seed 42

# PQC: 6,500 total (5,200 migration + 1,300 already-migrated)
python data/generate_synthetic_data.py \
    --num_scenarios 6500 --negative_ratio 0.20 \
    --output_dir ./data/raw_pqc_v2 --seed 42
```

---

## Step 2: Preprocess

```bash
# DDI
python data/preprocess.py \
    --input ./data/raw_ddi_v2/instructions.jsonl \
    --output_dir ./data/processed_ddi_v2 \
    --model_name ./base_model --max_length 1024

# Crypto
python data/preprocess.py \
    --input ./data/raw_crypto_v2/instructions.jsonl \
    --output_dir ./data/processed_crypto_v2 \
    --model_name ./base_model --max_length 1024

# PQC
python data/preprocess.py \
    --input ./data/raw_pqc_v2/instructions.jsonl \
    --output_dir ./data/processed_pqc_v2 \
    --model_name ./base_model --max_length 1024
```

---

## Step 3: Train (GPU Required)

### Single-Model Training

```bash
# DDI (2 epochs, batch 4, ~1.5h on RTX 4090)
python src/train_lora.py \
    --config ./configs/config_ddi_v2.yaml \
    --output_dir ./outputs/models_ddi_v2

# Crypto (2 epochs, batch 4, ~45min on RTX 4090)
python src/train_lora.py \
    --config ./configs/config_crypto_v2.yaml \
    --output_dir ./outputs/models_crypto_v2

# PQC (2 epochs, batch 4, ~25min on RTX 4090)
python src/train_lora.py \
    --config ./configs/config_pqc_v2.yaml \
    --output_dir ./outputs/models_pqc_v2
```

### All Models at Once

```bash
chmod +x scripts/run_training_gpu_all.sh
./scripts/run_training_gpu_all.sh
```

### Hardware Requirements

| GPU | VRAM | DDI Time | Crypto Time | PQC Time |
|-----|------|----------|-------------|----------|
| A100 40GB | 40 GB | ~30 min | ~15 min | ~8 min |
| RTX 4090 | 24 GB | ~90 min | ~45 min | ~25 min |
| RTX 3090 | 24 GB | ~100 min | ~50 min | ~30 min |
| V100 16GB | 16 GB | ~120 min | ~60 min | ~35 min |

---

## Step 4: Validate Against Real-World Ground Truth

```bash
# DDI clinical validation
python src/validate_clinical.py \
    --config ./configs/config_ddi_v2.yaml \
    --model_path ./outputs/models_ddi_v2/checkpoint-final \
    --ground_truth ./data/clinical_ground_truth.json

# Crypto real-world validation
python src/validate_crypto_realworld.py \
    --config ./configs/config_crypto_v2.yaml \
    --model_path ./outputs/models_crypto_v2/checkpoint-final

# PQC real-world validation
python src/validate_pqc_realworld.py \
    --config ./configs/config_pqc_v2.yaml \
    --model_path ./outputs/models_pqc_v2/checkpoint-final
```

### Production Validation Targets

| Metric | DDI Target | Crypto Target | PQC Target |
|--------|-----------|---------------|------------|
| Sensitivity / Detection | > 95% | > 90% | > 90% |
| Specificity / Safe Case Accuracy | > 85% | > 85% | N/A |
| Structural Score | > 75% | > 75% | > 70% |
| Token Confidence (mean) | > 0.65 | > 0.65 | > 0.60 |

---

## Step 5: Production Inference with Confidence Scoring

The production pipeline adds **confidence scoring, rejection thresholds, and safety guardrails**:

```bash
# Interactive / single case
python src/inference_production.py \
    --model ddi \
    --input "Warfarin, Aspirin, Omeprazole"

# Batch processing
python src/inference_production.py \
    --model crypto \
    --input_file ./test_cases.jsonl \
    --output_file ./outputs/production_results.json
```

### Confidence Scoring

Two scores are computed per output:

1. **Token Confidence** (0-1): Mean probability of generated tokens. Higher = model more certain.
2. **Structural Score** (0-1): Coverage of expected sections (severity, mechanism, evidence, etc.).

### Rejection Thresholds

| Model | Threshold | Action if Below |
|-------|-----------|-----------------|
| DDI | 0.35 | Reject + flag "VERIFY_WITH_PHARMACIST" |
| Crypto | 0.35 | Reject + flag "VERIFY_WITH_SECURITY_TEAM" |
| PQC | 0.40 | Reject + flag "REVIEW_WITH_ARCHITECT" |

### Safety Flags

| Flag | Meaning | Action |
|------|---------|--------|
| `VERIFY_WITH_PHARMACIST` | Low confidence or missing critical sections | Human review required |
| `VERIFY_WITH_SECURITY_TEAM` | Crypto analysis seems incomplete | Penetration test required |
| `MISSING_RECOMMENDATIONS` | Critical severity without mitigations | Re-flag for review |
| `NEGATIVE_CASE` | Model correctly identified safe case | Proceed with standard care |

---

## Step 6: ONNX Export for Edge Deployment

```bash
# Export DDI model + INT8 quantization
python src/export_onnx_int8.py \
    --model_path ./outputs/models_ddi_v2/checkpoint-final \
    --output_dir ./deployment/onnx/ddi_v2 \
    --task_name "Drug Interaction Predictor v2"

# Export Crypto model
python src/export_onnx_int8.py \
    --model_path ./outputs/models_crypto_v2/checkpoint-final \
    --output_dir ./deployment/onnx/crypto_v2 \
    --task_name "Quantum-Resistant Crypto Analyzer v2"
```

---

## Step 7: Upload Models to HuggingFace

```bash
export HF_TOKEN=hf_...

python src/upload_hf_ddi.py \
    --model_path ./outputs/models_ddi_v2/checkpoint-final \
    --repo_id QuantumindSSI/09_drug_interaction_predictor

python src/upload_hf_crypto.py \
    --model_path ./outputs/models_crypto_v2/checkpoint-final \
    --repo_id quantumindssi/01_quantum_resistant_crypto_analyzer
```

---

## Step 8: Upload Datasets to Kaggle

After generating new datasets:

```bash
# DDI v2 is already live:
# https://www.kaggle.com/datasets/quantumind/drug-interaction-dataset-with-negatives

# Upload Crypto v2 and PQC v2 similarly via Kaggle API
```

---

## Common Issues

### Issue: `LoraConfig.__init__() got unexpected keyword argument 'alora_invocation_tokens'`
**Fix:** Install exact PEFT version or use the provided custom adapter loading in `src/validate_*.py` and `src/inference_production.py`.

### Issue: Training is extremely slow on CPU
**Fix:** GPU is mandatory for production training. CPU is only suitable for data generation and preprocessing (~2min/step vs ~2s/step on GPU).

### Issue: False positive rate still high after v2 training
**Remedy:** Increase `--negative_ratio` to 0.30 and retrain for 3 epochs instead of 2.

### Issue: Model hallucinates during inference
**Fix:** Lower `temperature` in config (e.g., 0.3 instead of 0.7) and enable the production confidence filter (`inference_production.py`).

---

## Production Checklist

- [ ] Balanced dataset generated (all 3 models)
- [ ] Preprocessing complete (all 3 models)
- [ ] GPU training complete (all 3 models, 2 epochs)
- [ ] Real-world validation passed (all targets green)
- [ ] ONNX export + INT8 quantization tested
- [ ] Production inference script tested with edge cases
- [ ] Confidence threshold calibrated per model
- [ ] Safety flags tested and documented
- [ ] Model cards updated and pushed to HuggingFace
- [ ] Datasets uploaded to Kaggle
- [ ] GitHub repo updated with all new configs and scripts
- [ ] Clinical / security advisory disclaimer included in API responses
- [ ] Monitoring and logging configured for production deployment
