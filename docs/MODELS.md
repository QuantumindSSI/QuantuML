# Model Technical Documentation

This document provides full technical details for every model in the QuantuML suite.

---

## Table of Contents

1. [00: QuantuML PQC POC](#00-quantuml-pqc-poc)
2. [01: Quantum-Resistant Cryptographic Protocol Analyzer](#01-quantum-resistant-cryptographic-protocol-analyzer)
3. [02: Post-Quantum Key Migration Advisor](#02-post-quantum-key-migration-advisor)
4. [09: Drug Interaction Predictor](#09-drug-interaction-predictor)

---

## 00: QuantuML PQC POC

**HuggingFace**: [quantumindssi/quantuml-pqc-0.5b-poc](https://huggingface.co/quantumindssi/quantuml-pqc-0.5b-poc)

### Overview
Base model proof-of-concept used to validate the entire training, evaluation, and deployment pipeline before scaling to task-specific models.

### Configuration (`configs/config_pqc_migration.yaml`)

| Parameter | Value |
|-----------|-------|
| Base Model | `Qwen/Qwen2-0.5B-Instruct` |
| Fine-tuning | LoRA (r=16, alpha=32) |
| Torch dtype | float32 |
| Device map | auto |
| Batch size | 4 |
| Gradient accumulation | 1 |
| Learning rate | 2.0e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Max grad norm | 0.3 |
| Epochs | 1 |
| Max sequence length | 1024 |
| Seed | 42 |

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| r | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.05 |
| bias | none |
| task_type | CAUSAL_LM |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| use_rslora | false |

### Training Command

```bash
python src/train_lora.py \
  --config ./configs/config_pqc_migration.yaml \
  --output_dir ./outputs/models_poc
```

### Evaluation Command

```bash
python src/evaluate_pqc.py \
  --config ./configs/config_pqc_migration.yaml \
  --model_path ./outputs/models_poc/checkpoint-final \
  --output_file ./outputs/evaluations/poc_eval_results.json
```

---

## 01: Quantum-Resistant Cryptographic Protocol Analyzer

**HuggingFace**: [quantumindssi/01_quantum_resistant_crypto_analyzer](https://huggingface.co/quantumindssi/01_quantum_resistant_crypto_analyzer)

### Overview
Fine-tuned SLM that analyzes cryptographic protocol implementations (TLS, SSH, VPN, etc.) and identifies quantum-vulnerable patterns, attack vectors, and NIST-aligned mitigations.

### Configuration (`configs/config_crypto_analyzer.yaml`)

| Parameter | Value |
|-----------|-------|
| Base Model | `./base_model` (Qwen2-0.5B-Instruct) |
| Fine-tuning | LoRA (r=16, alpha=32) |
| Torch dtype | float32 |
| Device map | auto |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | 2.0e-4 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Weight decay | 0.01 |
| Max grad norm | 0.3 |
| Epochs | 1 |
| Max sequence length | 1024 |
| Seed | 42 |

### Quick Training (POC)

```bash
# Generate data
python data/generate_crypto_vulnerability_data.py \
  --num_scenarios 10500 --output_dir ./data/raw_vuln

# Preprocess
python data/preprocess.py \
  --input ./data/raw_vuln/instructions.jsonl \
  --output_dir ./data/processed_vuln_quick \
  --model_name ./base_model --max_length 1024

# Train
python src/train_lora.py \
  --config ./configs/config_crypto_analyzer_quick.yaml \
  --output_dir ./outputs/models_crypto_quick

# Evaluate
python src/evaluate_crypto.py \
  --config ./configs/config_crypto_analyzer_quick.yaml \
  --model_path ./outputs/models_crypto_quick/checkpoint-final \
  --num_samples 20
```

### Training Data

See [DATASETS.md](DATASETS.md) for full specifications.

| Statistic | Value |
|-----------|-------|
| Total scenarios | 10,500 |
| Train | 8,925 |
| Eval | 1,050 |
| Test | 525 |
| Protocols | 48 |
| Vulnerability types | 10 |
| Attack vectors | 10 |
| Algorithm families | 15 |
| Code templates | 24 (Python, C/OpenSSL, Go, Java, Node.js, Rust) |

### Evaluation Metrics

| Metric | POC Score | Full Score (Target) |
|--------|-----------|---------------------|
| Perplexity | 1.45 | < 10.0 |
| Task Quality | 83.3% | > 80% |
| Vulnerability Detection | 7/8 checks | > 85% |
| Attack Vector Recognition | Present | > 80% |
| Eval Loss | 0.094 | < 0.5 |
| Mean Latency (CPU) | 23.5s | < 1000ms (GPU) |
| Memory (CPU) | 17.95 GB | < 4GB (INT8) |

### Task Quality Heuristic

The evaluator checks 8 criteria per generated response:

1. `mentions_vulnerability_type` - Does output name the vulnerability?
2. `mentions_attack_vector` - Does it identify the quantum attack vector?
3. `mentions_severity` - Is severity mentioned?
4. `has_mitigations` - Are there numbered mitigations?
5. `mentions_nist` - Are NIST/FIPS/CNSA references included?
6. `mentions_algorithm` - Is the specific algorithm discussed?
7. `mentions_protocol` - Is the protocol name mentioned?
8. `has_headers` - Is the response well-structured with headers?

---

## 02: Post-Quantum Key Migration Advisor

**HuggingFace**: [quantumindssi/02_post_quantum_key_migration_advisor](https://huggingface.co/quantumindssi/02_post_quantum_key_migration_advisor)

### Overview
Enterprise-grade model providing step-by-step migration guidance for organizations transitioning from classical cryptography to quantum-safe systems.

### Configuration (`configs/config_pqc_migration.yaml`)

Same base configuration as 00 POC but trained on the hybrid dataset.

### Data Pipeline

```bash
# Step 1: Generate synthetic data
python data/generate_pqc_migration_data.py \
  --num_scenarios 5500 --output_dir ./data/raw

# Step 2: Generate real curated data (manual or via generate_real_data.py)
python data/real_sources/generate_real_data.py

# Step 3: Compile hybrid dataset
python data/compile_hybrid_dataset.py \
  --real_data ./data/real_sources/curated_real_instructions.jsonl \
  --synthetic_data ./data/raw/instructions.jsonl \
  --output_dir ./data/hybrid \
  --target_total 5500

# Step 4: Preprocess
python data/preprocess.py \
  --input ./data/hybrid/train.jsonl \
  --output_dir ./data/processed_poc \
  --model_name ./base_model --max_length 1024
```

### Evaluation Metrics

| Metric | Score | Target |
|--------|-------|--------|
| Perplexity | 1.59 | < 10.0 |
| Task Quality | 87.5% | > 80% |
| Train Loss | 0.300 | < 0.5 |
| Eval Loss | 0.073 | < 0.5 |

### Task Quality Heuristic

1. `has_phases` - Discovery, Assessment, Planning, Implementation, Validation, Deployment
2. `has_algorithms` - ML-KEM, ML-DSA, DILITHIUM, KYBER, SLH-DSA
3. `has_tools` - HSM, OpenSSL, CI/CD, scanner
4. `has_timeline` - Month, week, timeline, roadmap
5. `has_compliance` - FIPS, NIST, GDPR, HIPAA, PCI DSS
6. `has_rollback` - Rollback or fallback plan
7. `has_testing` - Testing strategy mentioned
8. `has_budget` - Dollar amount or budget mentioned

---

## 09: Drug Interaction Predictor

**HuggingFace**: [QuantumindSSI/09_drug_interaction_predictor](https://huggingface.co/QuantumindSSI/09_drug_interaction_predictor)

### Overview
Pharmacovigilance SLM that predicts drug-drug interactions (DDI) from medication lists, including severity, mechanism, predicted outcomes, clinical recommendations, and evidence.

### Configuration (`configs/config_ddi.yaml`)

| Parameter | Value |
|-----------|-------|
| Base Model | `./base_model` (Qwen2-0.5B-Instruct) |
| Fine-tuning | LoRA (r=16, alpha=32) |
| Torch dtype | float32 |
| Device map | auto |
| Batch size | 1 |
| Gradient accumulation | 8 |
| Learning rate | 2.0e-4 |
| LR scheduler | cosine |
| Max steps | 55 (quick POC) |
| Max sequence length | 1024 |
| Seed | 42 |

### Training Command

```bash
# Generate data
python data/generate_drug_interaction_data.py \
  --num_scenarios 102000 --output_dir ./data/raw_ddi

# Preprocess
python data/preprocess.py \
  --input ./data/raw_ddi/instructions.jsonl \
  --output_dir ./data/processed_ddi_quick \
  --model_name ./base_model --max_length 1024

# Train
python src/train_lora.py \
  --config ./configs/config_ddi.yaml \
  --output_dir ./outputs/models_ddi_quick

# Evaluate
python src/evaluate_ddi.py \
  --config ./configs/config_ddi.yaml \
  --model_path ./outputs/models_ddi_quick/checkpoint-final \
  --num_samples 20
```

### Training Data

| Statistic | Value |
|-----------|-------|
| Total scenarios | 102,000 |
| Train | 86,700 |
| Eval | 10,200 |
| Test | 5,100 |
| Drug classes | 30 |
| Interaction templates | 17 |
| Fallback interactions | 9 |
| Patient contexts | 12 |
| Severity levels | Critical, High, Medium, Low |

### Evaluation Metrics

| Metric | POC Score | Target |
|--------|-----------|--------|
| Perplexity | 3.38 | < 10.0 |
| Task Quality | 71.4% | > 70% |
| Eval Loss | 0.280 | < 0.5 |
| Test Loss | 0.308 | < 0.5 |
| Mean Latency (CPU) | 18.5s | < 1000ms (GPU) |
| Memory (CPU) | 18.2 GB | < 4GB (INT8) |

### Task Quality Heuristic

1. `mentions_severity` - Critical, High, Medium, Low
2. `mentions_mechanism` - CYP, pharmacodynamic, pharmacokinetic
3. `has_recommendations` - Numbered recommendations
4. `mentions_evidence` - FDA, guideline, trial, label
5. `mentions_drug_pair` - Specific drug names from input
6. `has_headers` - Structured with headers
7. `mentions_outcomes` - Outcome, risk, toxicity, bleeding

---

## Training Methods Deep Dive

### LoRA (Low-Rank Adaptation)

The default training method. Adds low-rank matrices to frozen pretrained weights.

**Trainable parameters**: ~0.1-0.5% of total model params
**VRAM requirement**: ~16GB for 0.5B model at FP32
**Checkpoint size**: ~35MB (adapter only)

### QLoRA (4-bit Quantized LoRA)

Uses BitsAndBytes 4-bit quantization for the base model, training only LoRA adapters.

**Trainable parameters**: Same as LoRA
**VRAM requirement**: ~8GB for 0.5B model
**Trade-offs**: Slightly slower training, minimal quality loss

### DPO (Direct Preference Optimization)

Trains on preference pairs (chosen vs rejected outputs) without explicit reward model.

**Requires**: Preference dataset with `prompt`, `chosen`, `rejected` columns
**Beta**: 0.1 (controls divergence from reference)
**Learning rate**: 1.0e-6

### PPO (Proximal Policy Optimization)

RLHF-style training with a separate reward model.

**Requires**: Policy model, reference model, reward model, prompt dataset
**Learning rate**: 1.41e-5
**Batch size**: 128 (mini-batch: 4)

### ORPO (Odds Ratio Preference Optimization)

Single-stage alignment that combines SFT and preference optimization.

**Requires**: Dataset with `prompt`, `chosen`, `rejected`
**Beta**: 0.1
**Learning rate**: 8.0e-6

---

## Model Comparison

| Attribute | 00 POC | 01 Crypto | 02 Migration | 09 DDI |
|-----------|--------|-----------|--------------|--------|
| HF Repo | quantumindssi/quantuml-pqc-0.5b-poc | 01_quantum_resistant_crypto_analyzer | 02_post_quantum_key_migration_advisor | 09_drug_interaction_predictor |
| Dataset Size | 5,500 | 10,500 | 5,317 | 102,000 |
| Perplexity | 1.59 | 1.45 | 1.59 | 3.38 |
| Task Quality | 87.5% | 83.3% | 87.5% | 71.4% |
| Eval Loss | 0.073 | 0.094 | 0.073 | 0.280 |
| Training Steps | Full epoch | Full epoch | Full epoch | 55 (quick) |
| Batch Size | 4 | 1 | 4 | 1 |
| Grad Accum | 1 | 8 | 1 | 8 |
