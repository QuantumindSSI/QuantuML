# System Architecture

This document describes the end-to-end architecture of the QuantuML platform.

---

## High-Level Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Data Generation | --> |   Training Pipeline| --> |  Evaluation Suite |
|   (Synthetic +    |     |  (LoRA/QLoRA/     |     |  (Perplexity +   |
|    Real Curated)  |     |   DPO/PPO/ORPO)   |     |   Task Quality)  |
+-------------------+     +-------------------+     +-------------------+
         |                         |                         |
         v                         v                         v
+-------------------+     +-------------------+     +-------------------+
|   Kaggle Datasets |     |  HuggingFace Hub  |     |  Benchmark Reports|
|   (Public Access) |     |  (Model Registry) |     |  (JSON/MD Output) |
+-------------------+     +-------------------+     +-------------------+
                                 |
                                 v
                        +-------------------+
                        |  Deployment Layer |
                        |  (API + Edge)     |
                        +-------------------+
                                 |
                    +------------+------------+
                    |                         |
                    v                         v
          +-------------------+     +-------------------+
          |   FastAPI Server  |     |  Victron Edge Box |
          |   (Cloud/On-prem) |     |  (CPU/INT8/ONNX)  |
          +-------------------+     +-------------------+
```

---

## Data Flow

### 1. Data Generation Layer

```
+-------------------------------------------------------------+
|                    DATA GENERATION                           |
+-------------------------------------------------------------+
|                                                             |
|  +----------------+    +----------------+    +-------------+|
|  | Drug Generator |    | Crypto Vuln    |    | PQC Migration||
|  | 102,000 rows   |    | 10,500 rows    |    | 5,500 rows   ||
|  +-------+--------+    +-------+--------+    +------+-------+|
|          |                     |                    |        |
|          v                     v                    v        |
|  +----------------+    +----------------+    +-------------+|
|  |  raw_ddi/      |    |  raw_vuln/     |    |  raw/        ||
|  |  instructions  |    |  instructions  |    |  instructions||
|  +----------------+    +----------------+    +-------------+|
|                                                             |
+-------------------------------------------------------------+
```

**Generators:**
- `data/generate_drug_interaction_data.py` - Pharmacological scenario generator
- `data/generate_crypto_vulnerability_data.py` - Cryptographic audit scenario generator
- `data/generate_pqc_migration_data.py` - Enterprise migration plan generator
- `data/compile_hybrid_dataset.py` - Merges synthetic + real curated data

### 2. Preprocessing Layer

```
+----------------+     +----------------+     +----------------+
|  Raw JSONL     | --> |  Chat Template | --> |  Tokenized     |
|  (instruction, |     |  Formatting    |     |  HF Datasets   |
|   input,       |     |  (Alpaca/Chat) |     |  (train/eval/  |
|   output)      |     |                |     |   test)        |
+----------------+     +----------------+     +----------------+
```

**Preprocessing:**
- `data/preprocess.py` - Loads tokenizer, applies chat template, tokenizes, splits
- Supports both native chat templates (Phi-3, Qwen2) and Alpaca-style fallback
- Produces `DatasetDict` with train/eval/test splits

### 3. Training Layer

```
+----------------+     +----------------+     +----------------+
|  Base Model    | --> |  LoRA Adapters | --> |  Fine-tuned    |
|  (Qwen2-0.5B)  |     |  (r=16,        |     |  Adapter       |
|                |     |   alpha=32)    |     |  (~35MB)       |
+----------------+     +----------------+     +----------------+
       |                                              |
       v                                              v
+----------------+                           +----------------+
|  QLoRA (4-bit) |                           |  Merged Model  |
|  for consumer  |                           |  (optional)    |
|  GPUs          |                           |  (~2GB)        |
+----------------+                           +----------------+
```

**Training Scripts:**
- `src/train_lora.py` - Standard LoRA fine-tuning
- `src/train_qlora.py` - 4-bit quantized LoRA
- `src/train_dpo.py` - Direct Preference Optimization
- `src/train_ppo.py` - Proximal Policy Optimization (RLHF)
- `src/train_orpo.py` - Odds Ratio Preference Optimization

### 4. Evaluation Layer

```
+----------------+     +----------------+     +----------------+
|  Model +       | --> |  Perplexity    | --> |  Task Quality  |
|  Tokenizer     |     |  Computation   |     |  Heuristics    |
+----------------+     +----------------+     +----------------+
       |                                              |
       v                                              v
+----------------+                           +----------------+
|  Latency       |                           |  Edge Compat   |
|  Benchmarks    |                           |  Check         |
+----------------+                           +----------------+
```

**Evaluation Scripts:**
- `src/evaluate_pqc.py` - Migration plan quality (8 criteria)
- `src/evaluate_crypto.py` - Vulnerability detection quality (8 criteria)
- `src/evaluate_ddi.py` - Drug interaction quality (7 criteria)

### 5. Deployment Layer

```
+-------------------+     +-------------------+     +-------------------+
|   Model Artifact  | --> |   Export/Package  | --> |   Runtime        |
|   (LoRA Adapter   |     |   (ONNX/INT8/     |     |   (PyTorch/     |
|    or Merged)     |     |    Victron)       |     |    ONNX/        |
|                   |     |                   |     |    FastAPI)    |
+-------------------+     +-------------------+     +-------------------+
```

**Deployment Scripts:**
- `src/export_onnx_int8.py` - ONNX export + INT8 quantization + benchmarking
- `deployment/api/api_server_*.py` - FastAPI REST servers
- `deployment/victron/deploy_victron_*.py` - Edge device packaging

---

## Component Details

### Base Model Adapter

All models share the same base: `Qwen/Qwen2-0.5B-Instruct`

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype=torch.float32,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
```

### LoRA Configuration

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
# Trainable params: ~1.6M / 494M = 0.33%
```

### API Server Architecture

```
Client Request --> FastAPI Router --> Model Loading --> Tokenization
                                              |
                                              v
                                    Response Generation
                                              |
                                              v
                                    Response Formatting --> JSON Output
```

All API servers expose:
- `POST /health` - Health check
- `POST /predict` or `POST /analyze` - Main inference endpoint
- `GET /info` - Model metadata

### Victron Edge Package Structure

```
victron_package/
├── model/
│   ├── adapter_model.safetensors  # LoRA weights
│   ├── adapter_config.json        # LoRA config
│   ├── tokenizer.json             # Tokenizer
│   └── tokenizer_config.json      # Tokenizer config
├── manifest.json                  # Deployment metadata
├── start.sh                       # Startup script
└── <service>.service             # systemd service file
```

---

## Security Considerations

1. **Model Safety**: All models are fine-tuned for narrow tasks and should not be used for general chat
2. **Data Privacy**: Edge deployment ensures no data leaves the device
3. **Synthetic Data**: Training data is synthetic; real patient data or production keys should never be used
4. **Verification**: Model outputs require human expert verification before operational use

---

## Scalability

| Scale | Hardware | Models | Throughput |
|-------|----------|--------|------------|
| Development | Single GPU | 1 | ~10 req/s |
| Production | Multi-GPU | 1-3 | ~100 req/s |
| Edge | Victron CPU | 1 | ~0.2 req/s |

For higher throughput, consider:
- vLLM or TGI for GPU serving
- ONNX Runtime with batching for CPU
- Model distillation to smaller architectures
