# Quantum-Resistant Cryptographic Protocol Analyzer - Training Report

## Summary

Fine-tuned a Qwen2-0.5B-Instruct base model using LoRA on 10,500 synthetic cryptographic protocol vulnerability analyses. Training was performed on CPU as a proof-of-concept. Full-scale training on GPU is recommended for production deployment.

## Dataset

- **Total Scenarios**: 10,500
- **Train / Eval / Test**: 8,925 / 1,050 / 525
- **Quick POC Train / Eval / Test**: 434 / 52 / 26 (subsampled for CPU training)
- **Protocols Covered**: TLS/SSL, SSH, IPsec, WireGuard, OpenVPN, WPA, S/MIME, OpenPGP, DNSSEC, Kerberos, Bitcoin, Ethereum, gRPC, MQTT, Bluetooth, Signal, Noise
- **Vulnerability Types**: 10 categories including Shor-vulnerable, Grover-amplified, HNDL, downgrade attacks, weak randomness, transition gaps
- **Attack Vectors**: 10 detailed vectors including Shor factoring, Shor DLP, Grover search, quantum collision finding, HNDL, quantum MITM

## Model Configuration

- **Base Model**: Qwen/Qwen2-0.5B-Instruct (494M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Trainable Parameters**: 8,798,208 (1.75% of total)
- **Sequence Length**: 1024
- **Batch Size**: 1
- **Gradient Accumulation**: 8
- **Learning Rate**: 2e-4
- **Epochs**: 1 (POC)

## Training Results (POC)

- **Runtime**: 1h 46m on CPU (Intel Xeon or equivalent, 62GB RAM)
- **Train Loss**: 0.6526
- **Eval Loss**: 0.0945
- **Test Loss**: 0.0936
- **Final Perplexity**: 1.45
- **Task Quality Score**: 83.3% (3 samples)

### Task Quality Breakdown

| Check | Pass Rate |
|-------|-----------|
| Mentions Vulnerability Type | 100% |
| Mentions Attack Vector | 100% |
| Mentions Severity | 33% |
| Has Mitigations | 100% |
| Mentions NIST | 100% |
| Mentions Algorithm | 100% |
| Mentions Protocol | 33% |
| Has Headers / Structure | 100% |

## Inference Examples

### Example 1: TLS 1.2 with RSA

**Input:**
```python
context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
context.set_ciphers('RSA-AES256-GCM-SHA384')
```

**Output Excerpt:**
- Recommends hybrid post-quantum key exchange (X25519Kyber768)
- Mentions crypto-agility and NIST SP 800-208
- Suggests monitoring NIST PQC standardization updates

## Deployment Artifacts

- **LoRA Adapter**: `outputs/models_crypto_quick/checkpoint-final/`
- **Merged Model**: `outputs/models_crypto_quick/merged-final/`
- **Victron Package**: `deployment/victron/crypto_merged/victron_package/`
- **API Server**: `deployment/api/api_server_crypto.py`
- **Evaluation Results**: `outputs/evaluations/eval_crypto_results.json`

## Limitations

- POC trained on 512 samples; full training on 10,500 samples recommended
- CPU inference latency is ~10-24s for 100-200 tokens; GPU or quantized inference required for edge targets
- Some outputs contain minor hallucinations (e.g., invented algorithm names) typical of very small models with limited training
- Severity mention and protocol name mention rates need improvement with more training data

## Next Steps

1. **Full GPU Training**: Run 3 epochs on 10,500 samples with A100/L4 GPU (~1-2 hours)
2. **QLoRA**: Reduce memory to ~6GB with 4-bit quantization for larger batch sizes
3. **DPO Fine-tuning**: Use preference pairs to improve output structure and reduce hallucinations
4. **Quantization**: Export to ONNX Runtime with INT8 for <2GB edge deployment
5. **Evaluation Scale**: Run evaluation on full 525-sample test set with automated metric computation
6. **Hugging Face Upload**: Push final merged model to `quantumindssi/01_quantum_resistant_crypto_analyzer`
