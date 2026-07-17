# Real-World Validation Report

**Date:** July 2026
**Models Tested:** All three QuantuML task-specific models against real-world ground truth

---

## Executive Summary

All three models in the QuantuML suite were tested against **real-world ground truth datasets** built from FDA labels, CVE databases, NIST standards, and vendor documentation. The results reveal consistent weaknesses across the suite:

| Model | Synthetic Accuracy | Real-World Accuracy | Specificity on Negatives | Clinically Ready |
|-------|-------------------|---------------------|--------------------------|------------------|
| **DDI Predictor** | 71.4% | 53.8% | **0%** (100% FP) | **NO** |
| **Crypto Analyzer** | 83.3% | 51.7% | **0%** (100% FP) | **NO** |
| **PQC Migration Advisor** | 87.5% | 22.5% | N/A (generation task) | **NO** |

**Universal finding:** Models trained exclusively on synthetic positive examples hallucinate findings for every negative (safe) case. The synthetic-to-real gap is significant across all tasks.

---

## 09: Drug Interaction Predictor

### Ground Truth
- **30 positive cases** from FDA Boxed Warnings, ACC/AHA/CHEST/ESC guidelines
- **10 negative cases** from DrugBank (no clinically significant interaction)

### Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Sensitivity | 100% | > 95% | Pass |
| Specificity | **0%** | > 85% | **FAIL** |
| Severity Accuracy | 26.7% | > 70% | **FAIL** |
| Mechanism Presence | 20.0% | > 60% | **FAIL** |
| Clinical Score | 53.8% | > 75% | **FAIL** |

### Root Cause
Original training data (102,000 examples) contained **zero negative examples**. The model learned: "Every drug pair interacts."

### Remediation Applied
- `data/generate_drug_interaction_data.py` now generates `--negative_ratio 0.20`
- New dataset (25,000 examples, 5,000 negative) uploaded to Kaggle

---

## 01: Quantum-Resistant Crypto Analyzer

### Ground Truth
- **12 vulnerable cases** from CVE database (Logjam, KRACK, BEAST, POODLE, etc.)
- **3 non-vulnerable cases** (TLS 1.3 + Kyber, WireGuard, Signal Protocol)

### Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Vulnerability Detection | 83.3% | > 85% | Near Pass |
| Attack Vector Recognition | 83.3% | > 80% | Pass |
| NIST Reference Present | **0%** | > 60% | **FAIL** |
| Severity Mentioned | **8.3%** | > 60% | **FAIL** |
| Mitigations Present | 83.3% | > 80% | Pass |
| Specificity on Safe Code | **0%** | > 85% | **FAIL** |
| Average Score | 51.7% | > 75% | **FAIL** |

### Root Cause
Like the DDI model, training data contained only vulnerable examples. Safe configurations (TLS 1.3, WireGuard) were never shown to the model.

### Required Remediation
1. Add non-vulnerable code examples to training data (~20%)
2. Add explicit NIST/FIPS citation requirements to prompt templates
3. Train severity as a classification head, not free text

---

## 02: Post-Quantum Key Migration Advisor

### Ground Truth
- **5 migration scenarios** based on NIST IR 8547, NSA CNSA 2.0, PCI DSS, HIPAA

### Results

| Criterion | Pass Rate | Target |
|-----------|-----------|--------|
| Has Migration Phases | **0%** | > 80% |
| Mentions PQC Algorithms | **0%** | > 80% |
| Mentions Tools | **0%** | > 60% |
| Mentions Compliance | 20% | > 80% |
| Has Timeline | 20% | > 60% |
| Has Rollback Plan | **0%** | > 60% |
| Has Testing | 100% | > 80% |
| Has Budget | 40% | > 60% |
| **Average Score** | **22.5%** | > 75% |

### Root Cause
The model was trained on only **5,500 synthetic scenarios** with a **very quick POC run** (not full training). The model outputs generic text but fails structured quality checks because:
1. Training data emphasized narrative over structure
2. No explicit loss function for section coverage
3. Very limited training steps

### Required Remediation
1. Increase training data to 20,000+ examples
2. Add structured output templates as training targets
3. Use constrained decoding or JSON-mode output
4. Add a coverage scoring reward function (RLHF)

---

## Cross-Model Findings

### Finding 1: The "False Positive Epidemic"

Both DDI and Crypto models have **100% false positive rates** on negative cases. This is a direct consequence of:
- Training data containing only positive examples
- No rejection mechanism ("I don't know" / "No interaction found")
- No confidence calibration

**Fix:** Every model needs ~20% negative examples in training data.

### Finding 2: Structural Quality Collapse

Synthetic evaluation (perplexity, task quality heuristics) suggested 71-87% quality. Real-world structured evaluation showed **22-54% quality**. The gap comes from:
- Synthetic evaluators checking for keyword presence, not correctness
- Real-world evaluators checking against specific standards/FDA labels/CVEs
- Severity/mechanism details are hallucinated or generic

### Finding 3: Citation Amnesia

None of the models reliably cite real standards (NIST, FDA, ACC/AHA) even though these appear in training data. Citation keywords are treated as decorative text, not verified facts.

---

## Kaggle Dataset Updates

| Dataset | Status | Link |
|---------|--------|------|
| DDI v1 (positive only) | Archived | [Link](https://www.kaggle.com/datasets/quantumind/synthetic-drug-drug-interaction-dataset) |
| **DDI v2 (with negatives)** | **Live** | [Link](https://www.kaggle.com/datasets/quantumind/drug-interaction-dataset-with-negatives) |
| Crypto Vulnerability | Live | [Link](https://www.kaggle.com/datasets/quantumind/quantum-vulnerability-crypto-protocol-dataset) |
| PQC Hybrid | Live | [Link](https://www.kaggle.com/datasets/quantumind/post-quantum-key-migration-hybrid-dataset) |

**Crypto and PQC datasets still lack negative examples.** These should be updated next.

---

## Validation Scripts

| Script | Purpose | Ground Truth Source |
|--------|---------|---------------------|
| `src/validate_clinical.py` | DDI clinical accuracy | FDA, guidelines, DrugBank |
| `src/validate_crypto_realworld.py` | Crypto vulnerability detection | CVE database, NIST standards |
| `src/validate_pqc_realworld.py` | Migration plan quality | NIST IR 8547, NSA CNSA 2.0 |

---

## Roadmap to Clinical Readiness

| Milestone | DDI | Crypto | PQC |
|-----------|-----|--------|-----|
| Add negatives to training data | Done | Pending | N/A |
| Retrain with full epochs | Pending | Pending | Pending |
| Add structured output loss | N/A | N/A | Needed |
| Add citation verification layer | Needed | Needed | Needed |
| Real-world beta testing | Needed | Needed | Needed |
| Regulatory review (FDA/CE) | Long-term | N/A | N/A |

---

## Citation

```bibtex
@misc{quantuml_rw_val_2026,
  title={QuantuML Real-World Validation Report},
  author={QuantumIndSSI Ltd},
  year={2026},
  howpublished={\url{https://github.com/QuantumindSSI/QuantuML/blob/main/docs/REAL_WORLD_VALIDATION.md}}
}
```
