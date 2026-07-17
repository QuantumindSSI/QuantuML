# Clinical Validation Report

**Date:** July 2026
**Model:** Drug Interaction Predictor (09)
**Validator:** `src/validate_clinical.py`
**Ground Truth:** `data/clinical_ground_truth.json`

---

## Executive Summary

The Drug Interaction Predictor was tested against **40 clinically validated drug pairs**: 30 known interactions sourced from FDA labels, ACC/AHA guidelines, CHEST guidelines, and major RCTs; plus 10 drug pairs with **no clinically significant interaction** per DrugBank.

### Key Findings

| Metric | POC Model (55 steps) | Target | Status |
|--------|---------------------|--------|--------|
| **Sensitivity (Recall)** | **100.0%** | > 95% | Pass |
| **Specificity** | **0.0%** | > 80% | **FAIL** |
| **Severity Accuracy** | **26.7%** | > 70% | **FAIL** |
| **Mechanism Presence** | **20.0%** | > 60% | **FAIL** |
| **Outcome Presence** | **36.7%** | > 60% | **FAIL** |
| **Recommendation Presence** | **30.0%** | > 60% | **FAIL** |
| **Overall Clinical Score** | **53.8%** | > 75% | **FAIL** |

> **Verdict:** The model detects interactions reliably but **hallucinates interactions for every safe drug pair** (100% false positive rate). It is **not ready for clinical deployment** without remediation.

---

## Root Cause Analysis

### 1. Training Data Bias

The original training dataset (`generate_drug_interaction_data.py`, default `num_scenarios=102000`) contained **zero negative examples**. Every single training sample described a drug interaction. Consequently, the model learned:

> *"Given any two drugs, always report an interaction."*

This is a classic **class imbalance** problem. The model was never taught to say *"no significant interaction found."*

### 2. Limited Training Steps

The POC model was trained for only **55 steps** (~1 epoch on a tiny subset). A full training run on the complete 102,000 examples with proper hyperparameter tuning is expected to significantly improve severity and mechanism accuracy.

### 3. Severity Classification Weakness

Severity was correctly predicted for only **8 of 30** known interactions. The model often defaulted to generic severity labels or omitted severity entirely. This suggests the severity signal was not strongly learned in 55 steps.

---

## Detailed Results

### Positive Cases (Known Interactions)

| Drug Pair | Expected Severity | Predicted Severity | Score | Status |
|-----------|-------------------|--------------------|-------|--------|
| Warfarin + Aspirin | Critical | High | 6/8 (75%) | Pass |
| Simvastatin + Clarithromycin | High | Low | 5/8 (62%) | Pass |
| Warfarin + Metronidazole | High | High | 7/8 (88%) | Pass |
| Omeprazole + Clopidogrel | High | Unknown | 4/8 (50%) | Fail |
| Morphine + Lorazepam | Critical | Unknown | 6/8 (75%) | Pass |
| Lisinopril + Losartan | High | Unknown | 4/8 (50%) | Fail |
| Metformin + Furosemide | Medium | High | 3/8 (38%) | Fail |
| Carbamazepine + Oral Contraceptives | High | Unknown | 2/8 (25%) | Fail |
| Amoxicillin + Warfarin | High | High | 6/8 (75%) | Pass |
| Fluconazole + Warfarin | Critical | Unknown | 6/8 (75%) | Pass |
| Amlodipine + Simvastatin | Medium | High | 3/8 (38%) | Fail |
| Verapamil + Metoprolol | High | High | 7/8 (88%) | Pass |
| Sertraline + Tramadol | Critical | High | 4/8 (50%) | Fail |
| Enoxaparin + Aspirin | High | Unknown | 3/8 (38%) | Fail |
| Ritonavir + Simvastatin | Critical | Low | 3/8 (38%) | Fail |
| Digoxin + Amiodarone | High | High | 5/8 (62%) | Pass |
| Lithium + Furosemide | High | High | 6/8 (75%) | Pass |
| Phenytoin + Warfarin | High | Unknown | 3/8 (38%) | Fail |
| Levothyroxine + Calcium Carbonate | Medium | Low | 3/8 (38%) | Fail |
| Ibuprofen + Lisinopril | High | Unknown | 3/8 (38%) | Fail |
| Allopurinol + Azathioprine | Critical | Critical | 5/8 (62%) | Pass |
| Methotrexate + TMP-SMX | Critical | Low | 3/8 (38%) | Fail |
| Prednisone + Ibuprofen | High | Unknown | 2/8 (25%) | Fail |
| Sildenafil + Nitroglycerin | Critical | Unknown | 5/8 (62%) | Pass |
| Fluoxetine + Tramadol | Critical | High | 5/8 (62%) | Pass |
| Cyclosporine + Fluconazole | Critical | Unknown | 3/8 (38%) | Fail |
| Metformin + Iodinated Contrast | High | Low | 3/8 (38%) | Fail |
| Rifampin + Warfarin | High | High | 6/8 (75%) | Pass |
| Diltiazem + Atorvastatin | High | High | 5/8 (62%) | Pass |
| Ketoconazole + Warfarin | Critical | Low | 3/8 (38%) | Fail |

### Negative Cases (No Known Interaction)

**All 10 negative cases produced false positives** (100% false positive rate):

| Drug Pair | Result |
|-----------|--------|
| Paracetamol + Amlodipine | **False Positive** |
| Metformin + Amlodipine | **False Positive** |
| Loratadine + Omeprazole | **False Positive** |
| Fexofenadine + Metformin | **False Positive** |
| Atorvastatin + Levothyroxine | **False Positive** |
| Cetirizine + Aspirin | **False Positive** |
| Pantoprazole + Metoprolol | **False Positive** |
| Glipizide + Atenolol | **False Positive** |
| Furosemide + Albuterol | **False Positive** |
| Ranitidine + Lisinopril | **False Positive** |

---

## Remediation Plan

### 1. Add Negative Examples to Training Data

**Status:** Implemented in `data/generate_drug_interaction_data.py`

A new `--negative_ratio` parameter (default `0.20`) generates drug pairs with **no clinically significant interaction**. Each negative example teaches the model to output:

> *"No clinically significant interaction... No dose adjustment required."*

**Recommended generation:**
```bash
python data/generate_drug_interaction_data.py \
    --num_scenarios 125000 \
    --negative_ratio 0.20 \
    --output_dir ./data/raw_ddi_v2
```

This produces:
- 100,000 positive interaction examples
- 25,000 negative (no-interaction) examples

### 2. Retrain with Full Dataset

The POC model used only **55 training steps**. A full training run should use:
- `max_steps: ~1500` (or full epoch)
- `per_device_train_batch_size: 2`
- `gradient_accumulation_steps: 4`
- `learning_rate: 2.0e-4`
- `num_train_epochs: 1-2`

### 3. Calibrate Severity Classification

Add a supervised classification head or prompt-based severity calibration:
- Fine-tune with explicit severity labels as classification targets
- Use weighted loss for underrepresented severity classes (Critical, High)

### 4. Add Mechanism Extraction Head

Train a secondary task to extract mechanism keywords from the output and compare against ground truth. This can be done with:
- Multi-task learning (generation + keyword extraction)
- Post-hoc NER-style tagging

### 5. Re-validate Against Clinical Ground Truth

After retraining, rerun `src/validate_clinical.py` and target:

| Metric | Minimum Target |
|--------|---------------|
| Sensitivity | > 95% |
| Specificity | > 85% |
| Severity Accuracy | > 70% |
| Mechanism Presence | > 60% |
| Outcome Presence | > 60% |
| Recommendation Presence | > 60% |
| Overall Clinical Score | > 75% |

---

## Validation Methodology

### Ground Truth Sources

| Source | Examples | Reliability |
|--------|----------|-------------|
| FDA Drug Labels / Boxed Warnings | 8 | Gold standard |
| ACC/AHA / ESC / CHEST Guidelines | 7 | Gold standard |
| ASH / ACR / ADA / AAN Guidelines | 6 | Gold standard |
| Major RCTs (ONTARGET, COGENT, WOEST) | 3 | Gold standard |
| DrugBank (negative cases) | 10 | High |

### Scoring Rubric (per positive case)

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Interaction Detected | 2 | Model claims an interaction exists |
| Severity Correct | 2 | Matches FDA/guideline severity |
| Mechanism Present | 1 | Mentions CYP, pharmacodynamic, etc. |
| Outcome Present | 1 | Lists predicted clinical outcomes |
| Recommendation Present | 1 | Provides clinical recommendations |
| Drug Pair Mentioned | 1 | Names both drugs from the prompt |
| **Max Score** | **8** | |

A case scores **PASS** if clinical accuracy >= 60% (5/8 points).

### False Positive Detection

For negative cases, the model output is scanned for:
- Positive signals: "interaction", "contraindicated", "avoid", "caution", "severe", "major"
- Negative signals: "no interaction", "no known interaction", "not expected"

If positive signals > negative signals, the case is marked as a **false positive**.

---

## Clinical Safety Warning

> **This model is NOT approved for clinical use.**
>
> - It has a **100% false positive rate** on safe drug pairs.
> - It misclassifies severity for **73%** of known interactions.
> - It omits mechanisms and recommendations for most cases.
>
> **Always verify model outputs against FDA labels, clinical guidelines, and licensed pharmacists before making patient care decisions.**

---

## Citation

If you use this validation framework, please cite:

```bibtex
@misc{quantuml_clinical_val_2026,
  title={QuantuML Clinical Validation: Drug Interaction Predictor},
  author={QuantumIndSSI Ltd},
  year={2026},
  howpublished={\url{https://github.com/QuantumindSSI/QuantuML/blob/main/docs/CLINICAL_VALIDATION.md}}
}
```
