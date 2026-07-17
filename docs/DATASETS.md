# Dataset Documentation

All datasets are publicly available on [Kaggle](https://www.kaggle.com/quantumind).

---

## Dataset Registry

| # | Dataset | Kaggle Link | Size | Format | Notes |
|---|---------|-------------|------|--------|-------|
| 1 | Synthetic Drug-Drug Interaction | [Link](https://www.kaggle.com/datasets/quantumind/synthetic-drug-drug-interaction-dataset) | 102,000 | JSONL | v1: positive only |
| 1b | **Drug Interaction Dataset with Negatives** | [Link](https://www.kaggle.com/datasets/quantumind/drug-interaction-dataset-with-negatives) | 25,000 | JSONL | **v2: includes 5,000 negative (no-interaction) examples** |
| 2 | Quantum Vulnerability Crypto Protocol | [Link](https://www.kaggle.com/datasets/quantumind/quantum-vulnerability-crypto-protocol-dataset) | 10,500 | JSONL | Positive only |
| 3 | Post-Quantum Key Migration Hybrid | [Link](https://www.kaggle.com/datasets/quantumind/post-quantum-key-migration-hybrid-dataset) | 5,317 | JSONL | Real + synthetic |
| 4 | Post-Quantum Key Migration Synthetic | [Link](https://www.kaggle.com/datasets/quantumind/post-quantum-key-migration-synthetic-dataset) | 5,300 | JSONL | Synthetic only |
| 5 | PQC Real Curated Instructions | [Link](https://www.kaggle.com/datasets/quantumind/pqc-real-curated-instructions-dataset) | 18 | JSONL | Manually curated |

---

## 1. Synthetic Drug-Drug Interaction (DDI) Dataset

**Kaggle**: [synthetic-drug-drug-interaction-dataset](https://www.kaggle.com/datasets/quantumind/synthetic-drug-drug-interaction-dataset)

### Overview
102,000 synthetic drug-drug interaction examples with severity, mechanism, evidence, and clinical recommendations.

### Schema

```json
{
  "instruction": "Analyze the following medication list for potential drug-drug interactions...",
  "input": "## Medication List\n- Warfarin\n- Aspirin\n\n## Patient Context\n65-year-old male with atrial fibrillation",
  "output": "# Drug Interaction Analysis Report\n\n## Identified Interaction\n- **Drug Pair**: Warfarin + Aspirin\n...",
  "scenario_id": "DDI-000001",
  "drug_pair": ["Warfarin", "Aspirin"],
  "severity": "Critical"
}
```

### Splits

| Split | Count | Percentage |
|-------|-------|------------|
| Train | 86,700 | 85% |
| Eval | 10,200 | 10% |
| Test | 5,100 | 5% |

### Drug Classes (30)

Anticoagulants, Antiplatelets, Statins, ACE Inhibitors, ARBs, Beta-Blockers, Calcium Channel Blockers, Diuretics, PPIs, H2 Blockers, Antibiotics, Antifungals, Antivirals, NSAIDs, Opioids, Benzodiazepines, Antidepressants, Antipsychotics, Anticonvulsants, Oral Hypoglycemics, Insulins, Corticosteroids, Immunosuppressants, Antihistamines, Bronchodilators, Antigout, Thyroid, Bone Health, Erectile Dysfunction, Anticholinergics

### Interaction Types

- **Pharmacokinetic**: CYP3A4/2D6/2C9/2C19/1A2 inhibition/induction, P-glycoprotein, protein binding displacement, renal excretion competition, hepatic uptake inhibition, gastric pH alteration
- **Pharmacodynamic**: Additive sedation, QT prolongation, hypotension, hyperkalemia, hypoglycemia, nephrotoxicity, hepatotoxicity, serotonergic effects, anticholinergic burden, bleeding, bradycardia

### Severity Distribution

| Severity | Approximate % |
|----------|---------------|
| Critical | 20% |
| High | 30% |
| Medium | 30% |
| Low | 20% |

---

## 2. Quantum Vulnerability Cryptographic Protocol Dataset

**Kaggle**: [quantum-vulnerability-crypto-protocol-dataset](https://www.kaggle.com/datasets/quantumind/quantum-vulnerability-crypto-protocol-dataset)

### Overview
10,500 synthetic cryptographic protocol vulnerability analyses for training quantum threat detection models.

### Schema

```json
{
  "instruction": "Analyze the following TLS 1.2 implementation for quantum-vulnerable patterns...",
  "input": "## Protocol Description\nEnterprise TLS 1.2...\n## Implementation Code\n```python\n...\n```\n## Algorithm Used\nRSA-2048",
  "output": "# Quantum Vulnerability Analysis Report\n\n## Protocol\nTLS 1.2\n\n## Vulnerability Classification\n- **Type**: Shor's Algorithm Vulnerable\n- **Severity**: Critical (9.5/10.0)\n...",
  "scenario_id": "QRCA-00001",
  "protocol_name": "TLS 1.2",
  "vulnerability_category": "shor_vulnerable",
  "severity": "Critical"
}
```

### Splits

| Split | Count | Percentage |
|-------|-------|------------|
| Train | 8,925 | 85% |
| Eval | 1,050 | 10% |
| Test | 525 | 5% |

### Protocols (48)

TLS 1.0-1.3, SSL 3.0, DTLS, SSH variants, IPsec IKEv1/v2, WireGuard, OpenVPN, WPA2/WPA3, S/MIME, OpenPGP, DNSSEC, DNS over TLS/HTTPS, Kerberos 5, OAuth 2.0 JWT, gRPC, REST API HTTPS, Bitcoin, Ethereum, Hyperledger Fabric, PostgreSQL/MySQL/MongoDB TLS, MQTT, CoAP, Zigbee, Bluetooth variants, JSON Web Encryption, SAML, WS-Security, IPMI, LDAP, RADIUS, SIP, QUIC, Signal Protocol, Double Ratchet, Noise Protocol Framework

### Vulnerability Types

1. Shor's Algorithm Vulnerable
2. Grover's Algorithm Amplified
3. Harvest Now, Decrypt Later (HNDL)
4. Quantum-Enabled Downgrade Attack
5. Insufficient Key Size for Quantum Era
6. Deprecated Quantum-Vulnerable Protocol
7. Weak Randomness
8. Small Subgroup / Invalid Curve
9. Post-Quantum Transition Gap
10. Side-Channel Leakage

### Code Templates

24 templates across 6 languages: Python, C/OpenSSL, Go, Java, Node.js, Rust

---

## 3. Post-Quantum Key Migration Hybrid Dataset

**Kaggle**: [post-quantum-key-migration-hybrid-dataset](https://www.kaggle.com/datasets/quantumind/post-quantum-key-migration-hybrid-dataset)

### Overview
5,317 combined real + synthetic PQC migration scenarios. 18 real curated examples anchor quality; synthetic examples provide scale and diversity.

### Schema

```json
{
  "instruction": "You are a Post-Quantum Cryptography Migration Advisor...",
  "input": "",
  "output": "# Post-Quantum Migration Plan for TLS/SSL Infrastructure\n\n## Overview\n- **Industry**: Financial Services\n...",
  "scenario_id": "PQC-MIG-00001",
  "industry": "Financial Services",
  "system_type": "TLS/SSL Infrastructure",
  "risk_level": "Critical",
  "_source": "synthetic",
  "_id": "a1b2c3d4e5f67890"
}
```

### Splits

| Split | Count | Percentage |
|-------|-------|------------|
| Train | 4,521 | 85% |
| Eval | 531 | 10% |
| Test | 265 | 5% |

### Source Breakdown

| Source | Count |
|--------|-------|
| Synthetic | ~5,299 |
| Real Curated | 18 |

---

## 4. Post-Quantum Key Migration Synthetic Dataset

**Kaggle**: [post-quantum-key-migration-synthetic-dataset](https://www.kaggle.com/datasets/quantumind/post-quantum-key-migration-synthetic-dataset)

### Overview
5,300 synthetic PQC migration scenarios.

### Files

| File | Description |
|------|-------------|
| `instructions.jsonl` | Instruction-response pairs for SFT |
| `scenarios.json` | Full structured scenarios with all fields |

### Schema (Raw Scenario)

```json
{
  "scenario_id": "PQC-MIG-00001",
  "industry": "Financial Services",
  "system_type": "TLS/SSL Infrastructure",
  "risk_level": "Critical",
  "compliance_frameworks": ["NIST FIPS 140-3", "PCI DSS"],
  "current_crypto_stack": {"key_exchange": "ECDH P-256", "authentication": "ECDSA P-256"},
  "target_pqc_schemes": {"key_encapsulation": "ML-KEM-768", "authentication": "ML-DSA-65"},
  "estimated_timeline_months": 12,
  "budget_estimate_usd": 750000,
  "migration_steps": [...],
  "rollback_plan": [...],
  "testing_strategy": {...}
}
```

---

## 5. PQC Real Curated Instructions Dataset

**Kaggle**: [pqc-real-curated-instructions-dataset](https://www.kaggle.com/datasets/quantumind/pqc-real-curated-instructions-dataset)

### Overview
18 manually curated high-quality PQC migration instructions from industry sources.

### Files

| File | Description |
|------|-------------|
| `curated_real_instructions.jsonl` | Real instruction-response pairs |
| `generate_real_data.py` | Script that generated/structures these |

### Topics

- PKCS#11 HSM infrastructure migration to FIPS 203/204/205
- Algorithm performance (ML-KEM, ML-DSA, SLH-DSA)
- Vendor requirements (Thales Luna, Utimaco, nCipher)
- Key capacity impacts
- TLS/SSL certificate lifecycle
- Cloud KMS integration

---

## Data Generation Methodology

### Synthetic Data Pipeline

```
Domain Research --> Template Design --> Randomized Generation --> Deduplication --> Splitting
```

1. **Domain Research**: NIST SP 800-208, FIPS 203/204/205, NSA CNSA 2.0, FDA guidelines, clinical pharmacology references
2. **Template Design**: Interaction/migration/vulnerability templates with parameterized fields
3. **Randomized Generation**: Random sampling from validated drug/algorithm/protocol pools with contextual patient/industry data
4. **Deduplication**: SHA-256 hash on normalized instruction + output prefix to remove exact duplicates
5. **Splitting**: Stratified random split (85/10/5) with fixed seed

### Quality Assurance

- All generated scenarios include references to real standards/guidelines
- Severity scores use CVSS-like scoring adapted for quantum threats
- Drug interactions based on established pharmacological mechanisms
- Code snippets are syntactically valid (though simplified for illustration)

---

## Usage

### Loading from Kaggle

```python
# Using kagglehub
import kagglehub
path = kagglehub.dataset_download("quantumind/synthetic-drug-drug-interaction-dataset")

# Or download manually and load with datasets
from datasets import load_dataset
ds = load_dataset("json", data_files="train.jsonl")
```

### Loading Locally

```python
import json

with open("data/raw_ddi/train.jsonl", "r") as f:
    data = [json.loads(line) for line in f]
```

---

## Licensing

All datasets are released under **Apache License 2.0**.

**Disclaimer**: These are synthetic datasets for research and model training. They are not substitutes for professional medical advice, certified security audits, or regulatory compliance review.
