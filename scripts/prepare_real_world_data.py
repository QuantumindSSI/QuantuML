#!/usr/bin/env python3
"""Convert downloaded real-world datasets into QuantuML instruction JSONL.

Produces files in the same {"instruction", "input", "output"} schema as the
synthetic generators (data/generate_*.py), plus provenance fields
("origin", "source_license") so mixing scripts can report exact
real-vs-synthetic splits.

Tasks:
  pqc    data/real_world/pqc_neura_parse (Neura-parse, CC-BY-4.0,
         source-verified against NIST FIPS 203/204/205, IR 8547)
         -> data/real_world/real_pqc_instructions.jsonl
  ddi    data/real_world/ddi_corpus_2013 (DDIExtraction 2013 SemEval corpus,
         real MEDLINE/DrugBank sentences, CC-BY-NC-4.0)
         -> data/real_world/real_ddi_instructions.jsonl
  crypto NVD CVE API (real CVE records, U.S. Government public data) filtered
         to cryptographic weakness CWEs
         -> data/real_world/real_crypto_instructions.jsonl
  ddi_openfda
         openFDA drug label API (real FDA prescribing information,
         U.S. Government public data - commercial-safe, unlike the
         CC-BY-NC DDI-2013 corpus which should be used for evaluation only)
         -> data/real_world/real_ddi_openfda_instructions.jsonl

Usage:
    ./venv/bin/python scripts/prepare_real_world_data.py --task pqc [--max_records N]
    ./venv/bin/python scripts/prepare_real_world_data.py --task ddi
    ./venv/bin/python scripts/prepare_real_world_data.py --task crypto
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REAL_DIR = PROJECT_ROOT / "data" / "real_world"

# ---------------------------------------------------------------------------
# PQC: Neura-parse quantum-cryptography-and-post-quantum-security
# ---------------------------------------------------------------------------

def convert_pqc(max_records: int, seed: int) -> list[dict]:
    """Convert Neura-parse parquet rows to instruction pairs.

    Uses generative record types only (qa_open, instruction, concept).
    MCQ rows are excluded: multiple-choice format teaches option-picking,
    not the advisory generation this model performs.
    """
    import pandas as pd

    parquet = REAL_DIR / "pqc_neura_parse" / "data" / "train-00000-of-00001.parquet"
    if not parquet.is_file():
        raise FileNotFoundError(f"missing {parquet}; download the dataset first")
    df = pd.read_parquet(parquet)

    records = []
    for row in df.itertuples(index=False):
        rt = row.record_type
        if rt == "qa_open":
            q, a = row.question, row.answer
            if _nonempty(q) and _nonempty(a):
                out = str(a)
                if _nonempty(row.rationale):
                    out = f"{out}\n\nRationale: {row.rationale}"
                records.append(_rec(str(q), "", out, "neura-parse/nist-pqc", "CC-BY-4.0"))
        elif rt == "instruction":
            ins, outp = row.instruction, row.output
            if _nonempty(ins) and _nonempty(outp):
                inp = str(row.input) if _nonempty(row.input) else ""
                records.append(_rec(str(ins), inp, str(outp), "neura-parse/nist-pqc", "CC-BY-4.0"))
        elif rt == "concept":
            term, definition = row.term, row.definition
            if _nonempty(term) and _nonempty(definition):
                out = str(definition)
                if _nonempty(row.explanation):
                    out = f"{out}\n\n{row.explanation}"
                records.append(_rec(str(term), "", out, "neura-parse/nist-pqc", "CC-BY-4.0"))

    rng = random.Random(seed)
    rng.shuffle(records)
    if max_records > 0:
        records = records[:max_records]
    return records


# ---------------------------------------------------------------------------
# DDI: DDIExtraction 2013 corpus (real MEDLINE/DrugBank sentences)
# ---------------------------------------------------------------------------

RELATION_DESCRIPTIONS = {
    "MECHANISM": (
        "Interaction detected - type: MECHANISM (pharmacokinetic). "
        "The text describes a pharmacokinetic mechanism by which {d1} and {d2} interact "
        "(e.g., altered absorption, metabolism, or clearance)."
    ),
    "EFFECT": (
        "Interaction detected - type: EFFECT (pharmacodynamic). "
        "The text describes a clinical effect resulting from the combination of {d1} and {d2} "
        "(e.g., potentiation, antagonism, or increased adverse-event risk)."
    ),
    "ADVISE": (
        "Interaction detected - type: ADVISE (clinical recommendation). "
        "The text contains a recommendation or warning about co-administering {d1} and {d2}."
    ),
    "INT": (
        "Interaction detected - type: INT (interaction stated without detail). "
        "The text states that {d1} and {d2} interact but does not specify mechanism or effect."
    ),
    "NONE": (
        "No interaction detected. "
        "The text mentions both {d1} and {d2} but does not describe a drug-drug interaction between them."
    ),
}


def convert_ddi() -> list[dict]:
    """Convert DDI-2013 sentence pairs to interaction-classification instructions."""
    src = REAL_DIR / "ddi_corpus_2013" / "data" / "train.jsonl"
    if not src.is_file():
        raise FileNotFoundError(f"missing {src}; download the dataset first")

    records = []
    with open(src, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            row = json.loads(line)
            relation = row["relation"]
            if relation not in RELATION_DESCRIPTIONS:
                raise ValueError(f"{src}:{lineno}: unknown relation '{relation}'")
            d1, d2 = row["drug1"], row["drug2"]
            instruction = (
                "You are a Drug Interaction Predictor. Analyze the following statement "
                "from the biomedical literature and determine whether it describes a "
                f"drug-drug interaction between {d1} and {d2}. "
                "Classify the interaction type (MECHANISM, EFFECT, ADVISE, INT, or NONE) "
                "and explain your classification."
            )
            output = RELATION_DESCRIPTIONS[relation].format(d1=d1, d2=d2)
            records.append(_rec(instruction, row["sentence"], output,
                                "ddi-extraction-2013/drugbank-medline", "CC-BY-NC-4.0"))
    return records


# ---------------------------------------------------------------------------
# Crypto: real CVE records from the NVD API, crypto-weakness CWEs
# ---------------------------------------------------------------------------

CRYPTO_CWES = {
    "CWE-327": "Use of a Broken or Risky Cryptographic Algorithm",
    "CWE-326": "Inadequate Encryption Strength",
    "CWE-325": "Missing Cryptographic Step",
    "CWE-328": "Use of Weak Hash",
    "CWE-295": "Improper Certificate Validation",
    "CWE-330": "Use of Insufficiently Random Values",
    "CWE-320": "Key Management Errors",
    "CWE-311": "Missing Encryption of Sensitive Data",
}

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"
NVD_PAGE_SIZE = 2000
NVD_SLEEP_SECONDS = 7  # public rate limit: 5 requests / 30s
NVD_MAX_RETRIES = 5


def _fetch_nvd_page(params: dict) -> dict:
    """One NVD API request with bounded retries and backoff."""
    import requests

    last_err = None
    for attempt in range(1, NVD_MAX_RETRIES + 1):
        try:
            resp = requests.get(NVD_URL, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            last_err = RuntimeError(f"NVD HTTP {resp.status_code}")
        except requests.RequestException as exc:
            last_err = exc
        sleep = NVD_SLEEP_SECONDS * attempt
        print(f"  NVD retry {attempt}/{NVD_MAX_RETRIES} in {sleep}s ({last_err})", flush=True)
        time.sleep(sleep)
    raise RuntimeError(f"NVD API failed after {NVD_MAX_RETRIES} retries: {last_err}")


def convert_crypto(max_per_cwe: int) -> list[dict]:
    """Fetch real crypto-weakness CVEs from NVD and convert to analysis instructions."""
    records = []
    seen_cves = set()
    for cwe_id, cwe_name in CRYPTO_CWES.items():
        print(f"Fetching {cwe_id} ({cwe_name})...", flush=True)
        start = 0
        fetched = 0
        while fetched < max_per_cwe:
            data = _fetch_nvd_page({
                "cweId": cwe_id,
                "resultsPerPage": min(NVD_PAGE_SIZE, max_per_cwe - fetched),
                "startIndex": start,
            })
            vulns = data.get("vulnerabilities", [])
            if not vulns:
                break
            for item in vulns:
                cve = item.get("cve", {})
                rec = _cve_to_record(cve, cwe_id, cwe_name)
                if rec is not None and cve.get("id") not in seen_cves:
                    seen_cves.add(cve["id"])
                    records.append(rec)
            fetched += len(vulns)
            start += len(vulns)
            total = data.get("totalResults", 0)
            print(f"  {cwe_id}: {fetched} fetched (total available {total})", flush=True)
            if start >= total:
                break
            time.sleep(NVD_SLEEP_SECONDS)
        time.sleep(NVD_SLEEP_SECONDS)
    return records


def _cve_to_record(cve: dict, cwe_id: str, cwe_name: str):
    """Map one NVD CVE object to an instruction pair; None if unusable."""
    cve_id = cve.get("id")
    descriptions = [d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en"]
    if not cve_id or not descriptions:
        return None
    description = descriptions[0]
    if len(description) < 60:  # too short to teach anything
        return None

    severity, score, vector = "UNKNOWN", None, None
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        if key in metrics and metrics[key]:
            m = metrics[key][0].get("cvssData", {})
            severity = metrics[key][0].get("baseSeverity") or m.get("baseSeverity", "UNKNOWN")
            score = m.get("baseScore")
            vector = m.get("vectorString")
            break

    instruction = (
        "You are a Quantum-Resistant Cryptographic Protocol Analyzer. "
        "Analyze the following real-world vulnerability description, identify the "
        "cryptographic weakness class, assess its severity, and recommend mitigations "
        "aligned with current NIST guidance."
    )
    output_lines = [
        "# Cryptographic Vulnerability Analysis",
        "",
        f"## Identifier\n{cve_id}",
        "",
        f"## Weakness Classification\n- **CWE**: {cwe_id} ({cwe_name})",
    ]
    if score is not None:
        output_lines.append(f"- **CVSS**: {score} ({severity})")
        if vector:
            output_lines.append(f"- **Vector**: {vector}")
    output_lines += [
        "",
        "## Assessment",
        f"This vulnerability falls under {cwe_id} ({cwe_name}). {description}",
        "",
        "## Recommended Mitigations",
        _cwe_mitigation(cwe_id),
    ]
    return _rec(instruction, description, "\n".join(output_lines), "nvd-cve", "US-Gov public data")


def _cwe_mitigation(cwe_id: str) -> str:
    mitigations = {
        "CWE-327": "Replace broken/deprecated algorithms with NIST-approved primitives (AES-256-GCM, SHA-2/SHA-3). For quantum resistance, adopt ML-KEM (FIPS 203) for key establishment and ML-DSA (FIPS 204) for signatures.",
        "CWE-326": "Increase key lengths to meet NIST SP 800-57 minimums (AES-128+ symmetric, RSA-3072+/P-256+ classical asymmetric) and plan migration to PQC per NIST IR 8547.",
        "CWE-325": "Ensure all required cryptographic steps (padding verification, MAC validation, nonce handling) are implemented; use vetted high-level libraries instead of custom constructions.",
        "CWE-328": "Replace weak hashes (MD5, SHA-1) with SHA-256/SHA-3; use dedicated password-hashing functions (Argon2id, scrypt) for credentials.",
        "CWE-295": "Enforce full certificate chain validation, hostname verification, and revocation checking; do not disable TLS verification in production code.",
        "CWE-330": "Use CSPRNGs (e.g., /dev/urandom, OS crypto APIs) for all security-relevant values; never seed from time or PIDs.",
        "CWE-320": "Apply NIST SP 800-57 key-management practice: rotation schedules, HSM-backed storage, separation of key-encryption and data-encryption keys.",
        "CWE-311": "Encrypt sensitive data in transit (TLS 1.3) and at rest (AES-256-GCM); classify data flows to identify unencrypted paths.",
    }
    return mitigations[cwe_id]


# ---------------------------------------------------------------------------
# DDI (commercial-safe): openFDA drug label interaction sections
# ---------------------------------------------------------------------------

OPENFDA_URL = "https://api.fda.gov/drug/label.json"
OPENFDA_PAGE_SIZE = 100
OPENFDA_SLEEP_SECONDS = 2   # public limit: 40 req/min without API key
OPENFDA_MAX_RETRIES = 5
OPENFDA_MIN_TEXT = 200      # skip stub sections
OPENFDA_MAX_TEXT = 4000     # cap very long label sections


def _fetch_openfda_page(skip: int) -> dict:
    """One openFDA label API request with bounded retries and backoff."""
    import requests

    params = {
        "search": "_exists_:drug_interactions AND _exists_:openfda.generic_name",
        "limit": OPENFDA_PAGE_SIZE,
        "skip": skip,
    }
    last_err = None
    for attempt in range(1, OPENFDA_MAX_RETRIES + 1):
        try:
            resp = requests.get(OPENFDA_URL, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            last_err = RuntimeError(f"openFDA HTTP {resp.status_code}")
        except requests.RequestException as exc:
            last_err = exc
        sleep = OPENFDA_SLEEP_SECONDS * attempt
        print(f"  openFDA retry {attempt}/{OPENFDA_MAX_RETRIES} in {sleep}s ({last_err})", flush=True)
        time.sleep(sleep)
    raise RuntimeError(f"openFDA API failed after {OPENFDA_MAX_RETRIES} retries: {last_err}")


def convert_ddi_openfda(max_records: int) -> list[dict]:
    """Fetch real FDA structured product label interaction sections.

    openFDA drug label data is U.S. Government public data (no copyright),
    safe for commercial training use. One record per unique generic drug.
    """
    records = []
    seen_drugs = set()
    skip = 0
    max_skip = 25000  # openFDA hard pagination limit
    while len(records) < max_records and skip < max_skip:
        data = _fetch_openfda_page(skip)
        results = data.get("results", [])
        if not results:
            break
        for label in results:
            rec = _label_to_record(label, seen_drugs)
            if rec is not None:
                records.append(rec)
                if len(records) >= max_records:
                    break
        skip += len(results)
        total = data.get("meta", {}).get("results", {}).get("total", 0)
        print(f"  openFDA: {len(records)} usable records from {skip} labels "
              f"(total available {total})", flush=True)
        if skip >= total:
            break
        time.sleep(OPENFDA_SLEEP_SECONDS)
    return records


def _label_to_record(label: dict, seen_drugs: set):
    """Map one FDA label to an instruction pair; None if unusable/duplicate."""
    openfda = label.get("openfda", {})
    generic_names = openfda.get("generic_name", [])
    interactions = label.get("drug_interactions", [])
    if not generic_names or not interactions:
        return None
    drug = generic_names[0].strip().lower()
    if not drug or drug in seen_drugs:
        return None
    text = " ".join(t.strip() for t in interactions if t and t.strip())
    # Strip the redundant section heading FDA labels often start with.
    for prefix in ("7 DRUG INTERACTIONS", "DRUG INTERACTIONS", "Drug Interactions"):
        if text.startswith(prefix):
            text = text[len(prefix):].lstrip(" :.-")
            break
    if len(text) < OPENFDA_MIN_TEXT:
        return None
    if len(text) > OPENFDA_MAX_TEXT:
        cut = text.rfind(". ", 0, OPENFDA_MAX_TEXT)
        text = text[:cut + 1] if cut > 0 else text[:OPENFDA_MAX_TEXT]
    seen_drugs.add(drug)

    instruction = (
        "You are a Drug Interaction Predictor. Based on FDA prescribing "
        f"information, describe the clinically significant drug interactions for {drug}. "
        "Include interacting drugs or drug classes, mechanisms where known, and "
        "clinical recommendations."
    )
    output = f"# Drug Interactions: {drug}\n\n{text}"
    return _rec(instruction, "", output, "openfda-drug-label", "US-Gov public data")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _nonempty(value) -> bool:
    """True when a parquet/JSON scalar contains usable text."""
    if value is None:
        return False
    s = str(value).strip()
    return bool(s) and s.lower() not in ("nan", "none")


def _rec(instruction: str, input_text: str, output: str, origin: str, license_: str) -> dict:
    return {
        "instruction": instruction.strip(),
        "input": input_text.strip(),
        "output": output.strip(),
        "origin": origin,
        "data_kind": "real",
        "source_license": license_,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert real-world datasets to instruction JSONL")
    parser.add_argument("--task", choices=["pqc", "ddi", "ddi_openfda", "crypto"], required=True)
    parser.add_argument("--max_records", type=int, default=0,
                        help="Cap output records (0 = no cap; pqc only)")
    parser.add_argument("--max_per_cwe", type=int, default=800,
                        help="Max CVEs fetched per CWE class (crypto only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.task == "pqc":
        records = convert_pqc(args.max_records, args.seed)
        out_path = REAL_DIR / "real_pqc_instructions.jsonl"
    elif args.task == "ddi":
        records = convert_ddi()
        out_path = REAL_DIR / "real_ddi_instructions.jsonl"
    elif args.task == "ddi_openfda":
        records = convert_ddi_openfda(args.max_records if args.max_records > 0 else 6000)
        out_path = REAL_DIR / "real_ddi_openfda_instructions.jsonl"
    else:
        records = convert_crypto(args.max_per_cwe)
        out_path = REAL_DIR / "real_crypto_instructions.jsonl"

    if not records:
        print("ERROR: no records produced", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} real-world instruction records to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
