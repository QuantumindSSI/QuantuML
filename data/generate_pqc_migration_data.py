#!/usr/bin/env python3
"""
Synthetic Data Generator for Post-Quantum Key Migration Advisor
Generates 5,000+ realistic migration scenarios based on NIST guidelines
and industry best practices.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any

# NIST PQC Algorithm families mapped to scenarios
PQC_ALGORITHMS = {
    "key_encapsulation": [
        "ML-KEM-512", "ML-KEM-768", "ML-KEM-1024",
        "KYBER-512", "KYBER-768", "KYBER-1024"
    ],
    "digital_signature": [
        "ML-DSA-44", "ML-DSA-65", "ML-DSA-87",
        "DILITHIUM-2", "DILITHIUM-3", "DILITHIUM-5",
        "SLH-DSA-SHA2-128s", "SLH-DSA-SHAKE-256s"
    ],
    "hybrid": [
        "ECDH+ML-KEM-768", "X25519+ML-KEM-1024",
        "RSA-3072+ML-KEM-512"
    ]
}

CLASSICAL_ALGORITHMS = {
    "rsa": ["RSA-2048", "RSA-3072", "RSA-4096"],
    "ecdsa": ["ECDSA P-256", "ECDSA P-384", "ECDSA P-521"],
    "dh": ["DH-2048", "DH-3072", "DH-4096"],
    "ecdh": ["ECDH P-256", "ECDH P-384"],
    "symmetric": ["AES-128-GCM", "AES-256-GCM", "ChaCha20-Poly1305"]
}

INDUSTRIES = [
    "Financial Services", "Healthcare", "Government", "Defense",
    "Telecommunications", "Energy", "Manufacturing", "Retail",
    "Cloud Service Provider", "Blockchain/Cryptocurrency"
]

SYSTEM_TYPES = [
    "TLS/SSL Infrastructure", "VPN Gateway", "Code Signing",
    "Document Signing", "Email Encryption (S/MIME)", "SSH Infrastructure",
    "Database Encryption", "API Authentication", "PKI Certificate Authority",
    "Blockchain Consensus", "IoT Device Authentication", "File Encryption",
    "Secure Messaging", "Hardware Security Module (HSM)", "Smart Card"
]

COMPLIANCE_FRAMEWORKS = [
    "NIST FIPS 140-3", "Common Criteria", "PCI DSS", "HIPAA",
    "GDPR Article 32", "FISMA", "SOX", "ISO 27001"
]

RISK_LEVELS = ["Critical", "High", "Medium", "Low"]

PHASES = ["Discovery", "Assessment", "Planning", "Implementation", "Validation", "Deployment"]


def generate_migration_scenario(idx: int) -> Dict[str, Any]:
    """Generate a single realistic migration scenario."""
    industry = random.choice(INDUSTRIES)
    system = random.choice(SYSTEM_TYPES)
    risk = random.choice(RISK_LEVELS)
    compliance = random.sample(COMPLIANCE_FRAMEWORKS, k=random.randint(1, 3))

    # Determine classical stack
    classical = {}
    if "TLS" in system or "VPN" in system or "API" in system:
        classical["key_exchange"] = random.choice(CLASSICAL_ALGORITHMS["ecdh"] + CLASSICAL_ALGORITHMS["dh"])
        classical["authentication"] = random.choice(CLASSICAL_ALGORITHMS["ecdsa"] + CLASSICAL_ALGORITHMS["rsa"])
    elif "Signing" in system or "Certificate" in system:
        classical["signature"] = random.choice(CLASSICAL_ALGORITHMS["ecdsa"] + CLASSICAL_ALGORITHMS["rsa"])
    else:
        classical["encryption"] = random.choice(CLASSICAL_ALGORITHMS["rsa"])

    # Determine target PQC stack
    target = {}
    if "key_exchange" in classical:
        target["key_encapsulation"] = random.choice(PQC_ALGORITHMS["key_encapsulation"])
        target["authentication"] = random.choice(PQC_ALGORITHMS["digital_signature"])
    elif "signature" in classical:
        target["signature"] = random.choice(PQC_ALGORITHMS["digital_signature"])
    else:
        target["hybrid_kem"] = random.choice(PQC_ALGORITHMS["hybrid"])
        target["signature"] = random.choice(PQC_ALGORITHMS["digital_signature"])

    # Generate migration steps
    steps = generate_migration_steps(system, classical, target, risk)

    return {
        "scenario_id": f"PQC-MIG-{idx:05d}",
        "industry": industry,
        "system_type": system,
        "risk_level": risk,
        "compliance_frameworks": compliance,
        "current_crypto_stack": classical,
        "target_pqc_schemes": target,
        "estimated_timeline_months": random.randint(3, 24),
        "budget_estimate_usd": random.randint(50000, 2000000),
        "migration_steps": steps,
        "rollback_plan": generate_rollback_plan(system, classical),
        "testing_strategy": generate_testing_strategy(system),
    }


def generate_migration_steps(system: str, classical: Dict, target: Dict, risk: str) -> List[Dict[str, Any]]:
    """Generate realistic step-by-step migration guidance."""
    steps = []
    phases = PHASES.copy()

    for i, phase in enumerate(phases):
        if phase == "Discovery":
            steps.append({
                "step_number": i + 1,
                "phase": phase,
                "title": f"Inventory all {system} cryptographic assets",
                "description": f"Catalog all instances using {list(classical.values())}. Document key lifecycles, certificate chains, and dependency graphs.",
                "tools": ["Cryptography Discovery Scanner", "Certificate Transparency Logs", "Network Traffic Analyzer"],
                "deliverables": ["Cryptographic Asset Inventory", "Risk Heat Map", "Dependency Graph"],
                "estimated_effort_hours": random.randint(40, 80),
                "owner": "Security Architecture Team"
            })
        elif phase == "Assessment":
            steps.append({
                "step_number": i + 1,
                "phase": phase,
                "title": f"Evaluate quantum threat exposure for {system}",
                "description": f"Assess vulnerability to Harvest Now, Decrypt Later (HNDL) attacks. Evaluate data sensitivity and retention periods against quantum timeline.",
                "tools": ["Quantum Risk Assessment Framework", "NIST SP 800-208 Analysis"],
                "deliverables": ["Quantum Vulnerability Report", "Business Impact Analysis"],
                "estimated_effort_hours": random.randint(60, 120),
                "owner": "Risk Management Team"
            })
        elif phase == "Planning":
            steps.append({
                "step_number": i + 1,
                "phase": phase,
                "title": f"Design hybrid cryptographic architecture",
                "description": f"Plan transition to {list(target.values())} using hybrid/composite schemes. Define fallback strategies and interop requirements.",
                "tools": ["PKI Design Tool", "Protocol Analyzer", "NIST Migration Guidance"],
                "deliverables": ["Target Architecture Document", "Migration Roadmap", "Test Plan"],
                "estimated_effort_hours": random.randint(80, 160),
                "owner": "Cryptography Engineering Team"
            })
        elif phase == "Implementation":
            steps.append({
                "step_number": i + 1,
                "phase": phase,
                "title": f"Deploy {list(target.values())[0]} in staging environment",
                "description": "Implement new algorithms in non-production. Update libraries (OpenSSL 3.x, BoringSSL, wolfSSL). Configure certificate profiles.",
                "tools": ["HSM Configuration Utility", "CI/CD Pipeline", "Static Code Analyzer"],
                "deliverables": ["Staging Deployment", "Configuration Baselines", "Updated Policies"],
                "estimated_effort_hours": random.randint(120, 240),
                "owner": "DevSecOps Team"
            })
        elif phase == "Validation":
            steps.append({
                "step_number": i + 1,
                "phase": phase,
                "title": "Perform cryptographic validation and performance testing",
                "description": "Execute ACVP testing, throughput benchmarks, and interoperability tests. Validate against NIST CAVP vectors.",
                "tools": ["ACVP Test Server", "Performance Benchmark Suite", "FIPS Test Harness"],
                "deliverables": ["Validation Report", "Performance Baseline", "Interoperability Matrix"],
                "estimated_effort_hours": random.randint(80, 160),
                "owner": "QA / Cryptography Validation Team"
            })
        elif phase == "Deployment":
            steps.append({
                "step_number": i + 1,
                "phase": phase,
                "title": "Phased production rollout with monitoring",
                "description": "Canary release, monitor for latency regressions and certificate errors. Maintain classical fallback during transition period.",
                "tools": ["Observability Platform", "A/B Testing Framework", "Incident Response Playbook"],
                "deliverables": ["Production Deployment", "Monitoring Dashboard", "Post-Deployment Review"],
                "estimated_effort_hours": random.randint(60, 120),
                "owner": "Site Reliability Engineering"
            })

    return steps


def generate_rollback_plan(system: str, classical: Dict) -> List[str]:
    return [
        f"Maintain active {list(classical.values())[0]} certificates during transition period",
        "Ensure dual-stack algorithm support in all endpoints",
        "Pre-stage classical configuration snapshots in HSM",
        "Define automated rollback triggers (>5% error rate, >200ms latency increase)",
        "Test emergency reversion procedure quarterly"
    ]


def generate_testing_strategy(system: str) -> Dict[str, Any]:
    return {
        "unit_tests": ["Algorithm correctness (KAT vectors)", "Boundary condition handling", "Memory safety (ASAN/MSAN)"],
        "integration_tests": ["End-to-end TLS handshake", "Certificate chain validation", "Mixed-mode interoperability"],
        "performance_tests": ["Handshake latency (p50/p99)", "Throughput (connections/sec)", "CPU/memory overhead"],
        "compliance_tests": ["FIPS 140-3 Level 3 validation", "Common Criteria EAL4+", "NIST SP 800-56B conformance"],
    }


def generate_instruction_response(scenario: Dict[str, Any]) -> Dict[str, str]:
    """Convert scenario into instruction-response pair for SLM training."""
    current = json.dumps(scenario["current_crypto_stack"], indent=2)
    target = json.dumps(scenario["target_pqc_schemes"], indent=2)
    steps = "\n".join([
        f"{s['step_number']}. [{s['phase']}] {s['title']}\n   Description: {s['description']}\n   Tools: {', '.join(s['tools'])}\n   Deliverables: {', '.join(s['deliverables'])}\n   Effort: {s['estimated_effort_hours']} hours\n   Owner: {s['owner']}"
        for s in scenario["migration_steps"]
    ])

    instruction = (
        f"You are a Post-Quantum Cryptography Migration Advisor. "
        f"An organization in the {scenario['industry']} sector needs to migrate their "
        f"{scenario['system_type']} from classical cryptography to quantum-safe algorithms. "
        f"The risk level is {scenario['risk_level']} and they must comply with {', '.join(scenario['compliance_frameworks'])}.\n\n"
        f"Current cryptographic stack:\n{current}\n\n"
        f"Provide a step-by-step migration plan including tools, deliverables, effort estimates, and responsible teams."
    )

    response = (
        f"# Post-Quantum Migration Plan for {scenario['system_type']}\n\n"
        f"## Overview\n"
        f"- **Industry**: {scenario['industry']}\n"
        f"- **Risk Level**: {scenario['risk_level']}\n"
        f"- **Compliance**: {', '.join(scenario['compliance_frameworks'])}\n"
        f"- **Timeline**: ~{scenario['estimated_timeline_months']} months\n"
        f"- **Budget**: ${scenario['budget_estimate_usd']:,}\n\n"
        f"## Target PQC Schemes\n{target}\n\n"
        f"## Migration Steps\n{steps}\n\n"
        f"## Rollback Plan\n"
        + "\n".join([f"- {r}" for r in scenario["rollback_plan"]])
    )

    return {
        "instruction": instruction,
        "input": "",
        "output": response,
        "scenario_id": scenario["scenario_id"],
        "industry": scenario["industry"],
        "system_type": scenario["system_type"],
        "risk_level": scenario["risk_level"],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PQC migration training data")
    parser.add_argument("--num_scenarios", type=int, default=5500, help="Number of scenarios to generate")
    parser.add_argument("--output_dir", type=str, default="./raw", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    instruction_data = []

    print(f"Generating {args.num_scenarios} synthetic migration scenarios...")

    for i in range(1, args.num_scenarios + 1):
        scenario = generate_migration_scenario(i)
        scenarios.append(scenario)
        instruction_data.append(generate_instruction_response(scenario))

        if i % 500 == 0:
            print(f"  Generated {i}/{args.num_scenarios} scenarios...")

    # Save raw scenarios
    raw_path = output_dir / "scenarios.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(scenarios, f, indent=2)
    print(f"Saved raw scenarios to {raw_path}")

    # Save instruction-response pairs
    instruct_path = output_dir / "instructions.jsonl"
    with open(instruct_path, "w", encoding="utf-8") as f:
        for item in instruction_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Saved instruction data to {instruct_path}")

    print("\nDataset generation complete!")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Total instruction pairs: {len(instruction_data)}")


if __name__ == "__main__":
    main()
