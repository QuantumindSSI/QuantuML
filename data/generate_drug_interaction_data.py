#!/usr/bin/env python3
"""
Synthetic Data Generator for Drug Interaction Predictor
Generates 100,000+ drug-drug interaction (DDI) examples with severity,
mechanism, evidence, and clinical recommendations.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any

# ==================== DRUG DATABASE ====================
DRUGS = {
    "anticoagulants": ["Warfarin", "Apixaban", "Rivaroxaban", "Dabigatran", "Edoxaban", "Heparin", "Enoxaparin"],
    "antiplatelets": ["Aspirin", "Clopidogrel", "Ticagrelor", "Prasugrel", "Dipyridamole"],
    "statins": ["Atorvastatin", "Simvastatin", "Rosuvastatin", "Pravastatin", "Lovastatin", "Fluvastatin"],
    "ace_inhibitors": ["Lisinopril", "Enalapril", "Ramipril", "Captopril", "Perindopril", "Quinapril", "Benazepril"],
    "arbs": ["Losartan", "Valsartan", "Irbesartan", "Candesartan", "Telmisartan", "Olmesartan"],
    "beta_blockers": ["Metoprolol", "Atenolol", "Propranolol", "Bisoprolol", "Carvedilol", "Nebivolol", "Labetalol"],
    "calcium_channel_blockers": ["Amlodipine", "Nifedipine", "Diltiazem", "Verapamil", "Felodipine"],
    "diuretics": ["Furosemide", "Hydrochlorothiazide", "Spironolactone", "Torsemide", "Indapamide", "Chlorthalidone"],
    "ppi": ["Omeprazole", "Esomeprazole", "Pantoprazole", "Lansoprazole", "Rabeprazole", "Dexlansoprazole"],
    "h2_blockers": ["Ranitidine", "Famotidine", "Cimetidine", "Nizatidine"],
    "antibiotics": ["Amoxicillin", "Azithromycin", "Ciprofloxacin", "Clarithromycin", "Doxycycline", "Metronidazole", "Trimethoprim-Sulfamethoxazole"],
    "antifungals": ["Fluconazole", "Itraconazole", "Ketoconazole", "Voriconazole", "Posaconazole"],
    "antivirals": ["Acyclovir", "Oseltamivir", "Ritonavir", "Remdesivir", "Sofosbuvir"],
    "nsaids": ["Ibuprofen", "Naproxen", "Diclofenac", "Celecoxib", "Meloxicam", "Indomethacin"],
    "opioids": ["Morphine", "Oxycodone", "Hydrocodone", "Fentanyl", "Tramadol", "Codeine", "Methadone"],
    "benzodiazepines": ["Diazepam", "Lorazepam", "Alprazolam", "Clonazepam", "Midazolam", "Temazepam"],
    "antidepressants": ["Sertraline", "Fluoxetine", "Citalopram", "Escitalopram", "Venlafaxine", "Duloxetine", "Amitriptyline", "Bupropion"],
    "antipsychotics": ["Risperidone", "Olanzapine", "Quetiapine", "Aripiprazole", "Haloperidol", "Clozapine"],
    "anticonvulsants": ["Carbamazepine", "Phenytoin", "Valproic Acid", "Lamotrigine", "Levetiracetam", "Gabapentin", "Pregabalin", "Topiramate"],
    "oral_hypoglycemics": ["Metformin", "Glipizide", "Glyburide", "Sitagliptin", "Linagliptin", "Empagliflozin", "Dapagliflozin", "Pioglitazone"],
    "insulins": ["Insulin Glargine", "Insulin Aspart", "Insulin Lispro", "Insulin NPH", "Insulin Detemir", "Insulin Degludec"],
    "corticosteroids": ["Prednisone", "Methylprednisolone", "Dexamethasone", "Hydrocortisone", "Budesonide", "Fluticasone"],
    "immunosuppressants": ["Methotrexate", "Cyclosporine", "Tacrolimus", "Mycophenolate Mofetil", "Azathioprine", "Sirolimus"],
    "antihistamines": ["Cetirizine", "Loratadine", "Fexofenadine", "Diphenhydramine", "Promethazine"],
    "bronchodilators": ["Albuterol", "Salmeterol", "Formoterol", "Tiotropium", "Ipratropium"],
    "antigout": ["Allopurinol", "Colchicine", "Febuxostat", "Probenecid"],
    "thyroid": ["Levothyroxine", "Methimazole", "Propylthiouracil"],
    "bone_health": ["Alendronate", "Risedronate", "Raloxifene", "Denosumab", "Zoledronic Acid"],
    "erectile_dysfunction": ["Sildenafil", "Tadalafil", "Vardenafil"],
    "proton_pump": ["Omeprazole", "Esomeprazole", "Pantoprazole"],
    "anticholinergics": ["Atropine", "Scopolamine", "Oxybutynin", "Tolterodine", "Solifenacin"],
}

# Flatten all drugs
ALL_DRUGS = [d for group in DRUGS.values() for d in group]

# ==================== INTERACTION MECHANISMS ====================
MECHANISMS = {
    "pharmacokinetic": {
        "CYP3A4_inhibition": " inhibits CYP3A4-mediated metabolism of ",
        "CYP3A4_induction": " induces CYP3A4, reducing plasma levels of ",
        "CYP2D6_inhibition": " inhibits CYP2D6 metabolism of ",
        "CYP2C9_inhibition": " inhibits CYP2C9-mediated metabolism of ",
        "CYP2C19_inhibition": " inhibits CYP2C19 metabolism of ",
        "CYP1A2_inhibition": " inhibits CYP1A2, increasing concentrations of ",
        "P_glycoprotein_inhibition": " inhibits P-glycoprotein-mediated efflux of ",
        "P_glycoprotein_induction": " induces P-glycoprotein, reducing absorption of ",
        "protein_binding_displacement": " displaces {drug_b} from plasma protein binding sites, increasing free fraction of ",
        "renal_excretion_competition": " reduces renal tubular secretion of ",
        "hepatic_uptake_inhibition": " inhibits OATP1B1-mediated hepatic uptake of ",
        "gastric_pH_alteration": " alters gastric pH, affecting absorption of ",
    },
    "pharmacodynamic": {
        "additive_sedation": " has additive CNS depressant effects with ",
        "additive_QT_prolongation": " and {drug_b} both prolong QTc interval, increasing torsades de pointes risk",
        "additive_hypotension": " has additive hypotensive effects with ",
        "additive_hyperkalemia": " has additive hyperkalemic effects with ",
        "additive_hypoglycemia": " increases hypoglycemic risk when combined with ",
        "additive_nephrotoxicity": " has additive nephrotoxic effects with ",
        "additive_hepatotoxicity": " increases hepatotoxicity risk with ",
        "additive_serotonergic": " has additive serotonergic effects with ",
        "additive_anticholinergic": " has additive anticholinergic burden with ",
        "pharmacological_antagonism": " pharmacologically antagonizes the effect of ",
        "additive_bleeding": " has additive anticoagulant/antiplatelet effects with ",
        "additive_bradycardia": " causes additive bradycardia with ",
    },
}

# ==================== INTERACTION OUTCOMES & RECOMMENDATIONS ====================
INTERACTION_TEMPLATES = [
    {
        "drug_classes": ("anticoagulants", "antiplatelets"),
        "interaction_type": "Major Bleeding Risk",
        "severity": "Critical",
        "mechanism_cat": "pharmacodynamic",
        "mechanism_key": "additive_bleeding",
        "outcomes": ["Major gastrointestinal bleeding", "Intracranial hemorrhage", "Prolonged bleeding time", "Hemorrhagic stroke"],
        "recommendations": [
            "Avoid concurrent use unless absolutely necessary",
            "If unavoidable, monitor INR/anti-Xa levels closely",
            "Consider proton pump inhibitor for GI protection",
            "Patient education on bleeding signs"
        ],
        "evidence": ["FDA Boxed Warning", "ACC/AHA Guideline Class III", "Major RCT (WOEST trial)"],
    },
    {
        "drug_classes": ("statins", "antibiotics"),
        "interaction_type": "Statin Toxicity / Myopathy",
        "severity": "High",
        "mechanism_cat": "pharmacokinetic",
        "mechanism_key": "CYP3A4_inhibition",
        "outcomes": ["Rhabdomyolysis", "Severe myopathy", "Acute kidney injury", "Elevated CK/AST/ALT"],
        "recommendations": [
            "Use pravastatin or rosuvastatin (non-CYP3A4 substrates)",
            "Temporarily hold statin during antibiotic course",
            "Monitor CK and renal function",
            "Dose reduction if combination unavoidable"
        ],
        "evidence": ["FDA Label Warning", "Am J Cardiol Meta-analysis", "AHA Scientific Statement"],
    },
    {
        "drug_classes": ("warfarin_class", "antibiotics"),
        "interaction_type": "Warfarin Potentiation / Bleeding",
        "severity": "Critical",
        "mechanism_cat": "pharmacokinetic",
        "mechanism_key": "CYP2C9_inhibition",
        "outcomes": ["Supratherapeutic INR", "Life-threatening bleeding", "Hematuria", "GI bleed"],
        "recommendations": [
            "Increase INR monitoring frequency to every 2-3 days",
            "Counsel patient on dietary vitamin K consistency",
            "Consider temporary warfarin dose reduction (10-25%)",
            "Switch to DOAC if long-term antibiotic use required"
        ],
        "evidence": ["FDA Adverse Event Database", "Chest Guideline", "ASH Guideline"],
    },
    {
        "drug_classes": ("anticoagulants", "nsaids"),
        "interaction_type": "GI Bleeding and Anticoagulant Potentiation",
        "severity": "High",
        "mechanism_cat": "pharmacodynamic",
        "mechanism_key": "additive_bleeding",
        "outcomes": ["Peptic ulcer disease", "Upper GI bleeding", "Nephrotoxicity", "Hypertension"],
        "recommendations": [
            "Avoid NSAIDs; use acetaminophen for analgesia",
            "If NSAID unavoidable, use lowest effective dose + PPI",
            "Monitor renal function and blood pressure",
            "Patient education on black/tarry stools"
        ],
        "evidence": ["FDA Drug Safety Communication", "British Journal of Haematology", "Gastroenterology Guideline"],
    },
    {
        "drug_classes": ("opioids", "benzodiazepines"),
        "interaction_type": "Respiratory Depression / Overdose",
        "severity": "Critical",
        "mechanism_cat": "pharmacodynamic",
        "mechanism_key": "additive_sedation",
        "outcomes": ["Profound sedation", "Respiratory depression", "Coma", "Death (overdose)"],
        "recommendations": [
            "Avoid concurrent use when possible (FDA Black Box Warning)",
            "If co-prescribed, use lowest effective doses",
            "Provide naloxone prescription",
            "Monitor for signs of oversedation"
        ],
        "evidence": ["FDA Black Box Warning", "CDC Guideline for Opioid Prescribing", "NEJM Case Series"],
    },
    {
        "drug_classes": ("ace_inhibitors", "arbs"),
        "interaction_type": "Dual RAAS Blockade / Hyperkalemia / AKI",
        "severity": "High",
        "mechanism_cat": "pharmacodynamic",
        "mechanism_key": "pharmacological_antagonism",
        "outcomes": ["Hyperkalemia", "Acute kidney injury", "Hypotension", "No additional cardiovascular benefit"],
        "recommendations": [
            "Avoid routine dual RAAS blockade (ONTARGET trial)",
            "If indicated (proteinuric CKD), monitor K+ and creatinine closely",
            "Consider SGLT2 inhibitor or MR antagonist instead",
            "Discontinue if K+ >5.5 or creatinine rises >30%"
        ],
        "evidence": ["ONTARGET Trial (NEJM)", "KDIGO Guideline", "ACC/AHA Heart Failure Guideline"],
    },
    {
        "drug_classes": ("diuretics", "nsaids"),
        "interaction_type": "Diuretic Resistance / Renal Impairment",
        "severity": "Medium",
        "mechanism_cat": "pharmacodynamic",
        "mechanism_key": "pharmacological_antagonism",
        "outcomes": ["Reduced diuretic efficacy", "Acute kidney injury", "Fluid retention", "Worsening heart failure"],
        "recommendations": [
            "Avoid NSAIDs in heart failure patients",
            "Use acetaminophen as first-line analgesic",
            "If NSAID required, use short course + monitor weight/BUN/Cr",
            "Consider colchicine for gout instead of NSAIDs"
        ],
        "evidence": ["ESC Heart Failure Guideline", "FDA Label", "Cochrane Review"],
    },
    {
        "drug_classes": ("antidepressants", "tramadol"),
        "interaction_type": "Serotonin Syndrome",
        "severity": "Critical",
        "mechanism_cat": "pharmacodynamic",
        "mechanism_key": "additive_serotonergic",
        "outcomes": ["Serotonin syndrome", "Hyperthermia", "Autonomic instability", "Mental status changes", "Muscle rigidity"],
        "recommendations": [
            "Avoid tramadol with SSRIs/SNRIs when possible",
            "If unavoidable, monitor for serotonergic symptoms",
            "Use lower tramadol doses; avoid MAOIs",
            "Patient education on warning signs"
        ],
        "evidence": ["FDA Warning", "Hunter Serotonin Toxicity Criteria", "Australasian Psychiatry Review"],
    },
    {
        "drug_classes": ("ppi", "clopidogrel"),
        "interaction_type": "Antiplatelet Efficacy Reduction",
        "severity": "Medium",
        "mechanism_cat": "pharmacokinetic",
        "mechanism_key": "CYP2C19_inhibition",
        "outcomes": ["Reduced clopidogrel activation", "Increased cardiovascular event risk", "Stent thrombosis"],
        "recommendations": [
            "Use pantoprazole if PPI needed (weakest CYP2C19 interaction)",
            "Consider H2 blocker (famotidine) as alternative",
            "Use ticagrelor or prasugrel instead of clopidogrel if PPI unavoidable",
            "Avoid omeprazole and esomeprazole with clopidogrel"
        ],
        "evidence": ["FDA Drug Safety Communication", "COGENT Trial", "ESC Guideline"],
    },
    {
        "drug_classes": ("calcium_channel_blockers", "statins"),
        "interaction_type": "Statin Toxicity Risk",
        "severity": "Medium",
        "mechanism_cat": "pharmacokinetic",
        "mechanism_key": "CYP3A4_inhibition",
        "outcomes": ["Myopathy", "Rhabdomyolysis", "Increased statin plasma concentrations"],
        "recommendations": [
            "Use amlodipine (weak interaction) over diltiazem/verapamil",
            "Prefer pravastatin or rosuvastatin in CCB patients",
            "Limit simvastatin dose to 20 mg with diltiazem/verapamil",
            "Monitor CK levels"
        ],
        "evidence": ["FDA Label Restriction", "AHA Scientific Statement", "Product Labeling"],
    },
    {
        "drug_classes": ("oral_hypoglycemics", "ACE_inhibitors"),
        "interaction_type": "Hypoglycemia Risk",
        "severity": "Low",
        "mechanism_cat": "pharmacodynamic",
        "mechanism_key": "additive_hypoglycemia",
        "outcomes": ["Hypoglycemia", "Unexplained low blood glucose"],
        "recommendations": [
            "Monitor blood glucose more frequently when initiating ACE-I",
            "Patient education on hypoglycemia recognition",
            "Consider slight reduction in sulfonylurea dose if needed",
            "Generally beneficial interaction; no need to avoid"
        ],
        "evidence": ["ADA Standards of Care", "Clinical Pharmacology & Therapeutics", "Case Reports"],
    },
    {
        "drug_classes": ("anticonvulsants", "oral_contraceptives"),
        "interaction_type": "Contraceptive Failure",
        "severity": "High",
        "mechanism_cat": "pharmacokinetic",
        "mechanism_key": "CYP3A4_induction",
        "outcomes": ["Unplanned pregnancy", "Breakthrough bleeding", "Reduced contraceptive efficacy"],
        "recommendations": [
            "Use non-hormonal contraception or IUD",
            "Consider increasing estrogen dose if hormonal only option",
            "Lamotrigine levels also reduced by oral contraceptives (monitor seizures)",
            "Counsel on back-up contraception"
        ],
        "evidence": ["ACOG Practice Bulletin", "AAN Guideline", "Epilepsia Review"],
    },
    {
        "drug_classes": ("antibiotics", "anticoagulants"),
        "interaction_type": "Anticoagulant Potentiation / Vitamin K Disruption",
        "severity": "High",
        "mechanism_cat": "pharmacokinetic",
        "mechanism_key": "CYP2C9_inhibition",
        "outcomes": ["Major bleeding", "Supratherapeutic INR", "Hematoma"],
        "recommendations": [
            "Increase INR monitoring frequency",
            "Temporary warfarin dose reduction",
            "For DOACs, monitor renal function and bleeding signs",
            "Patient alert card for emergency providers"
        ],
        "evidence": ["ASH Guideline", "Chest Guideline", "British Journal of Clinical Pharmacology"],
    },
    {
        "drug_classes": ("immunosuppressants", "antibiotics"),
        "interaction_type": "Immunosuppressant Toxicity / Rejection Risk",
        "severity": "Critical",
        "mechanism_cat": "pharmacokinetic",
        "mechanism_key": "CYP3A4_inhibition",
        "outcomes": ["Nephrotoxicity", "Neurotoxicity", "Graft rejection (if levels drop)", "Bone marrow suppression"],
        "recommendations": [
            "Therapeutic drug monitoring (TDM) for tacrolimus/cyclosporine",
            "Monitor creatinine and drug trough levels",
            "Dose adjustment based on levels",
            "Coordinate with transplant team"
        ],
        "evidence": ["AST/ASTS Guideline", "Transplantation Journal", "FDA Label"],
    },
    {
        "drug_classes": ("beta_blockers", "calcium_channel_blockers"),
        "interaction_type": "Excessive Bradycardia / Heart Block",
        "severity": "High",
        "mechanism_cat": "pharmacodynamic",
        "mechanism_key": "additive_bradycardia",
        "outcomes": ["Severe bradycardia", "Atrioventricular block", "Syncope", "Heart failure exacerbation"],
        "recommendations": [
            "Use dihydropyridine CCBs (amlodipine, nifedipine) with beta-blockers",
            "Avoid verapamil or diltiazem with beta-blockers",
            "Monitor heart rate and ECG if combination used",
            "Dose reduction if HR <60 bpm"
        ],
        "evidence": ["ACC/AHA Guideline", "ESC Guideline", "FDA Label"],
    },
    {
        "drug_classes": ("antihistamines", "cns_depressants"),
        "interaction_type": "Additive CNS Depression",
        "severity": "Medium",
        "mechanism_cat": "pharmacodynamic",
        "mechanism_key": "additive_sedation",
        "outcomes": ["Excessive sedation", "Impaired driving", "Cognitive dysfunction", "Falls (elderly)"],
        "recommendations": [
            "Prefer non-sedating antihistamines (loratadine, fexofenadine)",
            "Avoid first-generation antihistamines in elderly",
            "Counsel on drowsiness and avoid driving",
            "Monitor for confusion or falls"
        ],
        "evidence": ["Beers Criteria", "AGS Guideline", "Cochrane Review"],
    },
]

# General fallback interactions for random pairs
FALLBACK_INTERACTIONS = [
    ("pharmacokinetic", "CYP3A4_inhibition", " plasma levels may be increased, leading to toxicity"),
    ("pharmacokinetic", "CYP3A4_induction", " plasma levels may be decreased, reducing efficacy"),
    ("pharmacokinetic", "P_glycoprotein_inhibition", " absorption and bioavailability may increase"),
    ("pharmacodynamic", "additive_QT_prolongation", " combined QTc prolongation risk; monitor ECG"),
    ("pharmacodynamic", "additive_hypotension", " combined hypotensive effects; risk of syncope"),
    ("pharmacodynamic", "additive_sedation", " additive CNS depression; caution with driving/operating machinery"),
    ("pharmacodynamic", "pharmacological_antagonism", " pharmacological antagonism may reduce efficacy of one or both drugs"),
    ("pharmacokinetic", "protein_binding_displacement", " free drug concentrations may increase transiently"),
    ("pharmacokinetic", "gastric_pH_alteration", " altered absorption; consider timing separation"),
]


def build_mechanism(drug_a: str, drug_b: str, mech_cat: str, mech_key: str) -> str:
    template = MECHANISMS[mech_cat][mech_key]
    if "{drug_b}" in template:
        return f"{drug_a}" + template.format(drug_b=drug_b)
    return f"{drug_a}" + template + f"{drug_b}"


def generate_interaction(idx: int) -> Dict[str, Any]:
    """Generate a single drug interaction scenario."""
    # Pick a template or fallback
    if random.random() < 0.75:
        template = random.choice(INTERACTION_TEMPLATES)
        class_a, class_b = template["drug_classes"]
        # Handle special class mappings
        if class_a == "warfarin_class":
            drugs_a = ["Warfarin"]
        else:
            drugs_a = DRUGS.get(class_a, ALL_DRUGS)
        drugs_b = DRUGS.get(class_b, ALL_DRUGS)
        drug_a = random.choice(drugs_a)
        drug_b = random.choice([d for d in drugs_b if d != drug_a])
        severity = template["severity"]
        interaction_type = template["interaction_type"]
        mechanism = build_mechanism(drug_a, drug_b, template["mechanism_cat"], template["mechanism_key"])
        outcomes = template["outcomes"]
        recommendations = template["recommendations"]
        evidence = template["evidence"]
    else:
        # Random fallback pair
        drug_a, drug_b = random.sample(ALL_DRUGS, 2)
        mech_cat, mech_key, desc = random.choice(FALLBACK_INTERACTIONS)
        severity = random.choice(["Low", "Medium", "High"])
        interaction_type = f"Potential {mech_key.replace('_', ' ').title()} Interaction"
        mechanism = build_mechanism(drug_a, drug_b, mech_cat, mech_key)
        if severity == "High":
            outcomes = ["Potential toxicity or reduced efficacy", "Requires monitoring"]
            recommendations = ["Monitor drug levels or clinical effect", "Consider alternative agent"]
            evidence = ["Theoretical interaction", "Product labeling"]
        else:
            outcomes = ["Mild to moderate effect", "Usually manageable"]
            recommendations = ["Monitor for symptoms", "No specific action usually needed"]
            evidence = ["Limited clinical data", "Theoretical concern"]

    # Build medication list (2-5 drugs, including the interacting pair + decoys)
    num_extra = random.randint(0, 3)
    extras = random.sample([d for d in ALL_DRUGS if d not in (drug_a, drug_b)], num_extra)
    medication_list = [drug_a, drug_b] + extras
    random.shuffle(medication_list)

    scenario = {
        "scenario_id": f"DDI-{idx:06d}",
        "medication_list": medication_list,
        "drug_pair": [drug_a, drug_b],
        "interaction_type": interaction_type,
        "severity": severity,
        "mechanism": mechanism,
        "predicted_outcomes": outcomes,
        "recommendations": recommendations,
        "evidence": evidence,
        "patient_context": generate_patient_context(),
    }
    return scenario


def generate_patient_context() -> str:
    contexts = [
        "65-year-old male with atrial fibrillation and osteoarthritis",
        "72-year-old female with heart failure, diabetes, and chronic kidney disease (eGFR 35)",
        "58-year-old post-MI patient on dual antiplatelet therapy with hyperlipidemia",
        "45-year-old female with depression, migraines, and contraceptive needs",
        "82-year-old nursing home resident with dementia, hypertension, and insomnia",
        "50-year-old male with type 2 diabetes, obesity, and knee pain",
        "38-year-old female post-renal transplant on immunosuppression with UTI",
        "60-year-old male with COPD, osteoporosis, and erectile dysfunction",
        "28-year-old female with epilepsy planning pregnancy",
        "55-year-old male with gout, hypertension, and recent stent placement",
        "70-year-old female with atrial fibrillation, GERD, and history of GI bleed",
        "35-year-old male with HIV on ART requiring pain management",
    ]
    return random.choice(contexts)


def generate_instruction_response(scenario: Dict[str, Any]) -> Dict[str, str]:
    """Convert scenario into instruction-response pair for SLM training."""
    meds = ", ".join(scenario["medication_list"])
    drug_a, drug_b = scenario["drug_pair"]

    instruction_templates = [
        f"Analyze the following medication list for potential drug-drug interactions: {meds}. Report any predicted interactions with severity, mechanism, evidence, and clinical recommendations.",
        f"You are a pharmacovigilance system. Given the patient on {meds}, identify significant drug interactions and provide actionable recommendations.",
        f"Review this medication regimen ({meds}) for a {scenario['patient_context']}. What interactions should be monitored?",
        f"Predict drug-drug interactions between the medications in this list: {meds}. Include severity and evidence.",
        f"Clinical pharmacist review requested for: {meds}. Provide interaction analysis with severity, mechanism, and recommendations.",
    ]
    instruction = random.choice(instruction_templates)

    input_text = (
        f"## Medication List\n" + "\n".join([f"- {m}" for m in scenario["medication_list"]]) + "\n\n"
        f"## Patient Context\n{scenario['patient_context']}"
    )

    response = (
        f"# Drug Interaction Analysis Report\n\n"
        f"## Identified Interaction\n"
        f"- **Drug Pair**: {drug_a} + {drug_b}\n"
        f"- **Interaction Type**: {scenario['interaction_type']}\n"
        f"- **Severity**: {scenario['severity']}\n"
        f"- **Mechanism**: {scenario['mechanism']}\n\n"
        f"## Predicted Outcomes\n"
        + "\n".join([f"- {o}" for o in scenario["predicted_outcomes"]])
        + f"\n\n## Clinical Recommendations\n"
        + "\n".join([f"{i+1}. {r}" for i, r in enumerate(scenario["recommendations"])])
        + f"\n\n## Evidence\n"
        + "\n".join([f"- {e}" for e in scenario["evidence"]])
    )

    return {
        "instruction": instruction,
        "input": input_text,
        "output": response,
        "scenario_id": scenario["scenario_id"],
        "drug_pair": scenario["drug_pair"],
        "severity": scenario["severity"],
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic drug interaction training data")
    parser.add_argument("--num_scenarios", type=int, default=102000, help="Number of scenarios to generate")
    parser.add_argument("--output_dir", type=str, default="./data/raw_ddi", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = []
    instruction_data = []

    print(f"Generating {args.num_scenarios} drug-drug interaction scenarios...")

    for i in range(1, args.num_scenarios + 1):
        scenario = generate_interaction(i)
        scenarios.append(scenario)
        instruction_data.append(generate_instruction_response(scenario))

        if i % 5000 == 0:
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

    # Splits
    eval_split = int(args.num_scenarios * 0.1)
    test_split = int(args.num_scenarios * 0.05)
    train_data = instruction_data[: -(eval_split + test_split)]
    eval_data = instruction_data[-(eval_split + test_split) : -test_split]
    test_data = instruction_data[-test_split:]

    for split_name, data in [("train", train_data), ("eval", eval_data), ("test", test_data)]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {split_name} split ({len(data)} samples) to {path}")

    print("\nDataset generation complete!")
    print(f"Total scenarios: {len(scenarios)}")
    print(f"Total instruction pairs: {len(instruction_data)}")


if __name__ == "__main__":
    main()
