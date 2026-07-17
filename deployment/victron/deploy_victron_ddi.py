#!/usr/bin/env python3
"""
Victron Edge Deployment Script for Drug Interaction Predictor
Packages model for edge hardware deployment.
"""

import os
import yaml
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_victron_package(model_dir: str, output_dir: Path, cfg: dict):
    pkg_dir = output_dir / "victron_package"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copytree(model_dir, pkg_dir / "model", dirs_exist_ok=True)
    startup_script = pkg_dir / "start.sh"
    with open(startup_script, "w") as f:
        f.write("#!/bin/bash\n# Victron Edge Inference Server - Drug Interaction Predictor\ncd \"$(dirname \"$0\")\"\npython3 -m uvicorn api_server_ddi:app --host 0.0.0.0 --port 8000\n")
    os.chmod(startup_script, 0o755)
    service_file = pkg_dir / "ddi-predictor.service"
    with open(service_file, "w") as f:
        f.write("""[Unit]
Description=Drug Interaction Predictor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ddi-predictor
ExecStart=/opt/ddi-predictor/start.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
""")
    manifest = {
        "name": cfg["project"]["name"],
        "version": cfg["project"]["version"],
        "model_path": "./model",
        "api_port": 8000,
        "max_memory_mb": cfg["edge"]["max_memory_mb"],
        "max_latency_ms": cfg["edge"]["max_latency_ms"],
        "requirements": ["torch", "transformers", "fastapi", "uvicorn"],
    }
    with open(pkg_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Created Victron package at {pkg_dir}")
    return str(pkg_dir)


def main():
    parser = argparse.ArgumentParser(description="Deploy Drug Interaction Predictor to Victron Edge")
    parser.add_argument("--config", type=str, default="./configs/config_ddi.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./deployment/victron/ddi")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model"], trust_remote_code=cfg["model"].get("trust_remote_code", False))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]), device_map="cpu", trust_remote_code=cfg["model"].get("trust_remote_code", False))
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()
    model.eval()

    final_model_dir = args.model_path
    pkg_dir = create_victron_package(final_model_dir, output_dir, cfg)
    print(f"\nVictron deployment package ready at: {pkg_dir}")
    print("To deploy:")
    print(f"  1. Copy {pkg_dir} to Victron device /opt/ddi-predictor")
    print(f"  2. Install: sudo cp ddi-predictor.service /etc/systemd/system/")
    print(f"  3. Enable: sudo systemctl enable --now ddi-predictor")


if __name__ == "__main__":
    main()
