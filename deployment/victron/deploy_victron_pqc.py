#!/usr/bin/env python3
"""
Victron Edge Deployment Script for Post-Quantum Key Migration Advisor
Quantizes, exports to ONNX, and packages for edge hardware.
"""

import os
import yaml
import argparse
import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def quantize_int8(model, tokenizer, output_dir: Path):
    """Export model with INT8 dynamic quantization for CPU inference."""
    print("Applying INT8 dynamic quantization...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    quantized_model.config.use_cache = False

    save_path = output_dir / "int8"
    save_path.mkdir(parents=True, exist_ok=True)
    quantized_model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Saved INT8 model to {save_path}")
    return str(save_path)


def quantize_int4(model, tokenizer, output_dir: Path):
    """Export model with 4-bit NF4 quantization using bitsandbytes."""
    print("Applying INT4 NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Save config for load-time quantization
    save_path = output_dir / "int4"
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))

    # Save quantization config
    with open(save_path / "quantization_config.json", "w") as f:
        json.dump(bnb_config.to_dict(), f, indent=2)

    print(f"Saved INT4 config to {save_path}")
    return str(save_path)


def export_onnx(model, tokenizer, output_dir: Path, max_length: int = 4096):
    """Export to ONNX for optimized CPU inference."""
    try:
        from optimum.onnxruntime import ORTModelForCausalLM
    except ImportError:
        print("optimum[onnxruntime] not installed. Skipping ONNX export.")
        print("Install with: pip install optimum[onnxruntime]")
        return None

    print("Exporting to ONNX...")
    save_path = output_dir / "onnx"
    save_path.mkdir(parents=True, exist_ok=True)

    ort_model = ORTModelForCausalLM.from_pretrained(
        model,
        export=True,
        provider="CPUExecutionProvider",
    )
    ort_model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Saved ONNX model to {save_path}")
    return str(save_path)


def create_victron_package(model_dir: str, output_dir: Path, cfg: dict):
    """Create deployment package for Victron hardware."""
    pkg_dir = output_dir / "victron_package"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # Copy model
    import shutil
    shutil.copytree(model_dir, pkg_dir / "model", dirs_exist_ok=True)

    # Create startup script
    startup_script = pkg_dir / "start.sh"
    with open(startup_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Victron Edge Inference Server\n")
        f.write("cd \"$(dirname \"$0\")\"\n")
        f.write("python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000\n")
    os.chmod(startup_script, 0o755)

    # Create systemd service
    service_file = pkg_dir / "pqc-advisor.service"
    with open(service_file, "w") as f:
        f.write("""[Unit]
Description=Post-Quantum Key Migration Advisor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/pqc-advisor
ExecStart=/opt/pqc-advisor/start.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
""")

    # Create manifest
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


def benchmark_edge(model_path: str, tokenizer, num_runs: int = 10):
    """Benchmark inference on CPU."""
    print("Running edge benchmark...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    prompt = (
        "You are a Post-Quantum Cryptography Migration Advisor. "
        "A financial services firm needs to migrate their TLS from ECDSA P-256 to ML-DSA-65. "
        "Provide a migration plan."
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    import time
    latencies = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=100)
        latencies.append(time.time() - start)

    avg_latency = sum(latencies) / len(latencies)
    print(f"Average CPU latency: {avg_latency:.3f}s")

    del model
    return avg_latency


def main():
    parser = argparse.ArgumentParser(description="Deploy to Victron Edge")
    parser.add_argument("--config", type=str, default="./configs/config.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./deployment/victron")
    parser.add_argument("--quantize", type=str, default="int8", choices=["int8", "int4", "none"])
    parser.add_argument("--export_onnx", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model"]["base_model"],
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=getattr(torch, cfg["model"]["torch_dtype"]),
        device_map="cpu",
        trust_remote_code=cfg["model"].get("trust_remote_code", False),
    )

    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    model.eval()

    # Quantization
    final_model_dir = args.model_path
    if args.quantize == "int8":
        final_model_dir = quantize_int8(model, tokenizer, output_dir)
    elif args.quantize == "int4":
        final_model_dir = quantize_int4(model, tokenizer, output_dir)

    # ONNX export
    if args.export_onnx:
        onnx_dir = export_onnx(model, tokenizer, output_dir)
        if onnx_dir:
            final_model_dir = onnx_dir

    # Benchmark
    if args.benchmark:
        benchmark_edge(final_model_dir, tokenizer)

    # Package
    pkg_dir = create_victron_package(final_model_dir, output_dir, cfg)

    print(f"\nVictron deployment package ready at: {pkg_dir}")
    print("To deploy:")
    print(f"  1. Copy {pkg_dir} to Victron device /opt/pqc-advisor")
    print(f"  2. Install: sudo cp pqc-advisor.service /etc/systemd/system/")
    print(f"  3. Enable: sudo systemctl enable --now pqc-advisor")


if __name__ == "__main__":
    main()
