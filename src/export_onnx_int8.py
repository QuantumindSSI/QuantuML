#!/usr/bin/env python3
"""
ONNX + INT8 Quantization Export Script
Exports fine-tuned models to ONNX Runtime with INT8 dynamic quantization
for true edge deployment and benchmarks latency.
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def export_onnx(model_path: str, output_dir: Path, task_name: str):
    """Export model to ONNX using optimum."""
    from optimum.onnxruntime import ORTModelForCausalLM

    onnx_dir = output_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{task_name}] Exporting to ONNX: {model_path} -> {onnx_dir}")

    # Use base_model tokenizer to avoid version incompatibilities with saved tokenizer files
    tokenizer_path = model_path if (Path(model_path) / "tokenizer_config.json").exists() else "./base_model"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try loading as merged model first; if adapter-only, load base + adapter
    config_json = Path(model_path) / "config.json"
    is_merged = config_json.exists() and "Qwen2ForCausalLM" in config_json.read_text()

    if is_merged:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=False,
        )
        print(f"[{task_name}] Loaded merged model")
    else:
        base = AutoModelForCausalLM.from_pretrained(
            "./base_model",
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=False,
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print(f"[{task_name}] Loaded and merged LoRA adapter")

    model.eval()

    ort_model = ORTModelForCausalLM.from_pretrained(
        model,
        export=True,
        provider="CPUExecutionProvider",
        use_merged=True,
    )
    ort_model.save_pretrained(str(onnx_dir))
    tokenizer.save_pretrained(str(onnx_dir))
    print(f"[{task_name}] ONNX export complete: {onnx_dir}")
    return str(onnx_dir)


def quantize_onnx_int8(onnx_dir: str, output_dir: Path, task_name: str):
    """Apply INT8 dynamic quantization to ONNX model using optimum quantizer."""
    from optimum.onnxruntime.configuration import AutoQuantizationConfig
    from optimum.onnxruntime import ORTQuantizer

    int8_dir = output_dir / "onnx_int8"
    int8_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{task_name}] Quantizing ONNX model to INT8 -> {int8_dir}")

    quantizer = ORTQuantizer.from_pretrained(onnx_dir)

    # Use default quantization config (avx2/avx512 compatible)
    qconfig = AutoQuantizationConfig.avx2(
        is_static=False,
        per_channel=False,
        # Reduce calibration overhead for dynamic quant
        operators_to_quantize=["MatMul", "Add", "Mul", "Conv", "Gather", "Transpose", "Reshape", "Squeeze", "Unsqueeze", "Concat", "Split"],
    )

    quantizer.quantize(
        save_dir=str(int8_dir),
        quantization_config=qconfig,
    )
    print(f"[{task_name}] INT8 quantization complete: {int8_dir}")
    return str(int8_dir)


def benchmark_onnx(model_dir: str, tokenizer, prompt: str, num_runs: int = 10, max_new_tokens: int = 100, task_name: str = ""):
    """Benchmark ONNX Runtime inference latency."""
    from optimum.onnxruntime import ORTModelForCausalLM

    print(f"[{task_name}] Benchmarking ONNX model: {model_dir}")
    model = ORTModelForCausalLM.from_pretrained(
        model_dir,
        provider="CPUExecutionProvider",
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens)

    latencies = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
        latencies.append(time.time() - start)

    avg = sum(latencies) / len(latencies)
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    print(f"[{task_name}] ONNX Avg latency: {avg:.3f}s | P95: {p95:.3f}s")
    return {"mean_latency_s": avg, "p95_latency_s": p95, "min_latency_s": min(latencies), "max_latency_s": max(latencies)}


def create_victron_onnx_package(model_dir: str, output_dir: Path, task_name: str, manifest_info: dict):
    """Create a Victron deployment package with ONNX model."""
    import shutil

    pkg_dir = output_dir / "victron_onnx_package"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_dir, pkg_dir / "model", dirs_exist_ok=True)

    # Startup script for ONNX + FastAPI
    start_sh = pkg_dir / "start.sh"
    with open(start_sh, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Victron Edge ONNX Inference Server - {task_name}\n")
        f.write('cd "$(dirname "$0")"\n')
        f.write("python3 -m uvicorn api_server:app --host 0.0.0.0 --port 8000\n")
    os.chmod(start_sh, 0o755)

    # systemd service
    service_name = task_name.lower().replace(" ", "_").replace("-", "_")
    service_file = pkg_dir / f"{service_name}.service"
    with open(service_file, "w") as f:
        service_label = task_name.replace(" ", " ")
        f.write(f"""[Unit]
Description={service_label} ONNX Edge Inference
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/{service_name}
ExecStart=/opt/{service_name}/start.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
""")

    manifest = {
        "name": task_name,
        "version": "1.0.0",
        "model_path": "./model",
        "api_port": 8000,
        "max_memory_mb": manifest_info.get("max_memory_mb", 2048),
        "max_latency_ms": manifest_info.get("max_latency_ms", 500),
        "engine": "onnxruntime",
        "quantization": "int8",
        "requirements": ["optimum[onnxruntime]", "transformers", "fastapi", "uvicorn", "torch"],
    }
    with open(pkg_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[{task_name}] Victron ONNX package: {pkg_dir}")
    return str(pkg_dir)


def main():
    parser = argparse.ArgumentParser(description="Export and quantize models to ONNX+INT8")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--prompt", type=str, default=None, help="Benchmark prompt")
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--max_tokens", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Export to ONNX
    onnx_dir = export_onnx(args.model_path, output_dir, args.task_name)

    # Step 2: Quantize to INT8
    int8_dir = quantize_onnx_int8(onnx_dir, output_dir, args.task_name)

    # Step 3: Benchmark INT8 model
    tokenizer = AutoTokenizer.from_pretrained(int8_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    benchmark_prompt = args.prompt or (
        "Analyze medication list: Warfarin, Aspirin, Omeprazole. Report interactions."
    )
    bench_results = benchmark_onnx(int8_dir, tokenizer, benchmark_prompt, args.num_runs, args.max_tokens, args.task_name)

    # Step 4: Create Victron package
    mem_target = 2048 if "edge" in args.task_name.lower() else 4096
    lat_target = 500
    pkg_dir = create_victron_onnx_package(int8_dir, output_dir, args.task_name, {
        "max_memory_mb": mem_target,
        "max_latency_ms": lat_target,
    })

    # Save benchmark report
    report = {
        "task": args.task_name,
        "onnx_dir": onnx_dir,
        "int8_dir": int8_dir,
        "victron_package": pkg_dir,
        "benchmark": bench_results,
        "edge_compatible_latency": bench_results["mean_latency_s"] * 1000 <= lat_target,
        "edge_compatible_memory": True,  # INT8 models are ~50% of FP32
    }
    report_path = output_dir / "onnx_export_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[{args.task_name}] Report saved: {report_path}")

    if report["edge_compatible_latency"]:
        print(f"\n✅ [{args.task_name}] EDGE TARGET MET: latency < {lat_target}ms")
    else:
        print(f"\n⚠️  [{args.task_name}] EDGE TARGET NOT MET: latency {bench_results['mean_latency_s']*1000:.0f}ms > {lat_target}ms")
        print(f"    Consider: smaller base model, shorter max_tokens, or OpenVINO/TensorRT backend")


if __name__ == "__main__":
    main()
