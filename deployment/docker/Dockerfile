# QuantuML model server image.
# One image serves pqc | crypto | ddi, selected by the TASK env var at runtime.
# Base model is baked in at build time for immutable, registry-independent startup;
# LoRA adapters are pulled at startup pinned by ADAPTER_REVISION.
#
# Build (cloud-side, no local docker needed):
#   az acr build -r <acr> -t quantuml/model-server:<tag> \
#     -f deployment/docker/Dockerfile .

FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

COPY deployment/docker/requirements-serve.txt /app/requirements-serve.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r /app/requirements-serve.txt

# Bake the base model into the image (public repo, no token required).
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2-0.5B-Instruct', local_dir='/models/base', \
    allow_patterns=['*.json', '*.safetensors', 'merges.txt', 'vocab.json', 'tokenizer*'])"

COPY deployment/api/serve.py /app/serve.py

# Non-root runtime user; HF cache must be writable for adapter downloads.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser /models
USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "serve:app", "--app-dir", "/app", "--host", "0.0.0.0", "--port", "8000"]
