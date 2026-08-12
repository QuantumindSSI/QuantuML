# QuantuML MLOps — Deployment State & Runbook

Last updated: 2026-08-12

## 1. Live Infrastructure (Azure, `rg-quantuml`, eastus)

| Resource | Name | Notes |
|---|---|---|
| AKS | `aks-quantuml` | 1x `Standard_D4s_v7` (autoscale 1–3), Azure Linux, workload identity, Container Insights |
| ACR | `acrquantumlcgd5tlzu2gf5y` | `quantuml/model-server` images; AcrPull granted to kubelet identity |
| Log Analytics | `log-quantuml` | 30-day retention |
| App Insights | `appi-quantuml` | Workspace-based; OTel export from pods |
| Alerts | `alert-quantuml-failed-requests`, `alert-quantuml-node-cpu` | Email to admin@adminquantumindssi.onmicrosoft.com |

Provisioning is fully described by `infra/main.bicep` (idempotent redeploy: `./infra/deploy.sh rg-quantuml eastus <alert-email>`).

## 2. Serving

- **Public ingress**: `http://48.206.254.82` (managed nginx, `webapprouting.kubernetes.azure.com`)
- **Routes**: `/pqc`, `/crypto`, `/ddi` → one Deployment per model (kustomize overlays in `deployment/k8s/overlays/`)
- **Image**: single parameterized image (`TASK` env selects model); base model baked in, LoRA adapter pulled at startup from HF Hub pinned by `ADAPTER_REVISION` (HF Hub = model registry; promotion = revision bump)
- **Current tag**: `quantuml/model-server:v3`

### Auth
`POST /generate` requires `X-API-Key`. Key lives in:
- k8s secret `api-auth` (namespace `quantuml`), key `api-key`
- local `.env` as `QUANTUML_API_KEY` (git-ignored; never commit)

Probes (`/healthz`, `/readyz`) and `/metrics` are unauthenticated by design (kubelet/Prometheus).

```bash
curl -X POST http://48.206.254.82/crypto/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $QUANTUML_API_KEY" \
  -d '{"prompt": "Analyze TLS with ECDHE P-256 and RSA-2048 certs.", "max_tokens": 200}'
```

### Observability
- Prometheus metrics per pod: `quantuml_requests_total`, `quantuml_generate_latency_seconds`, `quantuml_tokens_generated_total`, `quantuml_inflight_requests`, `quantuml_model_info{adapter_revision=...}`
- App Insights: request traces via OpenTelemetry (`APPLICATIONINSIGHTS_CONNECTION_STRING` from secret `appinsights`)
- Container Insights: node/pod logs + metrics in Log Analytics

## 3. CI/CD

### GitHub Actions (operational)
- `ci.yml`: lint (ruff), compileall, kustomize render check, Bicep compile; Docker build on PRs
- `cd.yml`: on `main` pushes touching `deployment/**` or `infra/**` → OIDC login → `az acr build` → deploy overlays with pinned SHA tag → rollout gates → in-cluster smoke tests
- Auth: OIDC app `ffe2664e-c72d-4b22-b699-b24b52feba9b` (`github-oidc-quantuml`), Contributor on `rg-quantuml`
  - **Important**: federated credential subjects use GitHub's immutable format
    `repo:QuantumindSSI@277343853/QuantuML@1303812230:ref:refs/heads/main` and
    `...:environment:production`. If the repo is renamed/recreated, update both credentials.
- Repo variables (set): `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `AZURE_RG`, `ACR_NAME`, `AKS_NAME`

### Azure DevOps (pending manual step)
`azure-pipelines.yml` is complete. Requires (portal-only):
1. Create org at https://aex.dev.azure.com + project
2. Service connection `quantuml-azure` (workload identity federation, subscription scope)
3. Variable group `quantuml-infra`: `AZURE_RG`, `ACR_NAME`, `AKS_NAME`

## 4. Model lifecycle / promotion

1. Train adapter → push to HF Hub (`QuantumindSSI/<model>`) → note commit SHA
2. Set `ADAPTER_REVISION` (overlay env patch or `workflow_dispatch` input) → push → CD rolls out with `maxUnavailable: 0`
3. Rollback: `kubectl rollout undo deployment/model-server-<task> -n quantuml` or redeploy previous revision/tag

## 5. Datasets (real/synthetic provenance)

Manifests are committed; datasets regenerate via scripts.

| Dataset | Mix | Real source | License |
|---|---|---|---|
| `data/mlx_pqc_v3` (11,000) | 50/50 | Neura-parse (NIST FIPS 203/204/205, IR 8547 verified) | CC-BY-4.0 |
| `data/mlx_ddi_v3` (4,536) | 50/50 | openFDA drug labels | US-Gov public data |
| `data/mlx_crypto_v3` (4,130) | 50/50 | NVD CVEs (8 crypto CWEs) | US-Gov public data |

DDI-2013 corpus (CC-BY-NC-4.0) is **evaluation-only** — do not train commercial models on it.

Rebuild: `scripts/prepare_real_world_data.py` (download/convert) → `scripts/build_mixed_dataset.py` (dedup, mix, manifest).

## 6. Training

- Local (Apple Silicon): MLX-LM configs `configs/mlx_lora_pqc*.yaml`; adapters in `outputs/mlx_adapters/`; v2 = synthetic-only baseline, v3 = 50/50 mixed (A/B pending)
- Cloud GPU: **quota is 0 for all GPU families** (Sponsored subscription). Manual portal request needed: Quotas → Compute → `Standard NCASv3_T4 Family` (eastus) → 4 vCPUs. Then: `Standard_NC4as_T4_v3` spot + `src/train_lora.py` (or Unsloth) produces PEFT adapters compatible with the ONNX INT8 edge pipeline
- Zero-cost alternative: Kaggle (30 GPU-h/week) with the same mixed datasets

## 7. Cost controls

- Stop cluster when idle: `az aks stop -g rg-quantuml -n aks-quantuml` (restart: `az aks start ...`)
- Idle burn ≈ $5–7/day (1 node + LB + logs)

## 8. Known pending items

- [ ] Azure DevOps org + service connection (portal)
- [ ] T4 GPU quota portal request
- [ ] TLS on ingress (needs domain; cert-manager or approuting managed certs)
- [ ] A/B eval: v2 synthetic-only vs v3 mixed adapters
- [ ] `/metrics` exposure: currently public via ingress; restrict via ingress rule if needed
