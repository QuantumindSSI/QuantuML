#!/bin/bash
# One-time setup: GitHub Actions OIDC federation to Azure (no stored secrets).
# Creates an app registration + service principal, grants Contributor on the
# resource group, and adds a federated credential for the repo's main branch
# and production environment.
#
# Usage: ./infra/setup_github_oidc.sh <resource-group> <github-org/repo>
set -euo pipefail

RG="${1:?usage: setup_github_oidc.sh <resource-group> <github-org/repo>}"
REPO="${2:?usage: setup_github_oidc.sh <resource-group> <github-org/repo>}"

APP_NAME="github-oidc-quantuml"
SUB_ID=$(az account show --query id -o tsv)
TENANT_ID=$(az account show --query tenantId -o tsv)

echo "==> Creating app registration ${APP_NAME}"
APP_ID=$(az ad app create --display-name "${APP_NAME}" --query appId -o tsv)
az ad sp create --id "${APP_ID}" -o none 2>/dev/null || true

echo "==> Granting Contributor on ${RG}"
az role assignment create \
  --assignee "${APP_ID}" \
  --role Contributor \
  --scope "/subscriptions/${SUB_ID}/resourceGroups/${RG}" \
  -o none

echo "==> Adding federated credentials for ${REPO}"
az ad app federated-credential create --id "${APP_ID}" --parameters "{
  \"name\": \"github-main\",
  \"issuer\": \"https://token.actions.githubusercontent.com\",
  \"subject\": \"repo:${REPO}:ref:refs/heads/main\",
  \"audiences\": [\"api://AzureADTokenExchange\"]
}" -o none

az ad app federated-credential create --id "${APP_ID}" --parameters "{
  \"name\": \"github-env-production\",
  \"issuer\": \"https://token.actions.githubusercontent.com\",
  \"subject\": \"repo:${REPO}:environment:production\",
  \"audiences\": [\"api://AzureADTokenExchange\"]
}" -o none

echo ""
echo "Set these as GitHub repository variables (Settings > Secrets and variables > Actions > Variables):"
echo "  AZURE_CLIENT_ID=${APP_ID}"
echo "  AZURE_TENANT_ID=${TENANT_ID}"
echo "  AZURE_SUBSCRIPTION_ID=${SUB_ID}"
