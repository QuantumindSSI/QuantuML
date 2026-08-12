#!/bin/bash
# Provision QuantuML Azure infrastructure (idempotent).
# Usage: ./infra/deploy.sh <resource-group> <location> <alert-email>
set -euo pipefail

RG="${1:?usage: deploy.sh <resource-group> <location> <alert-email>}"
LOCATION="${2:?usage: deploy.sh <resource-group> <location> <alert-email>}"
ALERT_EMAIL="${3:?usage: deploy.sh <resource-group> <location> <alert-email>}"

echo "==> Creating resource group ${RG} in ${LOCATION}"
az group create -n "${RG}" -l "${LOCATION}" -o none

echo "==> Deploying Bicep template"
az deployment group create \
  -g "${RG}" \
  -f "$(dirname "$0")/main.bicep" \
  -p "$(dirname "$0")/main.parameters.json" \
  -p alertEmail="${ALERT_EMAIL}" \
  --query 'properties.outputs' -o json | tee "$(dirname "$0")/.last_outputs.json"

echo "==> Done. Outputs saved to infra/.last_outputs.json"
