#!/usr/bin/env bash
# Upload (or update) config.yaml in the GCS bucket. Because the app re-reads
# config from GCS (TTL-cached ~60s), edits apply WITHOUT a redeploy or restart.
#
# Usage: ./scripts/upload_config.sh path/to/config.yaml
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/deploy.env"

CONFIG_FILE="${1:-config.yaml}"
OBJECT_PATH="${GCS_CONFIG_PATH:-config.yaml}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

echo ">> Uploading ${CONFIG_FILE} -> gs://${GCS_BUCKET_NAME}/${OBJECT_PATH}"
gcloud storage cp "${CONFIG_FILE}" "gs://${GCS_BUCKET_NAME}/${OBJECT_PATH}"
echo ">> Done. Changes take effect within ~60s (CONFIG_TTL_SECONDS)."
