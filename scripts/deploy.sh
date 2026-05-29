#!/usr/bin/env bash
# Deploy NOBS to the VM from your local machine (the main deploy mechanism).
#
# Strategy: build the image LOCALLY for linux/amd64, ship it to the VM with
# `docker save | ssh docker load` (no registry needed, no build on the 1 GB
# e2-micro), then start it with docker compose using the pre-loaded image.
#
# Only docker-compose.yml + .env (+ optional secrets) are copied to the VM —
# never the source or Dockerfile, since nothing is built there.
#
# Prereqs: provision_vm.sh has run; a local .env exists (see .env.example);
# local Docker with buildx (Docker Desktop has it by default).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
source "${HERE}/deploy.env"

IMAGE="${IMAGE:-nobs:latest}"
PLATFORM="${PLATFORM:-linux/amd64}"   # e2-micro is amd64; your Mac may be arm64
REMOTE_DIR="/opt/nobs"
SSH_BASE=(gcloud compute ssh "${VM_NAME}" --zone "${GCP_ZONE}" --tunnel-through-iap)

if [[ ! -f "${ROOT}/.env" ]]; then
  echo "ERROR: ${ROOT}/.env not found. Copy .env.example and fill it in." >&2
  exit 1
fi

echo ">> Building image locally for ${PLATFORM}: ${IMAGE}"
docker buildx build --platform "${PLATFORM}" -t "${IMAGE}" --load "${ROOT}"

echo ">> Preparing remote directory ${REMOTE_DIR}"
"${SSH_BASE[@]}" --command "sudo mkdir -p ${REMOTE_DIR} && sudo chown \$USER ${REMOTE_DIR}"

echo ">> Shipping image to VM (docker save | docker load)"
docker save "${IMAGE}" | gzip \
  | "${SSH_BASE[@]}" --command "gunzip | sudo docker load"

echo ">> Copying compose file, .env (and secrets, if present) to VM"
# Small files: tar them through the same SSH transport.
tar -czf - -C "${ROOT}" \
    docker-compose.yml .env \
    $( [[ -d "${ROOT}/secrets" ]] && echo secrets ) \
  | "${SSH_BASE[@]}" --command "tar -xzf - -C ${REMOTE_DIR}"

echo ">> Starting container from the pre-loaded image (no build on VM)"
"${SSH_BASE[@]}" --command "cd ${REMOTE_DIR} && sudo docker compose up -d && sudo docker compose ps"

echo ">> Pruning dangling images to keep the small (free-tier) disk lean"
"${SSH_BASE[@]}" --command "sudo docker image prune -f"

echo ">> Recent logs:"
"${SSH_BASE[@]}" --command "cd ${REMOTE_DIR} && sudo docker compose logs --tail 30"

echo ">> Deployed. Tail logs with:"
echo "   gcloud compute ssh ${VM_NAME} --zone ${GCP_ZONE} --command 'cd ${REMOTE_DIR} && sudo docker compose logs -f'"
