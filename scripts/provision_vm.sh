#!/usr/bin/env bash
# Provision the GCP "Always Free" infrastructure for NOBS:
#   * a regional GCS bucket for config.yaml
#   * ONE e2-micro VM (free shape) on a small standard PD, Docker + swap
#
# Stays within the GCP Always Free tier as long as deploy.env keeps:
#   - GCP_ZONE in us-west1 / us-central1 / us-east1
#   - MACHINE_TYPE=e2-micro, DISK_TYPE=pd-standard, DISK_SIZE<=30GB
#
# Idempotent-ish: safe to re-run; existing resources are left in place.
#
# Prereqs: gcloud CLI authenticated (`gcloud auth login`) and a billing project.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/deploy.env"

DISK_TYPE="${DISK_TYPE:-pd-standard}"
DISK_SIZE="${DISK_SIZE:-10GB}"

# Warn (don't block) if the chosen region/shape/disk would fall outside Always Free.
REGION="${GCP_ZONE%-*}"
case "${REGION}" in
  us-west1 | us-central1 | us-east1) ;;
  *) echo "WARNING: region '${REGION}' is NOT free-tier eligible (use us-west1/us-central1/us-east1)." >&2 ;;
esac
[[ "${MACHINE_TYPE}" == "e2-micro" ]] || echo "WARNING: '${MACHINE_TYPE}' is not the free e2-micro shape." >&2
[[ "${DISK_TYPE}" == "pd-standard" ]] || echo "WARNING: disk type '${DISK_TYPE}' is not free (use pd-standard)." >&2

echo ">> Setting project: ${GCP_PROJECT}"
gcloud config set project "${GCP_PROJECT}" >/dev/null

echo ">> Enabling required APIs"
gcloud services enable compute.googleapis.com storage.googleapis.com >/dev/null

echo ">> Ensuring config bucket: gs://${GCS_BUCKET_NAME}"
if ! gcloud storage buckets describe "gs://${GCS_BUCKET_NAME}" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${GCS_BUCKET_NAME}" \
    --location="${GCP_ZONE%-*}" --uniform-bucket-level-access
else
  echo "   bucket already exists"
fi

# Startup script (first boot): add swap as a safety net for the 1 GB instance,
# then install Docker. The image is built locally and loaded, so the VM never
# builds — but swap keeps the runtime stable under memory spikes. Idempotent.
STARTUP_SCRIPT="$(cat <<'EOS'
#!/usr/bin/env bash
set -e
# 1 GB swap file: cheap insurance against runtime OOM on the 1 GB instance.
if [ ! -f /swapfile ]; then
  fallocate -l 1G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=1024
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
  systemctl enable --now docker
fi
EOS
)"

echo ">> Ensuring VM: ${VM_NAME} (${MACHINE_TYPE})"
SA_FLAG=()
if [[ -n "${SERVICE_ACCOUNT}" ]]; then
  SA_FLAG=(--service-account="${SERVICE_ACCOUNT}")
fi

if ! gcloud compute instances describe "${VM_NAME}" --zone "${GCP_ZONE}" >/dev/null 2>&1; then
  gcloud compute instances create "${VM_NAME}" \
    --zone="${GCP_ZONE}" \
    --machine-type="${MACHINE_TYPE}" \
    --image-family=debian-12 \
    --image-project=debian-cloud \
    --boot-disk-type="${DISK_TYPE}" \
    --boot-disk-size="${DISK_SIZE}" \
    --scopes=storage-ro \
    "${SA_FLAG[@]}" \
    --metadata=startup-script="${STARTUP_SCRIPT}"
else
  echo "   VM already exists"
fi

echo ">> Done. Next:"
echo "   1) Upload config:  ./scripts/upload_config.sh config.yaml"
echo "   2) Deploy the app: ./scripts/deploy.sh"
