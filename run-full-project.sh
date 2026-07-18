#!/usr/bin/env bash
set -Eeuo pipefail

# Provision an H100 VM, execute the notebook remotely, download its artifacts,
# and stop the VM on every exit path. Run this script from any directory.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEBIUS_CLI="${NEBIUS_CLI:-$HOME/.nebius/bin/nebius}"
NEBIUS_PROFILE="${NEBIUS_PROFILE:-ddp-hw}"
NEBIUS_PROJECT_ID="${NEBIUS_PROJECT_ID:-project-e00sv4bvpr00wr5e5kr9xr}"
NEBIUS_SUBNET_ID="${NEBIUS_SUBNET_ID:-vpcsubnet-e00wjtjhajmze371gd}"
SSH_USER="${SSH_USER:-sapir}"
SSH_PRIVATE_KEY="${SSH_PRIVATE_KEY:-$HOME/.ssh/id_ed25519}"
SSH_PUBLIC_KEY_PATH="${SSH_PUBLIC_KEY_PATH:-${SSH_PRIVATE_KEY}.pub}"
REMOTE_DIR="${REMOTE_DIR:-/home/${SSH_USER}/Hometask_3_Quantization_and_Benchmarking}"
NOTEBOOK="quant_serving.ipynb"
INSTANCE_NAME="${INSTANCE_NAME:-hw3-quantization-$(date -u +%Y%m%d-%H%M%S)}"

INSTANCE_ID=""
PUBLIC_IP=""
TEMP_DIR="$(mktemp -d)"
KNOWN_HOSTS="${TEMP_DIR}/known_hosts"
REQUEST_JSON="${TEMP_DIR}/instance.json"

log() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

cleanup() {
  local exit_code=$?
  if [[ -n "$INSTANCE_ID" ]]; then
    log "Stopping Nebius VM ${INSTANCE_ID}"
    if ! "$NEBIUS_CLI" compute v1 instance stop "$INSTANCE_ID" \
      --profile "$NEBIUS_PROFILE" --format json >/dev/null; then
      printf 'WARNING: failed to stop VM %s; stop it in the Nebius console.\n' "$INSTANCE_ID" >&2
      exit_code=1
    fi
  fi
  rm -rf "$TEMP_DIR"
  trap - EXIT
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

for command in jq ssh ssh-keyscan rsync; do
  command -v "$command" >/dev/null || die "required command not found: ${command}"
done
[[ -x "$NEBIUS_CLI" ]] || die "Nebius CLI not executable: ${NEBIUS_CLI}"
[[ -f "$SSH_PRIVATE_KEY" ]] || die "SSH private key not found: ${SSH_PRIVATE_KEY}"
[[ -f "$SSH_PUBLIC_KEY_PATH" ]] || die "SSH public key not found: ${SSH_PUBLIC_KEY_PATH}"
[[ -f "${PROJECT_DIR}/${NOTEBOOK}" ]] || die "notebook not found: ${PROJECT_DIR}/${NOTEBOOK}"
[[ -f "${PROJECT_DIR}/requirements.txt" ]] || die "requirements.txt not found"

SSH_PUBLIC_KEY="$(tr -d '\r\n' < "$SSH_PUBLIC_KEY_PATH")"
CLOUD_INIT=$(printf '%s\n' \
  '#cloud-config' \
  'users:' \
  "  - name: ${SSH_USER}" \
  '    sudo: ALL=(ALL) NOPASSWD:ALL' \
  '    shell: /bin/bash' \
  '    ssh_authorized_keys:' \
  "      - ${SSH_PUBLIC_KEY}" \
  'package_update: true' \
  'packages:' \
  '  - python3-venv' \
  '  - python3-dev' \
  '  - build-essential' \
  '  - git' \
  '  - rsync')

jq -n \
  --arg project_id "$NEBIUS_PROJECT_ID" \
  --arg instance_name "$INSTANCE_NAME" \
  --arg disk_name "${INSTANCE_NAME}-disk" \
  --arg subnet_id "$NEBIUS_SUBNET_ID" \
  --arg cloud_init "$CLOUD_INIT" \
  '{
    metadata: {parent_id: $project_id, name: $instance_name},
    spec: {
      resources: {platform: "gpu-h100-sxm", preset: "1gpu-16vcpu-200gb"},
      boot_disk: {
        attach_mode: "READ_WRITE",
        managed_disk: {
          name: $disk_name,
          spec: {
            size_gibibytes: 300,
            type: "NETWORK_SSD",
            source_image_family: {
              image_family: "ubuntu24.04-cuda13.0",
              parent_id: "project-e00public-images"
            }
          }
        }
      },
      cloud_init_user_data: $cloud_init,
      network_interfaces: [{
        name: "eth0",
        subnet_id: $subnet_id,
        ip_address: {},
        public_ip_address: {}
      }],
      reservation_policy: {policy: "FORBID"}
    }
  }' > "$REQUEST_JSON"

log "Creating Nebius H100 VM ${INSTANCE_NAME}"
CREATE_RESPONSE="$($NEBIUS_CLI compute v1 instance create \
  --profile "$NEBIUS_PROFILE" --file "$REQUEST_JSON" --format json)"
INSTANCE_ID="$(jq -r '.metadata.id // empty' <<< "$CREATE_RESPONSE")"
[[ -n "$INSTANCE_ID" ]] || die "Nebius did not return an instance ID"
log "Created ${INSTANCE_ID}"

log "Waiting for a public IP"
for _ in $(seq 1 60); do
  INSTANCE_JSON="$($NEBIUS_CLI compute v1 instance get "$INSTANCE_ID" \
    --profile "$NEBIUS_PROFILE" --format json)"
  PUBLIC_IP="$(jq -r '.status.network_interfaces[0].public_ip_address.address // empty | split("/")[0]' <<< "$INSTANCE_JSON")"
  [[ -n "$PUBLIC_IP" ]] && break
  sleep 5
done
[[ -n "$PUBLIC_IP" ]] || die "VM did not receive a public IP"
log "VM public IP: ${PUBLIC_IP}"

log "Waiting for SSH"
SSH_READY=false
for _ in $(seq 1 90); do
  ssh-keyscan -T 5 -H "$PUBLIC_IP" > "$KNOWN_HOSTS" 2>/dev/null || true
  if [[ -s "$KNOWN_HOSTS" ]] && ssh -i "$SSH_PRIVATE_KEY" \
    -o UserKnownHostsFile="$KNOWN_HOSTS" -o StrictHostKeyChecking=yes \
    -o ConnectTimeout=5 "${SSH_USER}@${PUBLIC_IP}" true 2>/dev/null; then
    SSH_READY=true
    break
  fi
  sleep 5
done
[[ "$SSH_READY" == true ]] || die "SSH did not become ready"

SSH_OPTIONS=(-i "$SSH_PRIVATE_KEY" -o UserKnownHostsFile="$KNOWN_HOSTS" -o StrictHostKeyChecking=yes)
SSH_TARGET="${SSH_USER}@${PUBLIC_IP}"

log "Waiting for cloud-init"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" 'sudo cloud-init status --wait'

log "Uploading project"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "mkdir -p '$REMOTE_DIR'"
rsync -az --delete \
  --exclude '.git/' --exclude '.venv/' --exclude '.ipynb_checkpoints/' \
  -e "ssh -i '$SSH_PRIVATE_KEY' -o UserKnownHostsFile='$KNOWN_HOSTS' -o StrictHostKeyChecking=yes" \
  "${PROJECT_DIR}/" "${SSH_TARGET}:${REMOTE_DIR}/"

log "Creating remote virtual environment and installing requirements"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
  "cd '$REMOTE_DIR' && python3 -m venv .venv && .venv/bin/python -m pip install --upgrade pip && .venv/bin/python -m pip install -r requirements.txt && .venv/bin/python -m pip check"

log "Validating the H100"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
  "cd '$REMOTE_DIR' && .venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'"

log "Executing ${NOTEBOOK} in place"
ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" \
  "cd '$REMOTE_DIR' && if [[ -f .env ]]; then set -a; source .env; set +a; fi; source .venv/bin/activate; jupyter nbconvert --to notebook --execute --inplace '$NOTEBOOK' --ExecutePreprocessor.timeout=3600 --ExecutePreprocessor.kernel_name=python3"

log "Downloading executed notebook and results"
rsync -az \
  -e "ssh -i '$SSH_PRIVATE_KEY' -o UserKnownHostsFile='$KNOWN_HOSTS' -o StrictHostKeyChecking=yes" \
  "${SSH_TARGET}:${REMOTE_DIR}/${NOTEBOOK}" "${PROJECT_DIR}/${NOTEBOOK}"
rsync -az \
  -e "ssh -i '$SSH_PRIVATE_KEY' -o UserKnownHostsFile='$KNOWN_HOSTS' -o StrictHostKeyChecking=yes" \
  "${SSH_TARGET}:${REMOTE_DIR}/results/" "${PROJECT_DIR}/results/"

log "Project completed successfully; the VM will now be stopped"
