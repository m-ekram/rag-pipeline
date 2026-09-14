#!/usr/bin/env bash
# Provision an Amazon Linux 2023 EC2 instance and start the RAG service.
#
# From a fresh instance (as ec2-user):
#   curl -fsSL https://raw.githubusercontent.com/m-ekram/rag-pipeline/main/deploy/ec2/setup.sh -o setup.sh
#   OPENAI_API_KEY=sk-... bash setup.sh
#
# Or from an existing clone:  bash deploy/ec2/setup.sh
#
# Re-running is safe: it pulls, rebuilds, re-ingests only if there is no index,
# and restarts the container.
#
# Env overrides: REPO_URL, APP_DIR, HOST_PORT (default 80), OPENAI_API_KEY.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/m-ekram/rag-pipeline.git}"
HOST_PORT="${HOST_PORT:-80}"
CONTAINER_UID=10001

log() { printf '\n==> %s\n' "$*"; }

# Use the surrounding clone when run from inside one, else clone to ~/rag-pipeline.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || echo .)"
if [ -z "${APP_DIR:-}" ]; then
  if [ -f "$SCRIPT_DIR/../../docker-compose.yml" ]; then
    APP_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
  else
    APP_DIR="$HOME/rag-pipeline"
  fi
fi

log "Installing Docker, git, python3"
sudo dnf install -y -q docker git python3
sudo systemctl enable --now docker

if ! sudo docker compose version >/dev/null 2>&1; then
  log "Installing the Docker Compose plugin"
  sudo mkdir -p /usr/local/lib/docker/cli-plugins
  sudo curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
    "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)"
  sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
fi

# Small instances (t3.small, 2 GB) need headroom for the image build.
mem_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
if [ "$mem_kb" -lt 3500000 ] && ! swapon --show | grep -q .; then
  log "Adding 2 GB swap (instance has $((mem_kb / 1024)) MB RAM)"
  sudo fallocate -l 2G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile >/dev/null
  sudo swapon /swapfile
  echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

if [ -d "$APP_DIR/.git" ]; then
  log "Updating $APP_DIR"
  git -C "$APP_DIR" pull --ff-only
else
  log "Cloning $REPO_URL into $APP_DIR"
  git clone --depth 1 "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR"

if [ ! -f .env ]; then
  cp .env.example .env
fi
if ! grep -qE '^OPENAI_API_KEY=.+' .env; then
  key="${OPENAI_API_KEY:-}"
  if [ -z "$key" ] && [ -t 0 ]; then
    read -rsp "OpenAI API key: " key
    echo
  fi
  if [ -z "$key" ]; then
    echo "No OPENAI_API_KEY given. Add it to $APP_DIR/.env and re-run." >&2
    exit 1
  fi
  sed -i "s|^OPENAI_API_KEY=.*|OPENAI_API_KEY=${key}|" .env
  chmod 600 .env
fi

if [ -z "$(find data -type f ! -name .gitkeep -print -quit 2>/dev/null)" ]; then
  log "No documents in data/ - building the FastAPI docs corpus"
  python3 scripts/prepare_fastapi_docs.py
fi

# The container runs as uid $CONTAINER_UID; the bind-mounted index dir must be writable by it.
mkdir -p faiss_index
sudo chown -R "$CONTAINER_UID:$CONTAINER_UID" faiss_index

export HOST_PORT
log "Building the image"
sudo --preserve-env=HOST_PORT docker compose build

if [ ! -f faiss_index/index.faiss ]; then
  log "Building the index"
  sudo --preserve-env=HOST_PORT docker compose run --rm ingest
fi

log "Starting the service on port $HOST_PORT"
sudo --preserve-env=HOST_PORT docker compose up -d

for _ in $(seq 1 60); do
  if curl -fsS "http://localhost:${HOST_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done
curl -fsS "http://localhost:${HOST_PORT}/health" || { echo "Service did not become healthy; see: sudo docker compose logs api" >&2; exit 1; }

token="$(curl -fsS -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)"
public_ip="$(curl -fsS -H "X-aws-ec2-metadata-token: $token" http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)"
echo
echo "Up. Open http://${public_ip:-<instance-public-ip>}:${HOST_PORT}/ (the security group must allow inbound TCP ${HOST_PORT})."
