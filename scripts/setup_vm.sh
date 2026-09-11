#!/bin/bash
# ============================================================
# Fresh-VM setup for running the auto_memory SWE-bench-Verified
# eval agent (repo: minions, task: swebenchverified).
#
# Run this on a freshly created VM (e.g. from create_azurelinux_vm.sh)
# after SSH-ing in. It:
#   1. Installs system prerequisites (git, python3, pip, venv, Docker).
#   2. Clones the minions repo.
#   3. Creates a Python virtualenv and installs microbots[training].
#   4. Writes a .env with the model-provider auth vars (edit before running).
#   5. Creates the workdir and drops in task_config.yaml.
#
# Usage:
#   WORKDIR=$HOME/workdirs/pytest \
#   TASK_CONFIG_SRC=/path/to/task_config.yaml \
#   ./setup_vm.sh
#
# Adjust REPO_URL / BRANCH / WORKDIR / PYTHON_BIN / MODEL as needed.
# ============================================================

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/microbots}"
WORKDIR="${WORKDIR:-$HOME/workdir}"
MODEL="${MODEL:-azure-openai/gpt-6-astra}"
# Path to a task_config.yaml prepared beforehand (e.g. copied via scp).
# If unset, an empty workdir is created and you must place the file yourself.
TASK_CONFIG_SRC="${TASK_CONFIG_SRC:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "============================================"
echo "  Auto-memory eval agent VM setup"
echo "============================================"
echo "  Repo dir:       $REPO_DIR"
echo "  Workdir:        $WORKDIR"
echo "  Model:          $MODEL"
echo "  task_config.yaml source: ${TASK_CONFIG_SRC:-<none, create manually>}"
echo "============================================"

# --- Step 1: System prerequisites ---
echo "[1/6] Installing system prerequisites..."
if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y git python3 python3-venv python3-pip ca-certificates curl gnupg
elif command -v tdnf >/dev/null 2>&1; then
    # Azure Linux (Mariner)
    sudo tdnf install -y git python3 python3-pip python3-venv ca-certificates curl
else
    echo "  Unsupported package manager; install git/python3/pip/docker manually." >&2
fi

# --- Step 2: Docker ---
echo "[2/6] Ensuring Docker is installed and running..."
if ! command -v docker >/dev/null 2>&1; then
    if command -v tdnf >/dev/null 2>&1; then
        # Azure Linux (Mariner) ships Moby, not Docker CE; get.docker.com
        # doesn't support tdnf-based distros.
        sudo tdnf install -y moby-engine moby-cli
    elif command -v apt-get >/dev/null 2>&1; then
        curl -fsSL https://get.docker.com | sudo sh
    else
        echo "  Unsupported package manager; install Docker/Moby manually." >&2
    fi
fi
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER" || true

# Apply the new 'docker' group membership right away instead of requiring a
# re-login: re-exec this whole script under 'sg docker', which runs it in a
# shell that already has the group active. Every step above is idempotent
# (git fetch/checkout, package installs, docker install checks), so
# restarting from the top here is safe. Skipped if already a member (e.g.
# re-running the script) to avoid re-execing forever.
if ! id -nG "$USER" | grep -qw docker; then
    echo "  Applying 'docker' group membership now (re-exec via sg docker)..."
    exec sg docker "$0 $*"
fi


# --- Step 4: Virtualenv + install ---
echo "[4/6] Creating virtualenv and installing microbots[training]..."
cd "$REPO_DIR"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[training]"

# --- Step 5: Env vars for the model provider ---
echo "[5/6] Writing .env template (EDIT WITH REAL VALUES BEFORE RUNNING)..."
if [[ ! -f "$REPO_DIR/.env" ]]; then
    cat > "$REPO_DIR/.env" <<'EOF'
# Azure OpenAI (azure-openai/<deployment> models, e.g. gpt-6-astra, gpt-5.6-*)
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_VERSION=2025-03-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=
EOF
    echo "  Wrote $REPO_DIR/.env — fill in real values before running the CLI."
else
    echo "  $REPO_DIR/.env already exists; leaving as-is."
fi

# --- Step 6: Workdir + task_config.yaml ---
echo "[6/6] Setting up workdir..."
mkdir -p "$WORKDIR"
if [[ -n "$TASK_CONFIG_SRC" ]]; then
    cp "$TASK_CONFIG_SRC" "$WORKDIR/task_config.yaml"
    echo "  Copied task_config.yaml -> $WORKDIR/task_config.yaml"
else
    echo "  No TASK_CONFIG_SRC given; place task_config.yaml at $WORKDIR/task_config.yaml manually."
fi

echo ""
echo "Setup complete."
echo ""
echo "To run the eval agent:"
echo "  cd $REPO_DIR && source .venv/bin/activate && set -a && source .env && set +a"
echo "  python3 src/microbots/auto_memory/cli.py \\"
echo "    --model $MODEL \\"
echo "    --workdir $WORKDIR \\"
echo "    --task swebenchverified \\"
echo "    --max-rounds 5"
