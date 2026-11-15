#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
strata_dir="$repo_root/edge0-strata"
model_dir="${EDGE0_MODEL_DIR:-/mnt/ai-models/edge0/Edge0-120B-q3k}"
rocm_root="${EDGE0_ROCM_ROOT:-/opt/rocm/current}"
context="${EDGE0_CONTEXT:-16384}"
host="${EDGE0_HOST:-127.0.0.1}"
port="${EDGE0_PORT:-8085}"
config="$strata_dir/strata-edge0-120b-hip.json"

python3 "$repo_root/scripts/make-edge0-120b-strix-halo-config.py" \
  --repo-root "$repo_root" \
  --model-dir "$model_dir" \
  --rocm-root "$rocm_root" \
  --context "$context" \
  --output "$config" >/dev/null

export LD_LIBRARY_PATH="$rocm_root/lib:$rocm_root/lib/rocm_sysdeps/lib:$rocm_root/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "$strata_dir/.venv/bin/python" "$strata_dir/serve/server.py" \
  --engine strata \
  --config "$config" \
  --host "$host" \
  --port "$port"
