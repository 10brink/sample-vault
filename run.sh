#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PORT="${SAMPLE_VAULT_PORT:-8742}"
echo "Starting Sample Vault on http://127.0.0.1:${PORT} ..."
exec "$DIR/.venv/bin/python" "$DIR/cli.py" serve --port "$PORT" "$@"
