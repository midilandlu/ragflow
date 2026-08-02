#!/usr/bin/env bash
# Force a full restart of the RAGFlow backend + frontend dev processes,
# re-syncing Python/npm dependencies and the frontend source mirror first.
#
# Use this (instead of wsl_start_ragflow.sh) whenever:
#   - you pulled/merged updates from origin or upstream, OR
#   - you edited backend Python code yourself and need the change to take
#     effect (the backend has no auto-reload; frontend edits hot-reload on
#     their own and don't need this).
#
# Base services (MySQL/Redis/MinIO/Elasticsearch) are left running --
# nothing about a code update requires restarting those.
#
# Usage: bash scripts/wsl_restart_ragflow.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "== stopping backend/frontend =="
bash "$SCRIPT_DIR/wsl_stop_ragflow.sh"

# Give processes a moment to fully release their listening ports before we
# start new ones on the same ports.
sleep 2

echo
echo "== syncing deps and starting backend/frontend =="
bash "$SCRIPT_DIR/wsl_start_ragflow.sh"
