#!/usr/bin/env bash
# Stops the RAGFlow processes started by wsl_start_ragflow.sh (backend +
# frontend). Base services (MySQL/Redis/MinIO/Elasticsearch) are left
# running since they are systemd-managed and shared with other tools; stop
# them with `sudo systemctl stop mysql redis-server minio elasticsearch` if
# you really want to free the RAM.
set -uo pipefail

for pattern in "rag/svr/task_executor.py" "api/ragflow_server.py" "vite --host"; do
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    echo "[stop] $pattern"
    pkill -f "$pattern"
  else
    echo "[skip] $pattern not running"
  fi
done
