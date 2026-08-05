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
    # pkill only sends SIGTERM and returns immediately -- it does not wait
    # for the process to actually exit. wsl_restart_ragflow.sh used to
    # follow this script with a fixed `sleep 2` before calling
    # wsl_start_ragflow.sh, but a process still mid-shutdown at that point
    # still matches wsl_start_ragflow.sh's "already running?" pgrep check,
    # so it gets skipped instead of restarted -- then it finishes dying a
    # moment later, leaving nothing running at all. Seen in practice
    # 2026-08-05. Wait here instead so callers can rely on "stopped" really
    # meaning stopped.
    for _ in $(seq 1 30); do
      pgrep -f "$pattern" >/dev/null 2>&1 || break
      sleep 0.5
    done
    if pgrep -f "$pattern" >/dev/null 2>&1; then
      echo "[force] $pattern didn't exit after SIGTERM, sending SIGKILL"
      pkill -9 -f "$pattern"
    fi
  else
    echo "[skip] $pattern not running"
  fi
done
