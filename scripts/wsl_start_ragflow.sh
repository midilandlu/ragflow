#!/usr/bin/env bash
# One-click RAGFlow dev startup for WSL2 without Docker.
#
# Background: some environments block Docker Desktop by IT policy. This script
# runs RAGFlow's four base services (MySQL, Redis, MinIO, Elasticsearch) as
# native systemd services inside WSL2, and runs the Python backend and Vite
# frontend directly, with no containers involved.
#
# One-time setup (native package installs, systemd units, etc.) is documented
# in docs/develop/launch_ragflow_from_source_wsl2_no_docker.md and must be
# done once per WSL2 instance before this script will work. This script only
# handles the repeatable "start everything" step.
#
# Usage (run inside WSL2, from any directory):
#   bash scripts/wsl_start_ragflow.sh
#
# Safe to re-run: dependencies (Python + npm) and the frontend source mirror
# are always re-synced (cheap no-ops when nothing changed), but an
# already-running backend/frontend process is left alone -- it won't pick up
# new *backend* code or dependency changes until restarted. Use
# wsl_restart_ragflow.sh after a `git pull`/merge, or after editing backend
# Python code, to force that restart. Frontend edits hot-reload live and
# don't need a restart.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# node_modules on a Windows drive mounted into WSL2 (DrvFs, i.e. any path
# under /mnt/*) is extremely slow / can hang npm install and uv sync (dropped
# hardlinks, EPERM on metadata copies). The venv and the web/ frontend are
# therefore kept on WSL2's native ext4 filesystem, under $HOME, and only the
# repo checkout itself (source files) stays on the Windows drive. Re-run this
# script any time you change files under web/ so the native mirror picks up
# the change; the Python backend reads $REPO_ROOT directly so it needs no
# such mirroring.
VENV_DIR="$HOME/.venvs/ragflow"
WEB_NATIVE_DIR="$HOME/ragflow-web"
LOG_DIR="$REPO_ROOT/logs/wsl-dev"
mkdir -p "$LOG_DIR"

echo "== repo root : $REPO_ROOT"
echo "== venv      : $VENV_DIR"
echo "== web (native mirror): $WEB_NATIVE_DIR"
echo

# ---------------------------------------------------------------------------
# 1. Base services (MySQL / Redis / MinIO / Elasticsearch)
# ---------------------------------------------------------------------------
for svc in mysql redis-server minio elasticsearch; do
  if systemctl is-active --quiet "$svc" 2>/dev/null; then
    echo "[ok]    $svc already running"
  else
    echo "[start] $svc"
    sudo systemctl start "$svc"
  fi
done

echo -n "[wait]  elasticsearch responding"
for _ in $(seq 1 30); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:9200 || true)
  if [ "$code" = "200" ] || [ "$code" = "401" ]; then
    echo " - ok"
    break
  fi
  echo -n "."
  sleep 2
done

# ---------------------------------------------------------------------------
# 2. VERSION file. common/versions.py reads a VERSION file at repo root if
# present, only falling back to `git describe` when it's missing -- this
# mirrors the official Docker build, which bakes VERSION in at image build
# time (`echo "$version_info" > /ragflow/VERSION` in the Dockerfile) rather
# than shelling out to git at runtime.
#
# We regenerate it here instead of relying on the runtime fallback because a
# git worktree's `.git` file stores an absolute `gitdir:` path, and this
# repo's worktree was created from the Windows side, so that path is
# Windows-style (`D:/ragflow/.git/worktrees/...`). WSL2's own git can't
# resolve that ("fatal: not a git repository") and get_ragflow_version()
# silently falls back to "unknown". Windows git (`git.exe`, reachable here
# via WSL2 interop) resolves the same path fine, so use that instead.
# ---------------------------------------------------------------------------
echo "[sync]  VERSION file"
(
  cd "$REPO_ROOT"
  GIT_BIN="git"
  command -v git.exe >/dev/null 2>&1 && GIT_BIN="git.exe"
  version_info="$("$GIT_BIN" describe --tags --match='v*' --first-parent --always 2>/dev/null || echo unknown)"
  echo "$version_info" > "$REPO_ROOT/VERSION"
)

# ---------------------------------------------------------------------------
# 3. Python venv (native fs). Always re-synced: after `git pull` /
# `git merge upstream/main` changes pyproject.toml or uv.lock, this is what
# picks up new/updated dependencies. `uv sync` is a no-op (a couple seconds)
# when nothing changed, so there's no real cost to always running it.
# ---------------------------------------------------------------------------
UV_BIN="$(command -v uv || echo "$HOME/.local/bin/uv")"
[ -x "$UV_BIN" ] || pipx install uv
echo "[sync]  uv sync (python deps)"
( cd "$REPO_ROOT" && UV_PROJECT_ENVIRONMENT="$VENV_DIR" "$UV_BIN" sync --python 3.13 --frozen )

# ---------------------------------------------------------------------------
# 4. Backend: task_executor + ragflow_server
# ---------------------------------------------------------------------------
if pgrep -f "rag/svr/task_executor.py" >/dev/null 2>&1; then
  echo "[ok]    task_executor already running"
else
  echo "[start] task_executor -> $LOG_DIR/task_executor.log"
  (
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT"
    export JEMALLOC_PATH
    JEMALLOC_PATH="$(pkg-config --variable=libdir jemalloc)/libjemalloc.so"
    export LD_PRELOAD="$JEMALLOC_PATH"
    nohup "$VENV_DIR/bin/python" rag/svr/task_executor.py -i 1 \
      > "$LOG_DIR/task_executor.log" 2>&1 &
    disown
  )
fi

if pgrep -f "api/ragflow_server.py" >/dev/null 2>&1; then
  echo "[ok]    ragflow_server already running"
else
  echo "[start] ragflow_server -> $LOG_DIR/ragflow_server.log"
  (
    cd "$REPO_ROOT"
    export PYTHONPATH="$REPO_ROOT"
    nohup "$VENV_DIR/bin/python" api/ragflow_server.py \
      > "$LOG_DIR/ragflow_server.log" 2>&1 &
    disown
  )
fi

# ---------------------------------------------------------------------------
# 5. Frontend: mirror web/ to native fs, then npm install.
#
# The mirror step always runs, so source edits on the Windows side (git pulls
# or your own local edits) reach the native copy every time. If the dev
# server below is already running, Vite's file watcher (native fs, so real
# inotify events work) picks up the changed files and hot-reloads the page
# automatically -- no restart needed for plain frontend source edits.
#
# `npm install` always runs too so a `package.json`/`package-lock.json`
# change from a pull is picked up; like uv sync, it's a quick no-op when the
# lockfile hasn't changed.
# ---------------------------------------------------------------------------
mkdir -p "$WEB_NATIVE_DIR"
rsync -a --delete --exclude node_modules --exclude dist --exclude .git \
  "$REPO_ROOT/web/" "$WEB_NATIVE_DIR/"

echo "[sync]  npm install (frontend deps)"
( cd "$WEB_NATIVE_DIR" && npm install )

if pgrep -f "vite --host" >/dev/null 2>&1; then
  echo "[ok]    frontend dev server already running"
else
  echo "[start] frontend dev server -> $LOG_DIR/web_dev.log"
  (
    cd "$WEB_NATIVE_DIR"
    # This repo's web/.env.development defaults API_PROXY_SCHEME to 'go',
    # targeting the newer Go backend on :9384. We only start the Python
    # backend (task_executor.py / ragflow_server.py on :9380), so this must
    # be overridden to 'python' or every API call 404s against a server that
    # was never started.
    export API_PROXY_SCHEME=python
    nohup npm run dev > "$LOG_DIR/web_dev.log" 2>&1 &
    disown
  )
fi

sleep 2
echo
echo "================================================================"
echo " RAGFlow dev stack starting. Logs in: $LOG_DIR"
echo " Backend:  http://127.0.0.1:9380"
port="$(grep -oE 'Local:\s+http://localhost:[0-9]+' "$LOG_DIR/web_dev.log" 2>/dev/null | grep -oE '[0-9]+$' || echo '9222 (default, check web_dev.log)')"
echo " Frontend: http://localhost:${port}"
echo "================================================================"
