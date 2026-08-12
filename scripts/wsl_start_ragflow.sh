#!/usr/bin/env bash
# One-click RAGFlow dev startup for WSL2 without Docker.
#
# AGENTS (Claude Code / Codex / CI) READ THIS BEFORE RUNNING:
#   1. Run me from PowerShell, not Git Bash:
#        wsl -d Ubuntu-24.04 -- bash /mnt/d/.../scripts/wsl_start_ragflow.sh
#      Git Bash rewrites /mnt/c-style arguments into Windows paths.
#   2. Redirect my output to a file (`> /tmp/ragflow_start.log 2>&1`) and read
#      that file for progress. Do NOT pipe me through `tail`/`head` -- the pipe
#      buffers and you will see an empty log whether I am working or hung.
#   3. I take several minutes on a cold start (uv sync + npm install). Run me in
#      the background; do not poll.
#   4. I use `sudo -n` for the base services and fail loudly if a password is
#      needed. If you see that failure, a human must run the printed command
#      once in an interactive terminal -- no agent can answer a sudo prompt.
#   5. WSL2 reclaims the VM once the last process exits, which kills everything
#      I started. If you launched me from a one-shot `wsl -- bash ...`, verify
#      afterwards with healthz + a process list; do not trust my exit code.
#   Full write-ups: scripts/wsl_dev_README.md, "Non-obvious gotchas".
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
# `readlink -f` (not just `cd && pwd`) so this is the physical path even when
# invoked through the `~/ragflow` symlink -- otherwise it can't be compared
# against /proc/<pid>/cwd (also physical) or another invocation's REPO_ROOT
# below without false mismatches between "the same directory, two spellings".
REPO_ROOT="$(readlink -f "$SCRIPT_DIR/..")"

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
# 0. Worktree-alignment guard. This machine can have this same dev-tooling
# checked out in more than one git worktree (see wsl_dev_README.md's push-
# policy note) -- but only ONE worktree's backend/frontend can ever be "the"
# running dev stack at a time: fixed ports (9380/9222), one `~/ragflow`
# symlink, one WSL2 instance. Starting from a worktree other than whichever
# one already has processes running leaves the other instance running
# unnoticed while this worktree's edits silently don't show up in the
# browser -- this has actually happened (2026-08-03 and again 2026-08-05).
# Refuse instead of silently doing the wrong thing.
# ---------------------------------------------------------------------------
for proc_pattern in "rag/svr/task_executor.py" "api/ragflow_server.py"; do
  running_pid="$(pgrep -f "$proc_pattern" | head -1 || true)"
  if [ -n "$running_pid" ]; then
    running_root="$(readlink -f "/proc/$running_pid/cwd" 2>/dev/null || true)"
    if [ -n "$running_root" ] && [ "$running_root" != "$REPO_ROOT" ]; then
      echo "ERROR: a RAGFlow dev process is already running from a DIFFERENT worktree." >&2
      echo "  already running from  : $running_root (pid $running_pid, $proc_pattern)" >&2
      echo "  this script's worktree: $REPO_ROOT" >&2
      echo >&2
      echo "Starting here would leave that instance running while ~/ragflow and" >&2
      echo "wsl_dev_STATUS.md drift out of sync with what's actually serving requests." >&2
      echo "Stop it first (works from either worktree): bash scripts/wsl_stop_ragflow.sh" >&2
      exit 1
    fi
  fi
done

if [ -L "$HOME/ragflow" ]; then
  symlink_target="$(readlink -f "$HOME/ragflow" 2>/dev/null || true)"
  if [ "$symlink_target" != "$REPO_ROOT" ]; then
    echo "[fix]   ~/ragflow pointed at $symlink_target -- repointing to $REPO_ROOT"
    ln -sfn "$REPO_ROOT" "$HOME/ragflow"
  fi
elif [ ! -e "$HOME/ragflow" ]; then
  echo "[setup] creating ~/ragflow -> $REPO_ROOT"
  ln -sfn "$REPO_ROOT" "$HOME/ragflow"
fi

# ---------------------------------------------------------------------------
# 1. Base services (MySQL / Redis / MinIO / Elasticsearch)
# ---------------------------------------------------------------------------
for svc in mysql redis-server minio elasticsearch; do
  # These are all `enabled`, so on a cold WSL2 boot systemd is usually still
  # bringing them up when we get here. Give them a few seconds before deciding
  # they need starting -- otherwise we race the boot and needlessly hit sudo.
  for _ in $(seq 1 15); do
    systemctl is-active --quiet "$svc" 2>/dev/null && break
    sleep 1
  done

  if systemctl is-active --quiet "$svc" 2>/dev/null; then
    echo "[ok]    $svc already running"
    continue
  fi

  echo "[start] $svc"
  # -n (non-interactive): never block on a password prompt. A non-interactive
  # caller (CI, an agent session, `wsl -- bash script`) has no way to answer it
  # and would otherwise hang here forever with no output.
  if ! sudo -n systemctl start "$svc" 2>/dev/null; then
    echo "[fail]  cannot start $svc: sudo needs a password and this shell is non-interactive." >&2
    echo "        Run this once yourself in an interactive terminal:" >&2
    echo "          sudo systemctl start $svc" >&2
    exit 1
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
    # `disown` alone isn't enough here: `npm run dev` inherits the
    # launching pty's stdin ("stdio: inherit" for the vite child it
    # spawns), so when the wsl.exe session that ran this script exits and
    # the pty closes, the frontend process dies with it even though it was
    # nohup'd -- verified by watching it survive as long as the launching
    # session stayed open, then vanish the moment that session ended.
    # `setsid` gives it its own session (no controlling terminal) and
    # `</dev/null` detaches stdin outright, which is what actually keeps
    # it alive after this script's own session ends. The backend processes
    # above don't need this -- they're plain `python` execs with no
    # inherited-stdio child process, so nohup+disown is sufficient there.
    setsid nohup npm run dev < /dev/null > "$LOG_DIR/web_dev.log" 2>&1 &
    disown
  )
fi

sleep 2

# ---------------------------------------------------------------------------
# 6. Auto-record ground truth into wsl_dev_STATUS.md. The rest of that file
# ("Last updated", "Environment status" prose, "Known gaps") is hand-
# maintained and can drift from reality if whoever's working forgets to
# update it before handing off -- that's already happened once. This block
# can't drift: it's regenerated between fixed markers on every run and
# nothing else in the file is touched.
# ---------------------------------------------------------------------------
STATUS_FILE="$REPO_ROOT/scripts/wsl_dev_STATUS.md"
if [ -f "$STATUS_FILE" ]; then
  GIT_BIN="git"
  command -v git.exe >/dev/null 2>&1 && GIT_BIN="git.exe"
  branch="$(cd "$REPO_ROOT" && "$GIT_BIN" branch --show-current 2>/dev/null || echo unknown)"
  commit="$(cd "$REPO_ROOT" && "$GIT_BIN" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  te_pid="$(pgrep -f 'rag/svr/task_executor.py' | head -1 || true)"
  rs_pid="$(pgrep -f 'api/ragflow_server.py' | head -1 || true)"
  vite_pid="$(pgrep -f 'vite --host' | head -1 || true)"
  symlink_now="$(readlink -f "$HOME/ragflow" 2>/dev/null || echo '(missing)')"
  {
    echo "<!-- AUTO-STATUS:BEGIN (rewritten by wsl_start_ragflow.sh every run -- do not hand-edit this block, edit the prose sections below instead) -->"
    echo "**Auto-verified environment** (written by \`wsl_start_ragflow.sh\` itself, not hand-maintained -- trust this over the prose below if they ever disagree):"
    echo
    echo "- **When:** $(date -Iseconds)"
    echo "- **Worktree:** \`$REPO_ROOT\`"
    echo "- **Branch / commit:** \`$branch\` @ \`$commit\`"
    echo "- **~/ragflow symlink target:** \`$symlink_now\`"
    echo "- **task_executor.py:** $([ -n "$te_pid" ] && echo "running, pid $te_pid" || echo "not running")"
    echo "- **ragflow_server.py:** $([ -n "$rs_pid" ] && echo "running, pid $rs_pid" || echo "not running")"
    echo "- **vite dev server:** $([ -n "$vite_pid" ] && echo "running, pid $vite_pid" || echo "not running")"
    echo "<!-- AUTO-STATUS:END -->"
  } > "$LOG_DIR/.auto_status_block.md"
  awk '
    BEGIN { while ((getline line < ARGV[2]) > 0) { block = block line "\n" }; ARGV[2] = "" }
    /<!-- AUTO-STATUS:BEGIN/ { printf "%s", block; skip = 1; next }
    /<!-- AUTO-STATUS:END/ { skip = 0; next }
    skip { next }
    { print }
  ' "$STATUS_FILE" "$LOG_DIR/.auto_status_block.md" > "$STATUS_FILE.tmp" && mv "$STATUS_FILE.tmp" "$STATUS_FILE"
fi

echo
echo "================================================================"
echo " RAGFlow dev stack starting. Logs in: $LOG_DIR"
echo " Backend:  http://127.0.0.1:9380"
port="$(grep -oE 'Local:\s+http://localhost:[0-9]+' "$LOG_DIR/web_dev.log" 2>/dev/null | grep -oE '[0-9]+$' || echo '9222 (default, check web_dev.log)')"
echo " Frontend: http://localhost:${port}"
echo "================================================================"
