# Running RAGFlow from source on Windows without Docker (WSL2)

Handoff notes for anyone (human or AI coding agent) picking up local RAGFlow
development on a Windows machine where Docker Desktop is not available
(e.g. blocked by IT policy). This supplements the official guide at
[docs/develop/launch_ragflow_from_source.md](../docs/develop/launch_ragflow_from_source.md),
which assumes Docker is available for the four base services.

## TL;DR

```bash
# inside WSL2, from the repo root
bash scripts/wsl_start_ragflow.sh
```

Then open the URL it prints (frontend defaults to `http://localhost:9222`).
To stop the backend/frontend processes: `bash scripts/wsl_stop_ragflow.sh`.

The script is idempotent - safe to re-run any time (e.g. after a reboot, or
after pulling new frontend code) and it skips whatever is already running.

## Why this exists

Docker Desktop is blocked by corporate policy on this machine. WSL2 itself
(a built-in Windows feature, not third-party software) is not blocked, so
the approach is: run everything RAGFlow's docker-compose would normally
containerize as **native Linux services inside WSL2** instead - no
containers anywhere.

## One-time environment setup (per machine)

Do this once per WSL2 instance. The start script above only handles the
repeatable part.

1. **Install WSL2 + a distro.**
   - If `wsl --update` fails with `0x8024500c`, that's Windows Update/Store
     access being blocked by the same kind of policy that blocks Docker
     Desktop. Fix: download the standalone installer from
     `https://github.com/microsoft/WSL/releases` (the `wsl.<version>.x64.msi`
     asset) instead of relying on the Store-based update path.
   - `wsl --install -d Ubuntu-24.04` (installs to C: by default; see the
     `wsl --import` route in this repo's chat history / ask the person who
     set this up if you need it on another drive).

2. **Install the four base services natively, inside WSL2** (as root):
   - `apt-get install mysql-server redis-server` - both come with systemd
     units enabled out of the box.
   - MinIO: download the official Linux binary to `/usr/local/bin/minio`,
     create a `minio-user` system user and a systemd unit
     (`MINIO_ROOT_USER=rag_flow`, `MINIO_ROOT_PASSWORD=infini_rag_flow`,
     listening on `127.0.0.1:9000`).
   - Elasticsearch **8.11.3** (must match `docker/.env`'s `STACK_VERSION`):
     download the official `linux-x86_64` tarball to `/opt/elasticsearch`,
     run as a dedicated `elasticsearch` system user via a custom systemd
     unit. Required `elasticsearch.yml` settings (mirrors
     `docker/docker-compose-base.yml`'s `es01` service):
     ```yaml
     network.host: 127.0.0.1
     http.port: 9200
     discovery.type: single-node
     xpack.security.enabled: true
     xpack.security.http.ssl.enabled: false
     xpack.security.transport.ssl.enabled: false
     ```
     Also set `vm.max_map_count=262144` via `/etc/sysctl.d/`, and **cap the
     JVM heap** in `config/jvm.options.d/heap.options` (e.g. `-Xms2g` /
     `-Xmx2g`) - left on auto, ES grabs ~50% of WSL2's RAM allocation, which
     was 16GB on a 32GB host. Reset the `elastic` user's password to match
     `conf/service_conf.yaml`:
     `bin/elasticsearch-reset-password -u elastic -i -b --url http://127.0.0.1:9200`.
   - Enable all four: `systemctl enable --now mysql redis-server minio elasticsearch`.
   - Configure MySQL: `character-set-server=utf8mb4`, set the `root` user's
     password (native + TCP: `root@localhost`, `root@127.0.0.1`, `root@%`,
     all with `mysql_native_password`), then run `docker/init.sql` to create
     the `rag_flow` database.
   - Configure Redis: `requirepass`, `maxmemory`, `maxmemory-policy` per
     `docker/docker-compose-base.yml`'s `redis` service.
   - All passwords/users above are RAGFlow's well-known dev defaults
     (`infini_rag_flow`), the same ones baked into `docker/.env` - not
     real secrets, don't need to be handled specially.

3. **Point the app at the native services.** `conf/service_conf.yaml` in
   this repo is already committed with `host: localhost` and native default
   ports (mysql `3306`, minio `9000`, redis `6379`, es `9200`) - if you
   diff against a fresh checkout and it points at docker hostnames/ports
   instead, that customization was lost; redo it (see the
   `docker/service_conf.yaml.template` file for the full key list).

4. **Python 3.13 + uv.** `pipx install uv`, then let
   `scripts/wsl_start_ragflow.sh` run `uv sync` for you.

5. **Node.js 22.** `curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs`
   (repo requires Node `>=18.20.4`; 22 LTS was used here).

## Non-obvious gotchas (read before debugging for an hour)

- **Never let `uv sync` or `npm install` write into a path under `/mnt/*`
  (i.e. anywhere on a Windows drive mounted into WSL2).** DrvFs cannot
  handle the volume of small-file metadata operations these tools do:
  `uv sync` fails outright with `Operation not permitted (os error 1)`
  mid-install, and `npm install` doesn't error but becomes so slow it can
  look hung for 10+ minutes on a package set that installs in ~1 minute on
  native ext4. Fix used here: `.venv` lives at `~/.venvs/ragflow` (via
  `UV_PROJECT_ENVIRONMENT`), and the entire `web/` frontend is **mirrored**
  to `~/ragflow-web` via `rsync` and built/run from there. The Python
  backend does NOT need this treatment - it reads `.py` source files
  directly from the Windows-mounted repo path fine (it's specifically
  bulk package-installation metadata operations that DrvFs chokes on, not
  ordinary file reads).
  - Practical consequence: **edits under `web/` on the Windows-side
    checkout are invisible to the running frontend until you re-run
    `wsl_start_ragflow.sh`** (it re-syncs on every run). If you're actively
    developing frontend code, either re-run the script after each save, or
    edit directly inside WSL2 at `~/ragflow-web` (e.g. via VS Code's
    "WSL" remote extension) and periodically `rsync` back the other
    direction into the real repo checkout before committing.
  - A plain symlink (`web/node_modules -> ~/native-dir`) does **not** work
    around this: npm's install step detects the pre-existing symlink as a
    "non-directory", deletes it, and recreates a real directory on the
    slow mount, silently undoing the workaround. A `mount --bind` onto a
    DrvFs path also failed to take effect. Full mirroring is the only
    approach that reliably worked.

- **`web/.env.development` defaults `API_PROXY_SCHEME` to `go`**, i.e. the
  dev server proxies `/api` and `/v1` to port `9384`, expecting the newer
  Go backend (see recent `Go: ...` commits in this repo). The official
  from-source guide only covers the Python backend
  (`task_executor.py` + `ragflow_server.py`, port `9380`). If you only
  started the Python backend, you **must** override this:
  `API_PROXY_SCHEME=python npm run dev` - otherwise every API call in the
  browser 404s against a Go server that was never started. If someone sets
  up the Go backend instead/also, this doesn't apply.

- **`fatal: not a git repository` in the backend logs, `RAGFlow version:
  unknown`.** Cosmetic only. Caused by this being a git *worktree* whose
  `.git` file contains a Windows-style absolute path
  (`gitdir: D:/ragflow/.git/worktrees/...`), which Linux git run from
  inside WSL2 can't resolve. Doesn't affect functionality.

- **`wsl --update` failing with `0x8024500c`** - see step 1 above.

## What to hand off / keep in sync

If you're onboarding another person or agent (Codex, another Claude Code
session, etc.) onto this same environment or a fresh machine with the same
Docker restriction:

- This directory (`scripts/wsl_start_ragflow.sh`, `wsl_stop_ragflow.sh`,
  this README) - the repeatable part, committed to git so it travels with
  the branch/PR.
- The **one-time setup steps** above - not scripted end-to-end (they involve
  `sudo`, package downloads, and one interactive password reset), so they're
  written out narratively rather than as a blind-run script. Turning them
  into a single `setup_wsl_base_services.sh` would be a reasonable follow-up
  if this becomes a common onboarding path.
- `conf/service_conf.yaml`'s localhost customization (already committed in
  this branch - verify it survives merges/rebases against `main`, since a
  conflict could silently revert it to docker hostnames).
- Nothing here is a real secret - every credential in this doc and in
  `conf/service_conf.yaml` is RAGFlow's shared open-source dev default.
