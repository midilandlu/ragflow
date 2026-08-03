# RAGFlow WSL2 dev environment - current status

Living document. Unlike `wsl_dev_README.md` (setup steps, gotchas - rarely
changes), this file tracks **what state things are actually in right now**.
Whoever is ending a work session here updates this file before handing off;
whoever is picking up reads it first. See "Handoff protocol" below for the
exact prompts to use on both ends.

---

## Last updated

- **When:** 2026-08-03
- **By:** Claude Code (session that built the WSL2 no-Docker dev setup)
- **Branch / commit:** `claude/local-origin-upstream-diff-902ea8` @ `79b930423`
- **Working tree:** clean, in sync with `origin/claude/local-origin-upstream-diff-902ea8` (0 ahead / 0 behind)

## Environment status (as of last update)

| Component | State |
|---|---|
| WSL2 distro `Ubuntu-24.04` | installed, C: drive |
| MySQL / Redis / MinIO / Elasticsearch 8.11.3 | native systemd services, all `active` + `enabled` |
| Python venv (`~/.venvs/ragflow`) | present, `uv sync --frozen` done (no `test` group yet) |
| Frontend native mirror (`~/ragflow-web`) | present, `npm install` done |
| `task_executor.py` | running |
| `ragflow_server.py` | running, `http://127.0.0.1:9380` |
| Vite dev server | running, `http://localhost:9222`, `API_PROXY_SCHEME=python` |
| Browser smoke test | passed - login page loads, `/api/v1/*` calls return 200, no console errors |

To re-verify any of this yourself rather than trust this table, run
`bash scripts/wsl_start_ragflow.sh` (idempotent) and re-check.

## What's been done this "chapter" of work

1. Diagnosed local/origin/upstream had no divergence; found `origin` was
   misconfigured to point at `infiniflow/ragflow` directly instead of the
   user's fork - fixed (`origin` = fork, `upstream` = official repo).
2. Diagnosed Docker Desktop is IT-policy-blocked; built a full Docker-free
   RAGFlow-from-source setup on WSL2 instead (native MySQL/Redis/MinIO/ES).
3. Hit and worked around DrvFs performance/permission issues for both
   `uv sync` and `npm install` (native fs relocation for venv + web mirror).
4. Got backend + frontend running end-to-end, verified in browser.
5. Wrote `wsl_start_ragflow.sh` / `wsl_stop_ragflow.sh` /
   `wsl_restart_ragflow.sh` + `wsl_dev_README.md`, iterated on them based
   on real re-runs (fixed a path-naming mismatch, made dependency sync
   unconditional so pulls are picked up automatically).
6. Recorded the origin-only push policy for this tooling.
7. Set up `~/ragflow` convenience symlink and confirmed VS Code
   Remote-WSL editing works against it.

## Known gaps / not yet done

- `test` dependency group (`pytest` etc.) is not installed in the venv yet
  - see `wsl_dev_README.md`'s "Testing your own local changes" section for
    the one-time command.
- No CI/automation for the one-time WSL base-service setup - it's
  documented narratively in `wsl_dev_README.md`, not scripted end-to-end
  (see that file's "What to hand off" section for why).
- Nothing beyond a manual browser smoke test (login page) has been
  exercised yet - no document upload / chat / agent flow has been tried
  against this stack.
- No actual RAGFlow feature/bugfix work has started yet on this branch;
  everything so far is dev-environment tooling (see the push-policy note:
  none of this should go to `upstream`).

## Handoff protocol

### Before you stop working (update this file)

Give your agent (or do it yourself) a prompt like:

> Before we stop, update `scripts/wsl_dev_STATUS.md`: refresh the "Last
> updated" block (date, branch, commit, working-tree/sync state), the
> environment status table (re-check services/processes, don't just copy
> the old table), summarize what you actually did this session under
> "What's been done", and update "Known gaps" with anything you didn't
> finish or newly discovered. Commit it to `origin` only (per the push
> policy in `wsl_dev_README.md`) and push.

### Picking up (Codex, another Claude Code session, a colleague)

Paste this as your opening prompt:

> You're picking up RAGFlow development in this repo
> (`D:\ragflow\.claude\worktrees\update-understand-anything-analysis-21479f`
> or wherever this checkout lives on your machine). Docker is not
> available here; RAGFlow runs from source against native WSL2 services
> instead. Before doing anything else:
> 1. Read `scripts/wsl_dev_README.md` in full (one-time setup, gotchas,
>    push policy).
> 2. Read `scripts/wsl_dev_STATUS.md` (this file) for what's already
>    running and what's left to do.
> 3. Inside WSL2, run `bash scripts/wsl_start_ragflow.sh` to bring the
>    dev stack up (it's idempotent - safe even if things are already
>    running), and verify `http://localhost:9222` loads in a browser.
> 4. Confirm with me what to work on next based on "Known gaps" /
>    whatever I tell you directly, then proceed.
> 5. Before you end your own session, update this STATUS file per the
>    "Before you stop working" section above.
