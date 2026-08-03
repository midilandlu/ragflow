# RAGFlow WSL2 dev environment - current status

Living document. Unlike `wsl_dev_README.md` (setup steps, gotchas, and the
fixed handoff prompt templates - rarely changes), this file tracks **what
state things are actually in right now**, and gets fully rewritten each
handoff. Whoever is ending a work session here updates this file before
handing off; whoever is picking up reads it first. See
`wsl_dev_README.md`'s "Handoff protocol" section for the exact prompts to
use on both ends - they're kept there, not here, specifically so they
don't get lost or drift when this file's content gets replaced.

---

## Last updated

- **When:** 2026-08-03
- **By:** Claude Code (handoff to a colleague's Claude Code session)
- **Branch / commit:** `claude/local-origin-upstream-diff-902ea8` @ `7476b2003`
- **Working tree:** clean, in sync with `origin/claude/local-origin-upstream-diff-902ea8` (0 ahead / 0 behind)

## Environment status (as of last update)

**Backend/frontend processes were deliberately stopped as part of this
handoff** (`bash scripts/wsl_stop_ragflow.sh`), so the machine isn't left
running someone else's session unattended. Base services are left running
(they're systemd-managed and shared infrastructure, not part of "this dev
session"). To bring the app back up: `bash scripts/wsl_start_ragflow.sh`.

| Component | State |
|---|---|
| WSL2 distro `Ubuntu-24.04` | installed, C: drive |
| MySQL / Redis / MinIO / Elasticsearch 8.11.3 | native systemd services, all `active` + `enabled` (left running) |
| Python venv (`~/.venvs/ragflow`) | present, `uv sync --frozen` done (no `test` group yet) |
| Frontend native mirror (`~/ragflow-web`) | present, `npm install` done |
| `task_executor.py` | **stopped** for handoff - was running and healthy before stopping |
| `ragflow_server.py` | **stopped** for handoff - was running and healthy before stopping (`http://127.0.0.1:9380`) |
| Vite dev server | **stopped** for handoff - was running and healthy before stopping (`http://localhost:9222`, `API_PROXY_SCHEME=python`) |
| Last browser smoke test | passed (before this handoff) - login page loads, `/api/v1/*` calls return 200, no console errors |

Run `bash scripts/wsl_start_ragflow.sh` to bring everything back up and
re-verify rather than trusting this table.

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
6. Recorded the origin-only push policy for this tooling, then extended
   it to explicitly cover `AGENTS.md`/`CLAUDE.md` themselves (local/
   `origin`-only, never carried into an upstream PR).
7. Set up `~/ragflow` convenience symlink and confirmed VS Code
   Remote-WSL editing works against it.
8. Added a one-line breadcrumb in `AGENTS.md` pointing at `scripts/` for
   Docker-less environments.
9. Split the handoff prompt templates out of this STATUS file and into
   `wsl_dev_README.md` (stable file) so they can't drift/get lost when
   this file gets rewritten.
10. Stopped the backend/frontend for this handoff (`wsl_stop_ragflow.sh`).

## Known gaps / not yet done

- **This is the first real handoff to another person's Claude Code session
  - the one-time WSL setup steps in `wsl_dev_README.md` have never been
  validated by anyone other than the person who wrote them.** If the
  colleague picking this up hits something the README doesn't cover, that's
  the most likely place for a gap; please fix the README/STATUS file as
  you go rather than only solving it locally in your own head.
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

See `wsl_dev_README.md`'s "Handoff protocol" section for the standard
before-stopping and picking-up prompt templates.
