# RAGFlow WSL2 dev environment - current status

Living document. Unlike `wsl_dev_README.md` (setup steps, gotchas, and the
fixed handoff prompt templates - rarely changes), this file tracks **what
state things are actually in right now**, and gets fully rewritten each
handoff. Whoever is ending a work session here updates this file before
handing off; whoever is picking up reads it first. See
`wsl_dev_README.md`'s "Handoff protocol" section for the exact prompts to
use on both ends - they're kept there, not here, specifically so they
don't get lost or drift when this file's content gets replaced.

<!-- AUTO-STATUS:BEGIN (rewritten by wsl_start_ragflow.sh every run -- do not hand-edit this block, edit the prose sections below instead) -->
**Auto-verified environment** (written by `wsl_start_ragflow.sh` itself, not hand-maintained -- trust this over the prose below if they ever disagree):

- **When:** 2026-08-05T14:06:21+08:00
- **Worktree:** `/mnt/d/ragflow/.claude/worktrees/ragflow-wsl2-dev-setup-969743`
- **Branch / commit:** `claude/local-ragflow-service-check-0ba110` @ `65b115ae0`
- **~/ragflow symlink target:** `/mnt/d/ragflow/.claude/worktrees/ragflow-wsl2-dev-setup-969743`
- **task_executor.py:** running, pid 22679
- **ragflow_server.py:** running, pid 22682
- **vite dev server:** running, pid 24038
<!-- AUTO-STATUS:END -->

---

## Last updated

- **When:** 2026-08-05
- **By:** Claude Code, in worktree
  `D:\ragflow\.claude\worktrees\ragflow-wsl2-dev-setup-969743`, branch
  `claude/local-ragflow-service-check-0ba110`.
- **Branch / commit:** `claude/local-ragflow-service-check-0ba110` @
  `65b115ae0` (after fast-forward merging
  `claude/ragflow-wsl2-dev-setup-969743` into it - this branch had
  **no `scripts/` directory at all** before that; it was cut from `main`
  after the point this file previously described, so it never had the
  WSL dev-setup commits).
  - The `update-understand-anything-analysis-21479f` worktree (the one
    `~/ragflow` was pointing at coming into this session) is unchanged/
    untouched; its backend/frontend were stopped this session in favor of
    this worktree.
- **Working tree:** clean except the same pre-existing untracked `VERSION`
  file (generated, see below).

## Environment status (as of last update)

`~/ragflow` symlink now points at **this** worktree
(`ragflow-wsl2-dev-setup-969743`); backend + frontend were (re)started from
here and left **running**.

| Component | State |
|---|---|
| WSL2 distro `Ubuntu-24.04` | installed, C: drive |
| MySQL / Redis / MinIO / Elasticsearch 8.11.3 | native systemd services, all `active` + `enabled` |
| Python venv (`~/.venvs/ragflow`) | present, `uv sync --frozen` re-run this session |
| Frontend native mirror (`~/ragflow-web`) | present, `npm install` re-run, mirrors this worktree's `web/` |
| `task_executor.py` | **running**, healthy |
| `ragflow_server.py` | **running**, healthy (`http://127.0.0.1:9380`) |
| Vite dev server | **running**, healthy (`http://localhost:9222`, `API_PROXY_SCHEME=python`) - started via the now-fixed `setsid` launch, confirmed to survive session exit |
| Last browser smoke test | passed this session - login page renders (email/password/sign-in fields present), `/api/v1/language`, `/api/v1/auth/login/channels`, `/api/v1/system/config` all return 200, no console errors |

Run `bash scripts/wsl_start_ragflow.sh` to re-verify rather than trusting
this table.

## What's been done this "chapter" of work

1. Confirmed nothing was running (fresh check), then confirmed the user's
   own `wsl_start_ragflow.sh` run had started `ragflow_server.py` +
   `task_executor.py` - but from the **other** worktree
   (`update-understand-anything-analysis-21479f`), because `~/ragflow`
   still pointed there. Frontend hadn't been started at all yet.
2. Per user request: found the current worktree's branch
   (`claude/local-ragflow-service-check-0ba110`) had no `scripts/`
   directory - fast-forward merged `claude/ragflow-wsl2-dev-setup-969743`
   into it (clean, zero conflicts, this branch's HEAD was a strict
   ancestor) to pick up the WSL dev-setup tooling.
3. Stopped the backend/task_executor running against the old worktree,
   repointed `~/ragflow` -> this worktree, ran `wsl_start_ragflow.sh`.
4. **Fixed the frontend-dies-silently bug for real this time**: previous
   sessions had root-caused it (`npm run dev`'s `stdio: inherit` child
   keeps the launching pty's stdin, so it dies when that pty closes even
   with `nohup`/`disown`) and documented a manual `setsid` workaround, but
   never patched the script. Patched `wsl_start_ragflow.sh` step 5 to use
   `setsid nohup npm run dev < /dev/null > log 2>&1 &`, verified it now
   survives a full separate-session restart cycle via a fresh `curl`.
5. Updated `wsl_dev_README.md`'s gotcha entry to reflect the fix is now
   actually in the script, not just documented as a manual workaround.
6. Browser-verified the full stack end-to-end (see table above).

## Known gaps / not yet done

- **The `wsl_start_ragflow.sh` / `wsl_dev_README.md` / `VERSION`-fix /
  `setsid`-fix changes above are uncommitted** on
  `claude/local-ragflow-service-check-0ba110` - push to `origin` per the
  push policy once reviewed (local dev tooling, not upstream-bound).
- **Two worktrees still exist with this tooling and will keep diverging**
  (`ragflow-wsl2-dev-setup-969743` and
  `update-understand-anything-analysis-21479f`) - flagged 2026-08-03,
  discussed with the user 2026-08-05, three mitigations landed the same
  day:
  1. Pushed this tooling to `origin/main` (see below) so *new*
     branches/worktrees cut after this point inherit it automatically -
     today's root cause for gap #1 above (this branch was cut from `main`
     before the tooling existed) can't recur for anything based on current
     `main`.
  2. `wsl_start_ragflow.sh` now refuses to start (exit 1, clear error) if
     `task_executor.py`/`ragflow_server.py` are already running from a
     *different* worktree's physical path (compares `/proc/<pid>/cwd`
     against its own `$REPO_ROOT`, both `readlink -f`'d) - this exact
     scenario silently happened twice before this check existed.
  3. `wsl_start_ragflow.sh` now auto-writes an `<!-- AUTO-STATUS -->`
     block at the top of this file every run (worktree, branch/commit,
     symlink target, process pids) so that part can no longer drift from
     reality the way the hand-written sections below did.
  - **Still true, and not fixable by tooling alone:** only one worktree's
    code can ever be the live one at a time (fixed ports 9380/9222, one
    `~/ragflow` symlink, one WSL2 instance) - switching which worktree is
    active is still a manual, explicit step (stop, repoint symlink or `cd`
    to the other worktree, start).
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
