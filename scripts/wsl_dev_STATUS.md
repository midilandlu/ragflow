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

- **When:** 2026-08-05T19:23:34+08:00
- **Worktree:** `/mnt/d/ragflow/.claude/worktrees/ragflow-wsl2-dev-setup-969743`
- **Branch / commit:** `claude/local-ragflow-service-check-0ba110` @ `6e92a7cf6`
- **~/ragflow symlink target:** `/mnt/d/ragflow/.claude/worktrees/ragflow-wsl2-dev-setup-969743`
- **task_executor.py:** running, pid 83455
- **ragflow_server.py:** running, pid 83458
- **vite dev server:** running, pid 83551
<!-- AUTO-STATUS:END -->

> The AUTO-STATUS block is the snapshot from the most recent start-script
> run. After an explicit stop, the live checks and prose below are newer.

---

## Last updated

- **When:** 2026-08-06
- **By:** OpenAI Codex, continuing Claude Code's handoff in worktree
  `D:\ragflow\.claude\worktrees\ragflow-wsl2-dev-setup-969743`, branch
  `claude/local-ragflow-service-check-0ba110`.
- **Upstream sync base:** `upstream/main` @ `2e37997ab`; merged without
  conflict by `db348b9eb` (no rebase, force push, or upstream write).
- **Working tree:** the prior uncommitted `provider_api_service.py`
  `max_tokens: 16` change was discarded by explicit user decision. Only
  this handoff-document update remained before the final origin-only commit.

## Environment status (as of last update)

`~/ragflow` symlink points at **this** worktree
(`ragflow-wsl2-dev-setup-969743`). Backend/task_executor/frontend were
healthy through the end of the previous Claude session and then
**deliberately stopped** (`bash
scripts/wsl_stop_ragflow.sh`) as part of this handoff, so the machine
isn't left running someone else's session unattended. Base services
(MySQL/Redis/MinIO/Elasticsearch) are left running (systemd-managed,
shared infrastructure).

| Component | State |
|---|---|
| WSL2 distro `Ubuntu-24.04` | installed, C: drive |
| MySQL / Redis / MinIO / Elasticsearch 8.11.3 | native systemd services, all `active` + `enabled` (left running) |
| `task_executor.py` | **stopped** for handoff - was healthy before stopping |
| `ragflow_server.py` | **stopped** for handoff - was healthy before stopping |
| Vite dev server | **stopped** for handoff - was healthy before stopping |

To bring the app back up: `bash scripts/wsl_start_ragflow.sh`.

## What's been done this "chapter" of work

1. **Worktree/symlink alignment + tooling infra** (see prior entries in git
   history for full detail): fast-forward merged the WSL dev-setup tooling
   into this branch, fixed the frontend silently dying after script exit
   (`setsid` fix), added a worktree-alignment guard and auto-status
   writeback to `wsl_start_ragflow.sh`, fixed a stop-script race condition.
   Pushed this tooling to `origin/main` and to the sibling worktree
   (`update-understand-anything-analysis-21479f`, branch
   `claude/local-origin-upstream-diff-902ea8`) so both are in sync.
2. **Fixed a real app bug and sent it upstream**: Tongyi-Qianwen's
   international DashScope endpoint was hardcoded as
   `.../compatible-model/v1` instead of `.../compatible-mode/v1` in
   `api/apps/services/provider_api_service.py` (two places) - confirmed via
   repo-wide grep that every other reference uses the correct spelling.
   This caused every model verification call to 404 and blocked saving the
   provider. Verified locally against a real DashScope account (key
   rotated by the user after being pasted in chat - **not stored/used by
   me**). Committed (`fe485eaf4`), pushed to `origin`, and opened
   **[infiniflow/ragflow#17887](https://github.com/infiniflow/ragflow/pull/17887)**
   against upstream from a clean cherry-picked branch. The PR was merged on
   2026-08-05; no further review follow-up is pending.
3. **DeepInfra `Qwen/QwQ-32B` investigation - external issue remains**:
   - User reported "No valid response received" / warning-triangle on this
     model after saving DeepInfra credentials.
   - Investigated a possible broader gap: the health-check call in
   `verify_api_key()`'s `check_streamly()` didn't set `max_tokens` at
     all, so it inherited whatever default the provider applies - for some
     providers/models that default exceeds the model's actual limit
     (confirmed via direct curl to DeepInfra: unset `max_tokens` defaults
     to 65536, but QwQ-32B's real `max_total_tokens` is 40960, a hard
     rejection). An explicit `max_tokens: 16` was tried locally, but the
     user decided not to retain it; the working tree has been restored.
   - **This did NOT fix the QwQ-32B symptom** - after the fix, the same
     error persists but now root-caused precisely: it's litellm (installed
     version `1.84.0`) itself crashing inside `stream_chunk_builder()`
     with `IndexError: list index out of range` while assembling a
     completed response from DeepInfra's stream for this model, because
     none of the buffered chunks ever contain a populated `choices` list.
     This is a known *class* of litellm bug (unsafe indexing, especially
     around reasoning-model streaming) seen before for other
     providers/models (DeepSeek R1 via PR #8009, Vertex Gemini via
     #28884/#27928, Responses-API via #32051) but **no existing litellm
     issue/fix found for this specific DeepInfra+QwQ-32B combination**.
     Not fixable from RAGFlow's side. Workaround: use other DeepInfra
     models (`Qwen3-14B`, `Qwen3-235B-A22B` confirmed working).
   - Spawned a background task suggestion (`task_cd383eb7`) to check for a
     newer litellm release and draft (not auto-submit) a GitHub issue -
     **still pending, user has not acted on it yet.**
4. **Synchronized the active branch with upstream**: fetched
   `upstream/main` @ `2e37997ab` and merged it normally via `db348b9eb`.
   The merge was conflict-free; no history rewrite and no upstream push were
   performed.

## Known gaps / not yet done

- **QwQ-32B via DeepInfra remains broken** due to the external litellm bug
  described above - no action possible on our side beyond the pending
  litellm-version-check / issue-filing background task (`task_cd383eb7`).
- **Two worktrees still exist with this tooling and will keep diverging**
  (`ragflow-wsl2-dev-setup-969743` and
  `update-understand-anything-analysis-21479f`) - mitigated (not solved,
  can't be fully solved by tooling alone) by: pushing this tooling to
  `origin/main` so new worktrees inherit it automatically; a startup guard
  in `wsl_start_ragflow.sh` that refuses to start if another worktree's
  process is already running; and the AUTO-STATUS block above that can't
  drift from reality. Still true: only one worktree's code can be the live
  one at a time - switching is still a manual, explicit step.
- `test` dependency group (`pytest` etc.) is not installed in the venv yet
  - see `wsl_dev_README.md`'s "Testing your own local changes" section for
    the one-time command.
- No CI/automation for the one-time WSL base-service setup - it's
  documented narratively in `wsl_dev_README.md`, not scripted end-to-end.
- Nothing beyond login-page + Model Providers page has been exercised in
  the browser yet - no document upload / chat / agent flow tried.

## Handoff protocol

See `wsl_dev_README.md`'s "Handoff protocol" section for the standard
before-stopping and picking-up prompt templates.
