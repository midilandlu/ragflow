# 交接 → Claude Code：RAGFlow WSL2 no-Docker 本地開發

> 你是接手這個 fork 的 Claude Code session。這份檔案是**寫給你的交接信**，跟
> `wsl_dev_STATUS.md`（客觀的「現在狀態」快照，每次啟動腳本會覆寫一段）不同 ——
> 這裡記錄的是脈絡、決策原因、還沒收尾的事，採**累加式更新**（每次交接在最上面加一段
> `## 🆕 <日期> 更新`，不要整篇重寫，舊的交接記錄留著讓後面的人看得到演變過程）。

## 先讀這三份，順序如下

1. `scripts/wsl_dev_README.md` —— 穩定參考：一次性環境設置、已知 gotcha、
   push 政策（origin vs upstream）、交接 prompt 範本。很少變動。
2. `scripts/wsl_dev_STATUS.md` —— 現在的真實狀態：目前 branch/commit、
   服務有沒有在跑、上次做了什麼、已知缺口。**每次交接前都會被重寫**，永遠以它
   為準，不要相信這份 HANDOFF 檔裡任何「現在狀態」的描述（如果有寫，那是舊的）。
3. 這份檔案剩下的部分 —— 「為什麼」層級的脈絡，STATUS.md 不會記的東西。

## 🆕 2026-08-06 更新（Codex 接手與 upstream 同步）

本次 Codex 接手後，使用者做了以下明確決策：

1. **不保留 `verify_api_key()` 健康檢查的 `max_tokens: 16` 修改。**
   `api/apps/services/provider_api_service.py` 已還原到 committed state；不要再把
   這個修改當成待審或待提交工作。
2. **已同步最新 upstream。** 使用 `gpt-5.6-terra` 子代理 fetch
   `upstream/main`（`2e37997ab`），再以普通 merge、無 rebase／force 的方式併入
   本分支；merge commit 是 `db348b9eb`。同步過程無衝突，且沒有推送 upstream。
3. **Tongyi-Qianwen PR 已完成。**
   [infiniflow/ragflow#17887](https://github.com/infiniflow/ragflow/pull/17887)
   已於 2026-08-05 merge，不再需要追蹤 review。
4. **DeepInfra `Qwen/QwQ-32B` 仍是外部 LiteLLM 串流問題。** 本次沒有加入
   RAGFlow workaround；同供應商其他已驗證可用模型仍是現階段替代方案。

推送邊界維持不變：本次同步與 handoff 更新只推 `origin` 的
`claude/local-ragflow-service-check-0ba110`，不向 `upstream` 寫入任何內容。

## 🆕 2026-08-06 更新（Claude Code / Sonnet 5）

**背景**：這個 worktree（`ragflow-wsl2-dev-setup-969743`，branch
`claude/local-ragflow-service-check-0ba110`）這次 session 做的是「確認本地服務
狀態 + 排查兩個 Model Providers 存不進去的 bug」，不是功能開發。

**還沒收尾、下一個接手的人請注意**：

1. **`api/apps/services/provider_api_service.py` 有一個修正還沒 commit。**
   `verify_api_key()` 內 `check_streamly()` 這段驗證用的健康檢查呼叫沒有設
   `max_tokens`，導致某些供應商套用自己的預設值（可能超過該模型實際
   `max_total_tokens`，會被直接拒絕）。已加 `max_tokens: 16`，本地測試過確認
   backend 能正常跑，但**依使用者指示先不 commit**，留在 working tree 裡 ——
   接手前先 `git status` / `git diff` 確認它還在，決定要不要納入。
2. **[infiniflow/ragflow#17887](https://github.com/infiniflow/ragflow/pull/17887)**
   —— 已送出的 upstream PR，修正通義千問（Tongyi-Qianwen）國際版端點打錯字
   （`compatible-model` → `compatible-mode`）。**還沒去確認 review 狀態**，接手
   時可以先看一下有沒有人回應。
3. **DeepInfra 的 `Qwen/QwQ-32B` 模型驗證會失敗**（`No valid response
   received`），根因已查到是 **litellm（目前裝的版本 `1.84.0`）自己內部
   `stream_chunk_builder()` 的 bug**（串流回應完全沒有 `choices` 內容時會
   `IndexError` 崩潰），不是 RAGFlow 或這個環境的問題。查過 litellm GitHub，
   同類 bug 在其他供應商/推理模型上出現過（PR #8009、Issue #28884/#27928/
   #32051），但沒找到 DeepInfra + QwQ-32B 這個組合的既有 issue。**已建立一個
   背景任務建議（chip：`task_cd383eb7`）** 讓使用者決定要不要研究新版 litellm
   有沒有修好、或去開一個新 issue —— 接手時可以問使用者這個 chip 有沒有處理。
   同廠牌其他模型（`Qwen3-14B`、`Qwen3-235B-A22B`）驗證正常，這不是全面性
   問題。

**這次 session 順便修好、已經 commit + push 到 origin 的東西**（細節見
`wsl_dev_STATUS.md` 的 commit log 或直接 `git log`）：
- Vite 前端 dev server 在啟動腳本的 session 結束後會被悄悄殺掉的 bug
  （`setsid` + stdin 重導向修好，已進 `wsl_start_ragflow.sh`）。
- `wsl_stop_ragflow.sh` 的 race condition（送 SIGTERM 不等真的退出就返回，
  導致 restart 有時候會把服務停在「沒人重啟」的狀態）。
- worktree 對齊防呆：`wsl_start_ragflow.sh` 現在會拒絕在「另一個 worktree的
  process 還在跑」時啟動，並自動寫一段 `<!-- AUTO-STATUS -->` 到
  `wsl_dev_STATUS.md`，避免這份文件手動維護漂移。

## 這個 fork 的兩個 worktree

`ragflow-wsl2-dev-setup-969743`（這個）和
`update-understand-anything-analysis-21479f` 都裝了同一套 WSL2 dev tooling，
但**同一時間只有一個能是「活著」的那個**（固定 port 9380/9222、一個
`~/ragflow` symlink、一個 WSL2 實例）。啟動前務必先跑
`bash scripts/wsl_start_ragflow.sh` —— 如果它報「另一個 worktree 正在跑」，
照它的指示先 `wsl_stop_ragflow.sh` 再啟動，不要手動繞過這個檢查。

## Push 政策（跟 `AGENTS.md`/`CLAUDE.md` 一致）

- `scripts/` 底下這些 dev tooling 檔案（含這份 HANDOFF）：只推 `origin`，
  **永不進 upstream PR**。
- 真正驗證過的 app 程式碼修正（像 #17887 那個）：本地驗證過後可以送
  upstream PR。
- 完整規則見 `wsl_dev_README.md` 的「Push policy」章節。
