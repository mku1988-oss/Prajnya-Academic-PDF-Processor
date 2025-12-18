# GEMINI.md - Shastrasaara V4 Master Protocol

**Module**: Shastrasaara V4 Active Development
**Location**: `D:\LLMs\Shastrasaara\src\v4\gemini_refactor`
**Git Repository**: PRIMARY (Direct Access)
**Agent**: Google DeepMind Gemini (Antigravity)

---

## 1. 🧠 Memory & Context (Source of Truth)

**The File System is the ONLY Source of Truth.**
To maintain compatibility with Claude, we adhere to the **Session Handoff Protocol**.

### 📂 Memory Bank Locations
*   **Status**: `D:\LLMs\Shastrasaara\src\v4\shastrasaara_memory\memory_bank\status.md`
    *   *Purpose*: High-level current state, active sprint info, and technical context.
*   **Progress**: `D:\LLMs\Shastrasaara\src\v4\shastrasaara_memory\memory_bank\progress.md`
    *   *Purpose*: Chronological log of achievements and milestones.
*   **Session**: `D:\LLMs\Shastrasaara\src\v4\shastrasaara_memory\memory_bank\session_current.md`
    *   *Purpose*: Immediate context, active TODOs, and handoff notes.

### 🔄 Memory Workflow
1.  **Session Start**:
    *   **Read** `status.md` and `session_current.md`.
    *   If `session_current.md` indicates a pending handoff, acknowledge it.
2.  **During Execution**:
    *   Keep internal state (and `task.md`) updated.
3.  **Session End / Handoff**:
    *   **Update** `status.md` (Append new section if needed).
    *   **Update** `progress.md` (Append chronological entry).
    *   **REWRITE** `session_current.md` completely with the current context for the next agent.

---

## 2. 🛠️ Tool Usage & Agentic Workflow

### 🔧 Core Tools
*   **Exploration**: `list_dir`, `view_file_outline`, `view_file`.
*   **Editing**: `replace_file_content` (Single block), `multi_replace_file_content` (Multiple blocks).
*   **Creation**: `write_to_file`.
*   **Execution**: `run_command` (PowerShell).
    *   **CRITICAL**: ALWAYS use the specific Python environment: `D:\LLMs\env\shastrasaara\python.exe`
*   **Planning**: `task_boundary` (UI updates), `notify_user` (Questions/Reviews).

### 🚀 Git Workflow (Dual-Repo System)

> [!IMPORTANT]
> **Lab-to-Stage Protocol**: Development happens in "The Lab", Releases happen in "The Stage".
> See `WORKFLOW.md` for detailed instructions.

#### 1. The Laboratory (Development)
*   **Path**: `D:\LLMs\Shastrasaara\src\v4\gemini_refactor`
*   **Action**: Develop, Test, Experiment here.
*   **Git**: Private history, granular commits.

#### 2. The Stage (Release)
*   **Path**: `src/Prajnya_v5_release` (Nested Repo)
*   **Action**: Deploy verified files here, then run `tools/sterilize.py`.
*   **Git**: Public release history, clean commits.

#### 3. Standard Cycle
1.  **Check Status**: `git status` (in Lab)
2.  **Edit & Verify**: Make changes, run tests (in Lab)
3.  **Deploy**: Copy to `src/Prajnya_v5_release`, Sterilize, and Verify.
4.  **Publish**: Commit & Push in Release repo.
5.  **Backup**: Commit in Lab repo.

---

## 3. 🧪 Coding Standards: "Biology Research Scientist"

**Persona**: Helpful, academic, rigorous, and explicit.

*   **Documentation**:
    *   EVERY class and method must have a docstring.
    *   Explain *WHY* (scientific context), not just *WHAT* (code logic).
    *   Use biology/research terminology where appropriate.
*   **Simplicity**:
    *   Break complex logic into clear, numbered steps (e.g., `# Step 1: Load PDF`, `# Step 2: Extract text`).
    *   Variable names should be descriptive (e.g., `research_paper_metadata` vs `data`).
*   **Micro-Functions**:
    *   Keep functions under **400 lines**.
    *   Break large functions into 7-10 smaller "micro-functions".
    *   Helper methods go **AFTER** the main function.

---

## 4. 🛡️ Verification & Integrity (Inherited from XML)

### ✅ Skeleton Backup Strategy
*   **Mandate**: Create `.skeleton.py` before major refactors.
    *   *Content*: Imports, class definitions (docstrings only), method signatures (pass).
    *   *Why*: Quick recovery if implementation goes wrong.

### ✅ Class Integrity Protocol
When a class is fully implemented, perform this check:
1.  **Structure**: Are all methods present?
2.  **Completeness**: Are there any loose `TODO`s?
3.  **Syntax**: Does it compile? (`python -m py_compile <file>`)
4.  **Tests**: Do unit tests pass?
5.  **Growth**: Did file size increase appropriately (approx 15-40KB per class)?

### ✅ Test Suite
*   **Run All**: `python -m pytest src/tests/ -v`
*   **Run Specific**: `python -m pytest src/tests/test_file.py -v`

---

## 6. 📚 Key Documentation

**Always refer to these files for architectural context:**

### 🏗️ Backend Pipeline (Shastrasaara V4)
*   **Protocol**: `D:\LLMs\Shastrasaara\src\v4\gemini_refactor\docs\Shastrasaara V4 Pipeline Phase 3 - End-to-End Processing Flow_new.md`
    *   *Context*: End-to-end processing flow, 5-Tier Routing, 4-Signal Classification, and Atomic Ledger Persistence.
*   **Gemini 3 Integration**: Supports `gemini-3-flash` with `thinking_config` for improved reasoning during title page extraction and complex layout analysis.

### 🖥️ Frontend GUI (Prajnya Studio)
*   **Protocol**: `D:\LLMs\Shastrasaara\src\v4\gemini_refactor\docs\Prajnya studio V5 System Architecture and Processing flow.md`
    *   *Context*: "Persisted Async" architecture, NiceGUI interaction flow, ThreadPool execution, and Global State management.

---

## 7. ⚠️ Critical Rules & Troubleshooting
1.  **NEVER** leave the repository in a broken state (syntax errors) without explanation.
2.  **NEVER** delete `status.md` or `progress.md`.
3.  **ALWAYS** ask the user if you are stuck or unsure.

### 🚨 Troubleshooting Matrix
*   **File Corrupted?** -> Stop. Troubleshoot root cause. Restore from skeleton/git.
*   **Duplication Detected?** -> Stop. Do not attempt in-place repair. Restore from skeleton.
*   **Syntax Errors?** -> Fix before proceeding to next task.
