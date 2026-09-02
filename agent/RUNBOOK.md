# NIM Agent — Runbook

All commands assume you are in the repo root with the venv active:

```powershell
cd C:\Users\Roni_Azriel\Development\nim-api-adapter
.venv\Scripts\activate
```

---

## 0. Prerequisites

Copy `.env` and fill in all values before any run:

```
NGC_API_KEY=...
GITHUB_TOKEN=...          # classic PAT, repo scope, SSO-authorized for dataloop-ai-apps
BOT_EMAIL=...
BOT_PASSWORD=...
ENV=...                   # rc / prod
```

The token needs SSO authorization: GitHub → Settings → Developer settings → PAT → Configure SSO → Authorize `dataloop-ai-apps`.

---

## 1. Main agent run (state-aware, opens PR)

```powershell
python agent/nim_agent.py run-agentic
```

**Sanity check runs automatically** at the start — it verifies NGC_API_KEY, GITHUB_TOKEN (push permission), and Dataloop M2M login. Fix any failures before proceeding.

Common flags:

| Flag | Purpose |
|---|---|
| `--no-pr` | Skip PR creation (dry-run style) |
| `--skip-docker` | Skip Docker builds for downloadables |
| `--force-docker` | Rebuild Docker even for manifests that already have a `runnerImage` |
| `--anomaly-threshold 0.9` | Raise the deprecation safety threshold (use when many deprecations are expected) |
| `--limit 5` | Cap models processed per category |
| `--state-path PATH` | Use a custom `run_state.json` (default: `agent/agent/run_data/run_state.json`) |
| `--downloadable-preview` | Print which downloadables are resolvable — no builds, no PR |
| `--max-workers 10` | Parallelism (default: 10) |

Typical full run:
```powershell
python agent/nim_agent.py run-agentic --anomaly-threshold 0.9
```

---

## 2. Quick dry-run (no PR, no Docker, 2 models per category)

```powershell
python agent/nim_agent.py dry-run
python agent/nim_agent.py dry-run --limit 5
```

---

## 3. Check current state (quarantined models, last run)

```powershell
python agent/nim_agent.py status
```

---

## 4. Un-quarantine a model

```powershell
python agent/nim_agent.py clear-quarantine <model_id>
python agent/nim_agent.py clear-quarantine all
```

---

## 5. Print availability report

```powershell
python agent/nim_agent.py report
```

---

## 6. Reset run state (before a clean run)

Delete the state file so no models are quarantined and history is cleared:

```powershell
del agent\agent\run_data\run_state.json
```

---

## 7. Deprecated DPK cleanup

### 7a. Audit — who has what installed (read-only, safe)

From the latest report JSON:
```powershell
python agent/deprecated_cleanup.py audit --from-report agent/agent/run_data/report_<TIMESTAMP>.json
```

By DPK name:
```powershell
python agent/deprecated_cleanup.py audit nim-llama-3-1-8b-instruct nim-bge-m3
```

Custom CSV output path:
```powershell
python agent/deprecated_cleanup.py audit --from-report agent/agent/run_data/report_<TIMESTAMP>.json --output deprecated_audit.csv
```

CSV is auto-saved to `agent/audit_<timestamp>.csv` by default.

### 7b. Cleanup — dry run first (no changes)

```powershell
python agent/deprecated_cleanup.py cleanup --from-report agent/agent/run_data/report_<TIMESTAMP>.json
```

### 7c. Cleanup — execute (deletes models, uninstalls apps, deletes DPKs)

```powershell
python agent/deprecated_cleanup.py cleanup --from-report agent/agent/run_data/report_<TIMESTAMP>.json --execute
```

Order of operations per DPK:
1. Delete all models (`packageId` filter, across all projects)
2. Uninstall all apps (`dpkName` filter, across all projects)
3. Delete the DPK from the marketplace

---

## 8. Run outputs

After `run-agentic` completes, artifacts land in `agent/agent/run_data/`:

| File | Contents |
|---|---|
| `run_state.json` | Persistent state: quarantined models, seen models, last run |
| `report_<TIMESTAMP>.json` | Full run report: new, deprecated, failed, quarantined |
| `manifests_<TIMESTAMP>.json` | All fetched NIM manifests for this run |

---

## 9. Post-release platform sanity test

Installs a random sample of public NIM DPKs as apps, deploys the model, runs a predict/embed call, and reports pass/fail.

```powershell
# Default: test 10% of each model type (LLM / VLM / Embedding), leave resources running
python agent/nim_agent.py validate-release

# Custom percentage (e.g. 25%)
python agent/nim_agent.py validate-release --percentage 0.25

# Auto-cleanup after each test (uninstall apps + delete models)
python agent/nim_agent.py validate-release --percentage 0.10 --cleanup
```

How it works:
1. Lists all public NIM DPKs published from this repo (filtered by `codebase.gitUrl`)
2. Groups them by type: `llm`, `vlm`, `embedding`
3. Randomly samples `ceil(count × percentage)` from each group (minimum 1)
4. For each sampled DPK: installs as app → deploys model → runs predict/embed → records pass/fail
5. Prints a summary and writes `agent/agent/run_data/validate_release_<TIMESTAMP>.json`

Required env vars (same as main agent run):
```
NGC_API_KEY
DATALOOP_TEST_PROJECT   (defaults to "NVIDIA-AGENT-PROJECT")
DATALOOP_NGC_INTEGRATION_ID
BOT_EMAIL / BOT_PASSWORD / ENV
```

---

## Typical release workflow

```
1.  Reset state (if needed)       del agent\agent\run_data\run_state.json
2.  Verify credentials             .env is populated + GitHub SSO authorized
3.  Full run                       python agent/nim_agent.py run-agentic --anomaly-threshold 0.9
4.  Review the PR on GitHub
5.  Merge the PR
6.  Post-release sanity            python agent/nim_agent.py validate-release --percentage 0.10
7.  Review validate_release_*.json in agent/agent/run_data/
8.  Audit deprecated DPKs          python agent/deprecated_cleanup.py audit --from-report ...
9.  Review the CSV
10. Dry-run cleanup                python agent/deprecated_cleanup.py cleanup --from-report ...
11. Execute cleanup                python agent/deprecated_cleanup.py cleanup --from-report ... --execute
```
