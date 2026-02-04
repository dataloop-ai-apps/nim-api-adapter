"""
Refactor Repo – one PR for 91 models with new structure and clean configs.

Flow:
  1. Load 91 successful models from MODELS_CONSOLIDATED_REPORT.txt
  2. Create DPK manifests for all 91 (via DPK MCP)
  3. Run 10% platform tests on Dataloop; if any fail, abort
  4. If platform tests pass:
     - Remove all existing model manifests under models/
     - Write new structure: models/api/{type}/{provider}/{model_name}/dataloop.json
     - Clean .bumpversion.cfg (only new manifest paths)
     - Clean .dataloop.cfg (only new manifest paths)
     - Apply changes locally and on remote (branch + PR)

Usage:
  From repo root: python agent/refactor_repo.py
  Or from agent/:  python refactor_repo.py

Requires:
  - NGC_API_KEY, DATALOOP_TEST_PROJECT, OPENROUTER_API_KEY (for manifests + platform tests)
  - DPK_MCP_SERVER (for DPK manifest generation)
  - GITHUB_TOKEN (for opening the PR)

Run from repo root:  python agent/refactor_repo.py
Or from agent/:       python refactor_repo.py
"""

import os
import re
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

# Repo root (parent of agent/)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Report path (at repo root)
DEFAULT_REPORT_PATH = REPO_ROOT / "MODELS_CONSOLIDATED_REPORT.txt"

# When running from file (F5): load this manifest JSON and run platform tests
RUN_FROM_FILE_MANIFESTS = REPO_ROOT / "output" / "refactor_manifests_20260204_113338.json"
RUN_FROM_FILE_PLATFORM_TESTS = False  # Already passed - skip and go to git/PR
RUN_FROM_FILE_DRY_RUN = False  # Create PR after tests

# Specific models to test (if set, overrides fraction-based selection)
RUN_FROM_FILE_TEST_MODELS = []  # Empty - tests already passed

# Model type to folder name (same as github_client)
MODEL_TYPE_FOLDERS = {
    "embedding": "embeddings",
    "llm": "llm",
    "vlm": "vlm",
    "object_detection": "object_detection",
    "ocr": "ocr",
}


def _parse_model_id(model_id: str) -> tuple:
    """Return (publisher, model_name) for path building."""
    if "/" in model_id:
        parts = model_id.split("/", 1)
        publisher = parts[0].lower().replace("-", "_")
        model_name = parts[1].lower().replace(".", "_").replace("-", "_")
    else:
        publisher = "nvidia"
        model_name = model_id.lower().replace(".", "_").replace("-", "_")
    return publisher, model_name


def get_manifest_path(model_id: str, model_type: str) -> str:
    """Relative path: models/api/{type}/{provider}/{model_name}/dataloop.json"""
    type_folder = MODEL_TYPE_FOLDERS.get(model_type, "llm")
    publisher, model_name = _parse_model_id(model_id)
    return f"models/api/{type_folder}/{publisher}/{model_name}/dataloop.json"


# -----------------------------------------------------------------------------
# 1. Parse report
# -----------------------------------------------------------------------------


def parse_report(report_path: Path) -> list[tuple[str, str]]:
    """
    Parse MODELS_CONSOLIDATED_REPORT.txt and return list of (model_id, model_type).
    Only includes lines under "1. SUCCESSFUL MODELS" (sections 1a, 1b, 1c).
    """
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(f"Report not found: {path}")

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Find where "1. SUCCESSFUL MODELS" starts, then collect until "2. BROKEN" or "3. PASSED"
    results = []
    in_success = False
    # Model line: optional leading spaces, then "publisher/name (type)" - type is word chars
    model_line_re = re.compile(r"^\s*([a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+)\s*\(\s*(\w+)\s*\)\s*$")

    for line in lines:
        stripped = line.strip()
        # Enter success section
        if "1. SUCCESSFUL MODELS" in line or stripped.startswith("1a.") or stripped.startswith("1b.") or stripped.startswith("1c."):
            in_success = True
            continue
        if not in_success:
            continue
        # Leave success section: next major section (do not break on "====" or "---- 1x." or "(32)")
        if "2. BROKEN" in line or "3. PASSED API" in line or "4. SKIPPED" in line or stripped.startswith("---- 2"):
            break
        # Skip subsection headers like "---- 1a. Original pass ..."
        if stripped.startswith("----"):
            continue
        m = model_line_re.match(line)
        if m:
            model_id, model_type = m.group(1), m.group(2).lower()
            results.append((model_id, model_type))

    print(f"  Parsed {len(results)} models from report")
    return results


# -----------------------------------------------------------------------------
# 2. Create manifests
# -----------------------------------------------------------------------------


def create_manifests_for_models(models: list[tuple[str, str]]) -> list[dict]:
    """
    Create DPK manifest for each model via DPKGeneratorClient.
    Returns list of dicts: {model_id, model_type, manifest, dpk_name, error?}
    """
    from dpk_mcp_handler import DPKGeneratorClient

    client = DPKGeneratorClient()
    out = []
    for i, (model_id, model_type) in enumerate(models, 1):
        print(f"  [{i}/{len(models)}] Creating manifest: {model_id} ({model_type})")
        try:
            result = client.create_nim_dpk_manifest(model_id, model_type)
            if result.get("status") == "success":
                out.append({
                    "model_id": model_id,
                    "model_type": model_type,
                    "manifest": result["manifest"],
                    "dpk_name": result["dpk_name"],
                })
                print(f"    -> {result['dpk_name']}")
            else:
                err = result.get("error", "Unknown error")
                print(f"    FAILED: {err}")
                out.append({
                    "model_id": model_id,
                    "model_type": model_type,
                    "manifest": None,
                    "dpk_name": None,
                    "error": err,
                })
        except Exception as e:
            print(f"    ERROR: {e}")
            out.append({
                "model_id": model_id,
                "model_type": model_type,
                "manifest": None,
                "dpk_name": None,
                "error": str(e),
            })
    return out


def save_manifests_before_platform_test(manifest_entries: list[dict], repo_root: Path) -> Path:
    """
    Save all successful manifest entries to a JSON file before running platform tests.
    If platform tests fail or the script crashes later, manifests are not lost.
    """
    success = [e for e in manifest_entries if e.get("manifest")]
    if not success:
        return None
    out_dir = repo_root / "output"
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"refactor_manifests_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(success, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(success)} manifests to {path}")
    return path


def load_manifests_from_file(path: Path) -> list[dict]:
    """Load manifest entries from a refactor_manifests_*.json file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def get_latest_refactor_manifests_path(repo_root: Path) -> Path | None:
    """Return path to the most recent output/refactor_manifests_*.json, or None."""
    out_dir = repo_root / "output"
    if not out_dir.is_dir():
        return None
    candidates = sorted(out_dir.glob("refactor_manifests_*.json"), reverse=True)
    return candidates[0] if candidates else None


# -----------------------------------------------------------------------------
# 3. Platform tests (10%)
# -----------------------------------------------------------------------------


def run_platform_tests(manifest_entries: list[dict], fraction: float = 0.1, specific_models: list[str] = None) -> bool:
    """
    Run platform test (publish + deploy + predict) for a fraction of models.
    If specific_models is provided, test only those model IDs instead of fraction.
    Returns True only if all selected tests pass.
    """
    from tester import TestingTool

    success_entries = [e for e in manifest_entries if e.get("manifest") and not e.get("error")]
    if not success_entries:
        print("  No successful manifests to test.")
        return False

    if specific_models:
        # Filter to only the specified models
        to_test = [e for e in success_entries if e["model_id"] in specific_models]
        print(f"  Running platform tests for {len(to_test)} specific models: {specific_models}")
    else:
        n = max(1, int(len(success_entries) * fraction))
        to_test = success_entries[:n]
        print(f"  Running platform tests for {len(to_test)} models ({fraction*100:.0f}% of {len(success_entries)})")

    tester = TestingTool()
    all_passed = True
    for e in to_test:
        print(f"    Testing: {e['model_id']}")
        try:
            result = tester.publish_and_test_dpk(
                dpk_name=e["dpk_name"],
                manifest=e["manifest"],
                model_type=e["model_type"],
                cleanup=True,
            )
            if result.get("status") != "success":
                print(f"      FAILED: {result.get('error')}")
                all_passed = False
            else:
                print(f"      OK")
        except Exception as ex:
            print(f"      ERROR: {ex}")
            all_passed = False

    return all_passed


# -----------------------------------------------------------------------------
# 4. Local refactor: remove old, write new, configs
# -----------------------------------------------------------------------------


def find_existing_manifest_dirs(repo_root: Path) -> list[Path]:
    """All directories under models/ that contain a dataloop.json."""
    models_dir = repo_root / "models"
    if not models_dir.is_dir():
        return []
    dirs = []
    for root, _dirs, files in os.walk(models_dir):
        if "dataloop.json" in files:
            dirs.append(Path(root))
    return dirs


def remove_existing_models(repo_root: Path) -> list[Path]:
    """
    Remove every model folder under models/ that contains dataloop.json.
    Returns list of removed dirs. Does not remove base.py or top-level type folders.
    """
    dirs = find_existing_manifest_dirs(repo_root)
    # Sort by path depth descending so we remove deep dirs first
    dirs.sort(key=lambda p: len(p.parts), reverse=True)
    removed = []
    for d in dirs:
        if d.is_dir():
            print(f"  Removing: {d.relative_to(repo_root)}")
            shutil.rmtree(d, ignore_errors=True)
            removed.append(d)
    return removed


def write_new_structure(repo_root: Path, manifest_entries: list[dict]) -> list[str]:
    """
    Write models/api/{type}/{provider}/{model_name}/dataloop.json for each entry.
    Returns list of relative manifest paths (for configs).
    """
    paths = []
    for e in manifest_entries:
        if not e.get("manifest"):
            continue
        rel_path = get_manifest_path(e["model_id"], e["model_type"])
        full_path = repo_root / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(e["manifest"], f, indent=2, ensure_ascii=False)
        paths.append(rel_path)
        print(f"  Wrote: {rel_path}")
    return paths


def write_bumpversion_cfg(repo_root: Path, manifest_paths: list[str], current_version: str = "0.3.34") -> None:
    """Write .bumpversion.cfg with [bumpversion] and one [bumpversion:file:...] per path."""
    lines = [
        "[bumpversion]",
        f"current_version = {current_version}",
        "commit = True",
        "tag = True",
        'tag_name = {new_version}',
        "",
    ]
    for p in sorted(manifest_paths):
        lines.append(f"[bumpversion:file:{p}]")
        lines.append('search = "{current_version}"')
        lines.append('replace = "{new_version}"')
        lines.append("")
    path = repo_root / ".bumpversion.cfg"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote .bumpversion.cfg ({len(manifest_paths)} entries)")


def write_dataloop_cfg(repo_root: Path, manifest_paths: list[str], public_app: bool = False) -> None:
    """Write .dataloop.cfg with only the new manifest paths."""
    config = {
        "manifests": sorted(manifest_paths),
        "public_app": public_app,
    }
    path = repo_root / ".dataloop.cfg"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent="\t")
    print(f"  Wrote .dataloop.cfg ({len(manifest_paths)} manifests)")


# -----------------------------------------------------------------------------
# 5. Git and PR
# -----------------------------------------------------------------------------


def get_current_branch(repo_root: Path) -> str:
    """Current git branch name."""
    r = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Git branch failed: {r.stderr}")
    return r.stdout.strip()


def create_refactor_branch_and_push(
    repo_root: Path,
    branch_name: str,
    base_branch: str,
    commit_message: str,
) -> bool:
    """
    Create branch from current HEAD, add all changes, commit, push.
    Returns True on success.
    """
    # Check if branch already exists locally
    existing = subprocess.run(
        ["git", "branch", "--list", branch_name],
        cwd=repo_root, capture_output=True, text=True
    )
    if existing.stdout.strip():
        print(f"  Branch {branch_name} already exists locally, deleting...")
        subprocess.run(["git", "branch", "-D", branch_name], cwd=repo_root, capture_output=True)
    
    # Create branch from current HEAD (keeps our local changes)
    print(f"  Creating branch: {branch_name} from current HEAD")
    r = subprocess.run(
        ["git", "checkout", "-b", branch_name],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(f"  Failed to create branch: {r.stderr}")
        return False
    
    print("  Adding all changes...")
    subprocess.run(["git", "add", "-A"], cwd=repo_root, check=True, capture_output=True)
    
    status = subprocess.run(["git", "status", "--short"], cwd=repo_root, capture_output=True, text=True)
    if not status.stdout.strip():
        print("  No changes to commit.")
        return False
    
    print("  Committing...")
    r = subprocess.run(["git", "commit", "-m", commit_message], cwd=repo_root, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  Commit failed: {r.stderr}")
        return False
    
    print("  Pushing...")
    r = subprocess.run(["git", "push", "-u", "origin", branch_name], cwd=repo_root, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  Push failed: {r.stderr}")
        return False
    
    print(f"  Branch {branch_name} pushed to origin")
    return True


def create_pr_from_branch(repo_root: Path, branch_name: str, base_branch: str, title: str, body: str) -> dict | None:
    """Open a PR from branch_name to base_branch. Returns PR result or None."""
    try:
        from github_client import GitHubClient
        client = GitHubClient(base_branch=base_branch)
        pr = client.repository.create_pull(
            title=title,
            body=body,
            head=branch_name,
            base=base_branch,
        )
        return {"url": pr.html_url, "number": pr.number}
    except Exception as e:
        print(f"  PR creation failed: {e}")
        return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run(
    report_path: Path = None,
    platform_fraction: float = 0.1,
    dry_run: bool = False,
    skip_platform_tests: bool = False,
    base_branch: str = "main",
    load_manifests_from: Path = None,
    specific_models: list[str] = None,
):
    """
    Full refactor flow.
    If load_manifests_from is set, skip parse report + create manifests and load from that JSON file.
    If specific_models is set, test only those model IDs instead of fraction-based selection.
    """
    report_path = report_path or DEFAULT_REPORT_PATH
    print("=" * 60)
    print("REFACTOR REPO – 91 models, new structure, clean configs")
    print("=" * 60)

    if load_manifests_from and Path(load_manifests_from).is_file():
        # Load manifests from file – skip creating them
        print("\n[1/6] Skipping parse report (loading from file).")
        print("\n[2/6] Loading manifests from file (skipping MCP creation)...")
        manifest_entries = load_manifests_from_file(Path(load_manifests_from))
        success_count = sum(1 for e in manifest_entries if e.get("manifest"))
        print(f"  Loaded {len(manifest_entries)} entries, {success_count} with manifests.")
        if success_count == 0:
            print("  No manifests in file. Aborting.")
            return
    else:
        # 1. Parse report
        print("\n[1/6] Parsing report...")
        models = parse_report(report_path)
        if len(models) != 91:
            print(f"  Expected 91 models, got {len(models)}. Proceeding anyway.")
        print(f"  Total models: {len(models)}")

        # 2. Create manifests
        print("\n[2/6] Creating manifests for all models...")
        manifest_entries = create_manifests_for_models(models)
        success_count = sum(1 for e in manifest_entries if e.get("manifest"))
        fail_count = len(manifest_entries) - success_count
        print(f"  Manifests: {success_count} OK, {fail_count} failed")
        if success_count == 0:
            print("  No manifests created. Aborting.")
            return

        # Save manifests to output/ before platform tests (so they are not lost if tests fail)
        print("\n  Saving manifests to output/ before platform tests...")
        save_manifests_before_platform_test(manifest_entries, REPO_ROOT)

    # 3. Platform tests (10% or specific models)
    if not skip_platform_tests:
        print("\n[3/6] Running platform tests...")
        if not run_platform_tests(manifest_entries, fraction=platform_fraction, specific_models=specific_models):
            print("  Platform tests failed. Refactor aborted.")
            return
        print("  All platform tests passed.")
    else:
        print("\n[3/6] Skipping platform tests (skip_platform_tests=True)")

    if dry_run:
        print("\n[DRY RUN] Stopping before file changes and git.")
        return

    # 4. Local refactor
    print("\n[4/6] Local refactor: remove existing models, write new structure...")
    remove_existing_models(REPO_ROOT)
    new_paths = write_new_structure(REPO_ROOT, manifest_entries)
    if not new_paths:
        print("  No paths written. Aborting.")
        return

    # Read current version from existing .bumpversion.cfg if present
    bump_path = REPO_ROOT / ".bumpversion.cfg"
    current_version = "0.3.34"
    if bump_path.is_file():
        for line in bump_path.read_text().splitlines():
            if line.startswith("current_version = "):
                current_version = line.split("=", 1)[1].strip()
                break

    print("\n[5/6] Writing clean configs...")
    write_bumpversion_cfg(REPO_ROOT, new_paths, current_version=current_version)
    write_dataloop_cfg(REPO_ROOT, new_paths)

    # 6. Git: branch, commit, push, PR
    print("\n[6/6] Git: branch, commit, push, PR...")
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"refactor/nim-91-models-{ts}"
    commit_message = f"Refactor: 91 NIM models under models/api/{{type}}/{{provider}}/{{model}}/dataloop.json, clean bumpversion and dataloop.cfg"
    base = base_branch
    created = create_refactor_branch_and_push(
        REPO_ROOT,
        branch_name=branch_name,
        base_branch=base,
        commit_message=commit_message,
    )
    if not created:
        print("  Branch/commit/push skipped or failed. Check git status.")
        return

    title = "[NIM] Refactor: 91 models – models/api structure, clean configs"
    body = f"""## Refactor

- **91 models** from MODELS_CONSOLIDATED_REPORT (API + adapter passed)
- **Structure**: `models/api/{{type}}/{{provider}}/{{model_name}}/dataloop.json`
- Removed all previous model manifests under `models/`
- **.bumpversion.cfg** and **.dataloop.cfg** contain only the new manifest paths

### Model types
- LLM, VLM, Embeddings (api only)

---
*Generated by agent/refactor_repo.py*
"""
    pr = create_pr_from_branch(REPO_ROOT, branch_name, base, title, body)
    if pr:
        print(f"  PR opened: {pr['url']}")
    else:
        print("  Open PR manually from branch:", branch_name)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    import argparse

    # When run from IDE (e.g. Run from file / F5) there are no CLI args → use RUN_FROM_FILE_* constants
    debug_from_file = len(sys.argv) == 1
    if debug_from_file:
        manifest_path = RUN_FROM_FILE_MANIFESTS if RUN_FROM_FILE_MANIFESTS.is_file() else get_latest_refactor_manifests_path(REPO_ROOT)
        run_platform = RUN_FROM_FILE_PLATFORM_TESTS
        dry_run = RUN_FROM_FILE_DRY_RUN
        test_models = RUN_FROM_FILE_TEST_MODELS if RUN_FROM_FILE_TEST_MODELS else None
        print(f"(Run from file: manifests={manifest_path.name}, platform_tests={run_platform}, dry_run={dry_run})")
        if test_models:
            print(f"(Testing specific models: {test_models})\n")
        run(
            report_path=None,
            platform_fraction=0.1,
            dry_run=dry_run,
            skip_platform_tests=not run_platform,
            base_branch="main",
            load_manifests_from=manifest_path,
            specific_models=test_models,
        )
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Refactor repo: 91 models, new structure, one PR")
    parser.add_argument("--report", type=Path, default=None, help="Path to MODELS_CONSOLIDATED_REPORT.txt")
    parser.add_argument("--platform-fraction", type=float, default=0.1, help="Fraction for platform tests (default 0.1)")
    parser.add_argument("--dry-run", action="store_true", help="Stop after manifests and platform tests, no file/git changes")
    parser.add_argument("--skip-platform-tests", action="store_true", help="Skip 10%% platform tests (use with care)")
    parser.add_argument("--base-branch", type=str, default="main", help="Base branch for refactor (default: main)")
    parser.add_argument("--load-manifests", type=Path, default=None, help="Load manifest entries from JSON (skip parse + MCP create)")
    args = parser.parse_args()

    run(
        report_path=args.report,
        platform_fraction=args.platform_fraction,
        dry_run=args.dry_run,
        skip_platform_tests=args.skip_platform_tests,
        base_branch=args.base_branch,
        load_manifests_from=args.load_manifests,
    )
