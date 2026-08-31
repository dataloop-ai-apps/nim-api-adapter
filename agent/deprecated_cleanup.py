"""
Deprecated NIM DPK Cleanup

Three entry points:

  find_deprecated_dpks(dpk_names)
      Resolve DPK names → live dl.Dpk objects from the marketplace.
      Returns (found, missing) lists.

  audit(dpk_names)
      For every deprecated DPK, list every project that has it installed:
      project name/id, app name/id/creator, and every model name/id/creator
      linked to that app.  Prints a table and returns the raw data.

  cleanup(dpk_names, dry_run=True)
      For each DPK (by name), across all accessible projects:
        1. Delete all models linked to the DPK  (filter by packageId)
        2. Uninstall all apps linked to the DPK (filter by dpkName)
        3. Delete the DPK from the marketplace
      dry_run=True (default) only prints what would happen — safe to run first.

CLI
---
  python deprecated_cleanup.py audit   [--from-report PATH] [DPK_NAME ...]
  python deprecated_cleanup.py cleanup [--from-report PATH] [DPK_NAME ...] [--execute]
"""

import csv
import datetime
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import dtlpy as dl
from dotenv import load_dotenv

from dpk_handler import ensure_dataloop_login

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NIM_GIT_URLS = [
    "https://github.com/dataloop-ai-apps/nim-api-adapter.git",
    "https://github.com/dataloop-ai-apps/nim-api-adapter",
]


def _all_projects() -> list[dl.Project]:
    """Return all projects the current auth token can access."""
    try:
        return list(dl.projects.list())
    except Exception as e:
        print(f"  ⚠️  Could not list projects: {e}")
        return []


def _dpk_names_from_report(report_path: str) -> list[str]:
    """
    Extract deprecated DPK names from a run-report JSON
    (agent/run_data/report_*.json).

    Both api_deprecated and downloadable_deprecated are included.
    """
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    names = []
    for section in ("api_deprecated", "downloadable_deprecated"):
        for entry in report.get(section) or []:
            name = entry.get("name") if isinstance(entry, dict) else str(entry)
            if name and name not in names:
                names.append(name)
    return names


# ---------------------------------------------------------------------------
# 1. Find deprecated DPKs in the marketplace
# ---------------------------------------------------------------------------

def find_deprecated_dpks(dpk_names: list[str]) -> tuple[list[dl.Dpk], list[str]]:
    """
    Resolve a list of DPK names to live dl.Dpk objects.

    Returns:
        (found, missing)  — found is a list of dl.Dpk; missing is DPK names
        that are no longer in the marketplace (already deleted or never published).
    """
    found, missing = [], []
    for name in dpk_names:
        try:
            dpk = dl.dpks.get(dpk_name=name)
            found.append(dpk)
        except dl.exceptions.NotFound:
            missing.append(name)
        except Exception as e:
            print(f"  ⚠️  Error fetching DPK '{name}': {e}")
            missing.append(name)
    return found, missing


# ---------------------------------------------------------------------------
# 2. Audit — list all installed apps and models per project
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _safe_print(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def _scan_project(project: dl.Project, all_dpk_names: list[str], dpk_by_name: dict) -> list[dict]:
    """
    One API call per project: fetch ALL apps from any deprecated DPK in a
    single IN-filter query, then fetch models per app.

    Returns a flat list of installation dicts:
      {"dpk_name", "project_id", "project_name",
       "app_id", "app_name", "app_creator", "models": [...]}
    """
    try:
        app_filters = dl.Filters(resource=dl.FiltersResource.APP)
        app_filters.add(
            field="dpkName",
            values=all_dpk_names,
            operator=dl.FiltersOperations.IN,
        )
        apps = list(project.apps.list(filters=app_filters).all())
    except Exception as e:
        _safe_print(f"  ⚠️  Could not list apps in '{project.name}': {e}")
        return []

    results = []
    for app in apps:
        # resolve which DPK this app belongs to
        app_dpk_name = (
            getattr(app, "dpk_name", None)
            or getattr(app, "dpkName", None)
            or getattr(app, "package_name", None)
        )
        # fallback: match by display name against dpk_by_name keys
        if app_dpk_name not in dpk_by_name:
            app_dpk_name = next(
                (n for n in all_dpk_names if n in (app.name or "")), None
            )
        if app_dpk_name not in dpk_by_name:
            continue  # couldn't map → skip

        models_info = []
        try:
            model_filters = dl.Filters(resource=dl.FiltersResource.MODEL)
            model_filters.add(field="app.id", values=app.id)
            for model in project.models.list(filters=model_filters).all():
                models_info.append({
                    "model_id": model.id,
                    "model_name": model.name,
                    "model_creator": getattr(model, "creator", "") or "",
                })
        except Exception as e:
            _safe_print(f"  ⚠️  Could not list models for app '{app.name}': {e}")

        results.append({
            "dpk_name": app_dpk_name,
            "project_id": project.id,
            "project_name": project.name,
            "app_id": app.id,
            "app_name": app.name,
            "app_creator": getattr(app, "creator", "") or "",
            "models": models_info,
        })
    return results


def audit(dpk_names: list[str], max_workers: int = 20) -> list[dict]:
    """
    For every deprecated DPK, find every project that has it installed and
    list the app(s) and models linked to it.

    Strategy: one IN-filter query per project (not per DPK), parallelised
    across projects with ThreadPoolExecutor.  Complexity: O(N_projects)
    instead of O(N_projects × N_dpks).

    Return schema:
      [{"dpk_name", "dpk_id", "dpk_version",
        "installations": [{"project_id", "project_name",
                           "app_id", "app_name", "app_creator",
                           "models": [{"model_id","model_name","model_creator"}]}]}]
    """
    found, missing = find_deprecated_dpks(dpk_names)

    if missing:
        print(f"\n⚠️  DPKs not found in marketplace (already deleted or never published):")
        for name in missing:
            print(f"  - {name}")

    dpk_by_name = {dpk.name: dpk for dpk in found}
    all_dpk_names = list(dpk_by_name.keys())

    projects = _all_projects()
    total = len(projects)
    print(f"\n🔍 Scanning {total} project(s) for {len(found)} deprecated DPK(s)"
          f"  [workers={max_workers}] ...")

    installations_by_dpk: dict[str, list] = {name: [] for name in all_dpk_names}
    done_count = 0
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_scan_project, project, all_dpk_names, dpk_by_name): project.name
            for project in projects
        }
        for future in as_completed(futures):
            with lock:
                done_count += 1
                current = done_count
            if current % 100 == 0 or current == total:
                _safe_print(f"  ↳ {current}/{total} projects scanned ...")
            try:
                for inst in future.result():
                    dpk_n = inst.pop("dpk_name")
                    if dpk_n in installations_by_dpk:
                        installations_by_dpk[dpk_n].append(inst)
            except Exception as e:
                _safe_print(f"  ⚠️  {futures[future]}: {e}")

    audit_data = [
        {
            "dpk_name": dpk.name,
            "dpk_id": dpk.id,
            "dpk_version": getattr(dpk, "version", "?"),
            "installations": installations_by_dpk.get(dpk.name, []),
        }
        for dpk in found
    ]

    # --------------- Print report ---------------
    print("=" * 70)
    print("DEPRECATED NIM DPK AUDIT REPORT")
    print("=" * 70)

    total_apps = sum(len(e["installations"]) for e in audit_data)
    total_models = sum(
        len(inst["models"])
        for e in audit_data
        for inst in e["installations"]
    )
    print(f"  DPKs deprecated:   {len(found)}")
    print(f"  DPKs missing:      {len(missing)}")
    print(f"  Installed apps:    {total_apps}")
    print(f"  Linked models:     {total_models}")
    print()

    for entry in audit_data:
        installs = entry["installations"]
        status = f"{len(installs)} installation(s)" if installs else "not installed anywhere"
        print(f"📦 {entry['dpk_name']}  (id={entry['dpk_id']}, v={entry['dpk_version']})  → {status}")

        for inst in installs:
            print(f"  Project : {inst['project_name']}  (id={inst['project_id']})")
            print(f"  App     : {inst['app_name']}  (id={inst['app_id']}, creator={inst['app_creator']})")
            if inst["models"]:
                print(f"  Models  :")
                for m in inst["models"]:
                    print(f"    - {m['model_name']}  (id={m['model_id']}, creator={m['model_creator']})")
            else:
                print(f"  Models  : (none)")
            print()

    print("=" * 70)
    return audit_data


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "dpk_name", "dpk_id", "dpk_version",
    "project_name", "project_id",
    "app_name", "app_id", "app_creator",
    "model_name", "model_id", "model_creator",
]


def write_audit_csv(audit_data: list[dict], path: str) -> None:
    """
    Write audit data to a CSV file, one row per model.
    Apps with no models produce one row with empty model columns.
    DPKs with no installations produce one row with all installation columns empty.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for entry in audit_data:
            base = {
                "dpk_name": entry["dpk_name"],
                "dpk_id": entry["dpk_id"],
                "dpk_version": entry["dpk_version"],
            }
            if not entry["installations"]:
                writer.writerow({**base, "project_name": "", "project_id": "",
                                 "app_name": "", "app_id": "", "app_creator": "",
                                 "model_name": "", "model_id": "", "model_creator": ""})
                continue
            for inst in entry["installations"]:
                inst_base = {**base,
                             "project_name": inst["project_name"],
                             "project_id": inst["project_id"],
                             "app_name": inst["app_name"],
                             "app_id": inst["app_id"],
                             "app_creator": inst["app_creator"] or ""}
                if not inst["models"]:
                    writer.writerow({**inst_base,
                                     "model_name": "", "model_id": "", "model_creator": ""})
                else:
                    for m in inst["models"]:
                        writer.writerow({**inst_base,
                                         "model_name": m["model_name"],
                                         "model_id": m["model_id"],
                                         "model_creator": m["model_creator"] or ""})
    print(f"\n💾 Audit saved → {path}")


# ---------------------------------------------------------------------------
# 3. Cleanup — delete models → uninstall apps → delete DPK
# ---------------------------------------------------------------------------

def cleanup(dpk_names: list[str], dry_run: bool = True) -> dict:
    """
    For each DPK name, across all accessible projects:
      1. Delete all models linked to the DPK  (packageId filter)
      2. Uninstall all apps linked to the DPK  (dpkName filter)
      3. Delete the DPK from the marketplace

    Args:
        dpk_names: List of DPK names to clean up.
        dry_run:   If True (default), only print what would happen — no deletions.

    Returns:
        Summary dict with counts per DPK.
    """
    if dry_run:
        print("\n🔒 DRY RUN — no changes will be made. Pass dry_run=False to execute.\n")
    else:
        print("\n🚨 LIVE RUN — changes will be applied.\n")

    found, missing = find_deprecated_dpks(dpk_names)

    if missing:
        print(f"ℹ️  Already gone from marketplace (skipping):")
        for name in missing:
            print(f"  - {name}")
        print()

    projects = _all_projects()
    summary = {}

    for dpk in found:
        print(f"\n{'='*60}")
        print(f"DPK: {dpk.name}  (id={dpk.id})")
        print(f"{'='*60}")

        counts = {"models_deleted": 0, "apps_uninstalled": 0, "dpk_deleted": False, "errors": []}

        for project in projects:
            # -- Step 1: delete models linked to this DPK (by packageId) --
            try:
                model_filters = dl.Filters(resource=dl.FiltersResource.MODEL)
                model_filters.add(field="packageId", values=dpk.id)
                models = list(project.models.list(filters=model_filters).all())
                for model in models:
                    print(f"  🗑️  Model: {model.name}  (id={model.id}, project={project.name})")
                    if not dry_run:
                        try:
                            model.delete()
                            counts["models_deleted"] += 1
                        except Exception as e:
                            err = f"Failed to delete model {model.name}: {e}"
                            print(f"      ⚠️  {err}")
                            counts["errors"].append(err)
                    else:
                        counts["models_deleted"] += 1
            except Exception as e:
                err = f"Could not list models in project '{project.name}': {e}"
                print(f"  ⚠️  {err}")
                counts["errors"].append(err)

            # -- Step 2: uninstall apps linked to this DPK (by dpkName) --
            try:
                app_filters = dl.Filters(resource=dl.FiltersResource.APP)
                app_filters.add(field="dpkName", values=dpk.name)
                apps = list(project.apps.list(filters=app_filters).all())
                for app in apps:
                    print(f"  🗑️  App: {app.name}  (id={app.id}, project={project.name}, creator={getattr(app, 'creator', '?')})")
                    if not dry_run:
                        try:
                            app.uninstall()
                            counts["apps_uninstalled"] += 1
                        except Exception as e:
                            err_str = str(e).lower()
                            if "404" in err_str or "not found" in err_str:
                                print(f"      ℹ️  Already uninstalled")
                                counts["apps_uninstalled"] += 1
                            else:
                                err = f"Failed to uninstall app {app.name}: {e}"
                                print(f"      ⚠️  {err}")
                                counts["errors"].append(err)
                    else:
                        counts["apps_uninstalled"] += 1
            except Exception as e:
                err = f"Could not list apps in project '{project.name}': {e}"
                print(f"  ⚠️  {err}")
                counts["errors"].append(err)

        # -- Step 3: delete the DPK itself --
        print(f"  🗑️  DPK: {dpk.name}")
        if not dry_run:
            try:
                dpk.delete()
                counts["dpk_deleted"] = True
                print(f"      ✅ Deleted")
            except Exception as e:
                err = f"Failed to delete DPK {dpk.name}: {e}"
                print(f"      ⚠️  {err}")
                counts["errors"].append(err)
        else:
            counts["dpk_deleted"] = True  # would be deleted

        summary[dpk.name] = counts

    # --------------- Summary ---------------
    print(f"\n{'='*60}")
    label = "DRY RUN SUMMARY" if dry_run else "CLEANUP SUMMARY"
    print(label)
    print(f"{'='*60}")
    total_models = sum(v["models_deleted"] for v in summary.values())
    total_apps = sum(v["apps_uninstalled"] for v in summary.values())
    total_dpks = sum(1 for v in summary.values() if v["dpk_deleted"])
    total_errors = sum(len(v["errors"]) for v in summary.values())
    action = "Would delete" if dry_run else "Deleted"
    print(f"  {action}: {total_models} model(s), {total_apps} app(s), {total_dpks} DPK(s)")
    if total_errors:
        print(f"  Errors: {total_errors}")
    if dry_run:
        print(f"\n  Re-run with --execute (or dry_run=False) to apply.")
    print(f"{'='*60}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Audit and clean up deprecated NIM DPKs from the Dataloop marketplace.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- audit --
    p_audit = sub.add_parser("audit", help="List all installations of deprecated DPKs")
    p_audit.add_argument("dpk_names", nargs="*", help="DPK names to audit")
    p_audit.add_argument(
        "--from-report", metavar="PATH",
        help="Load deprecated DPK names from a run-report JSON (agent/run_data/report_*.json)",
    )
    p_audit.add_argument(
        "--output", metavar="CSV_PATH",
        help="CSV output path (default: audit_<timestamp>.csv next to the script)",
    )
    p_audit.add_argument(
        "--workers", type=int, default=20,
        help="Parallel workers for project scanning (default: 20)",
    )

    # -- cleanup --
    p_clean = sub.add_parser("cleanup", help="Delete models, uninstall apps, delete DPKs")
    p_clean.add_argument("dpk_names", nargs="*", help="DPK names to clean up")
    p_clean.add_argument(
        "--from-report", metavar="PATH",
        help="Load deprecated DPK names from a run-report JSON",
    )
    p_clean.add_argument(
        "--execute", action="store_true",
        help="Actually delete/uninstall (default is dry-run)",
    )

    args = parser.parse_args()

    # Collect DPK names
    names: list[str] = list(args.dpk_names)
    if args.from_report:
        names.extend(_dpk_names_from_report(args.from_report))
    if not names:
        print("Error: provide DPK names directly or via --from-report")
        sys.exit(1)
    names = list(dict.fromkeys(names))  # deduplicate, preserve order

    print(f"\nLogging in to Dataloop...")
    ensure_dataloop_login()

    if args.command == "audit":
        data = audit(names, max_workers=args.workers)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = args.output if args.output else str(
            Path(__file__).parent / f"audit_{ts}.csv"
        )
        write_audit_csv(data, csv_path)

    elif args.command == "cleanup":
        cleanup(names, dry_run=not args.execute)
