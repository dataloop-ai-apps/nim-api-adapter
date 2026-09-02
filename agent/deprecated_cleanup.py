"""
Deprecated NIM DPK Cleanup

CLI:
  python deprecated_cleanup.py find  [DPK_NAME ...] [--from-report PATH] [--output CSV] [--env rc|prod]
  python deprecated_cleanup.py clean [DPK_NAME ...] [--from-report PATH] [--execute] [--output CSV] [--env rc|prod]

Examples:
  python deprecated_cleanup.py find  --from-report agent/run_data/report_20260827_204809.json
  python deprecated_cleanup.py clean --from-report agent/run_data/report_20260827_204809.json --execute
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import dtlpy as dl
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

_CSV_FIELDS = ["dpk_name", "dpk_id", "app_name", "app_id", "project", "model_name", "model_id"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _names_from_report(path: str) -> list[str]:
    """Extract deprecated DPK names from a run-report JSON."""
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    names = []
    for section in ("api_deprecated", "downloadable_deprecated"):
        for entry in report.get(section) or []:
            name = entry.get("name") if isinstance(entry, dict) else str(entry)
            if name and name not in names:
                names.append(name)
    return names


def _dpk_rows(dpk: dl.Dpk) -> list[dict]:
    """Flat rows for CSV — one row per model, per app if no models, per DPK if no apps."""
    rows = []
    for app in dl.apps.list(filters=dl.Filters(field="dpkName", values=dpk.name, resource="apps")).all():
        base = {"dpk_name": dpk.name, "dpk_id": dpk.id,
                "app_name": app.name, "app_id": app.id, "project": app.project.name}
        models = list(dl.models.list(filters=dl.Filters(field="app.id", values=app.id, resource="models")).all())
        for model in models:
            rows.append(base | {"model_name": model.name, "model_id": model.id})
        if not models:
            rows.append(base | {"model_name": "", "model_id": ""})
    if not rows:
        rows.append({"dpk_name": dpk.name, "dpk_id": dpk.id,
                     "app_name": "", "app_id": "", "project": "", "model_name": "", "model_id": ""})
    return rows


def write_csv(rows: list[dict], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# find
# ---------------------------------------------------------------------------

def find_dpk(dpk_name: str) -> dl.Dpk | None:
    try:
        dpk = dl.dpks.get(dpk_name=dpk_name)
    except dl.exceptions.NotFound:
        log.warning("DPK not found: %s", dpk_name)
        return None

    print(f"\nDPK: {dpk.name}  id={dpk.id}")

    apps = list(dl.apps.list(filters=dl.Filters(field="dpkName", values=dpk.name, resource="apps")).all())
    print(f"\n--- Apps ({len(apps)}) ---")
    for app in apps:
        print(f"  app: {app.name}  id={app.id}  project={app.project.name}")
        models = list(dl.models.list(filters=dl.Filters(field="app.id", values=app.id, resource="models")).all())
        print(f"    models ({len(models)}):")
        for model in models:
            print(f"      model: {model.name}  id={model.id}  creator={model.creator}  project={model.project.name}")

    revisions = list(dpk.revisions.all())
    print(f"\n--- Revisions ({len(revisions)}) ---")
    for rev in revisions:
        print(f"  revision: {rev.version}  id={rev.id}")

    return dpk


# ---------------------------------------------------------------------------
# clean
# ---------------------------------------------------------------------------

def clean_dpk(dpk_name: str, dry_run: bool = True) -> dict:
    counts = {"models_deleted": 0, "apps_uninstalled": 0, "revisions_deleted": 0, "errors": []}

    try:
        dpk = dl.dpks.get(dpk_name=dpk_name)
    except dl.exceptions.NotFound:
        log.warning("DPK not found (already deleted?): %s", dpk_name)
        return counts

    print(f"\n{'='*60}")
    print(f"DPK: {dpk.name}  id={dpk.id}")
    print(f"{'='*60}")

    for app in dl.apps.list(filters=dl.Filters(field="dpkName", values=dpk.name, resource="apps")).all():
        print(f"  App: {app.name}  project={app.project.name}")

        for model in dl.models.list(filters=dl.Filters(field="app.id", values=app.id, resource="models")).all():
            print(f"    Deleting model: {model.name}  id={model.id}")
            if not dry_run:
                try:
                    model.delete()
                    counts["models_deleted"] += 1
                except Exception as e:
                    log.error("Failed to delete model %s: %s", model.name, e)
                    counts["errors"].append(f"model {model.name}: {e}")
            else:
                counts["models_deleted"] += 1

        print(f"    Uninstalling app: {app.name}")
        if not dry_run:
            try:
                app.uninstall()
                counts["apps_uninstalled"] += 1
            except Exception as e:
                err_str = str(e).lower()
                if "404" in err_str or "not found" in err_str:
                    log.info("App already uninstalled: %s", app.name)
                    counts["apps_uninstalled"] += 1
                else:
                    log.error("Failed to uninstall app %s: %s", app.name, e)
                    counts["errors"].append(f"app {app.name}: {e}")
        else:
            counts["apps_uninstalled"] += 1

    revisions = list(dpk.revisions.all())
    print(f"  Deleting {len(revisions)} revision(s)...")
    for rev in revisions:
        if not dry_run:
            try:
                rev.delete()
                counts["revisions_deleted"] += 1
            except Exception as e:
                log.error("Failed to delete revision %s: %s", rev.version, e)
                counts["errors"].append(f"revision {rev.version}: {e}")
        else:
            counts["revisions_deleted"] += 1

    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Find and clean up deprecated NIM DPKs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join([
            "Examples:",
            "  python deprecated_cleanup.py find  --from-report agent/run_data/report_*.json",
            "  python deprecated_cleanup.py clean --from-report agent/run_data/report_*.json --execute",
        ]),
    )
    parser.add_argument("--env", choices=["rc", "prod"], default="rc")
    sub = parser.add_subparsers(dest="command", required=True)

    p_find = sub.add_parser("find", help="List apps, models, and revisions for deprecated DPKs")
    p_find.add_argument("dpk_names", nargs="*")
    p_find.add_argument("--from-report", metavar="PATH", help="Load DPK names from a run-report JSON")
    p_find.add_argument("--output", metavar="CSV_PATH", help="Save results to CSV")

    p_clean = sub.add_parser("clean", help="Delete models, uninstall apps, delete revisions")
    p_clean.add_argument("dpk_names", nargs="*")
    p_clean.add_argument("--from-report", metavar="PATH", help="Load DPK names from a run-report JSON")
    p_clean.add_argument("--execute", action="store_true", help="Actually delete (default is dry-run)")
    p_clean.add_argument("--output", metavar="CSV_PATH", help="Save cleanup report to CSV")

    args = parser.parse_args()

    names: list[str] = list(args.dpk_names)
    if args.from_report:
        names.extend(_names_from_report(args.from_report))
    names = list(dict.fromkeys(names))
    if not names:
        parser.error("provide DPK names directly or via --from-report")

    dl.setenv(args.env)
    if dl.token_expired():
        dl.login()

    if args.command == "find":
        rows = []
        for name in names:
            dpk = find_dpk(name)
            if dpk and args.output:
                rows.extend(_dpk_rows(dpk))
        if args.output:
            write_csv(rows, args.output)

    elif args.command == "clean":
        dry_run = not args.execute
        if dry_run:
            print("\nDRY RUN - no changes. Pass --execute to apply.\n")
        else:
            print("\nLIVE RUN - changes will be applied.\n")

        rows, total = [], {"models_deleted": 0, "apps_uninstalled": 0, "revisions_deleted": 0, "errors": []}
        for name in names:
            if args.output:
                try:
                    rows.extend(_dpk_rows(dl.dpks.get(dpk_name=name)))
                except dl.exceptions.NotFound:
                    pass
            counts = clean_dpk(name, dry_run=dry_run)
            for key in ("models_deleted", "apps_uninstalled", "revisions_deleted"):
                total[key] += counts[key]
            total["errors"].extend(counts["errors"])

        print(f"\n{'='*60}")
        action = "Would delete" if dry_run else "Deleted"
        print(f"{action}: {total['models_deleted']} model(s), {total['apps_uninstalled']} app(s), {total['revisions_deleted']} revision(s)")
        if total["errors"]:
            log.warning("%d error(s) — see above", len(total["errors"]))
        if dry_run:
            print("Re-run with --execute to apply.")
        print(f"{'='*60}")

        if args.output and rows:
            write_csv(rows, args.output)


if __name__ == "__main__":
    main()
