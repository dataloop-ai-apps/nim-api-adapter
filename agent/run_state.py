"""
Run State Persistence for NIM Agent.

Tracks per-model history, quarantine, and run summaries across agent runs.
State is stored as JSON and persisted between CI runs via GitHub Action artifacts.
"""

import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path


DEFAULT_STATE_PATH = Path("agent/output/run_state.json")

DEFAULT_CONFIG = {
    "quarantine_after": 3,
    "probe_sample_size": 10,
    "quarantine_cooldown_days": 14,
    "pr_max_failure_rate": 0.80,
    "anomaly_deprecation_threshold": 0.50,
}


def classify_error(error: str) -> str:
    """
    Classify an error string into a category.

    Returns:
        "permanent"  -- 404, model not found (quarantine after N failures)
        "transient"  -- timeout, rate limit, 5xx (retry next run)
        "environment" -- auth/key issues (abort entire run)
    """
    if not error:
        return "transient"
    err = error.lower()
    if any(x in err for x in ["404", "not found", "no such model", "does not exist",
                               "not available", "model_not_found"]):
        return "permanent"
    if any(x in err for x in ["api key", "auth", "unauthorized", "forbidden",
                               "401", "403", "ngc_api_key"]):
        return "environment"
    if any(x in err for x in ["timeout", "rate limit", "429", "too many requests",
                               "502", "503", "504", "connection", "reset by peer"]):
        return "transient"
    return "transient"


class RunState:
    """Persisted state across NIM Agent runs."""

    def __init__(self, path: str = None):
        self.path = Path(path) if path else DEFAULT_STATE_PATH
        self.data = {
            "runs": [],
            "models": {},
            "config": dict(DEFAULT_CONFIG),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self):
        """Load state from disk. No-op if file doesn't exist."""
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                self.data = json.load(f)
            if "config" not in self.data:
                self.data["config"] = dict(DEFAULT_CONFIG)
        return self

    def save(self):
        """Write state to disk (creates parent dirs if needed)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Per-model tracking
    # ------------------------------------------------------------------

    def _get_model(self, model_id: str) -> dict:
        if model_id not in self.data["models"]:
            self.data["models"][model_id] = {
                "status": "pending",
                "last_error": None,
                "error_type": None,
                "consecutive_failures": 0,
                "total_attempts": 0,
                "last_attempt": None,
                "first_failure": None,
                "last_success": None,
            }
        return self.data["models"][model_id]

    def record_result(self, model_id: str, status: str, error: str = None):
        """
        Record the outcome of an onboarding attempt.

        Args:
            model_id: NVIDIA model ID
            status: "success", "error", or "skipped"
            error: Error message (only for status="error")
        """
        m = self._get_model(model_id)
        now = datetime.now().isoformat()
        m["last_attempt"] = now
        m["total_attempts"] = m.get("total_attempts", 0) + 1

        if status == "success":
            m["status"] = "success"
            m["consecutive_failures"] = 0
            m["last_success"] = now
            m["last_error"] = None
            m["error_type"] = None
        elif status == "error":
            error_type = classify_error(error or "")
            m["last_error"] = (error or "")[:500]
            m["error_type"] = error_type
            m["consecutive_failures"] = m.get("consecutive_failures", 0) + 1
            if m["first_failure"] is None:
                m["first_failure"] = now

            quarantine_after = self.data["config"].get("quarantine_after", 3)
            if error_type == "permanent" and m["consecutive_failures"] >= quarantine_after:
                m["status"] = "quarantined"
            else:
                m["status"] = "failed"

    def should_attempt(self, model_id: str) -> bool:
        """Return False if the model is quarantined and not due for re-probe."""
        m = self.data["models"].get(model_id)
        if m is None:
            return True
        if m["status"] != "quarantined":
            return True

        cooldown_days = self.data["config"].get("quarantine_cooldown_days", 14)
        last = m.get("last_attempt")
        if last:
            try:
                last_dt = datetime.fromisoformat(last)
                if datetime.now() - last_dt > timedelta(days=cooldown_days):
                    return True
            except (ValueError, TypeError):
                pass
        return False

    def get_quarantined(self) -> list[str]:
        """Return list of currently quarantined model IDs."""
        return [
            mid for mid, m in self.data["models"].items()
            if m.get("status") == "quarantined"
        ]

    def pick_probe_sample(self, n: int = None) -> list[str]:
        """
        Pick a random sample of quarantined models to re-test.

        This lets the agent discover if quarantined models have come back online
        without re-testing the entire quarantine list every run.
        """
        n = n or self.data["config"].get("probe_sample_size", 10)
        quarantined = self.get_quarantined()
        if len(quarantined) <= n:
            return quarantined
        return random.sample(quarantined, n)

    def clear_quarantine(self, model_id: str):
        """Un-quarantine a model (reset to pending)."""
        m = self.data["models"].get(model_id)
        if m:
            m["status"] = "pending"
            m["consecutive_failures"] = 0
            m["first_failure"] = None

    # ------------------------------------------------------------------
    # Run-level tracking
    # ------------------------------------------------------------------

    def start_run(self) -> dict:
        """Mark the beginning of a new run. Returns the run record."""
        run = {
            "started": datetime.now().isoformat(),
            "finished": None,
            "status": "running",
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "quarantined_total": len(self.get_quarantined()),
            "skipped_quarantined": 0,
            "probed": 0,
        }
        self.data["runs"].append(run)
        return run

    def end_run(self, summary: dict):
        """Finalize the current run with summary data."""
        if self.data["runs"]:
            run = self.data["runs"][-1]
            run["finished"] = datetime.now().isoformat()
            run.update(summary)

    def last_run_summary(self) -> dict | None:
        """Return the most recent run record, or None."""
        return self.data["runs"][-1] if self.data["runs"] else None

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    @property
    def pr_max_failure_rate(self) -> float:
        return self.data["config"].get("pr_max_failure_rate", 0.80)

    @property
    def anomaly_deprecation_threshold(self) -> float:
        return self.data["config"].get("anomaly_deprecation_threshold", 0.50)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def print_status(self):
        """Print a human-readable summary of the current state."""
        quarantined = self.get_quarantined()
        total = len(self.data["models"])
        success = sum(1 for m in self.data["models"].values() if m["status"] == "success")
        failed = sum(1 for m in self.data["models"].values() if m["status"] == "failed")

        print("=" * 60)
        print("NIM Agent State")
        print("=" * 60)
        print(f"  State file:  {self.path}")
        print(f"  Total runs:  {len(self.data['runs'])}")
        print(f"  Models tracked: {total}")
        print(f"    Success:      {success}")
        print(f"    Failed:       {failed}")
        print(f"    Quarantined:  {len(quarantined)}")

        last = self.last_run_summary()
        if last:
            print(f"\n  Last run:")
            print(f"    Started:  {last.get('started', '?')}")
            print(f"    Status:   {last.get('status', '?')}")
            print(f"    Attempted: {last.get('attempted', '?')}")
            print(f"    Succeeded: {last.get('succeeded', '?')}")
            print(f"    Failed:    {last.get('failed', '?')}")

        if quarantined:
            print(f"\n  Quarantined models ({len(quarantined)}):")
            for mid in quarantined[:20]:
                m = self.data["models"][mid]
                print(f"    - {mid}  ({m.get('last_error', '?')[:60]})")
            if len(quarantined) > 20:
                print(f"    ... and {len(quarantined) - 20} more")

        print("=" * 60)
