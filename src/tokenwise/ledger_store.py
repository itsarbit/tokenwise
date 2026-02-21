"""Persistent JSONL-based ledger store for cross-session spend tracking."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tokenwise.models import CostLedger

logger = logging.getLogger(__name__)

_DEFAULT_LEDGER_PATH = Path.home() / ".config" / "tokenwise" / "ledger.jsonl"


class LedgerStore:
    """Append-only JSONL store for plan execution records."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _DEFAULT_LEDGER_PATH

    def save(
        self,
        task: str,
        ledger: CostLedger,
        budget: float,
        success: bool,
        planner_cost: float = 0.0,
    ) -> None:
        """Append one JSON line recording a plan execution."""
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "budget": budget,
            "success": success,
            "total_cost": ledger.total_cost + planner_cost,
            "planner_cost": planner_cost,
            "wasted_cost": ledger.wasted_cost,
            "entries": [e.model_dump() for e in ledger.entries],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def load(self, limit: int = 50) -> list[dict[str, Any]]:
        """Read and return most recent N records."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed ledger line")
        return records[-limit:]

    def summary(self) -> dict[str, Any]:
        """Return aggregate spend statistics."""
        records = self.load(limit=0)  # 0 = load all via [-0:] = all
        if not records:
            return {
                "total_spend": 0.0,
                "total_wasted": 0.0,
                "total_planner_spend": 0.0,
                "num_tasks": 0,
                "num_succeeded": 0,
                "num_failed": 0,
            }
        # load(limit=0) returns records[-0:] which is all records
        all_records = self._load_all()
        return {
            "total_spend": sum(r.get("total_cost", 0.0) for r in all_records),
            "total_wasted": sum(r.get("wasted_cost", 0.0) for r in all_records),
            "total_planner_spend": sum(r.get("planner_cost", 0.0) for r in all_records),
            "num_tasks": len(all_records),
            "num_succeeded": sum(1 for r in all_records if r.get("success")),
            "num_failed": sum(1 for r in all_records if not r.get("success")),
        }

    def _load_all(self) -> list[dict[str, Any]]:
        """Load all records without limit."""
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed ledger line")
        return records
