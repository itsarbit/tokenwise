"""Tests for LedgerStore."""

from __future__ import annotations

from pathlib import Path

from tokenwise.ledger_store import LedgerStore
from tokenwise.models import CostLedger


class TestLedgerStore:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Save a record and load it back."""
        store = LedgerStore(path=tmp_path / "ledger.jsonl")
        ledger = CostLedger()
        ledger.record(
            reason="step 1 attempt 1",
            model_id="openai/gpt-4.1-mini",
            input_tokens=100,
            output_tokens=50,
            cost=0.001,
            success=True,
        )
        store.save(task="Test task", ledger=ledger, budget=1.0, success=True)

        records = store.load()
        assert len(records) == 1
        assert records[0]["task"] == "Test task"
        assert records[0]["budget"] == 1.0
        assert records[0]["success"] is True
        assert records[0]["total_cost"] == 0.001
        assert len(records[0]["entries"]) == 1

    def test_multiple_saves(self, tmp_path: Path):
        """Multiple saves should append records."""
        store = LedgerStore(path=tmp_path / "ledger.jsonl")
        ledger = CostLedger()
        ledger.record(reason="s1", model_id="m1", cost=0.01, success=True)

        store.save(task="Task 1", ledger=ledger, budget=1.0, success=True)
        store.save(task="Task 2", ledger=ledger, budget=2.0, success=False)

        records = store.load()
        assert len(records) == 2
        assert records[0]["task"] == "Task 1"
        assert records[1]["task"] == "Task 2"

    def test_load_with_limit(self, tmp_path: Path):
        """Load should respect limit parameter."""
        store = LedgerStore(path=tmp_path / "ledger.jsonl")
        ledger = CostLedger()
        for i in range(10):
            store.save(task=f"Task {i}", ledger=ledger, budget=1.0, success=True)

        records = store.load(limit=3)
        assert len(records) == 3
        assert records[0]["task"] == "Task 7"  # last 3

    def test_load_missing_file(self, tmp_path: Path):
        """Loading from a non-existent file should return empty list."""
        store = LedgerStore(path=tmp_path / "nonexistent.jsonl")
        assert store.load() == []

    def test_summary_aggregation(self, tmp_path: Path):
        """Summary should aggregate across all records."""
        store = LedgerStore(path=tmp_path / "ledger.jsonl")
        ledger1 = CostLedger()
        ledger1.record(reason="s1", model_id="m1", cost=0.01, success=True)

        ledger2 = CostLedger()
        ledger2.record(reason="s1", model_id="m1", cost=0.005, success=False)
        ledger2.record(reason="s1 esc", model_id="m2", cost=0.02, success=True)

        store.save(task="T1", ledger=ledger1, budget=1.0, success=True, planner_cost=0.001)
        store.save(task="T2", ledger=ledger2, budget=0.5, success=True, planner_cost=0.002)

        stats = store.summary()
        assert stats["num_tasks"] == 2
        assert stats["num_succeeded"] == 2
        assert stats["num_failed"] == 0
        assert stats["total_spend"] == 0.01 + 0.001 + 0.025 + 0.002
        assert stats["total_wasted"] == 0.005
        assert stats["total_planner_spend"] == 0.003

    def test_summary_empty(self, tmp_path: Path):
        """Summary on empty store should return zeros."""
        store = LedgerStore(path=tmp_path / "nonexistent.jsonl")
        stats = store.summary()
        assert stats["num_tasks"] == 0
        assert stats["total_spend"] == 0.0

    def test_creates_parent_dirs(self, tmp_path: Path):
        """Save should create parent directories if they don't exist."""
        deep_path = tmp_path / "a" / "b" / "c" / "ledger.jsonl"
        store = LedgerStore(path=deep_path)
        ledger = CostLedger()
        store.save(task="Test", ledger=ledger, budget=1.0, success=True)
        assert deep_path.exists()

    def test_planner_cost_in_total(self, tmp_path: Path):
        """Planner cost should be included in total_cost."""
        store = LedgerStore(path=tmp_path / "ledger.jsonl")
        ledger = CostLedger()
        ledger.record(reason="s1", model_id="m1", cost=0.01, success=True)
        store.save(task="T1", ledger=ledger, budget=1.0, success=True, planner_cost=0.005)

        records = store.load()
        assert records[0]["total_cost"] == 0.015
        assert records[0]["planner_cost"] == 0.005
