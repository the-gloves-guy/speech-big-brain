import sys
from types import SimpleNamespace

import pytest

from lid import cli, guard


def test_guard_sets_kernel_memory_and_swap_limits_and_propagates_failure(monkeypatch):
    calls = []

    def run(command, check):
        calls.append(command)
        assert check is False
        return SimpleNamespace(returncode=137)

    monkeypatch.setattr(guard.subprocess, "run", run)
    assert guard.launch(["worker", "argument"], memory="2G", unit="test-job") == 137
    assert "MemoryMax=2G" in calls[0]
    assert "MemorySwapMax=0" in calls[0]
    assert calls[0][-2:] == ["worker", "argument"]


def test_fleurs_cli_requires_guard_before_starting_work(monkeypatch):
    monkeypatch.delenv("LID_MEMORY_GUARDED", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        ["lid", "cache", "--fleurs", "--split", "test", "--output", "unused.pt"],
    )
    calls = []

    def launch(command, memory):
        calls.append((command, memory))
        return 137  # Failure to run the limited job must never fall back to an unlimited run.

    monkeypatch.setattr(guard, "launch", launch)
    monkeypatch.setattr(
        cli, "lazy", lambda *args: lambda a: pytest.fail("unguarded work started")
    )
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 137
    assert calls[0][1] == "2G"
    assert calls[0][0][1:3] == ["-m", "lid.cli"]


def test_cli_inherits_existing_guard_without_starting_another_scope(monkeypatch):
    monkeypatch.setenv("LID_MEMORY_GUARDED", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["lid", "cache", "--fleurs", "--split", "test", "--output", "unused.pt"],
    )
    calls = []
    monkeypatch.setattr(cli, "lazy", lambda *args: lambda a: calls.append(a.command))
    monkeypatch.setattr(
        guard, "launch", lambda *a, **kw: pytest.fail("unexpected nested scope")
    )
    cli.main()
    assert calls == ["cache"]
