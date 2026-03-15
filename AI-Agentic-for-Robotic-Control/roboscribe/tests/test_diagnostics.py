"""Tests for failure diagnostics."""

from roboscribe.sim.diagnostics import diagnose_failure
from roboscribe.sim.trajectory import TrajectoryResult


def test_diagnose_code_error():
    result = TrajectoryResult(error="SyntaxError: invalid syntax", error_type="CODE_ERROR")
    diag = diagnose_failure(result, "Lift")
    assert diag.category == "CODE_ERROR"
    assert "syntax" in diag.suggestions.lower() or "get_action" in diag.suggestions


def test_diagnose_timeout():
    result = TrajectoryResult(error="Simulation timed out", error_type="TIMEOUT")
    diag = diagnose_failure(result, "Lift")
    assert diag.category == "TIMEOUT"
    assert "loop" in diag.suggestions.lower()


def test_diagnose_runtime_error():
    result = TrajectoryResult(error="Shape mismatch", error_type="RUNTIME_ERROR")
    diag = diagnose_failure(result, "Lift")
    assert diag.category == "RUNTIME_ERROR"


def test_diagnose_no_movement():
    result = TrajectoryResult(
        success_rate=0.0,
        total_episodes=10,
        episode_rewards=[0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01],
    )
    diag = diagnose_failure(result, "Lift")
    assert diag.category == "NO_MOVEMENT"


def test_diagnose_missed_grasp():
    result = TrajectoryResult(
        success_rate=0.0,
        total_episodes=10,
        episode_rewards=[1.5, 1.8, 1.2, 1.6, 1.4, 1.3, 1.7, 1.5, 1.6, 1.4],
    )
    diag = diagnose_failure(result, "Lift")
    assert diag.category == "MISSED_GRASP"


def test_diagnose_partial_success():
    result = TrajectoryResult(
        success_rate=0.6,
        successes=6,
        total_episodes=10,
    )
    diag = diagnose_failure(result, "Lift")
    assert diag.category == "INCONSISTENT"
    assert "60%" in diag.summary
