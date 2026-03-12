"""Failure diagnostics — categorize and explain simulation failures."""

from __future__ import annotations

from dataclasses import dataclass

from roboscribe.sim.trajectory import TrajectoryResult


@dataclass
class Diagnosis:
    """Diagnosis of a simulation failure."""

    category: str       # e.g. CODE_ERROR, TIMEOUT, MISSED_GRASP, etc.
    summary: str        # one-line summary
    details: str        # full explanation
    suggestions: str    # actionable suggestions for the LLM


def diagnose_failure(result: TrajectoryResult, env_name: str) -> Diagnosis:
    """Analyze a TrajectoryResult and produce a structured diagnosis."""

    # Code error — policy couldn't even load or run
    if result.error_type == "CODE_ERROR":
        return Diagnosis(
            category="CODE_ERROR",
            summary="Policy code has an error and could not execute",
            details=result.error,
            suggestions=(
                "Fix the syntax or import error in the policy code. "
                "Make sure get_action(obs) is defined and returns a numpy array of the correct dimension. "
                "Check that all observation keys used actually exist in the obs dict."
            ),
        )

    # Timeout
    if result.error_type == "TIMEOUT":
        return Diagnosis(
            category="TIMEOUT",
            summary="Simulation timed out — possible infinite loop or very slow execution",
            details=result.error,
            suggestions=(
                "Check for infinite loops in get_action(). "
                "Make sure the function returns quickly (no heavy computation per step). "
                "Ensure the state machine transitions correctly and doesn't get stuck."
            ),
        )

    # Runtime error during simulation
    if result.error_type == "RUNTIME_ERROR":
        return Diagnosis(
            category="RUNTIME_ERROR",
            summary="Runtime error during simulation",
            details=result.error,
            suggestions=(
                "Check that the action array has the correct shape (7,) for OSC_POSE. "
                "Verify observation key names match the environment. "
                "Make sure numpy operations don't produce NaN or Inf values."
            ),
        )

    # Policy ran but had 0% success — analyze trajectory
    if result.success_rate == 0.0:
        return _diagnose_zero_success(result, env_name)

    # Partial success
    if result.success_rate < 0.8:
        return _diagnose_partial_success(result, env_name)

    # Should not reach here (success >= 0.8), but just in case
    return Diagnosis(
        category="NEAR_SUCCESS",
        summary=f"Policy achieves {result.success_rate:.0%} — close to target",
        details=result.trajectory_summary,
        suggestions="Minor tuning may improve reliability. Check edge cases in positioning.",
    )


def _diagnose_zero_success(result: TrajectoryResult, env_name: str) -> Diagnosis:
    """Diagnose a 0% success rate."""
    summary_lower = result.trajectory_summary.lower()

    # Check for no movement
    rewards = result.episode_rewards
    if rewards and all(r < 0.1 for r in rewards):
        return Diagnosis(
            category="NO_MOVEMENT",
            summary="Robot does not move meaningfully — rewards near zero",
            details=result.trajectory_summary,
            suggestions=(
                "The policy may be outputting zero or near-zero actions. "
                "Make sure get_action returns non-zero values for the position dimensions. "
                "Check that the state machine starts in the correct initial state. "
                "Verify the action scale — OSC_POSE expects small deltas (typically 0.01-0.1 range)."
            ),
        )

    # Some reward but no success — likely missed grasp or wrong target
    if rewards and max(rewards) > 1.0:
        if env_name == "NutAssemblySquare":
            # Check if nut was lifted (nut z changed from trajectory)
            summary = result.trajectory_summary
            if "SquareNut_pos" in summary:
                return Diagnosis(
                    category="MISSED_GRASP",
                    summary="Robot fails to grasp or place the nut — check grasp height and peg alignment",
                    details=result.trajectory_summary,
                    suggestions=(
                        "CRITICAL: The nut is VERY THIN (~2cm). You must lower the gripper BELOW the nut center. "
                        "Grasp height should be SquareNut_pos[2] - 0.01 (almost table level, ~z=0.82). "
                        "The peg is at FIXED position x=0.23, y=0.10, z=0.85 (NOT in obs, hardcode it). "
                        "Look at the trajectory data above: compare SquareNut_pos z at start vs end. "
                        "If nut z didn't change, the grasp failed — go LOWER. "
                        "If nut was lifted but success=False, the placement on peg failed — check peg x,y alignment. "
                        "Sequence: (1) align over nut, (2) lower to z=nut_z-0.01, (3) close gripper + wait 25 steps, "
                        "(4) lift to z>1.1, (5) move to x=0.23 y=0.10, (6) lower to z=0.88, (7) open gripper."
                    ),
                )
        if env_name in ("Lift", "Stack", "PickPlaceCan", "NutAssemblySquare"):
            return Diagnosis(
                category="MISSED_GRASP",
                summary="Robot moves toward object but fails to grasp",
                details=result.trajectory_summary,
                suggestions=(
                    "The robot is reaching the object area but not grasping successfully. "
                    "Common fixes: (1) improve x,y alignment before lowering — align eef with object center, "
                    "(2) lower further — the grasp height may be too high, "
                    "(3) wait a few steps after closing gripper before lifting, "
                    "(4) close gripper fully (action[-1] = 1.0, not 0.5)."
                ),
            )

        return Diagnosis(
            category="PARTIAL_PROGRESS",
            summary="Some progress but task not completed",
            details=result.trajectory_summary,
            suggestions=(
                "The robot makes progress but doesn't complete the task. "
                "Review the full task sequence and ensure all phases are implemented. "
                "Check state machine transitions — the policy may be getting stuck in an intermediate state."
            ),
        )

    return Diagnosis(
        category="UNKNOWN_FAILURE",
        summary="Policy runs without errors but achieves 0% success",
        details=result.trajectory_summary,
        suggestions=(
            "Review the task requirements carefully. "
            "Add debug prints to understand what observations the robot sees. "
            "Check that the state machine covers the full task sequence. "
            "Verify action dimensions and scale."
        ),
    )


def _diagnose_partial_success(result: TrajectoryResult, env_name: str) -> Diagnosis:
    """Diagnose partial success (0 < rate < 0.8)."""
    return Diagnosis(
        category="INCONSISTENT",
        summary=f"Policy succeeds {result.success_rate:.0%} of the time — inconsistent",
        details=result.trajectory_summary,
        suggestions=(
            "The policy works sometimes but not reliably. Common causes: "
            "(1) Hardcoded positions that don't adapt to object variation — use obs values instead. "
            "(2) Timing issues — add more tolerance in position checks. "
            "(3) Grasp reliability — ensure gripper is fully closed and wait before lifting. "
            "(4) State machine thresholds too tight — widen distance thresholds."
        ),
    )
