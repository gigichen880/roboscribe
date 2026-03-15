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
    failed_phase: str | None = None  # which phase of the state machine failed


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
        if env_name == "Door":
            return _diagnose_door(result)

        if env_name == "NutAssemblySquare":
            return _diagnose_nut_assembly(result)

        if env_name in ("Lift", "Stack", "PickPlaceCan"):
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


def _diagnose_door(result: TrajectoryResult) -> Diagnosis:
    """Diagnose Door env failures using obs snapshots from trajectory."""
    summary = result.trajectory_summary

    # Check if handle_qpos changed (handle turned)
    handle_turned = "handle_qpos" in summary and "Changed:" in summary and "handle_qpos" in summary.split("Changed:")[-1]
    # Check if hinge_pos changed (door opened)
    door_moved = "hinge_pos" in summary and "Changed:" in summary and "hinge_pos" in summary.split("Changed:")[-1]

    if not handle_turned:
        return Diagnosis(
            category="HANDLE_NOT_TURNED",
            summary="Robot did not turn the handle — check grasp and rotation",
            details=result.trajectory_summary,
            suggestions=(
                "The handle_qpos did not change, meaning the robot never successfully turned the handle. "
                "Check the trajectory Start/End obs values above. Common causes:\n"
                "1. GRASP FAILED: eef didn't reach handle_pos accurately — check approach/reach alignment\n"
                "2. GRIPPER NOT CLOSED: gripper must be fully closed (action[6]=1.0) and WAIT 15 steps before rotating\n"
                "3. WRONG ROTATION AXIS: handle turns via action[3] (rotation around x-axis), NOT action[4] or action[5]\n"
                "4. PID RESET BUG: do NOT call pid.reset() every step — only reset once per phase transition\n"
                "5. Check handle_pos in the obs — align eef_pos with handle_pos PRECISELY (error < 0.02) before gripping"
            ),
        )

    if handle_turned and not door_moved:
        return Diagnosis(
            category="HANDLE_TURNED_DOOR_STUCK",
            summary="Handle was turned but door didn't open — need to pull",
            details=result.trajectory_summary,
            suggestions=(
                "The handle_qpos changed (handle was turned) but hinge_pos barely moved (door didn't open). "
                "After turning the handle, you MUST pull the door open. Common causes:\n"
                "1. NO PULL PHASE: add a phase that moves eef in +Y direction (or toward robot) while maintaining grip\n"
                "2. HANDLE NOT FULLY TURNED: handle_qpos must reach ~1.0-1.5 before the latch releases — keep rotating longer\n"
                "3. LOST GRIP: the robot may have released the handle before pulling — keep action[6]=1.0 during pull\n"
                "4. PULL DIRECTION WRONG: check which axis moves the door open — typically +Y or toward the robot base"
            ),
        )

    if handle_turned and door_moved:
        return Diagnosis(
            category="DOOR_PARTIALLY_OPEN",
            summary="Door partially opened but not enough for success",
            details=result.trajectory_summary,
            suggestions=(
                "Handle was turned and door moved, but hinge_pos didn't reach the success threshold (~0.3 rad). "
                "The door needs to open further. Common fixes:\n"
                "1. PULL HARDER/LONGER: increase the pull force or duration\n"
                "2. MAINTAIN GRIP: ensure gripper stays closed (action[6]=1.0) throughout the pull phase\n"
                "3. COMBINE ROTATE + PULL: continue rotating the handle while pulling to keep the latch disengaged\n"
                "4. Check hinge_pos in the End obs — how close is it to 0.3?"
            ),
        )

    # Fallback
    return Diagnosis(
        category="DOOR_PARTIAL_PROGRESS",
        summary="Partial progress on door task — check trajectory obs for details",
        details=result.trajectory_summary,
        suggestions=(
            "Review the Start/End obs in the trajectory above. Key values to check:\n"
            "- handle_qpos: 0 → >1.0 means handle turned (good)\n"
            "- hinge_pos: 0 → >0.3 means door opened (success)\n"
            "- robot0_eef_pos vs handle_pos: should be very close during grip/rotate phases\n"
            "The Door task sequence: approach handle → grip → turn handle → pull door open"
        ),
    )


def _diagnose_nut_assembly(result: TrajectoryResult) -> Diagnosis:
    """Diagnose NutAssemblySquare env failures using obs snapshots."""
    summary = result.trajectory_summary

    # Check if nut was lifted (SquareNut_pos z changed)
    nut_lifted = False
    if "SquareNut_pos" in summary and "Changed:" in summary:
        nut_lifted = "SquareNut_pos" in summary.split("Changed:")[-1]

    if not nut_lifted:
        return Diagnosis(
            category="MISSED_GRASP",
            summary="Robot fails to grasp the nut — check grasp height",
            details=result.trajectory_summary,
            suggestions=(
                "CRITICAL: The nut is VERY THIN (~2cm). You must lower the gripper BELOW the nut center. "
                "Look at SquareNut_pos in the Start/End obs — if z didn't change, the grasp failed.\n"
                "Grasp height should be SquareNut_pos[2] - 0.01 (almost table level, ~z=0.82). "
                "The peg is at FIXED position x=0.23, y=0.10, z=0.85 (NOT in obs, hardcode it). "
                "Sequence: (1) align over nut, (2) lower to z=nut_z-0.01, (3) close gripper + wait 25 steps, "
                "(4) lift to z>1.1, (5) move to x=0.23 y=0.10, (6) lower to z=0.88, (7) open gripper."
            ),
        )

    return Diagnosis(
        category="PLACEMENT_FAILED",
        summary="Nut was grasped and lifted but placement on peg failed",
        details=result.trajectory_summary,
        suggestions=(
            "The nut was lifted (SquareNut_pos z changed) but placement on the peg failed. "
            "Check SquareNut_pos in End obs vs the peg position (x=0.23, y=0.10). Common fixes:\n"
            "1. ALIGN OVER PEG: move to x=0.23, y=0.10 BEFORE lowering\n"
            "2. LOWER CAREFULLY: lower to z=0.88 (just above peg top)\n"
            "3. OPEN GRIPPER: set action[6]=-1.0 to release the nut\n"
            "4. WAIT: the nut needs a few steps to settle on the peg"
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
