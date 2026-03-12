"""Prompt templates for policy generation and revision."""

from __future__ import annotations

from roboscribe.sim.env_registry import EnvInfo
from roboscribe.sim.diagnostics import Diagnosis
from roboscribe.agent.few_shot import FEW_SHOT_EXAMPLES


SYSTEM_PROMPT = """\
You are RoboScribe, an expert robotics programmer that generates scripted policies for robosuite.

## Your Task
Generate a Python function `get_action(obs) -> np.ndarray` that controls a robot in robosuite simulation.

## robosuite API Reference

### Controller: OSC_POSE
The action space is a 7-dimensional numpy array:
- action[0:3] = (dx, dy, dz) — end-effector position delta in world frame
- action[3:6] = (dax, day, daz) — end-effector orientation delta (axis-angle)
- action[6] = gripper command: -1.0 = fully open, 1.0 = fully closed

Actions are deltas (small displacements per step), not absolute positions.
Typical action magnitudes: 0.01 to 0.1 for position, use proportional control.

### Observations
The `obs` parameter is a dict with string keys mapping to numpy arrays.
Common keys:
- `robot0_eef_pos`: (3,) end-effector position [x, y, z]
- `robot0_eef_quat`: (4,) end-effector orientation quaternion
- `robot0_gripper_qpos`: (2,) gripper joint positions
- Object-specific keys vary per environment

### Important Notes
- Use proportional control: action = gain * (target - current)
- Typical gains: 5.0-15.0 for position control
- Always clip actions to [-1, 1] range
- Use a state machine pattern for multi-step tasks
- Gripper needs ~10-15 steps to fully close — wait before lifting

## Output Format
Return ONLY a Python code block with:
1. A `get_action(obs)` function that takes an observation dict and returns a 7D numpy action
2. A `reset()` function that resets any state between episodes
3. Use module-level variables for state machine state
4. Import only numpy (already available)
"""


def build_generation_prompt(
    task_description: str,
    env_info: EnvInfo,
) -> str:
    """Build the user prompt for initial policy generation."""
    parts = [
        f"## Task\n{task_description}\n",
        f"## Environment: {env_info.name}",
        f"Description: {env_info.description}",
        f"Goal: {env_info.goal_description}",
        f"\n## Available Observation Keys\n{env_info.obs_keys_str}",
        f"\n## Tips\n{env_info.tips}",
    ]

    # Add few-shot example if available
    example_env = _get_best_example(env_info.name)
    if example_env:
        parts.append(
            f"\n## Example: Working policy for {example_env}\n"
            f"Use this as a reference for the state-machine pattern and action format:\n"
            f"```python\n{FEW_SHOT_EXAMPLES[example_env]}\n```"
        )

    parts.append(
        "\n## Instructions\n"
        "Generate a complete policy for the task above. "
        "Use a state machine with clear phases (e.g. APPROACH → LOWER → GRASP → LIFT). "
        "Use proportional control based on observation values — do NOT hardcode positions. "
        "Output ONLY the Python code in a ```python block."
    )

    return "\n".join(parts)


def build_revision_prompt(
    task_description: str,
    env_info: EnvInfo,
    previous_code: str,
    diagnosis: Diagnosis,
    trajectory_summary: str,
    human_feedback: str = "",
) -> str:
    """Build the user prompt for revising a failed policy."""
    parts = [
        f"## Task\n{task_description}\n",
        f"## Environment: {env_info.name}",
        f"Goal: {env_info.goal_description}\n",
        f"## Previous Code (FAILED)\n```python\n{previous_code}\n```\n",
        f"## Failure Diagnosis",
        f"Category: {diagnosis.category}",
        f"Summary: {diagnosis.summary}",
        f"Details: {diagnosis.details}\n",
        f"## Simulation Trajectory\n{trajectory_summary}\n" if trajectory_summary else "",
        f"## Suggestions\n{diagnosis.suggestions}\n",
    ]

    if human_feedback:
        parts.append(
            "## Human Feedback (high-priority guidance)\n"
            f"{human_feedback}\n"
        )

    parts.append(
        "## Instructions\n"
        "Fix the policy based on the diagnosis above. "
        + ("Pay special attention to the Human Feedback section. " if human_feedback else "")
        + "Output the COMPLETE corrected policy (not just the changes). "
        "Make sure get_action(obs) and reset() are both defined. "
        "Output ONLY the Python code in a ```python block.",
    )

    return "\n".join(parts)


def _get_best_example(env_name: str) -> str | None:
    """Get the best few-shot example for an environment."""
    if env_name in FEW_SHOT_EXAMPLES:
        return env_name
    # For pick-and-place tasks, show the Stack example
    if env_name in ("PickPlaceCan", "NutAssemblySquare"):
        return "Stack"
    # For everything else, show Lift as a basic example
    if "Lift" in FEW_SHOT_EXAMPLES:
        return "Lift"
    return None
