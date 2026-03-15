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
The `obs` parameter is a dict with string keys mapping to numpy arrays OR scalars.
CRITICAL: Some observation values are SCALARS (float), not arrays. Always handle both:
  - If a value is a scalar, use it directly: `val = obs['key']`
  - If a value is an array, index normally: `val = obs['key'][0]`
  - Safe pattern: `val = float(obs['key'])` for scalars, `np.atleast_1d(obs['key'])` to force array

Common keys:
- `robot0_eef_pos`: (3,) end-effector position [x, y, z]
- `robot0_eef_quat`: (4,) end-effector orientation quaternion
- `robot0_gripper_qpos`: (2,) gripper joint positions
- Object-specific keys vary per environment — see the introspection report below for exact shapes

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
    introspection: str = "",
) -> str:
    """Build the user prompt for initial policy generation."""
    parts = [
        f"## Task\n{task_description}\n",
        f"## Environment: {env_info.name}",
        f"Description: {env_info.description}",
        f"Goal: {env_info.goal_description}",
    ]

    # Introspection data (ground truth) takes priority over static obs_keys
    if introspection:
        parts.append(
            f"\n## Environment Introspection (ground truth from actual env)\n"
            f"{introspection}"
        )
    else:
        parts.append(f"\n## Available Observation Keys\n{env_info.obs_keys_str}")

    parts.append(f"\n## Tips\n{env_info.tips}")

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


def build_env_selection_prompt(
    task_description: str,
    env_registry: dict[str, EnvInfo],
) -> str:
    """Build a prompt that asks the LLM to pick the best environment for a task."""
    env_list = "\n".join(
        f"- **{info.name}**: {info.description} (objects: {', '.join(info.objects)})"
        for info in env_registry.values()
    )
    return (
        f"## Task\n{task_description}\n\n"
        f"## Available Environments\n{env_list}\n\n"
        "## Instructions\n"
        "Which environment is the best match for the task above? "
        "Reply with ONLY the environment name (e.g. `Lift`), nothing else."
    )


ENV_SELECTION_SYSTEM_PROMPT = (
    "You are an environment selector for robosuite robotics simulation. "
    "Given a task description and a list of available environments, "
    "pick the single best-matching environment. Reply with ONLY the environment name."
)


# ─── Phase-design prompts ─────────────────────────────────────────────

PHASE_DESIGN_SYSTEM_PROMPT = """\
You are RoboScribe's phase designer. Given a robosuite manipulation task, \
you design a state-machine phase plan that a policy generator will implement.

## Your Job
Break the task into sequential phases (3-8 phases typical). Each phase is a \
discrete controller mode with a clear entry condition and exit condition.

## Phase Design Principles
- Each phase has ONE goal (e.g., "move above object", "close gripper", "rotate wrist")
- Exit conditions must be measurable from observations (position error < threshold, counter > N, etc.)
- Consider gripper orientation — some tasks require rotating the gripper BEFORE approaching
- Gripper takes 10-15 steps to close — always include a WAIT/GRIP phase after closing
- Use PID or proportional control for smooth motion — never hardcode positions
- For tasks involving handles, latches, or rotation: plan separate ORIENT, ROTATE, and PULL phases
- For multi-object tasks: plan pick-place sequences for each object

## Output Format
Return a JSON array of phase objects. Each phase has:
- `name`: short uppercase name (e.g., "ORIENT", "APPROACH", "GRIP")
- `goal`: one-sentence description of what this phase does
- `control`: what the controller should do (position target, orientation change, gripper state)
- `exit_condition`: measurable condition to transition to the next phase
- `notes`: any implementation hints (gains, offsets, special handling)

Example for a simple pick task:
```json
[
  {"name": "APPROACH", "goal": "Move above the object", "control": "Proportional XY + hover Z", "exit_condition": "XY error < 0.01", "notes": "Gripper open, gain=10"},
  {"name": "LOWER", "goal": "Lower to grasp height", "control": "Proportional XYZ to object + small Z offset", "exit_condition": "Z error < 0.02", "notes": "Gripper open"},
  {"name": "GRASP", "goal": "Close gripper and wait", "control": "Hold position, close gripper", "exit_condition": "counter > 15 steps", "notes": "Maintain XY tracking"},
  {"name": "LIFT", "goal": "Lift object upward", "control": "action[2] = 1.0 (up), hold gripper closed", "exit_condition": "eef_z > 1.0", "notes": "No XY correction needed"}
]
```

Return ONLY the JSON array in a ```json block. No other text.
"""


def build_phase_design_prompt(
    task_description: str,
    env_info: EnvInfo,
    introspection: str = "",
) -> str:
    """Build the user prompt for phase design."""
    parts = [
        f"## Task\n{task_description}\n",
        f"## Environment: {env_info.name}",
        f"Description: {env_info.description}",
        f"Goal: {env_info.goal_description}",
    ]

    if introspection:
        parts.append(
            f"\n## Environment Introspection (ground truth from actual env)\n"
            f"{introspection}"
        )

    parts.append(f"\n## Tips\n{env_info.tips}")

    # Show a reference phase design from few-shot if available
    example_env = _get_best_example(env_info.name)
    if example_env and example_env in FEW_SHOT_EXAMPLES:
        parts.append(
            f"\n## Reference: Working policy for {example_env}\n"
            f"Study this working policy's phase structure as a reference:\n"
            f"```python\n{FEW_SHOT_EXAMPLES[example_env]}\n```"
        )

    parts.append(
        "\n## Instructions\n"
        "Design a phase plan for this task. "
        "Think carefully about what gripper orientation is needed, "
        "what order of operations will succeed, and what exit conditions "
        "are measurable from the available observations.\n"
        "Output ONLY a JSON array of phase objects in a ```json block."
    )

    return "\n".join(parts)


def build_generation_prompt_with_phases(
    task_description: str,
    env_info: EnvInfo,
    phase_plan: list[dict],
    introspection: str = "",
) -> str:
    """Build a generation prompt that follows an approved phase design."""
    parts = [
        f"## Task\n{task_description}\n",
        f"## Environment: {env_info.name}",
        f"Description: {env_info.description}",
        f"Goal: {env_info.goal_description}",
    ]

    if introspection:
        parts.append(
            f"\n## Environment Introspection (ground truth from actual env)\n"
            f"{introspection}"
        )

    parts.append(f"\n## Tips\n{env_info.tips}")

    # The approved phase plan
    phase_text = "\n".join(
        f"  {i}. **{p['name']}** — {p['goal']}\n"
        f"     Control: {p['control']}\n"
        f"     Exit: {p['exit_condition']}\n"
        f"     Notes: {p.get('notes', 'none')}"
        for i, p in enumerate(phase_plan)
    )
    parts.append(
        f"\n## Approved Phase Plan (FOLLOW THIS EXACTLY)\n{phase_text}"
    )

    # Few-shot example
    example_env = _get_best_example(env_info.name)
    if example_env and example_env in FEW_SHOT_EXAMPLES:
        parts.append(
            f"\n## Reference: Working policy for {example_env}\n"
            f"Use this as a reference for code structure, PID usage, and action format:\n"
            f"```python\n{FEW_SHOT_EXAMPLES[example_env]}\n```"
        )

    parts.append(
        "\n## Policy Architecture\n"
        "For complex tasks (Door, NutAssembly, multi-step manipulation), use a CLASS-BASED policy:\n"
        "```python\n"
        "from roboscribe.pid import PID, RotationPID\n"
        "# PID(kp, ki, kd, target=np.array([x,y,z])) -> pid.update(current_pos, dt) returns 3D control\n"
        "# RotationPID(kp, ki, kd, target=float) -> rpid.update(current_angle, dt) returns float\n"
        "# pid.reset(new_target), pid.get_error() -> float (distance to target)\n\n"
        "class MyPolicy:\n"
        "    def __init__(self, obs):\n"
        "        # Initialize PID controllers, cache initial positions\n"
        "        ...\n"
        "    def get_action(self, obs):\n"
        "        # State machine with phases from the plan above\n"
        "        ...\n\n"
        "# Module-level shim (REQUIRED for runner compatibility)\n"
        "_policy = None\n\n"
        "def reset():\n"
        "    global _policy\n"
        "    _policy = None\n\n"
        "def get_action(obs):\n"
        "    global _policy\n"
        "    if _policy is None:\n"
        "        _policy = MyPolicy(obs)\n"
        "    return _policy.get_action(obs)\n"
        "```\n"
        "For simple tasks (Lift, basic pick), a flat state machine with module-level vars is fine."
    )

    parts.append(
        "\n## Instructions\n"
        "Implement the COMPLETE policy following the approved phase plan above. "
        "Each phase in the plan becomes a state in your state machine. "
        "Use PID controllers: `pid = PID(kp, ki, kd, target)` then `pid.update(pos, dt=0.05)` — NOT .step(). "
        "Use proportional control based on observation values — do NOT hardcode positions. "
        "Output ONLY the Python code in a ```python block."
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
