"""Guided tool-use agent for policy generation.

Split into two callable steps so the UI can control transitions:
  1. run_phase_design()  — introspect + design phases (returns for human review)
  2. run_with_phases()   — generate code + test + iterate (the main loop)

The old run() method is preserved as a convenience wrapper.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import click

from roboscribe.config import Config
from roboscribe.exceptions import RoboScribeError
from roboscribe.llm.base import LLMToolResponse
from roboscribe.llm.factory import create_backend
from roboscribe.sim.env_registry import ENV_REGISTRY
from roboscribe.sim.runner import SimulationRunner
from roboscribe.sim.trajectory import TrajectoryResult
from roboscribe.sim.introspect import introspect_env, format_obs_report
from roboscribe.sim.diagnostics import diagnose_failure
from roboscribe.agent.tools import TOOLS, ToolResult, execute_tool
from roboscribe.agent.prompts import (
    SYSTEM_PROMPT,
    ENV_SELECTION_SYSTEM_PROMPT,
    PHASE_DESIGN_SYSTEM_PROMPT,
    build_env_selection_prompt,
    build_generation_prompt,
    build_phase_design_prompt,
    build_generation_prompt_with_phases,
)
from roboscribe.agent.few_shot import FEW_SHOT_EXAMPLES
from roboscribe.output.writer import PolicyWriter


TOOL_AGENT_SYSTEM_PROMPT = """\
You are RoboScribe, an expert robotics policy engineer. Your job is to \
fix and improve scripted control policies for robosuite simulation environments.

You have already been given:
- Environment introspection (ground-truth obs shapes, reward source, success check)
- An approved phase plan designed by the phase designer
- A first-attempt policy with its test results and failure diagnosis

## Your Task
Fix the policy based on the diagnosis and test results. You have tools to:
- `test_policy`: run your revised code in simulation
- `read_robosuite_source`: read robosuite source code for env internals
- `submit_policy`: submit when you achieve >= 80% success rate

## Policy Format
Your policy MUST define:
- `get_action(obs) -> np.ndarray`: takes obs dict, returns 7D action array
  - action[0:3] = (dx, dy, dz) position deltas in WORLD FRAME
  - action[3:6] = (dax, day, daz) orientation deltas (axis-angle)
  - action[6] = gripper: -1.0 = open, 1.0 = close
- `reset()`: resets state machine state between episodes
- Import only numpy (already available)

## Policy Architecture
For complex tasks (Door, NutAssembly, multi-step manipulation), use a CLASS-BASED policy:
```python
from roboscribe.pid import PID, RotationPID
# PID API:
#   pid = PID(kp, ki, kd, target=np.array([x,y,z]))
#   control = pid.update(current_pos, dt=0.05)  -> np.ndarray (3D)
#   pid.reset(new_target)  # reset errors & set new target
#   pid.get_error() -> float  # euclidean distance to target
# RotationPID API (1D angular):
#   rpid = RotationPID(kp, ki, kd, target=float)
#   val = rpid.update(current_angle, dt=0.05) -> float

class MyPolicy:
    def __init__(self, obs):
        # Initialize PID controllers, cache initial positions
        ...
    def get_action(self, obs):
        # State machine with phases from the plan
        ...

# Module-level shim (REQUIRED for runner compatibility)
_policy = None

def reset():
    global _policy
    _policy = None

def get_action(obs):
    global _policy
    if _policy is None:
        _policy = MyPolicy(obs)
    return _policy.get_action(obs)
```
For simple tasks (Lift, basic pick), a flat state machine with module-level vars is fine.

## robosuite Controller: OSC_POSE
Actions are DELTAS (small displacements per step), not absolute positions.
Typical action magnitudes: 0.01 to 0.1 for position.

## Critical Rules
- Use proportional control: action = gain * (target - current), gain 5-15
- Use PID controllers: `pid = PID(kp, ki, kd, target)` then `pid.update(current_pos, dt)` — NOT .step()
- Clip all actions to [-1, 1] range
- SCALAR OBS: Some obs values are SCALARS (float), not arrays. \
Use `float(obs['key'])` for scalars. NEVER do `obs['key'][0]` on a scalar — it will crash.
- Safe pattern: `np.atleast_1d(obs['key'])` forces any value to be indexable
- Gripper needs 10-15 steps to fully close — WAIT before lifting
- Always use a state machine with clear phases matching the approved phase plan
- For orientation control: use action[3:6] for axis-angle deltas (e.g., rotate gripper)
- Always clip actions: `action[:6] = np.clip(action[:6], -1.0, 1.0)`
- PID BUG: NEVER call pid.reset() every step — it zeros the integral and derivative. \
Call reset() ONCE per phase transition. To update the target each step, set pid.target = new_target.

## Reading Trajectory Analysis (CRITICAL)
Each test result includes auto-analysis for the first 2 episodes. READ IT CAREFULLY:
- `!! eef is X cm from Y — TOO FAR` → Your approach/reach phase FAILED. Fix the targeting.
- `TIP: obs['X_to_eef_pos']=[...] — USE THIS as direct error signal` → The env provides a \
RELATIVE VECTOR from eef to the target. Use this obs key DIRECTLY as your error signal \
instead of computing `target_pos - eef_pos` manually. Example: `error = obs['handle_to_eef_pos']`.
- `!! Gripper is OPEN at episode end` → The grasp phase failed.
- `OK: eef reached X` → Approach succeeded, problem is elsewhere (grip timing, rotation, etc.).

When you see `TIP: obs['X_to_eef_pos']`, you MUST use that obs key in your policy. It is more \
accurate than computing positions manually because it accounts for offsets you cannot see.

## Iteration Strategy
- READ the Analysis section in trajectory output — it tells you EXACTLY what failed and why.
- If analysis says "TOO FAR from handle", fix your approach targeting, don't tweak rotation.
- If analysis says "USE obs['X_to_eef_pos']", rewrite your approach to use that key directly.
- If 2+ consecutive tests show 0% with similar rewards, you are STUCK. \
Change your approach fundamentally (different gains, different phase logic, different offsets).
- Before testing, explain what specific change you made and WHY based on the analysis.
- Use `read_robosuite_source` to check env internals when stuck (reward function, success criteria).
- Do NOT just test the same approach with minor number tweaks — that wastes turns.
"""

# Phase 2 only needs test, read_source, and submit (introspection already done)
PHASE2_TOOLS = [t for t in TOOLS if t["name"] != "inspect_env"]


@dataclass
class PhaseDesignResult:
    """Result from the phase design step (returned for human review)."""

    env_name: str
    phase_plan: list[dict]
    introspection_str: str
    tokens_used: int = 0


@dataclass
class ToolAgentResult:
    """Final result from the tool-use agent loop."""

    success: bool
    policy_code: str
    success_rate: float
    total_turns: int
    total_tokens: int = 0
    tool_history: list[dict] = field(default_factory=list)
    phase_plan: list[dict] = field(default_factory=list)


class ToolAgentLoop:
    """Guided agent: phase design → human review → code generation → iteration."""

    def __init__(self, config: Config):
        self.config = config
        self.llm = create_backend(config)
        self.runner = SimulationRunner(config)
        self.writer = PolicyWriter(config)

    def select_environment(self, task_description: str) -> str:
        """Ask the LLM to pick the best environment for a task."""
        prompt = build_env_selection_prompt(task_description, ENV_REGISTRY)
        resp = self.llm.generate(ENV_SELECTION_SYSTEM_PROMPT, prompt)
        raw = resp.raw_text.strip().strip("`").strip()
        for name in ENV_REGISTRY:
            if raw.lower() == name.lower():
                return name
        for name in ENV_REGISTRY:
            if name.lower() in raw.lower():
                return name
        raise RoboScribeError(
            f"LLM returned unknown environment '{raw}'. "
            f"Available: {', '.join(sorted(ENV_REGISTRY.keys()))}"
        )

    # ─── Step 1: Phase Design (returns for human review) ──────────

    def run_phase_design(
        self,
        task_description: str,
        env_name: str | None = None,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> PhaseDesignResult:
        """Detect env + introspect + design phases.

        Returns a PhaseDesignResult for human review before code generation.
        """
        def status(msg: str):
            if on_status:
                on_status(msg)

        # Auto-detect env if not provided
        if env_name is None:
            status("Detecting environment...")
            env_name = self.select_environment(task_description)
            status(f"Environment: {env_name}")

        if env_name not in ENV_REGISTRY:
            raise RoboScribeError(
                f"Unknown environment: {env_name}. "
                f"Available: {', '.join(sorted(ENV_REGISTRY.keys()))}"
            )

        env_info = ENV_REGISTRY[env_name]

        # Introspect
        status("Inspecting environment...")
        introspection = introspect_env(env_name, robot=self.config.robot)
        introspection_str = format_obs_report(introspection)

        # Design phases
        status("Designing phase plan...")
        phase_prompt = build_phase_design_prompt(
            task_description, env_info, introspection_str,
        )
        phase_resp = self.llm.generate(PHASE_DESIGN_SYSTEM_PROMPT, phase_prompt)
        phase_plan = _parse_phase_plan(phase_resp.raw_text)

        status(f"Phase plan ready ({len(phase_plan)} phases)")

        return PhaseDesignResult(
            env_name=env_name,
            phase_plan=phase_plan,
            introspection_str=introspection_str,
            tokens_used=phase_resp.tokens_used,
        )

    def redesign_phases(
        self,
        task_description: str,
        env_name: str,
        introspection_str: str,
        human_feedback: str,
        previous_plan: list[dict] | None = None,
        *,
        on_status: Callable[[str], None] | None = None,
    ) -> PhaseDesignResult:
        """Re-run phase design incorporating human feedback."""
        def status(msg: str):
            if on_status:
                on_status(msg)

        env_info = ENV_REGISTRY[env_name]
        prompt = build_phase_design_prompt(
            task_description, env_info, introspection_str,
        )

        if previous_plan:
            prev_text = "\n".join(
                f"  {i}. {p['name']} — {p['goal']}" for i, p in enumerate(previous_plan)
            )
            prompt += f"\n\n## Previous Phase Plan\n{prev_text}"

        prompt += (
            f"\n\n## Human Feedback\n{human_feedback}\n\n"
            "Revise the phase plan based on this feedback. "
            "Output ONLY a JSON array in a ```json block."
        )

        status("Redesigning phases with feedback...")
        resp = self.llm.generate(PHASE_DESIGN_SYSTEM_PROMPT, prompt)
        phase_plan = _parse_phase_plan(resp.raw_text)
        status(f"Revised plan ready ({len(phase_plan)} phases)")

        return PhaseDesignResult(
            env_name=env_name,
            phase_plan=phase_plan,
            introspection_str=introspection_str,
            tokens_used=resp.tokens_used,
        )

    # ─── Step 2: Generate + Iterate (with approved phase plan) ────

    def run_with_phases(
        self,
        task_description: str,
        env_name: str,
        phase_plan: list[dict],
        introspection_str: str,
        *,
        max_turns: int = 15,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
        on_frame: Callable | None = None,
        on_status: Callable[[str], None] | None = None,
        on_submit: Callable[[str, float], None] | None = None,
        get_human_feedback: Callable[[], str | None] | None = None,
    ) -> ToolAgentResult:
        """Generate code from an approved phase plan, test, and iterate.

        This is the main generation loop. Call after run_phase_design()
        and human approval of the phase plan.
        """
        if env_name not in ENV_REGISTRY:
            raise RoboScribeError(
                f"Unknown environment: {env_name}. "
                f"Available: {', '.join(sorted(ENV_REGISTRY.keys()))}"
            )

        def status(msg: str):
            click.secho(msg)
            if on_status:
                on_status(msg)

        env_info = ENV_REGISTRY[env_name]
        tool_history: list[dict] = []
        total_tokens = 0

        # ── Generate first policy from phase plan ──────────────────
        status("Generating policy from phase plan...")
        if phase_plan:
            gen_prompt = build_generation_prompt_with_phases(
                task_description, env_info, phase_plan, introspection_str,
            )
        else:
            gen_prompt = build_generation_prompt(
                task_description, env_info, introspection_str,
            )
        gen_resp = self.llm.generate(SYSTEM_PROMPT, gen_prompt)
        first_code = gen_resp.code
        total_tokens += gen_resp.tokens_used

        if on_tool_call:
            on_tool_call(
                "generate_policy", {},
                f"Generated {len(first_code)} chars of policy code",
            )

        # ── Test first policy ──────────────────────────────────────
        status("Testing initial policy...")
        sim_result = self.runner.run_policy(
            first_code, env_name, frame_callback=on_frame,
        )

        test_summary = _format_sim_result(sim_result)
        if on_tool_call:
            on_tool_call("test_policy", {"code": first_code}, test_summary)
        tool_history.append({
            "turn": 1,
            "tool": "test_policy",
            "args_summary": f"code=({len(first_code)} chars)",
            "result_summary": test_summary[:500],
            "is_error": bool(sim_result.error),
        })

        best_code = first_code
        best_rate = sim_result.success_rate

        # ── If already successful → done ───────────────────────────
        if best_rate >= self.config.success_threshold:
            status(f"Success on first attempt! Rate: {best_rate:.0%}")
            if on_submit:
                on_submit(best_code, best_rate)

            output_path = self.writer.write(
                best_code,
                env_name=env_name,
                task_description=task_description,
                success_rate=best_rate,
                attempts=1,
                robot=self.config.robot,
            )
            status(f"Policy saved to {output_path}")

            return ToolAgentResult(
                success=True,
                policy_code=best_code,
                success_rate=best_rate,
                total_turns=1,
                total_tokens=total_tokens,
                tool_history=tool_history,
                phase_plan=phase_plan,
            )

        # ── Diagnose failure ───────────────────────────────────────
        status("Diagnosing failure...")
        diagnosis = diagnose_failure(sim_result, env_name)

        if on_tool_call:
            diag_text = (
                f"[{diagnosis.category}] {diagnosis.summary}\n"
                f"Suggestions: {diagnosis.suggestions}"
            )
            on_tool_call("diagnose_failure", {}, diag_text)

        status(f"Diagnosis: [{diagnosis.category}] {diagnosis.summary}")

        # ── Agent-driven iteration (LLM uses tools freely) ────────
        status("Starting agent iteration...")

        phase2_user_msg = _build_phase2_context(
            task_description, env_name, env_info,
            introspection_str, phase_plan,
            first_code, sim_result, diagnosis,
        )

        messages: list[dict] = [{"role": "user", "content": phase2_user_msg}]

        for turn in range(2, max_turns + 1):
            status(f"Agent turn {turn}/{max_turns}...")

            response = self.llm.generate_with_tools(
                TOOL_AGENT_SYSTEM_PROMPT, messages, PHASE2_TOOLS,
            )
            total_tokens += response.tokens_used

            if response.has_tool_calls:
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                }
                messages.append(assistant_msg)

                for tc in response.tool_calls:
                    status(f"Calling tool: {tc.name}...")

                    tool_result = execute_tool(
                        tc.name, tc.arguments,
                        config=self.config,
                        env_name=env_name,
                        runner=self.runner,
                        frame_callback=on_frame,
                    )

                    tool_history.append({
                        "turn": turn,
                        "tool": tc.name,
                        "args_summary": _summarize_args(tc.arguments),
                        "result_summary": tool_result.result[:500],
                        "is_error": tool_result.is_error,
                    })

                    if on_tool_call:
                        on_tool_call(tc.name, tc.arguments, tool_result.result)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": tool_result.result,
                        "is_error": tool_result.is_error,
                    })

                    if tc.name == "test_policy" and tool_result.metadata:
                        rate = tool_result.metadata.get("success_rate", 0.0)
                        if rate > best_rate:
                            best_rate = rate
                            best_code = tool_result.metadata.get("code", "")

                    if tc.name == "submit_policy" and tool_result.metadata:
                        submitted_code = tool_result.metadata.get("code", "")
                        status(f"Policy submitted! Best rate: {best_rate:.0%}")
                        if on_submit:
                            on_submit(submitted_code, best_rate)
                        if not best_code:
                            best_code = submitted_code
                        if best_code:
                            self.writer.write(
                                best_code,
                                env_name=env_name,
                                task_description=task_description,
                                success_rate=best_rate,
                                attempts=turn,
                                robot=self.config.robot,
                            )
                        return ToolAgentResult(
                            success=best_rate >= self.config.success_threshold,
                            policy_code=best_code,
                            success_rate=best_rate,
                            total_turns=turn,
                            total_tokens=total_tokens,
                            tool_history=tool_history,
                            phase_plan=phase_plan,
                        )
            else:
                messages.append({
                    "role": "assistant",
                    "content": response.content,
                })

            # Human feedback injection
            if get_human_feedback is not None:
                feedback = get_human_feedback()
                if feedback:
                    messages.append({"role": "user", "content": feedback})
                    status(f"Human feedback injected: {feedback[:100]}...")

        # Exhausted turns
        status(f"Agent exhausted {max_turns} turns. Best rate: {best_rate:.0%}")
        if best_code:
            self.writer.write(
                best_code,
                env_name=env_name,
                task_description=task_description,
                success_rate=best_rate,
                attempts=max_turns,
                robot=self.config.robot,
            )

        return ToolAgentResult(
            success=best_rate >= self.config.success_threshold,
            policy_code=best_code,
            success_rate=best_rate,
            total_turns=max_turns,
            total_tokens=total_tokens,
            tool_history=tool_history,
            phase_plan=phase_plan,
        )

    # ─── Convenience: run everything in one call ──────────────────

    def run(
        self,
        task_description: str,
        env_name: str,
        *,
        max_turns: int = 15,
        on_tool_call: Callable[[str, dict, str], None] | None = None,
        on_frame: Callable | None = None,
        on_status: Callable[[str], None] | None = None,
        on_submit: Callable[[str, float], None] | None = None,
        on_phase_design: Callable[[list[dict]], None] | None = None,
        get_human_feedback: Callable[[], str | None] | None = None,
    ) -> ToolAgentResult:
        """Run the full pipeline: phase design → generate → iterate.

        For UI usage, prefer run_phase_design() + run_with_phases() separately
        so the user can review phases between steps.
        """
        design = self.run_phase_design(
            task_description, env_name, on_status=on_status,
        )

        if on_phase_design:
            on_phase_design(design.phase_plan)

        return self.run_with_phases(
            task_description, design.env_name,
            design.phase_plan, design.introspection_str,
            max_turns=max_turns,
            on_tool_call=on_tool_call,
            on_frame=on_frame,
            on_status=on_status,
            on_submit=on_submit,
            get_human_feedback=get_human_feedback,
        )


# ─── Helpers ──────────────────────────────────────────────────────

def _parse_phase_plan(raw_text: str) -> list[dict]:
    """Extract a JSON phase plan from LLM response text."""
    match = re.search(r"```json\s*\n?(.*?)\n?```", raw_text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    else:
        match = re.search(r"\[\s*\{.*\}\s*\]", raw_text, re.DOTALL)
        text = match.group(0) if match else ""

    if not text:
        return []

    try:
        plan = json.loads(text)
        if not isinstance(plan, list):
            return []
        validated = []
        for p in plan:
            if isinstance(p, dict) and "name" in p and "goal" in p:
                validated.append({
                    "name": p["name"],
                    "goal": p["goal"],
                    "control": p.get("control", ""),
                    "exit_condition": p.get("exit_condition", ""),
                    "notes": p.get("notes", ""),
                })
        return validated
    except (json.JSONDecodeError, TypeError):
        return []


def _build_phase2_context(
    task_description, env_name, env_info,
    introspection_str, phase_plan,
    first_code, sim_result, diagnosis,
) -> str:
    """Build the Phase 2 initial message with all context."""
    msg = (
        f"## Task\n{task_description}\n\n"
        f"## Environment: {env_name}\n"
        f"Description: {env_info.description}\n"
        f"Goal: {env_info.goal_description}\n\n"
        f"## Environment Introspection (ground truth)\n"
        f"{introspection_str}\n\n"
    )

    if phase_plan:
        phase_text = "\n".join(
            f"  {i}. **{p['name']}** — {p['goal']} "
            f"(exit: {p['exit_condition']})"
            for i, p in enumerate(phase_plan)
        )
        msg += f"## Approved Phase Plan\n{phase_text}\n\n"

    if env_info.tips:
        msg += f"## Environment-Specific Tips\n{env_info.tips}\n\n"

    example_env = _get_best_example(env_name)
    if example_env and example_env in FEW_SHOT_EXAMPLES:
        msg += (
            f"## Reference: Working policy for {example_env}\n"
            f"```python\n{FEW_SHOT_EXAMPLES[example_env]}\n```\n\n"
        )

    msg += (
        f"## First Attempt Code\n"
        f"```python\n{first_code}\n```\n\n"
        f"## First Attempt Result\n"
        f"Success Rate: {sim_result.success_rate:.0%} "
        f"({sim_result.successes}/{sim_result.total_episodes})\n"
    )

    if sim_result.error:
        msg += f"Error ({sim_result.error_type}): {sim_result.error}\n"
    if sim_result.trajectory_summary:
        msg += f"Trajectory:\n{sim_result.trajectory_summary}\n"
    if sim_result.episode_rewards:
        msg += f"Rewards: {[f'{r:.2f}' for r in sim_result.episode_rewards]}\n"

    msg += (
        f"\n## Failure Diagnosis\n"
        f"Category: {diagnosis.category}\n"
        f"Summary: {diagnosis.summary}\n"
        f"Details: {diagnosis.details}\n"
        f"Suggestions: {diagnosis.suggestions}\n\n"
        f"## Instructions\n"
        f"Fix this policy based on the diagnosis above. "
        f"You can call `test_policy` to test your revised code, "
        f"`read_robosuite_source` to read env source code, "
        f"and `submit_policy` when you achieve >= 80% success rate.\n"
        f"Iterate until success or you run out of turns."
    )

    return msg


def _format_sim_result(result: TrajectoryResult) -> str:
    """Format a TrajectoryResult into a human-readable string."""
    lines = []
    if result.error:
        lines.append(f"ERROR ({result.error_type}): {result.error}")
    else:
        lines.append(
            f"Success rate: {result.success_rate:.0%} "
            f"({result.successes}/{result.total_episodes})"
        )
        if result.trajectory_summary:
            lines.append(f"Trajectory:\n{result.trajectory_summary}")
        if result.episode_rewards:
            lines.append(f"Rewards: {[f'{r:.2f}' for r in result.episode_rewards]}")
    return "\n".join(lines)


def _summarize_args(args: dict) -> str:
    """Create a short summary of tool arguments for display."""
    parts = []
    for k, v in args.items():
        if k == "code":
            parts.append(f"code=({len(str(v))} chars)")
        elif isinstance(v, str) and len(v) > 80:
            parts.append(f"{k}={v[:80]}...")
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


def _get_best_example(env_name: str) -> str | None:
    """Get the best few-shot example environment name for a given env."""
    if env_name in FEW_SHOT_EXAMPLES:
        return env_name
    if env_name in ("PickPlaceCan", "NutAssemblySquare"):
        return "Stack"
    if "Lift" in FEW_SHOT_EXAMPLES:
        return "Lift"
    return None
