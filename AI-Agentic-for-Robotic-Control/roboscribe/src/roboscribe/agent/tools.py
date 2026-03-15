"""Tool definitions and execution for the agentic loop.

Four tools the LLM can call:
  - inspect_env: introspect a robosuite environment
  - test_policy: run a policy in simulation
  - read_robosuite_source: read robosuite module source code
  - submit_policy: submit the final working policy
"""

from __future__ import annotations

import inspect
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roboscribe.config import Config
from roboscribe.sim.introspect import introspect_env, format_obs_report
from roboscribe.sim.runner import SimulationRunner
from roboscribe.sim.diagnostics import diagnose_failure


# ── Tool schemas (provider-agnostic) ──────────────────────

TOOLS = [
    {
        "name": "inspect_env",
        "description": (
            "Inspect a robosuite environment. Returns observation keys with "
            "exact shapes/dtypes/sample values, reward function source code, "
            "and success check source code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "env_name": {
                    "type": "string",
                    "description": "Name of the robosuite environment (e.g. 'Lift', 'Door')",
                },
            },
            "required": ["env_name"],
        },
    },
    {
        "name": "test_policy",
        "description": (
            "Run a policy in simulation for N episodes. Returns success rate, "
            "full error traceback if crashed, and per-episode trajectory summary."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Complete Python policy code with get_action(obs) function",
                },
                "num_episodes": {
                    "type": "integer",
                    "description": "Number of episodes to run (default 5)",
                    "default": 5,
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "read_robosuite_source",
        "description": (
            "Read source code of a robosuite module. Use to understand env "
            "internals, controller behavior, or reward logic."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "module_path": {
                    "type": "string",
                    "description": (
                        "Path relative to the robosuite package "
                        "(e.g. 'environments/manipulation/door.py')"
                    ),
                },
            },
            "required": ["module_path"],
        },
    },
    {
        "name": "submit_policy",
        "description": (
            "Submit the final working policy. Call this when you're satisfied "
            "with the success rate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The final policy code to submit",
                },
            },
            "required": ["code"],
        },
    },
]


@dataclass
class ToolResult:
    """Result from executing a tool."""

    name: str
    result: str
    is_error: bool = False
    # Extra structured data for callbacks
    metadata: dict[str, Any] | None = None


def _find_robosuite_root() -> Path | None:
    """Find the robosuite package installation path."""
    try:
        import robosuite
        return Path(robosuite.__file__).parent
    except ImportError:
        return None


def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    config: Config,
    env_name: str,
    runner: SimulationRunner | None = None,
    frame_callback=None,
) -> ToolResult:
    """Execute a tool by name and return the result."""
    try:
        if name == "inspect_env":
            return _exec_inspect_env(arguments, config=config)
        elif name == "test_policy":
            return _exec_test_policy(
                arguments, config=config, env_name=env_name,
                runner=runner, frame_callback=frame_callback,
            )
        elif name == "read_robosuite_source":
            return _exec_read_robosuite_source(arguments)
        elif name == "submit_policy":
            return _exec_submit_policy(arguments)
        else:
            return ToolResult(
                name=name,
                result=f"Unknown tool: {name}",
                is_error=True,
            )
    except Exception as e:
        return ToolResult(
            name=name,
            result=f"Tool execution error: {traceback.format_exc()}",
            is_error=True,
        )


def _exec_inspect_env(args: dict, *, config: Config) -> ToolResult:
    target_env = args["env_name"]
    report = introspect_env(target_env, robot=config.robot)
    formatted = format_obs_report(report)
    return ToolResult(
        name="inspect_env",
        result=formatted,
        metadata={"env_name": target_env, "raw_report": report},
    )


def _exec_test_policy(
    args: dict,
    *,
    config: Config,
    env_name: str,
    runner: SimulationRunner | None = None,
    frame_callback=None,
) -> ToolResult:
    code = args["code"]
    # Cap episodes to config value — LLM may request more but we enforce the limit
    requested = args.get("num_episodes", config.num_episodes)
    num_episodes = min(requested, config.num_episodes)

    # Temporarily override num_episodes
    orig_episodes = config.num_episodes
    config.num_episodes = num_episodes

    if runner is None:
        runner = SimulationRunner(config)

    sim_result = runner.run_policy(
        code, env_name, frame_callback=frame_callback,
    )

    config.num_episodes = orig_episodes

    # Format result
    lines = []
    if sim_result.error:
        lines.append(f"ERROR ({sim_result.error_type}):")
        lines.append(sim_result.error)
        # Add diagnosis for context
        diagnosis = diagnose_failure(sim_result, env_name)
        lines.append(f"\nDiagnosis: {diagnosis.summary}")
        lines.append(f"Suggestions: {diagnosis.suggestions}")
    else:
        lines.append(
            f"Success rate: {sim_result.success_rate:.0%} "
            f"({sim_result.successes}/{sim_result.total_episodes})"
        )
        if sim_result.trajectory_summary:
            lines.append(f"\nTrajectory:\n{sim_result.trajectory_summary}")
        if sim_result.episode_rewards:
            lines.append(
                f"\nRewards: {[f'{r:.2f}' for r in sim_result.episode_rewards]}"
            )

    return ToolResult(
        name="test_policy",
        result="\n".join(lines),
        metadata={
            "success_rate": sim_result.success_rate,
            "successes": sim_result.successes,
            "total_episodes": sim_result.total_episodes,
            "error": sim_result.error,
            "sim_result": sim_result,
            "code": code,
        },
    )


def _exec_read_robosuite_source(args: dict) -> ToolResult:
    module_path = args["module_path"]

    root = _find_robosuite_root()
    if root is None:
        return ToolResult(
            name="read_robosuite_source",
            result="robosuite is not installed",
            is_error=True,
        )

    target = root / module_path
    if not target.exists():
        # Try without leading directory
        alt = root / module_path.lstrip("/")
        if alt.exists():
            target = alt
        else:
            return ToolResult(
                name="read_robosuite_source",
                result=(
                    f"File not found: {module_path}\n"
                    f"robosuite root: {root}\n"
                    f"Available top-level dirs: {sorted(p.name for p in root.iterdir() if p.is_dir())}"
                ),
                is_error=True,
            )

    # Read with size limit
    content = target.read_text()
    if len(content) > 50_000:
        content = content[:50_000] + "\n\n... (truncated, file too large)"

    return ToolResult(
        name="read_robosuite_source",
        result=content,
        metadata={"path": str(target)},
    )


def _exec_submit_policy(args: dict) -> ToolResult:
    code = args["code"]
    return ToolResult(
        name="submit_policy",
        result="Policy submitted successfully.",
        metadata={"code": code, "submitted": True},
    )
