"""
Environment introspection — spin up the env, take one step,
and report observation shapes, dtypes, sample values,
reward function source, and success criteria source.

This gives the LLM ground-truth information instead of
relying on hand-written tips that may be incomplete or wrong.
"""

from __future__ import annotations

import multiprocessing as mp
import numpy as np


def _introspect_worker(env_name: str, robot: str, result_queue: mp.Queue):
    """Subprocess: create env, reset, step, extract obs + reward + success source."""
    import robosuite as suite
    import inspect

    try:
        controller_config = suite.load_composite_controller_config(controller="BASIC")
        env = suite.make(
            env_name,
            robots=robot,
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            reward_shaping=True,
            horizon=10,
        )

        obs = env.reset()

        # Take one zero-action step
        action_dim = env.action_dim
        zero_action = np.zeros(action_dim)
        obs_after_step, reward, done, info = env.step(zero_action)

        # ── Build obs report ──────────────────────────────────
        obs_report = {}
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, np.ndarray):
                obs_report[key] = {
                    "shape": list(val.shape),
                    "dtype": str(val.dtype),
                    "sample": val.tolist() if val.size <= 10 else val.flat[:5].tolist(),
                    "is_scalar": False,
                }
            elif isinstance(val, (int, float, np.integer, np.floating)):
                obs_report[key] = {
                    "shape": "scalar",
                    "dtype": type(val).__name__,
                    "sample": float(val),
                    "is_scalar": True,
                }
            else:
                obs_report[key] = {
                    "shape": "unknown",
                    "dtype": str(type(val)),
                    "sample": str(val)[:100],
                    "is_scalar": False,
                }

        # ── Extract reward function source ────────────────────
        reward_source = ""
        try:
            # Get the reward method from the actual env class (not base)
            reward_method = type(env).reward
            reward_source = inspect.getsource(reward_method)
        except Exception:
            reward_source = "(Could not extract reward source)"

        # ── Extract success check source ──────────────────────
        success_source = ""
        try:
            success_method = type(env)._check_success
            success_source = inspect.getsource(success_method)
        except Exception:
            success_source = "(Could not extract success check source)"

        # ── Extract staged_rewards if available ───────────────
        staged_rewards_source = ""
        try:
            if hasattr(env, "staged_rewards"):
                staged_rewards_source = inspect.getsource(type(env).staged_rewards)
        except Exception:
            pass

        # ── Sample reward value from the zero-action step ─────
        sample_reward = float(reward)

        env.close()

        result_queue.put({
            "success": True,
            "action_dim": action_dim,
            "obs_report": obs_report,
            "reward_source": reward_source,
            "success_source": success_source,
            "staged_rewards_source": staged_rewards_source,
            "sample_reward": sample_reward,
        })

    except Exception as e:
        result_queue.put({
            "success": False,
            "error": str(e),
        })


def introspect_env(env_name: str, robot: str = "Panda", timeout: int = 30) -> dict:
    """Spin up an environment and return ground-truth obs + reward + success info."""
    result_queue = mp.Queue()
    proc = mp.Process(
        target=_introspect_worker,
        args=(env_name, robot, result_queue),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.kill()
        return {"success": False, "error": "Introspection timed out"}

    if not result_queue.empty():
        return result_queue.get_nowait()

    return {"success": False, "error": "No result from introspection"}


def format_obs_report(report: dict) -> str:
    """Format introspection results into a string the LLM can use in its prompt."""
    if not report.get("success"):
        return f"(Introspection failed: {report.get('error', 'unknown')})"

    lines = [
        f"Action dimension: {report['action_dim']}",
        f"Sample reward from zero action: {report['sample_reward']:.4f}",
        "",
        "Observation keys (ground truth from env.reset()):",
    ]

    for key, info in report["obs_report"].items():
        if info["is_scalar"]:
            lines.append(f"  - {key}: SCALAR ({info['dtype']}), example value: {info['sample']:.4f}")
        else:
            shape_str = f"({', '.join(str(s) for s in info['shape'])})" if isinstance(info['shape'], list) else info['shape']
            sample = info['sample']
            if isinstance(sample, list):
                sample_str = "[" + ", ".join(f"{v:.3f}" for v in sample[:5]) + "]"
                if isinstance(info['shape'], list) and len(info['shape']) > 0 and info['shape'][0] > 5:
                    sample_str += "..."
            else:
                sample_str = str(sample)
            lines.append(f"  - {key}: shape={shape_str}, dtype={info['dtype']}, example={sample_str}")

    # ── Reward function ───────────────────────────────────
    reward_src = report.get("reward_source", "")
    if reward_src and not reward_src.startswith("("):
        lines.append("")
        lines.append("Reward function (from env source code):")
        lines.append("```python")
        lines.append(reward_src.strip())
        lines.append("```")

    # ── Staged rewards (if available) ─────────────────────
    staged_src = report.get("staged_rewards_source", "")
    if staged_src:
        lines.append("")
        lines.append("Staged rewards breakdown (from env source code):")
        lines.append("```python")
        lines.append(staged_src.strip())
        lines.append("```")

    # ── Success check ─────────────────────────────────────
    success_src = report.get("success_source", "")
    if success_src and not success_src.startswith("("):
        lines.append("")
        lines.append("Success check (from env source code — THIS is what your policy must achieve):")
        lines.append("```python")
        lines.append(success_src.strip())
        lines.append("```")

    return "\n".join(lines)
