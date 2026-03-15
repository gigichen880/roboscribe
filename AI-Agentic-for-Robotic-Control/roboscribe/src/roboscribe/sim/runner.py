"""
Simulation runner — uses a subprocess worker to avoid macOS
'NSWindow must be on main thread' crash when called from a
background thread (e.g. inside Streamlit).

Frames are streamed back to the parent process via
multiprocessing shared memory so there is zero pickle overhead.
"""

from __future__ import annotations

import multiprocessing as mp
import multiprocessing.shared_memory as shm
import numpy as np
import struct
import time
import importlib.util
import tempfile
import os

from roboscribe.config import Config
from roboscribe.sim.env_registry import ENV_REGISTRY
from roboscribe.sim.trajectory import TrajectoryResult


# ──────────────────────────────────────────────────────────────
# Obs snapshot helpers (used inside subprocess)
# ──────────────────────────────────────────────────────────────

def _snapshot_obs(obs):
    """Capture small obs values for trajectory diagnostics."""
    snap = {}
    for key, val in sorted(obs.items()):
        if isinstance(val, (int, float)):
            snap[key] = round(float(val), 4)
        elif isinstance(val, np.ndarray) and val.ndim <= 1 and val.size <= 4:
            snap[key] = [round(float(v), 3) for v in val.flat]
        # Skip large arrays (images, joint vecs, etc.)
    return snap


def _format_snap(snap):
    """Format an obs snapshot into a compact string."""
    parts = []
    for key, val in snap.items():
        if isinstance(val, list):
            parts.append(f"{key}=[{', '.join(f'{v:.3f}' for v in val)}]")
        else:
            parts.append(f"{key}={val:.4f}")
    return ", ".join(parts)


def _obs_diff(start, end):
    """Show which obs keys changed significantly between start and end."""
    diffs = []
    for key in start:
        if key not in end:
            continue
        s, e = start[key], end[key]
        if isinstance(s, list) and isinstance(e, list):
            delta = max(abs(a - b) for a, b in zip(s, e))
            if delta > 0.01:
                diffs.append(key)
        elif isinstance(s, (int, float)) and isinstance(e, (int, float)):
            if abs(s - e) > 0.01:
                diffs.append(key)
    return ", ".join(diffs) if diffs else ""


def _analyze_episode_obs(end_snap):
    """Auto-analyze obs snapshot and return actionable insights.

    This replaces manual analysis — computes distances between eef and
    target objects, flags useful relative obs keys, checks gripper state.
    The agent sees these insights directly instead of raw numbers.
    """
    analysis = []

    # Find eef position
    eef = end_snap.get("robot0_eef_pos")
    if not isinstance(eef, list) or len(eef) < 3:
        return []

    # Known target object keys
    target_keys = {
        "cube_pos": "cube",
        "cubeA_pos": "cubeA",
        "cubeB_pos": "cubeB",
        "handle_pos": "handle",
        "SquareNut_pos": "nut",
        "Can_pos": "can",
    }

    for key, name in target_keys.items():
        obj = end_snap.get(key)
        if not isinstance(obj, list) or len(obj) < 3:
            continue
        dx = abs(obj[0] - eef[0])
        dy = abs(obj[1] - eef[1])
        dz = abs(obj[2] - eef[2])
        dist = (dx**2 + dy**2 + dz**2) ** 0.5

        if dist > 0.05:
            analysis.append(
                f"  !! eef is {dist*100:.1f}cm from {name} — TOO FAR for grasp "
                f"(need <2cm). Gap: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}"
            )
        elif dist > 0.02:
            analysis.append(
                f"  ~ eef is {dist*100:.1f}cm from {name} — borderline. "
                f"Gap: dx={dx:.3f}, dy={dy:.3f}, dz={dz:.3f}"
            )
        else:
            analysis.append(f"  OK: eef reached {name} ({dist*100:.1f}cm)")

    # Flag relative obs keys — these are DIRECT error signals
    # Patterns: *_to_eef_pos, *_to_robot0_eef_pos, eef_to_*
    # Skip quaternion keys (*_quat) — they are orientation, not position
    for key, val in end_snap.items():
        if "quat" in key:
            continue  # skip quaternion obs
        if ("_to_eef" in key or "eef_to_" in key or "_to_robot0_eef" in key) and isinstance(val, list):
            mag = sum(v**2 for v in val) ** 0.5
            analysis.append(
                f"  TIP: obs['{key}']={[round(v, 3) for v in val]} "
                f"(mag={mag*100:.1f}cm) — USE THIS as direct error signal "
                f"instead of computing positions manually!"
            )

    # Check gripper state
    grip = end_snap.get("robot0_gripper_qpos")
    if isinstance(grip, list):
        avg = sum(grip) / len(grip)
        if avg > 0.03:
            analysis.append("  !! Gripper is OPEN at episode end — object NOT grasped")
        elif avg < 0.005:
            analysis.append("  Gripper is CLOSED at episode end")

    return analysis


# ──────────────────────────────────────────────────────────────
# Worker  (runs in subprocess — owns the macOS main thread)
# ──────────────────────────────────────────────────────────────

def _simulation_worker(
    code: str,
    env_name: str,
    config_dict: dict,
    result_queue: mp.Queue,
    shm_name: str,
    shm_meta_name: str,
    frame_ready: mp.Event,
    frame_consumed: mp.Event,
    stop_event: mp.Event,
):
    """Subprocess entry point. Runs sim and streams frames via shared memory."""
    import robosuite as suite
    import numpy as np

    # ── load policy ──────────────────────────────────────────
    with tempfile.NamedTemporaryFile(mode="w", suffix="_policy.py", delete=False) as f:
        f.write(code)
        policy_path = f.name

    try:
        spec = importlib.util.spec_from_file_location("policy", policy_path)
        policy_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(policy_mod)
    except Exception as e:
        result_queue.put(TrajectoryResult(
            error=str(e), error_type="CODE_ERROR",
            total_episodes=config_dict["num_episodes"]
        ))
        os.unlink(policy_path)
        return

    if not hasattr(policy_mod, "get_action"):
        result_queue.put(TrajectoryResult(
            error="Policy must define get_action(obs)",
            error_type="CODE_ERROR",
            total_episodes=config_dict["num_episodes"]
        ))
        os.unlink(policy_path)
        return

    get_action = policy_mod.get_action

    # ── attach shared memory ─────────────────────────────────
    W, H = 256, 256
    frame_bytes = W * H * 3

    try:
        frame_shm  = shm.SharedMemory(name=shm_name)
        meta_shm   = shm.SharedMemory(name=shm_meta_name)
    except Exception as e:
        result_queue.put(TrajectoryResult(
            error=f"Shared memory attach failed: {e}",
            error_type="RUNTIME_ERROR",
            total_episodes=config_dict["num_episodes"]
        ))
        return

    frame_buf = np.ndarray((H, W, 3), dtype=np.uint8, buffer=frame_shm.buf)
    # meta layout: [episode(i32), step(i32)]
    meta_buf  = meta_shm.buf

    # ── create env ───────────────────────────────────────────
    try:
        controller_config = suite.load_composite_controller_config(controller="BASIC")
        env = suite.make(
            env_name,
            robots=config_dict["robot"],
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="agentview",
            use_camera_obs=False,
            camera_names=["agentview"],
            camera_heights=H,
            camera_widths=W,
            reward_shaping=True,
            horizon=config_dict["max_episode_steps"],
        )
    except Exception as e:
        result_queue.put(TrajectoryResult(
            error=f"Env creation failed: {e}",
            error_type="RUNTIME_ERROR",
            total_episodes=config_dict["num_episodes"]
        ))
        frame_shm.close()
        meta_shm.close()
        return

    successes = 0
    episode_rewards = []
    trajectory_lines = []
    last_frame_time = 0.0

    for ep in range(config_dict["num_episodes"]):
        if stop_event.is_set():
            break

        obs = env.reset()
        if hasattr(policy_mod, "reset"):
            policy_mod.reset()

        start_snap = _snapshot_obs(obs)
        ep_reward = 0.0
        num_steps = 0

        for step in range(config_dict["max_episode_steps"]):
            if stop_event.is_set():
                break

            try:
                action = get_action(obs)
                action = np.array(action, dtype=np.float64).flatten()
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                num_steps = step + 1
            except Exception as e:
                result_queue.put(TrajectoryResult(
                    error=str(e), error_type="RUNTIME_ERROR",
                    total_episodes=config_dict["num_episodes"]
                ))
                env.close()
                frame_shm.close()
                meta_shm.close()
                os.unlink(policy_path)
                return

            # stream frame at ~15 fps
            now = time.time()
            if now - last_frame_time >= 0.067:
                last_frame_time = now
                try:
                    raw = env.sim.render(camera_name="agentview", width=W, height=H)
                    raw = raw[:, :, ::-1]   # BGR→RGB
                    np.copyto(frame_buf, raw)
                    struct.pack_into("ii", meta_buf, 0, ep, step)
                    frame_ready.set()
                    # wait up to 50 ms for consumer to pick it up, then continue
                    frame_consumed.wait(timeout=0.05)
                    frame_consumed.clear()
                except Exception:
                    pass  # never let render errors kill the sim

            if done:
                break

        success = env._check_success()
        if success:
            successes += 1
        episode_rewards.append(ep_reward)

        # Build trajectory line — full obs detail for first 2 episodes, summary for rest
        end_snap = _snapshot_obs(obs)
        line = f"Episode {ep}: reward={ep_reward:.2f}, success={success}, steps={num_steps}"
        if ep < 2:
            # Full obs snapshots for diagnostic visibility
            changed = _obs_diff(start_snap, end_snap)
            if changed:
                line += f"\n  Start: {_format_snap(start_snap)}"
                line += f"\n  End:   {_format_snap(end_snap)}"
                line += f"\n  Changed: {changed}"
            # Auto-analysis: compute distances, flag useful obs keys
            insights = _analyze_episode_obs(end_snap)
            if insights:
                line += "\n  Analysis:"
                line += "\n" + "\n".join(insights)
        trajectory_lines.append(line)

    env.close()
    frame_shm.close()
    meta_shm.close()
    os.unlink(policy_path)

    result_queue.put(TrajectoryResult(
        success_rate=successes / config_dict["num_episodes"],
        successes=successes,
        total_episodes=config_dict["num_episodes"],
        trajectory_summary="\n".join(trajectory_lines),
        episode_rewards=episode_rewards,
        error="",
        error_type="",
    ))


# ──────────────────────────────────────────────────────────────
# Public runner
# ──────────────────────────────────────────────────────────────

class SimulationRunner:

    def __init__(self, config: Config):
        self.config = config

    def run_policy(
        self,
        code: str,
        env_name: str,
        frame_callback=None,
        render: bool = False,
    ) -> TrajectoryResult:

        if env_name not in ENV_REGISTRY:
            raise RuntimeError(f"Unknown environment {env_name}")

        W, H = 256, 256
        frame_bytes = W * H * 3

        # ── allocate shared memory ────────────────────────────
        frame_shm = shm.SharedMemory(create=True, size=frame_bytes)
        meta_shm  = shm.SharedMemory(create=True, size=8)   # 2 x int32

        frame_ready    = mp.Event()
        frame_consumed = mp.Event()
        stop_event     = mp.Event()
        result_queue   = mp.Queue()

        config_dict = {
            "robot":             self.config.robot,
            "num_episodes":      self.config.num_episodes,
            "max_episode_steps": self.config.max_episode_steps,
        }

        proc = mp.Process(
            target=_simulation_worker,
            args=(
                code, env_name, config_dict,
                result_queue,
                frame_shm.name, meta_shm.name,
                frame_ready, frame_consumed, stop_event,
            ),
            daemon=True,
        )
        proc.start()

        # ── parent: consume frames until subprocess exits ──────
        frame_view = np.ndarray((H, W, 3), dtype=np.uint8, buffer=frame_shm.buf)
        meta_view  = meta_shm.buf

        timeout = self.config.sim_timeout
        deadline = time.time() + timeout

        while proc.is_alive() and time.time() < deadline:
            if frame_ready.wait(timeout=0.1):
                frame_ready.clear()
                if frame_callback is not None:
                    ep, step = struct.unpack_from("ii", meta_view, 0)
                    frame_callback(frame_view.copy(), ep, step)
                frame_consumed.set()

        if proc.is_alive():
            stop_event.set()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()

        # ── clean up shared memory ─────────────────────────────
        frame_shm.close()
        frame_shm.unlink()
        meta_shm.close()
        meta_shm.unlink()

        # ── retrieve result ────────────────────────────────────
        if not result_queue.empty():
            return result_queue.get_nowait()

        return TrajectoryResult(
            error="Simulation timed out or crashed",
            error_type="TIMEOUT",
            total_episodes=self.config.num_episodes,
        )