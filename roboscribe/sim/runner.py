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

        ep_reward = 0.0

        for step in range(config_dict["max_episode_steps"]):
            if stop_event.is_set():
                break

            try:
                action = get_action(obs)
                action = np.array(action, dtype=np.float64).flatten()
                obs, reward, done, info = env.step(action)
                ep_reward += reward
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
        trajectory_lines.append(f"Episode {ep}: reward={ep_reward:.2f}, success={success}")

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