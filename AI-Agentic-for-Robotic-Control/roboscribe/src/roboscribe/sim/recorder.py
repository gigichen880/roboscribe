"""Video recorder for robosuite policy rollouts."""

from __future__ import annotations

import importlib.util
import os
import tempfile

import numpy as np


def record_policy(
    policy_path: str,
    env_name: str,
    output_path: str = "rollout.mp4",
    robot: str = "Panda",
    num_episodes: int = 1,
    max_steps: int = 200,
    camera: str = "agentview",
    width: int = 640,
    height: int = 480,
    fps: int = 30,
) -> str:
    """Record a policy rollout as an MP4 video.

    Args:
        policy_path: Path to a .py file with get_action(obs) function.
        env_name: Robosuite environment name.
        output_path: Where to save the MP4.
        robot: Robot name.
        num_episodes: Number of episodes to record.
        max_steps: Max steps per episode.
        camera: Camera name for rendering.
        width: Video width.
        height: Video height.
        fps: Video frames per second.

    Returns:
        Path to the saved video file.
    """
    import imageio
    import mujoco
    import robosuite as suite

    # Load policy module
    spec = importlib.util.spec_from_file_location("policy", policy_path)
    policy_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy_mod)
    get_action = policy_mod.get_action

    # Create environment
    controller_config = suite.load_composite_controller_config(controller="BASIC")
    env = suite.make(
        env_name,
        robots=robot,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping=True,
        horizon=max_steps,
    )

    # Create MuJoCo native renderer
    renderer = mujoco.Renderer(env.sim.model._model, height=height, width=width)

    writer = imageio.get_writer(output_path, fps=fps, quality=8)

    try:
        for ep in range(num_episodes):
            obs = env.reset()
            if hasattr(policy_mod, "reset"):
                policy_mod.reset()

            for step in range(max_steps):
                # Render frame
                renderer.update_scene(env.sim.data._data, camera=camera)
                frame = renderer.render()
                writer.append_data(frame)

                # Step
                action = get_action(obs)
                action = np.array(action, dtype=np.float64).flatten()
                obs, reward, done, info = env.step(action)

                if done:
                    break

            success = env._check_success()
            print(f"Episode {ep}: success={success}, steps={step + 1}")

    finally:
        writer.close()
        renderer.close()
        env.close()

    print(f"Video saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Record robosuite policy rollout as video")
    parser.add_argument("policy", help="Path to policy .py file")
    parser.add_argument("--env", "-e", required=True, help="Environment name")
    parser.add_argument("--output", "-o", default="rollout.mp4", help="Output video path")
    parser.add_argument("--episodes", "-n", type=int, default=1, help="Number of episodes")
    parser.add_argument("--camera", default="agentview", help="Camera name")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    args = parser.parse_args()

    record_policy(
        args.policy,
        args.env,
        output_path=args.output,
        num_episodes=args.episodes,
        camera=args.camera,
        fps=args.fps,
    )
