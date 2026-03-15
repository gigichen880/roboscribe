"""Registry of supported robosuite environments."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnvInfo:
    """Information about a robosuite environment."""

    name: str
    description: str
    objects: list[str] = field(default_factory=list)
    action_dim: int = 7  # OSC_POSE default
    obs_keys: list[str] = field(default_factory=list)
    tips: str = ""
    goal_description: str = ""

    @property
    def obs_keys_str(self) -> str:
        return ", ".join(self.obs_keys)


ENV_REGISTRY: dict[str, EnvInfo] = {
    "Lift": EnvInfo(
        name="Lift",
        description="Pick up a cube from the table",
        objects=["cube"],
        obs_keys=[
            "robot0_eef_pos",       # (3,) end-effector position
            "robot0_eef_quat",      # (4,) end-effector quaternion
            "robot0_gripper_qpos",  # (2,) gripper joint positions
            "cube_pos",             # (3,) cube position
            "cube_quat",            # (4,) cube quaternion
        ],
        goal_description="Lift the cube above a height threshold (z > 0.85)",
        tips=(
            "The cube starts on the table around z=0.82. "
            "Approach from above, lower to grasp height (~cube_z + 0.01), "
            "close gripper (action[-1] = 1), then lift upward. "
            "Key: be precise with x,y alignment before lowering. "
            "Gripper: -1 = open, 1 = close."
        ),
    ),
    "Stack": EnvInfo(
        name="Stack",
        description="Stack cube A (red) on top of cube B (green)",
        objects=["cubeA", "cubeB"],
        obs_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "cubeA_pos",            # (3,) red cube position
            "cubeA_quat",
            "cubeB_pos",            # (3,) green cube position
            "cubeB_quat",
        ],
        goal_description="Place cubeA on top of cubeB",
        tips=(
            "Two-phase task: (1) pick up cubeA, (2) place on cubeB. "
            "Approach cubeA, grasp it, lift high enough to clear cubeB, "
            "move over cubeB, lower, release. "
            "Important: lift cubeA high enough (z > cubeB_z + 0.05) before moving laterally. "
            "Gripper: -1 = open, 1 = close."
        ),
    ),
    "PickPlaceCan": EnvInfo(
        name="PickPlaceCan",
        description="Pick up a can and place it in a bin",
        objects=["can", "bin"],
        obs_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "Can_pos",
            "Can_quat",
        ],
        goal_description="Place the can into the correct bin",
        tips=(
            "Pick up the can and move it to the bin location. "
            "The bin is at a fixed position. Approach can, grasp, lift, "
            "move to bin, lower, release. Be careful with grasp alignment."
        ),
    ),
    "NutAssemblySquare": EnvInfo(
        name="NutAssemblySquare",
        description="Pick up a square nut and place it on a peg",
        objects=["square_nut", "peg"],
        obs_keys=[
            "robot0_eef_pos",       # (3,) end-effector xyz
            "robot0_eef_quat",      # (4,) end-effector quaternion
            "robot0_gripper_qpos",  # (2,) gripper joint positions
            "SquareNut_pos",        # (3,) nut center xyz — NOTE: nut is THIN/FLAT, ~2cm tall
            "SquareNut_quat",       # (4,) nut orientation
        ],
        goal_description="Place the square nut onto the square peg",
        tips=(
            "CRITICAL INFO for NutAssemblySquare:\n"
            "- The nut is FLAT and THIN (~2cm). Its center z is ~0.83 on the table.\n"
            "- To grasp: lower eef to z = SquareNut_pos[2] - 0.01 (BELOW center, almost table level).\n"
            "- The peg position is NOT in obs. It is fixed at approximately x=0.23, y=0.10, z=0.85.\n"
            "- Phase 1: Approach above nut (z = nut_z + 0.08), align x,y precisely (error < 0.005).\n"
            "- Phase 2: Lower to grasp height (z = nut_z - 0.01), keep gripper open.\n"
            "- Phase 3: Close gripper (action[6]=1.0), WAIT 25+ steps for firm grasp.\n"
            "- Phase 4: Lift HIGH (z > 1.1) before moving laterally.\n"
            "- Phase 5: Move to peg position (x=0.23, y=0.10) while keeping z high.\n"
            "- Phase 6: Lower slowly onto peg (target z ~0.88), then open gripper.\n"
            "- Use proportional control: action[i] = gain * error, gain=10-15 for approach, 5-8 for fine.\n"
            "- Gripper: -1 = open, 1 = close."
        ),
    ),
    "Door": EnvInfo(
        name="Door",
        description="Open a door by turning the handle",
        objects=["door", "handle"],
        obs_keys=[
            "robot0_eef_pos",       # (3,) end-effector position [x, y, z]
            "robot0_eef_quat",      # (4,) end-effector orientation quaternion
            "robot0_gripper_qpos",  # (2,) gripper joint positions
            "door_pos",             # (3,) door body position
            "handle_pos",           # (3,) handle position [x, y, z]
            "hinge_qpos",           # scalar (float) — door hinge angle, NOT an array
        ],
        goal_description="Open the door past a threshold angle",
        tips=(
            "CRITICAL INFO for Door:\n"
            "- hinge_qpos is a SCALAR (float), not an array. Use it directly: angle = obs['hinge_qpos'], do NOT index it.\n"
            "- Some obs values may be scalars. Always check: if np.ndim(val) == 0, use val directly.\n"
            "- Phase 1: Move to handle position (align eef x,y,z with handle_pos).\n"
            "- Phase 2: Close gripper on handle (action[6]=1.0), wait 15+ steps.\n"
            "- Phase 3: Pull handle toward robot (negative y direction typically) to open door.\n"
            "- Use proportional control: action[i] = gain * (target - current), gain=10-15.\n"
            "- Gripper: -1 = open, 1 = close.\n"
            "- Success: hinge_qpos exceeds threshold (door is open)."
        ),
    ),
    "Wipe": EnvInfo(
        name="Wipe",
        description="Wipe markers off the table surface",
        objects=["markers"],
        obs_keys=[
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ],
        goal_description="Move end-effector over all marker positions to wipe them",
        tips=(
            "Move the end-effector in a sweeping pattern across the table. "
            "Lower to table surface and sweep systematically. "
            "No grasping needed — this is a contact/coverage task."
        ),
    ),
}
