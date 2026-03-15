"""Hand-written few-shot example policies for prompt engineering."""

LIFT_POLICY = '''\
import numpy as np

# State machine states
APPROACH = 0
LOWER = 1
GRASP = 2
LIFT = 3
DONE = 4

state = APPROACH
grasp_counter = 0


def reset():
    """Reset state machine between episodes."""
    global state, grasp_counter
    state = APPROACH
    grasp_counter = 0


def get_action(obs):
    """Generate action to pick up the cube.

    Controller: OSC_POSE (7D) -> [dx, dy, dz, dax, day, daz, gripper]
    Gripper: -1 = open, 1 = close
    """
    global state, grasp_counter

    eef_pos = obs["robot0_eef_pos"]
    cube_pos = obs["cube_pos"]

    # Compute position error
    dx = cube_pos[0] - eef_pos[0]
    dy = cube_pos[1] - eef_pos[1]
    dz = cube_pos[2] - eef_pos[2]

    action = np.zeros(7)

    if state == APPROACH:
        # Move above the cube with gripper open
        action[0] = dx * 10.0  # proportional control for x
        action[1] = dy * 10.0  # proportional control for y
        action[2] = (cube_pos[2] + 0.08 - eef_pos[2]) * 10.0  # hover above
        action[6] = -1  # open gripper

        # Transition when aligned in x,y and above cube
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            state = LOWER

    elif state == LOWER:
        # Lower to grasp height
        action[0] = dx * 10.0
        action[1] = dy * 10.0
        action[2] = (cube_pos[2] + 0.01 - eef_pos[2]) * 10.0  # just above cube
        action[6] = -1  # keep gripper open

        if abs(dz) < 0.03:  # close enough vertically to grasp
            state = GRASP
            grasp_counter = 0

    elif state == GRASP:
        # Close gripper and wait
        action[0] = dx * 5.0  # maintain position
        action[1] = dy * 5.0
        action[2] = 0.0  # hold height
        action[6] = 1  # close gripper

        grasp_counter += 1
        if grasp_counter > 15:
            state = LIFT

    elif state == LIFT:
        # Lift the cube
        action[0] = 0.0
        action[1] = 0.0
        action[2] = 1.0  # move up
        action[6] = 1  # keep gripper closed

        if eef_pos[2] > 1.0:
            state = DONE

    elif state == DONE:
        # Hold position
        action[6] = 1  # keep gripper closed

    # Clip actions to valid range
    action[:6] = np.clip(action[:6], -1.0, 1.0)
    return action
'''

STACK_POLICY = '''\
import numpy as np

# State machine states
APPROACH_A = 0
LOWER_A = 1
GRASP_A = 2
LIFT_A = 3
MOVE_TO_B = 4
LOWER_TO_B = 5
RELEASE = 6
RETREAT = 7
DONE = 8

state = APPROACH_A
counter = 0


def reset():
    """Reset state machine between episodes."""
    global state, counter
    state = APPROACH_A
    counter = 0


def get_action(obs):
    """Generate action to stack cubeA on cubeB.

    Controller: OSC_POSE (7D) -> [dx, dy, dz, dax, day, daz, gripper]
    Gripper: -1 = open, 1 = close
    """
    global state, counter

    eef_pos = obs["robot0_eef_pos"]
    cubeA_pos = obs["cubeA_pos"]
    cubeB_pos = obs["cubeB_pos"]

    action = np.zeros(7)

    if state == APPROACH_A:
        # Move above cubeA
        dx = cubeA_pos[0] - eef_pos[0]
        dy = cubeA_pos[1] - eef_pos[1]
        action[0] = dx * 10.0
        action[1] = dy * 10.0
        action[2] = (cubeA_pos[2] + 0.08 - eef_pos[2]) * 10.0
        action[6] = -1

        if abs(dx) < 0.01 and abs(dy) < 0.01:
            state = LOWER_A

    elif state == LOWER_A:
        dx = cubeA_pos[0] - eef_pos[0]
        dy = cubeA_pos[1] - eef_pos[1]
        action[0] = dx * 10.0
        action[1] = dy * 10.0
        action[2] = (cubeA_pos[2] + 0.01 - eef_pos[2]) * 10.0
        action[6] = -1

        dz = cubeA_pos[2] + 0.01 - eef_pos[2]
        if abs(dz) < 0.03:
            state = GRASP_A
            counter = 0

    elif state == GRASP_A:
        dx = cubeA_pos[0] - eef_pos[0]
        dy = cubeA_pos[1] - eef_pos[1]
        action[0] = dx * 5.0
        action[1] = dy * 5.0
        action[6] = 1

        counter += 1
        if counter > 15:
            state = LIFT_A

    elif state == LIFT_A:
        action[2] = 1.0
        action[6] = 1

        if eef_pos[2] > cubeB_pos[2] + 0.15:
            state = MOVE_TO_B

    elif state == MOVE_TO_B:
        dx = cubeB_pos[0] - eef_pos[0]
        dy = cubeB_pos[1] - eef_pos[1]
        action[0] = dx * 10.0
        action[1] = dy * 10.0
        action[2] = (cubeB_pos[2] + 0.15 - eef_pos[2]) * 5.0  # maintain height
        action[6] = 1

        if abs(dx) < 0.01 and abs(dy) < 0.01:
            state = LOWER_TO_B

    elif state == LOWER_TO_B:
        dx = cubeB_pos[0] - eef_pos[0]
        dy = cubeB_pos[1] - eef_pos[1]
        action[0] = dx * 10.0
        action[1] = dy * 10.0
        action[2] = (cubeB_pos[2] + 0.05 - eef_pos[2]) * 10.0
        action[6] = 1

        dz = cubeB_pos[2] + 0.05 - eef_pos[2]
        if abs(dz) < 0.03:
            state = RELEASE
            counter = 0

    elif state == RELEASE:
        action[6] = -1
        counter += 1
        if counter > 10:
            state = RETREAT

    elif state == RETREAT:
        action[2] = 1.0
        action[6] = -1
        if eef_pos[2] > 1.0:
            state = DONE

    elif state == DONE:
        action[6] = -1

    action[:6] = np.clip(action[:6], -1.0, 1.0)
    return action
'''

DOOR_POLICY = '''\
import numpy as np
from roboscribe.pid import PID

class DoorPolicy:
    """6-phase state machine: orient gripper, approach, reach, grip, rotate handle, pull door.

    Phase Sequence:
        0: ORIENT    - Rotate gripper so fingers can wrap the handle bar
        1: APPROACH   - Move to offset position near handle (gripper open)
        2: REACH      - Move to handle grip position (gripper open)
        3: GRIP       - Close gripper and wait for secure grip
        4: ROTATE     - Rotate wrist to turn handle and unlatch door
        5: PULL       - Pull door open (+Y) while maintaining grip and rotation

    IMPORTANT: PID.reset() is only called on phase transitions, NOT every step.
    Use self.pid.target = new_target to update target without resetting state.
    """

    KP = 10.0
    KI = 0.0
    KD = 0.5
    DT = 0.05

    THRESHOLD = 0.03
    ORIENT_STEPS = 20
    WAIT_STEPS = 15
    HANDLE_THRESHOLD = 0.25  # handle_qpos angle to trigger pull (must fit in 200-step horizon)
    ROTATE_MAX_STEPS = 100   # safety timeout for rotate phase

    GRIPPER_OPEN = -1.0
    GRIPPER_CLOSE = 1.0

    APPROACH_OFFSET = np.array([0.10, 0.0, 0.05])
    GRIP_OFFSET = np.array([0.0, -0.01, -0.01])
    PULL_FORCE = 0.5

    ORIENT_ROTATION = np.array([-0.3, 0.0, 0.0])   # pitch gripper down
    HANDLE_ROTATION = np.array([0.0, -0.3, 0.0])    # rotate handle via wrist

    def __init__(self, obs):
        handle_pos = np.array(obs["handle_pos"])
        eef_pos = np.array(obs["robot0_eef_pos"])
        self.initial_handle_pos = handle_pos.copy()
        self.initial_eef_pos = eef_pos.copy()
        self.phase = 0
        self.step_counter = 0
        self.pid = PID(kp=self.KP, ki=self.KI, kd=self.KD, target=eef_pos)

    def get_action(self, obs):
        current_pos = np.array(obs["robot0_eef_pos"])
        handle_pos = np.array(obs["handle_pos"])
        action = np.zeros(7)

        if self.phase == 0:  # ORIENT — hold position, rotate gripper
            self.pid.target = self.initial_eef_pos
            control = self.pid.update(current_pos, self.DT)
            action[0:3] = control
            action[3:6] = self.ORIENT_ROTATION
            action[6] = self.GRIPPER_OPEN
            self.step_counter += 1
            if self.step_counter >= self.ORIENT_STEPS:
                self.phase = 1
                self.step_counter = 0
                self.pid.reset(target=self.initial_handle_pos + self.APPROACH_OFFSET)

        elif self.phase == 1:  # APPROACH — move near handle
            target = self.initial_handle_pos + self.APPROACH_OFFSET
            self.pid.target = target
            control = self.pid.update(current_pos, self.DT)
            action[0:3] = control
            action[6] = self.GRIPPER_OPEN
            if np.linalg.norm(current_pos - target) < self.THRESHOLD:
                self.phase = 2
                self.pid.reset(target=handle_pos + self.GRIP_OFFSET)

        elif self.phase == 2:  # REACH — move to handle
            target = handle_pos + self.GRIP_OFFSET
            self.pid.target = target
            control = self.pid.update(current_pos, self.DT)
            action[0:3] = control
            action[6] = self.GRIPPER_OPEN
            if np.linalg.norm(current_pos - target) < self.THRESHOLD:
                self.phase = 3
                self.step_counter = 0
                self.pid.reset(target=handle_pos + self.GRIP_OFFSET)

        elif self.phase == 3:  # GRIP — close gripper and wait
            target = handle_pos + self.GRIP_OFFSET
            self.pid.target = target
            control = self.pid.update(current_pos, self.DT)
            action[0:3] = control
            action[6] = self.GRIPPER_CLOSE
            self.step_counter += 1
            if self.step_counter >= self.WAIT_STEPS:
                self.phase = 4
                self.step_counter = 0
                self.pid.reset(target=handle_pos + self.GRIP_OFFSET)

        elif self.phase == 4:  # ROTATE — turn handle
            target = handle_pos + self.GRIP_OFFSET
            self.pid.target = target
            control = self.pid.update(current_pos, self.DT)
            action[0:3] = control
            action[3:6] = self.HANDLE_ROTATION
            action[6] = self.GRIPPER_CLOSE
            handle_qpos = float(obs.get("handle_qpos", 0))
            self.step_counter += 1
            if handle_qpos >= self.HANDLE_THRESHOLD or self.step_counter >= self.ROTATE_MAX_STEPS:
                self.phase = 5
                self.step_counter = 0
                self.pid.reset(target=handle_pos + self.GRIP_OFFSET)

        elif self.phase == 5:  # PULL — open door
            target = handle_pos + self.GRIP_OFFSET
            self.pid.target = target
            control = self.pid.update(current_pos, self.DT)
            pull = np.array([0.0, self.PULL_FORCE, 0.0])
            action[0:3] = control + pull
            action[3:6] = self.HANDLE_ROTATION * 0.5  # lighter rotation to prevent spring-back
            action[6] = self.GRIPPER_CLOSE

        action[:6] = np.clip(action[:6], -1.0, 1.0)
        return action


# Module-level shim for compatibility with roboscribe runner
_policy = None

def reset():
    global _policy
    _policy = None

def get_action(obs):
    global _policy
    if _policy is None:
        _policy = DoorPolicy(obs)
    return _policy.get_action(obs)
'''

# Map env names to their few-shot examples
FEW_SHOT_EXAMPLES: dict[str, str] = {
    "Lift": LIFT_POLICY,
    "Stack": STACK_POLICY,
    "Door": DOOR_POLICY,
}
