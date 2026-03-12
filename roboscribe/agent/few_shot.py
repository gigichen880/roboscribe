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

        if abs(dz - 0.01) < 0.02:
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
        if abs(dz) < 0.02:
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
        if abs(dz) < 0.02:
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

# Map env names to their few-shot examples
FEW_SHOT_EXAMPLES: dict[str, str] = {
    "Lift": LIFT_POLICY,
    "Stack": STACK_POLICY,
}
