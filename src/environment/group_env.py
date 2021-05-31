import numpy as np

import pybullet
import pybullet_utils.bullet_client as bc
import pybullet_data


TARGET_LOC = np.array([0.0, 0.0, 0.18])
TARGET_ORIENT = np.array([1, 1, 0])
JOINT_AT_LIMIT_COST = 0.1
TORQUE_COST = 0.4
STEP_ACTION_RATE = 5
REWARD_SCALE = 10
GROUND_CONTACT_COST = 100


class GroupEnv:
    def __init__(
            self,
            name,
            vis=False):
        self.name = name
        self.client = bc.BulletClient(connection_mode=pybullet.GUI) if vis \
            else bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.client.setGravity(0, 0, -10)
        self.plain_id = self.client.loadURDF(
            "plane.urdf",
            [0, 0, 0])

        self.last_states = []
        self.current_states = []
        self.robot_ids = []

    def add_actor(self, x_y_loc):
        robot_start_pos = [*x_y_loc, 0.25]
        robot_start_orientation = self.client.getQuaternionFromEuler([0, 0, 0])
        self.robot_ids.append(self.client.loadURDF(
            "src/environment/urdf/robot-simple.urdf",
            robot_start_pos,
            robot_start_orientation))
        state = self.get_state(self.robot_ids[-1])
        self.last_states.append(state)
        self.current_states.append(state)

    def get_state(self, robot_id):
        state_ls = [self.client.getLinkState(robot_id, i)[0]
                    for i in range(self.client.getNumJoints(robot_id))]
        base_link_state = self.client \
            .getBasePositionAndOrientation(robot_id)[0]
        state = np.array([
            *[self.client.getJointState(robot_id, i)[0]
              for i in range(self.client.getNumJoints(robot_id))],
            *[item for subls in state_ls for item in subls],
            *base_link_state
        ])
        return state

    def get_states(self):
        return [self.get_state(r_id) for r_id in self.robot_ids]

    def step_i(self, i, action):
        self.last_states[i] = self.current_states[i]
        for joint_i, joint_action in enumerate(action):
            maxForce = 175
            self.client.setJointMotorControl2(
                self.robot_ids[i], joint_i,
                controlMode=self.client.POSITION_CONTROL,
                targetPosition=joint_action,
                force=maxForce)
        self.current_states = self.get_states()
