# https://github.com/openai/gym/issues/585#issuecomment-370015441


from random import random, seed
from datetime import datetime
import numpy as np
from numpy import float32, inf
from src.environment.spaces import Box
from src.params import STEP_ACTION_RATE
# from math import sin, cos, pi

seed(datetime.now())

A_JOINTS = {
    b'core_left_shoulder',
    b'core_right_shoulder',
    b'core_back_left_shoulder',
    b'core_back_right_shoulder',
}

B_JOINTS = {
    b'left_shoulder_left_leg_top',
    b'right_shoulder_right_leg_top',
    b'back_left_shoulder_back_right_leg_top',
    b'back_right_shoulder_back_right_leg_top',
}

C_JOINTS = {
    b'left_leg_top_left_leg_bottom',
    b'right_leg_top_right_leg_bottom',
    b'back_left_leg_top_back_left_leg_bottom',
    b'back_right_leg_top_back_right_leg_bottom',
}

JOINTS = {*A_JOINTS, *B_JOINTS, *C_JOINTS}
# b'left_leg_bottom_left_foot',
# b'right_leg_bottom_right_foot',
# b'back_left_leg_bottom_back_left_foot',
# b'back_right_leg_bottom_back_right_foot'


class BaseEnv:
    def __init__(
            self,
            name,
            var=0.1,
            vis=False):

        import pybullet
        import pybullet_utils.bullet_client as bc
        import pybullet_data

        self.var = var
        self.vis = vis
        self.name = name
        self.last_state = None
        self.current_state = None
        self.client = bc.BulletClient(connection_mode=pybullet.GUI) if vis \
            else bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.i = 0
        self.reset()
        self.describe_space()

    def describe_space(self):
        all_state = self._get_state()
        joint_lower_bounds, joint_upper_bounds = [], []
        num_joints = self.client.getNumJoints(self.robot_id)

        for joint_i in range(num_joints):
            lower, upper = self.client \
                .getJointInfo(self.robot_id, joint_i)[8:10]
            joint_lower_bounds.append(lower)
            joint_upper_bounds.append(upper)

        obs_space_upper_bounds = joint_upper_bounds \
            + [inf for _ in range(num_joints, len(all_state))] + [1]
        obs_space_lower_bounds = joint_lower_bounds \
            + [inf for _ in range(num_joints, len(all_state))] + [1]
        self.observation_space = Box((len(obs_space_upper_bounds) + 1,),
                                     obs_space_upper_bounds,
                                     obs_space_lower_bounds)

        self.action_space = Box((num_joints, ),
                                np.array(joint_upper_bounds, dtype=float32),
                                np.array(joint_lower_bounds, dtype=float32))

        return self.current_state

    def reset(self):
        self.plane_id = None
        self.robot_id = None
        for body_id in range(self.client.getNumBodies()):
            self.client.removeBody(body_id)

        slope = self.client.getQuaternionFromEuler([
            self.var * random() - self.var / 2,
            self.var * random() - self.var / 2,
            0])

        # slope = self.client.getQuaternionFromEuler([0, 0, 0])

        self.plane_id = self.client.loadURDF(
            "plane.urdf",
            [0, 0, 0],
            slope)

        # self.client.changeDynamics(
        #     self.plane_id, -1, lateralFriction=1, restitution=0)

        robot_start_pos = [0, 0, 0.4]
        robot_start_orientation = self.client.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = self.client.loadURDF(
            "src/environment/urdf/robot-simple.urdf",
            robot_start_pos,
            robot_start_orientation)

        # create dictionary for each bodypart/joint
        num_joints = self.client.getNumJoints(self.robot_id)
        self.joints = {}
        for joint_i in range(num_joints):
            joint_data = self.client.getJointInfo(self.robot_id, joint_i)
            self.joints[joint_data[1]] = joint_data[0]

        self.shoulder_joints = set(self.joints[name] for name in A_JOINTS)
        self.hip_joints = set(self.joints[name] for name in B_JOINTS)
        self.knee_joints = set(self.joints[name] for name in C_JOINTS)

        self.action_set = sorted([self.joints[name] for name in JOINTS])

        self.client.setGravity(0, 0, -10)
        state = self._get_state()
        self.last_state = state
        self.current_state = state

        # self.client.changeDynamics(
        #     self.robot_id, -1, lateralFriction=1, restitution=0)
        return state

    def take_action(self, actions):
        for joint_i, action in zip(self.action_set, actions):
            maxForce = 500
            self.client.setJointMotorControl2(
                self.robot_id, joint_i,
                controlMode=self.client.POSITION_CONTROL,
                targetPosition=action,
                force=maxForce)

    def step(self, actions):
        self.i += 1
        self.last_state = self.current_state
        for _ in range(STEP_ACTION_RATE):
            self.take_action(actions)
        self.client.stepSimulation()
        # note _get_state must happen before _get_reward or _get_reward
        # will return nonsense!
        self.current_state = self._get_state()
        reward, done = self._get_reward()
        return self.current_state, reward, done, None

    def _get_state(self):
        base_link_pos, base_link_orient = self.client \
            .getBasePositionAndOrientation(self.robot_id)
        joint_states = [self.client.getJointState(self.robot_id, i)[0:2]
                        for i in self.action_set]
        joint_states = [state for joint_state in joint_states
                        for state in joint_state]
        state = np.array([
            *base_link_pos,
            *base_link_orient,
            *joint_states
        ])
        return state

    def close(self):
        self.client.disconnect()
