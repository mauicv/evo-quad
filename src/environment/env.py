from random import random, seed
from datetime import datetime
import numpy as np
from numpy import float32, inf
from src.environment.spaces import Box
from math import sin, pi

seed(datetime.now())

TARGET_LOC = np.array([0.0, 0.0, 0.18])
TARGET_ORIENT = np.array([1, 1, 0])
JOINT_AT_LIMIT_COST = 0.1
TORQUE_COST = 0.4
STEP_ACTION_RATE = 5
REWARD_SCALE = 10
GROUND_CONTACT_COST = 100
OSC_PERIOD = 200


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

        robot_start_pos = [0, 0, 0.25]
        robot_start_orientation = self.client.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = self.client.loadURDF(
            "src/environment/urdf/robot-simple.urdf",
            robot_start_pos,
            robot_start_orientation)

        self.client.setGravity(0, 0, -10)
        state = self._get_state()
        self.last_state = state
        self.current_state = state

        self.shoulder_joints = set(3*i for i in range(4))
        self.hip_joints = set(3*i + 1 for i in range(4))
        self.knee_joints = set(3*i + 2 for i in range(4))

        return state

    def take_action(self, actions):
        for joint_i, action in enumerate(actions):
            maxForce = 175
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
        state_ls = [self.client.getLinkState(self.robot_id, i)[0]
                    for i in range(self.client.getNumJoints(self.robot_id))]
        base_link_state = self.client \
            .getBasePositionAndOrientation(self.robot_id)[0]
        state = np.array([
            *[self.client.getJointState(self.robot_id, i)[0]
              for i in range(self.client.getNumJoints(self.robot_id))],
            *[item for subls in state_ls for item in subls],
            *base_link_state,
            sin(self.i*2*pi/OSC_PERIOD)*10
        ])
        return state

    def close(self):
        self.client.disconnect()
