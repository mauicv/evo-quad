
# https://www.etedal.net/2020/04/pybullet-panda_2.html

import numpy as np
from src.environment.env import BaseEnv
from src.params import REWARD_SCALE, TORQUE_COST, \
    GROUND_CONTACT_COST, JOINT_AT_LIMIT_COST

TARGET_HEIGHT = 0.30487225144721647


class WalkingEnv(BaseEnv):
    def __init__(
            self,
            name,
            var=0.1,
            vis=False):
        super().__init__(name, var, vis)

    def _torque_cost(self):
        torques = [abs(self.client.getJointState(self.robot_id, i)[3]) / 1500
                   for i in range(self.client.getNumJoints(self.robot_id))]
        torque_sum = sum(torques)
        return - torque_sum * TORQUE_COST

    def _check_done(self):
        if self.client.getContactPoints(
                bodyA=self.robot_id,
                linkIndexA=-1,
                bodyB=self.plane_id,
                linkIndexB=-1):
            return True
        return False

    def _get_reward(self):
        costs = np.array([
            self._joints_at_limit_cost(),
            self._progress_reward(),
            # self._torque_cost()
        ])
        done = self._check_done()
        return (costs.sum(), done) if not done \
            else (costs.sum() - GROUND_CONTACT_COST, done)

    def _progress_reward(self):
        forwards_movement = self.current_state[1] - self.last_state[1]
        vertical_movement = 4 * abs(self.current_state[2] - TARGET_HEIGHT) ** 2
        return forwards_movement * REWARD_SCALE - vertical_movement

    def _joints_at_limit_cost(self):
        count = 0
        for joint_i in self.action_set:
            j_rad = self.client.getJointState(self.robot_id, joint_i)[0]
            joint_per_loc = \
                (j_rad + abs(self.observation_space.low[joint_i])) / \
                self.observation_space.arc_sizes[joint_i]
            if joint_per_loc < 0.05 or joint_per_loc > 0.95:
                count += 1
        return - count * JOINT_AT_LIMIT_COST

    def take_action(self, actions):
        self.last_state = self.current_state
        for joint_i, action in zip(self.action_set, actions):
            # restrict joints.
            if joint_i in self.hip_joints or joint_i in self.knee_joints:
                self.client.setJointMotorControl2(
                    self.robot_id, joint_i,
                    controlMode=self.client.VELOCITY_CONTROL,
                    targetVelocity=action,
                    force=1500)
