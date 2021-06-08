
import numpy as np
from src.environment.env import BaseEnv

TARGET_LOC = np.array([0.0, 0.0, 0.18])
TARGET_ORIENT = np.array([1, 1, 0])
JOINT_AT_LIMIT_COST = 0.01
TORQUE_COST = 0.4
STEP_ACTION_RATE = 5
REWARD_SCALE = 10
GROUND_CONTACT_COST = 100


class WalkingEnv(BaseEnv):
    def __init__(
            self,
            name,
            var=0.1,
            vis=False):
        super().__init__(name, var, vis)

    def _standing_reward(self):
        """Dependent on base link difference from target location and
        target orientation. If base link contacts ground 100 point penalty is
        added and a small cost is added per unit of torque used.
        """

        base_data = self.client.getBasePositionAndOrientation(self.robot_id)
        base_loc = np.array(base_data[0])
        orient = np.array(self.client.getEulerFromQuaternion(base_data[1]))

        dist_from_target = np.linalg.norm(base_loc - TARGET_LOC) \
            + 6 * np.linalg.norm(orient * TARGET_ORIENT)
        return REWARD_SCALE / max(dist_from_target, 0.01)

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
            # self._standing_reward(),
            self._progress_reward(),
            # self._torque_cost()
        ])
        done = self._check_done()
        return (costs.sum(), done) if not done \
            else (costs.sum() - GROUND_CONTACT_COST, done)

    def _progress_reward(self):
        forwards_movement = self.current_state[1] - self.last_state[1]
        return forwards_movement * REWARD_SCALE

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
        for joint_i, action in zip(self.action_set, actions):
            # restrict joints.
            if joint_i in self.hip_joints or joint_i in self.knee_joints:
                maxForce = 175
                self.client.setJointMotorControl2(
                    self.robot_id, joint_i,
                    controlMode=self.client.POSITION_CONTROL,
                    targetPosition=action,
                    force=maxForce)
