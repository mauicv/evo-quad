from src.environment.walking_env import WalkingEnv # noqa
import numpy as np


def setup_env(var=0, vis=True):
    env = WalkingEnv('quadruped', var=var, vis=vis)
    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.shape[0]
    state_norm_array = env.observation_space.high
    min_action = env.action_space.low.min()
    max_action = env.action_space.high.max()
    if np.any(np.isinf(state_norm_array)):
        state_norm_array = np.ones_like(state_norm_array)
    return env, state_space_dim, action_space_dim, state_norm_array, \
        min_action, max_action
