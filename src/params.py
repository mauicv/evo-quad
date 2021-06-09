ENV_NAME = 'walking-quadruped'
EPISODES = 10000
STATE_DIMS = 33
ACTION_DIMS = 12
MIN_ACTION = -0.785398
MAX_ACTION = 0.785398
STEPS = 500
LAYER_DIMS = [20, 20]
BATCH_SIZE = 50
POPULATION_SIZE = 300
JOINT_AT_LIMIT_COST = 0.01
TORQUE_COST = 0.4
STEP_ACTION_RATE = 5
REWARD_SCALE = 10
GROUND_CONTACT_COST = 100
DIR = './data/default/'
ALPHA = 0.5
STD_DEV = 1.5
WEIGHT_LOW = -2
WEIGHT_HIGH = 2
