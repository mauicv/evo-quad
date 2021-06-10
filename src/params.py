import os
from dotenv import load_dotenv
load_dotenv()

EPISODES = int(os.environ['EPISODES'])
STEPS = int(os.environ['STEPS'])
BATCH_SIZE = int(os.environ['BATCH_SIZE'])
POPULATION_SIZE = int(os.environ['POPULATION_SIZE'])

ENV_NAME = 'walking-quadruped'
STATE_DIMS = 33
ACTION_DIMS = 12
MIN_ACTION = -0.785398
MAX_ACTION = 0.785398
LAYER_DIMS = [20, 20]
JOINT_AT_LIMIT_COST = 0.01
TORQUE_COST = 0.4
REWARD_SCALE = 10
GROUND_CONTACT_COST = 100
DIR = './data/default/'
ALPHA = 0.5
STD_DEV = 1.5
WEIGHT_LOW = -2
WEIGHT_HIGH = 2
INPUT_SCALING_VAL = 6
