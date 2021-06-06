from src.environment import setup_env
import shutil
import click
from src.training.batch import BatchJob
import os
import numpy as np
import sys
from src.environment.walking_env import WalkingEnv
from src.training.stream_redirect import RedirectAllOutput
from src.training.mappings import action_map
from time import time

batch_job = BatchJob()


@batch_job
def test_cpu_speeds(array):
    envs = [WalkingEnv('', var=0, vis=False)
            for _ in range(len(array))]
    states = [np.array(env.current_state, dtype='float32') for env in envs]
    action_size = len(states)

    for _ in range(100):
        for index, env in \
                enumerate(envs):
            action = np.random.uniform(action_size, -1, 1)
            action = action_map(action)
            next_state, _, _, _ = env.step(action)

    # Closing envs fixes memory leak:
    for env in envs:
        env.close()


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")


@cli.command()
def deets():
    env, state_space_dim, action_space_dim, state_norm_array, min_action, \
        max_action = setup_env(var=0, vis=False)
    print(len(env._get_state()))
    print('')
    print('--------------------')
    print('ENV_NAME: \t', env.name)
    print('STATE_DIMS: \t', state_space_dim)
    print('ACTION_DIMS: \t', action_space_dim)
    print('MIN_ACTION \t', min_action)
    print('MAX_ACTION \t', max_action)
    print('')


@cli.command()
def cpu_test():
    with RedirectAllOutput(sys.stdout, file=os.devnull), \
            RedirectAllOutput(sys.stderr, file=os.devnull):
        batch_data = [(100, 1), (50, 2), (25, 4), (20, 5), (10, 10), (5, 20),
                      (2, 50), (1, 100)]
        output = []
        for batch_size, batches in batch_data:

            start = time()
            test_cpu_speeds([[0 for i in range(batch_size)]
                             for _ in range(batches)])
            end = time()
            t = end - start
            s = f'BATCH_SIZE={batch_size}, BATCHES={batches}, time: {t}'
            output.append(s)

        for o in output:
            print(o)


@cli.command()
@click.pass_context
@click.option('--dir', '-nt', default='default', help='Save file location')
@click.option('--all', '-a', is_flag=True, help='Delete all files')
def clean(ctx, dir, all):
    for dir in os.listdir('data'):
        shutil.rmtree(f'./data/{dir}/')


if __name__ == '__main__':
    cli()
