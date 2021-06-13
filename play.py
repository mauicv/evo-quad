import click
import time
import numpy as np

from src.environment.walking_env import WalkingEnv
from src.training.mappings import action_map
from src.util import get_genome, load_genome
from src.params import STEPS, ENV_NAME, DIR, INPUT_SCALING_VAL

from gerel.model.model import Model
from gerel.algorithms.RES.mutator import RESMutator


def play(model, steps=STEPS, record=False):
    done = False
    env = WalkingEnv(ENV_NAME, var=0, vis=True, record=record)
    state = env.current_state
    rewards = 0
    i = 0
    while not done and i < steps:
        i += 1
        time.sleep(0.007)
        action = np.array(model(state))/INPUT_SCALING_VAL
        action = action_map(action)
        env.take_action(action)
        env.step()
        next_state, reward, done, _ = env.get_state()
        rewards += reward
        state = next_state

    env.close()
    return rewards


@click.group()
@click.option('--debug/--no-debug', default=False)
def cli(debug):
    click.echo(f"Debug mode is {'on' if debug else 'off'}")


@cli.command()
@click.option('--steps', '-s', default=STEPS, type=int,
              help='Max number of steps per episode')
@click.option('--generation', '-g', default=None, type=int,
              help='Generation to play')
@click.option('--dir', '-d', default=DIR,
              help='working folder')
@click.option('--mutate', '-m', default=0, type=float,
              help='mutation rate')
@click.option('--trial', '-t', is_flag=True,
              help='test random network')
@click.option('--record', '-r', is_flag=True,
              help='record scene')
def best(steps, generation, dir, mutate, trial, record):
    if trial:
        genome = get_genome()
    else:
        genome = load_genome(generation, dir=dir)

    if mutate:
        mutator = RESMutator(
            initial_mu=genome.weights,
            std_dev=mutate,
            alpha=0.5
        )
        mutator(genome)

    model = Model(genome.to_reduced_repr)
    rewards = play(model, steps, record)
    print(f'generation: {generation}, rewards: {rewards}')


if __name__ == '__main__':
    cli()
