import os
import click
import time
import numpy as np

from src.environment.walking_env import WalkingEnv
from src.training.mappings import action_map
from src.environment.group_env import GroupEnv

from gerel.util.datastore import DataStore
from gerel.model.model import Model
from gerel.algorithms.RES.population import RESPopulation
from gerel.genome.factories import from_genes
from gerel.algorithms.RES.mutator import RESMutator
from gerel.populations.genome_seeders import curry_genome_seeder
from gerel.util.activations import build_leaky_relu, build_sigmoid

STATE_DIMS = 51
ACTION_DIMS = 12
MIN_ACTION = -0.785398
MAX_ACTION = 0.785398
LAYER_DIMS = [20, 20]
BATCH_SIZE = 25
DIR = './data/default/'
ENV_NAME = 'walking-quadruped'
STEPS = 1000


def play(genome, steps=STEPS):
    done = False
    sigmoid = build_sigmoid()
    model = Model(genome, activation=sigmoid)
    env = WalkingEnv(ENV_NAME, var=0, vis=True)
    state = env.current_state
    rewards = 0
    i = 0
    while not done and i < STEPS:
        i += 1
        time.sleep(0.007)
        action = np.array(model(state))/6
        action = action_map(action)
        next_state, reward, done, _ = env.step(action)
        rewards += reward
        state = next_state
    return rewards


def play_population(population, steps=STEPS):
    sigmoid = build_sigmoid()
    models = [Model(genome.to_reduced_repr,
                    activation=sigmoid)
              for genome in population.genomes]
    env = GroupEnv(ENV_NAME, vis=True)
    for x in range(-3, 3, 2):
        for y in range(-3, 3, 2):
            env.add_actor([x, y])

    i = 0
    while i < STEPS:
        i += 1
        for j, (state, model) \
                in enumerate(zip(env.get_states(), models)):
            action = model(state)
            action = action_map(action)
            env.step_i(j, action)
        env.client.stepSimulation()


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
def play_best(steps, generation, dir):
    if not generation:
        generation = max([int(i) for i in os.listdir(dir)])

    ds = DataStore(name=dir)
    data = ds.load(generation)
    rewards = play(data['best_genome'], steps)
    print(f'generation: {generation}, rewards: {rewards}')


@cli.command()
@click.option('--steps', '-s', default=STEPS, type=int,
              help='Max number of steps per episode')
@click.option('--generation', '-g', default=None, type=int,
              help='Generation to play')
@click.option('--dir', '-d', default=DIR,
              help='working folder')
def play_gen(steps, generation, dir):
    if not generation:
        generation = max([int(i) for i in os.listdir(dir)])

    ds = DataStore(name=dir)
    data = ds.load(generation)
    nodes, edges = data['best_genome']
    input_num = len([n for n in nodes if n[4] == 'input'])
    output_num = len([n for n in nodes if n[4] == 'output'])
    nodes = [n for n in nodes if n[4] == 'hidden']
    genome = from_genes(
        nodes, edges,
        input_size=input_num,
        output_size=output_num,
        depth=len(LAYER_DIMS))

    init_mu = np.array(genome.weights)

    mutator = RESMutator(
        initial_mu=init_mu,
        std_dev=0.5,
        alpha=1
    )

    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[genome]
    )

    population = RESPopulation(
        population_size=9,
        genome_seeder=seeder
    )

    play_population(population)


if __name__ == '__main__':
    cli()
