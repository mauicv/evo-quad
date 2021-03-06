import os
import numpy as np
import time
import sys
# import pybullet_envs  # noqa
from src.environment.walking_env import WalkingEnv

import datetime

from gerel.algorithms.RES.population import RESPopulation
from gerel.algorithms.RES.mutator import RESMutator
from gerel.populations.genome_seeders import curry_genome_seeder
from gerel.genome.factories import dense, from_genes
from gerel.util.datastore import DataStore
from gerel.model.model import Model

from src.training.batch import BatchJob
from src.training.stream_redirect import RedirectAllOutput
from src.training.mappings import action_map
from src.params import ENV_NAME, STEPS, LAYER_DIMS, EPISODES, STATE_DIMS, \
    ACTION_DIMS, BATCH_SIZE, POPULATION_SIZE, STD_DEV, ALPHA, WEIGHT_LOW, \
    WEIGHT_HIGH, INPUT_SCALING_VAL

batch_job = BatchJob()


@batch_job
def compute_fitness(genomes):
    with RedirectAllOutput(sys.stdout, file=os.devnull), \
            RedirectAllOutput(sys.stderr, file=os.devnull):
        envs = [WalkingEnv(ENV_NAME, var=0, vis=False)
                for _ in range(len(genomes))]
        models = [Model(genome) for genome in genomes]
        dones = [False for _ in range(len(genomes))]
        states = [np.array(env.current_state, dtype='float32') for env in envs]
        rewards = [0 for _ in range(len(genomes))]
        for _ in range(STEPS):
            for index, (model, env, done, state) in \
                    enumerate(zip(models, envs, dones, states)):
                if done:
                    continue
                action = np.array(model(state))/INPUT_SCALING_VAL
                action = action_map(action)
                env.take_action(action)
                env.step()
                next_state, reward, done, _ = env.get_state()
                rewards[index] += reward
                dones[index] = done
                states[index] = next_state

        # Closing envs fixes memory leak:
        for env_i, env in enumerate(envs):
            # rewards[env_i] = env.current_state[-1]
            env.close()
    return rewards


def make_counter_fn(lim=5):
    def counter_fn():
        if not hasattr(counter_fn, 'count'):
            counter_fn.count = 0
        counter_fn.count += 1
        if counter_fn.count == lim:
            counter_fn.count = 0
            return True
        return False
    return counter_fn


def partition(ls, ls_size):
    parition_num = int(len(ls)/ls_size) + 1
    return [ls[i*ls_size:(i + 1)*ls_size] for i in range(parition_num)]


def departition(ls):
    return [item for sublist in ls for item in sublist]


def print_progress(data):
    data_string = f'{data["run_time"]}: gen: {data["generation"]} '
    for val in ['best_fitness', 'worst_fitness',
                'mean_fitness']:
        data_string += f' {val}: {round(data[val])}'
    data_string += f" ep_time: {data['time']}"
    print(data_string)


def train_walk(dir):
    ds = DataStore(name=f'{dir}')
    generation_inds = [int(i) for i in os.listdir(f'./{dir}')]
    if generation_inds:
        last_gen_ind = max(generation_inds)
        last_gen = ds.load(last_gen_ind)
        nodes, edges = last_gen['best_genome']
        input_num = len([n for n in nodes if n[4] == 'input'])
        output_num = len([n for n in nodes if n[4] == 'output'])
        nodes = [n for n in nodes if n[4] == 'hidden']
        genome = from_genes(
            nodes, edges,
            input_size=input_num,
            output_size=output_num,
            depth=len(LAYER_DIMS))
        next_gen = last_gen_ind + 1
        print(f'seeding generation {next_gen} with last best genome: {genome}')
    else:
        genome = dense(
            input_size=STATE_DIMS,
            output_size=ACTION_DIMS,
            layer_dims=LAYER_DIMS,
            weight_low=WEIGHT_LOW,
            weight_high=WEIGHT_HIGH
        )
        print(f'seeding generation 0, with  genome: {genome}')

    init_mu = np.array(genome.weights)

    mutator = RESMutator(
        initial_mu=init_mu,
        std_dev=STD_DEV,
        alpha=ALPHA
    )

    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[genome]
    )

    population = RESPopulation(
        population_size=POPULATION_SIZE,
        genome_seeder=seeder
    )

    print('population size:', POPULATION_SIZE)
    print('population type: RESPopulation')
    print('STD_DEV:', STD_DEV)
    print('ALPHA:', ALPHA)
    print('mutator type = RESMutator')
    print('num episodes:', EPISODES)
    print('num steps:', STEPS)
    print(f'running algorithm on: {batch_job.num_processes} CPUs')
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    print("Current Time:", current_time)

    init_time = time.time()
    for episode in range(EPISODES):
        start = time.time()
        genes = [g.to_reduced_repr for g in population.genomes]
        partitioned_population = partition(genes, BATCH_SIZE)
        scores = departition(compute_fitness(partitioned_population))
        for genome, fitness in zip(population.genomes, scores):
            genome.fitness = fitness
        data = population.to_dict()
        mutator(population)
        ds.save(data)
        end = time.time()
        episode_time = f'{round(end - start)} secs'
        current_run_time = str(
            datetime.timedelta(seconds=round(end - init_time)))
        print_progress({
            **data,
            'time': episode_time,
            'run_time': current_run_time
        })
