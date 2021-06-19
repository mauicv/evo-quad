from gerel.genome.factories import dense
from gerel.model.model import Model

from gerel.util.datastore import DataStore
from gerel.genome.factories import from_genes
import os


from src.params import STATE_DIMS, ACTION_DIMS, LAYER_DIMS, \
    DIR, WEIGHT_LOW, WEIGHT_HIGH


def get_genome():
    return dense(
        input_size=STATE_DIMS,
        output_size=ACTION_DIMS,
        layer_dims=LAYER_DIMS,
        weight_low=WEIGHT_LOW,
        weight_high=WEIGHT_HIGH
    )


def get_model():
    genome = get_genome()
    model = Model(genome.to_reduced_repr)
    return model


def load_genome(generation=None, dir=DIR):
    if not generation:
        generation = max([int(i) for i in os.listdir(dir)])

    ds = DataStore(name=dir)
    data = ds.load(generation)
    nodes, edges = data['best_genome']
    nodes = [n for n in nodes if n[4] == 'hidden']
    return from_genes(
        nodes, edges,
        input_size=STATE_DIMS,
        output_size=ACTION_DIMS,
        depth=len(LAYER_DIMS))


def load_model(generation=None):
    genome = load_genome(generation)
    return Model(genome.to_reduced_repr)
