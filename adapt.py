from play import play
from gerel.genome.factories import dense


STATE_DIMS = 51
ACTION_DIMS = 12
LAYER_DIMS = [3, 3]


def test_random(steps):
    genome = dense(
        input_size=STATE_DIMS,
        output_size=ACTION_DIMS,
        layer_dims=LAYER_DIMS
    )
    play(genome.to_reduced_repr, steps)


if __name__ == '__main__':
    test_random(100)
