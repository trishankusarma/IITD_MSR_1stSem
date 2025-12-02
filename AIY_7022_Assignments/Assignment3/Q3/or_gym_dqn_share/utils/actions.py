import itertools
import numpy as np

ACTION_VALUES = [-2, -1, 0, 1, 2]
ALL_ACTIONS = np.array(list(itertools.product(ACTION_VALUES, repeat=5)), dtype=np.int32)

def encode_action_index(idx):
    """Convert index to 5D action vector."""
    return ALL_ACTIONS[idx]

def decode_action_vector(vec):
    """Convert vector to index."""
    vec_tuple = tuple(vec.tolist())
    return np.where((ALL_ACTIONS == vec_tuple).all(axis=1))[0][0]
