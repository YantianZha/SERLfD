# utils.py
# ----------------------------
# some utils for the entire framework

import numpy as np


def manhattanDistance(xy1, xy2):
    """ Returns the Manhattan distance between points xy1 and xy2 """
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


def to_grayscale(rgb):
    """ the rgb should in the format of (height, width, 3) """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def concat_nested_list(nested_list):
    """
    concatenate nested list: [list 1, list 2, ...] -> [single numpy array containing all the elements]
    """
    n_elements = 0
    for sub_list in nested_list:
        n_elements += len(sub_list)

    is_scalar = False
    if isinstance(nested_list[0][0], np.ndarray):
        flat_array = np.zeros(shape=tuple([n_elements] + list(nested_list[0][0].shape)))
    else:
        is_scalar = True
        flat_array = np.zeros(shape=(n_elements, 1))

    i_element = 0
    for sub_list in nested_list:
        for element in sub_list:
            flat_array[i_element, :] = element if is_scalar else element[:]
            i_element += 1

    return flat_array
