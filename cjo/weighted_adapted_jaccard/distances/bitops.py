from numba import njit

from functions.general_functions import listified
import numpy as np


def int2bitvector(i, c):
    """
    Get the bitvector representation of an integer.

    Parameters
    ----------
    i: int
        The integer to vectorize
    c: int or None
        The length of the resulting vector. If the bit length of :math:`i` is larger than c, c is ignored. If None, the
        resulting length will be such that there are no leading 0's.

    Returns
    -------
    vec: list
        The vector value of the integer

    """
    if c is None:
        return [int(j) for j in f'{i:b}']
    else:
        return [int(j) for j in f'{i:0{c}b}']


# @njit
# def hamming_weight(x):
#     count = 0
#     x = np.int64(x)
#     while x:
#         count += 1
#         x = np.bitwise_and(x, x-1)
#     return count


@njit
def hamming_weight(x):
    """
    Computes the hamming weight of x.
    Implementation as per https://expobrain.net/2013/07/29/hamming-weights-python-implementation/

    Parameters
    ----------
    x: int
        The value for which to compute the Hamming weight.

    Returns
    -------
        The Hamming weight of x.

    """

    x = np.uint64(x)
    x -= np.bitwise_and(np.right_shift(np.uint64(x), np.uint64(1)), np.uint64(0x5555555555555555))

    x = np.bitwise_and(x, np.uint64(0x3333333333333333)) + \
        np.bitwise_and(np.right_shift(x, np.uint64(2)), np.uint64(0x3333333333333333))

    x += np.right_shift(x, np.uint64(4))
    x = np.bitwise_and(x, np.uint64(0x0f0f0f0f0f0f0f0f))

    # the following line will actually produce an overflow error, but this is fine. The reason is that you *want* to
    # truncate this to 64 bits anyway
    x *= np.uint64(0x0101010101010101)

    # The following line is used in the hamming_weight_old implementation, because python integers are not truncated
    # over 64 bits. It is not needed here, because of the truncation that numpy does apply.
    # x = np.bitwise_and(x, np.uint64(0xffffffffffffffff))

    x = np.right_shift(x, np.uint64(56))
    return x

def bitvector2int(bitvector):
    """
    Computes the integer representation of a bit vector. The inverse of int2bitvector

    Parameters
    ----------
    bitvector: iterable(int)
        The bit vector

    Returns
    -------
    i: int
        The integer representation of the bit vector

    Raises
    ------
    AssertionError:
        If any of the values in bitvector is not an int in [0, 1]
    """
    bitvector = listified(bitvector, int, validation=lambda x: x in [0, 1])
    return int(''.join([str(i) for i in bitvector]), 2)


def subset2bitvector(subset, superset_list):
    """
    Computes the bitvector representation of a subset.

    Parameters
    ----------
    subset: iterable
        The subset.
    superset_list: list
        The superset.

    Returns
    -------
    v: list(int)
        The bitvector representation of the subset based on the superset.
    """
    assert isinstance(superset_list, list)
    return [(1 if s in subset else 0) for s in superset_list]


def subset2int(subset, superset_list):
    """
    Computes the integer representation of a subset.

    Parameters
    ----------
    subset: iterable
        The subset.
    superset_list: list
        The superset.

    Returns
    -------
    i: int
        The integer representation of the subset based on the superset.

    Raises
    ------
    AssertionError
        If the list is not sorted.
    """
    assert sorted(superset_list) == superset_list, 'The superset list should be sorted. ' \
                                                   'You might have forgotten this somewhere else too'
    return bitvector2int(subset2bitvector(subset, superset_list))


def ints2bitvectors(ints):
    """
    Converts multiple integers to same-length bitvectors

    Parameters
    ----------
    ints: iterable of int
        The integers

    Returns
    -------
    bit_vectors: iterable of list of ints
        The bit_vector representation of the integers in ints, all with the same length.
    """

    ints = listified(ints, int)
    bit_length = len(f'{max(ints):b}')
    return (int2bitvector(i, bit_length) for i in ints)


def super_category_projection(ci, a):
    """
    Project a receipt onto a super-category.

    Parameters
    ----------
    ci: int
        Super category integer representation
    a: int
        Receipt integer representation

    Returns
    -------
    The integer representation of the super-category projection.

    """
    ci_vec, a_vec = ints2bitvectors([ci, a])
    vec_y = np.array(a_vec)[np.array(ci_vec) == 1]
    vec_y = [int(i) for i in vec_y]
    return bitvector2int(vec_y)
