"""
Useful functions that do not belong to any other file
"""
import datetime


def listified(values, t, validation=lambda x: True, filtering=lambda x: True, sort=False):
    """
    Transform single ``t`` or collection of ``t`` to a List of ``t``, with several extra options

    Parameters
    ----------
    values : t or iterable of t
        The values to listify
    t : Type
        The required type of the values
    validation: Callable ``t`` -> Boolean or iterable, optional
        Each of the values needs to match this condition. Alternatively, if an iterable is given, each of the value must
        be in this iterable.
    filtering: Callable ``t`` -> Boolean, optional
        Values that to not match this condition are removed
    sort
        Whether to sort the resulting list before returning

    Returns
    -------
    values : List of t
        A list of type ``t``, with only the values that match the filtering condition. Sorted if ``sort`` is True.

    Raises
    ------
    AssertionError:
        If a filtered value is not of the correct type
        If a filtered value does not meet the ``validation`` condition

    """

    if isinstance(values, t):
        values = [values]

    values = [v for v in values if filtering(v)]

    def check(v):
        if callable(validation):
            return validation(v)
        else:
            return v in validation

    for v in values:
        assert isinstance(v, t), f'value [{v}] not of type [{t}]'
        assert check(v), f'value [{v}] does not match given validation'

    if sort:
        values.sort()

    return values


def get_time_str(add_ms=False):
    """
    Get current time as string

    Parameters
    ----------
    add_ms: bool
        Add the current ms to the string

    Returns
    -------
    time_str: str
        The current time as string, formatted as "%Y-%m-%d_%H%M%S" or "%Y-%m-%d_%H%M%S.%f"
    """
    f = "%Y-%m-%d_%H%M%S"
    if add_ms:
        f += ".%f"
    return datetime.datetime.now().strftime(f)


def assert_valid_partition(full_set, partition):
    """
    Assert that the given iterable of sets is a partition of the entire set

    Parameters
    ----------
    full_set: set
        The complete set
    partition: iterable of set
        The partition

    Raises
    ------
    AssertionError
        - If the partition contains an empty set
        - The union of the partitions do not recreate the original set
        - There are elements that are in multiple partition sets

    """
    partition = [set(pi) for pi in partition]
    full_set = set(full_set)

    foo = []
    for p in partition:
        # Each set in the partition is non-empty
        assert len(partition) > 0, 'Partition contains an empty set'

        foo.extend(p)

    assert set(foo) == full_set, 'Union of the partition is not the original set'
    assert len(foo) == len(set(foo)), 'Duplicate elements in the partition'


def large_number_formatting(v, decimals=0, __post_fix=''):
    """
    Formats large numbers (1000 -> K), (1e6 -> M), (1e9 -> G).

    Parameters
    ----------
    v : float
        The value
    decimals : int
        number of decimals in the final answer
    __post_fix : str
        what to add after the results (for recursive computations of numbers of at least than 1e12)

    Returns
    -------
    fmt: str
        K/M/G string representation of the large number
    """
    if v < 1e3:
        return f'{v:.{decimals}f}{__post_fix}'
    for border, prefix in zip([1e6, 1e9, 1e12], 'KMG'):
        if v < border:
            return large_number_formatting(v / border * 1e3, decimals, prefix + __post_fix)
    return large_number_formatting(v / 1e12, decimals, 'T' + __post_fix)
