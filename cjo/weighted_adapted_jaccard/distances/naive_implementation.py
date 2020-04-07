import itertools
import pandas as pd
from cjo.base import classes
from cjo.base.stringconstants import ISC_WEIGHT
from functions.general_functions import assert_valid_partition


def compute_distances(dataset, hierarchy, weights):
    """
    Computes the distances of a dataset given the hierarchy, weights and multiplicity, in a naive way. The method was
    created to validate the implementation of all optimizations.

    Parameters
    ----------
    dataset: pd.DataFrame, classes.DataManager, or classes.ActivationLog
        Source dataset. The code is on a DataFrame, but it will retrieve it from other sources.
        (i.e. :math:`\mathcal{R}`)
    hierarchy: dict(String -> set(String))
        The hierarchy; listing the super-categories with their categories.
        (i.e. :math:`\mathcal{C}_1, ... \mathcal{C}_h`)
    weights: dict(String -> number)
        The weights; one for each super-category, and one for :math:`c_0`. (i.e. :math:`\\vec{w}`)

    Returns
    -------
    distances: dict((String, String) -> float)
        The distance between each two representations, corrected for multiplicity. (i.e. each element has key
        :math:`(a,b)` and value :math:`\delta_{\\vec{w}}(a,b)`)

    Raises
    ------
    AssertionError
        If dataset and multiplicity have different keys.
        If base.stringconstants.ISC_WEIGHTS is in the hierarchy.
        If the weights have different keys that (the keys of the hierarchy, and base.stringconstants.ISC_WEIGHT)

    """
    if isinstance(dataset, classes.DataManager):
        dataset = dataset.al
    if isinstance(dataset, classes.ActivationLog):
        dataset = dataset.df
    assert isinstance(dataset, pd.DataFrame)

    assert isinstance(hierarchy, dict)
    assert_valid_partition(dataset.columns, hierarchy.values())
    hierarchy = {k: list(v) for k, v in hierarchy.items()}
    assert ISC_WEIGHT not in hierarchy

    assert {*hierarchy.keys(), ISC_WEIGHT} == set(weights.keys())

    def si(r, ci):
        """
        The value of :math:`\\vec{s}(\\vec{r})_i`

        Parameters
        ----------
        r: pd.Series
            The vector :math:`\\vec{r}`
        ci: list of Str
            The set :math:`\mathcal{C}_i`

        Returns
        -------
        sr_i: int
            The value of :math:`\\vec{s}(\\vec{r})_i`

        """
        assert isinstance(r, pd.Series)
        return 1 if (max(r[ci]) > 0) else 0

    def distance(ra, rb):
        """
        The distance between receipts a and b. (i.e. :math:`\delta_{\\vec{w}}(a,b)`

        Parameters
        ----------
        ra: pd.Series
            The first receipt (i.e. :math:`a`)
        rb: pd.Series
            The other receipt (i.e. :math:`b`)

        Returns
        -------
        d: the adapted weighted jaccard distance :math:`\delta_{\\vec{w}}(a,b)` between :math:`a` and :math:`b`.

        """
        assert isinstance(ra, pd.Series)
        assert isinstance(rb, pd.Series)
        numerator_inter_scd = 0
        denominator_inter_scd = 0
        numerator_distance = 0
        denominator_distance = 0
        for h, ci in hierarchy.items():
            v_and = si(ra, ci) * si(rb, ci)
            v_or = si(ra, ci) + si(rb, ci) - si(ra, ci) * si(rb, ci)
            numerator_inter_scd += v_and
            denominator_inter_scd += v_or

            if v_or > 1:
                raise Exception('v_or > 1!')

            if v_or == 1:
                d_ci = 1 - (sum(ra[ci] * rb[ci]) / sum(ra[ci] + rb[ci] - ra[ci] * rb[ci]))
                numerator_distance += weights[h] * d_ci
                denominator_distance += weights[h]
        numerator_distance += weights[ISC_WEIGHT] * (1 - numerator_inter_scd / denominator_inter_scd)
        denominator_distance += weights[ISC_WEIGHT]
        return numerator_distance / denominator_distance

    distances = dict()
    for (ia, za), (ib, zb) in itertools.product(dataset.iterrows(), dataset.iterrows()):
        distances[(ia, ib)] = distance(za, zb)
    return distances
