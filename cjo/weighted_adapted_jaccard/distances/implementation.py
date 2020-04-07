from collections import Counter
from collections.abc import Iterable
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit
from cjo.base import classes
from cjo.base.stringconstants import supercat, cat, size, ISC_WEIGHT
from functions.general_functions import listified, assert_valid_partition
from cjo.weighted_adapted_jaccard.distances import bitops
from cjo.weighted_adapted_jaccard.distances.bitops import hamming_weight


def data_2_dataset_x(dataset, hierarchy, cat_list):
    """
    Wrapper to adapt the dataset and the hierarchy to the new formulations of Subsection \\ref{sec:compopt:ws2w}

    Parameters
    ----------
    dataset: Counter, classes.ActivationLog, classes.DataManager, str, or Path.
        the value of :math:`\mathcal{D'}`. If ActivationLog, str or Path, it is (imported and) transformed into a bag
    hierarchy: dict(str -> set(str))
        the integer value of each of the actual super-categories
    cat_list: iterable(str)
        categories that are in the dataset (for verification)

    Returns
    -------
    dataset: DSX
        D initiated with the adaption presented in Subsection \\ref{sec:compopt:ws2w}

        - :math:`r'`, :math:`\mathcal{R}'`
        - :math:`c_0` added to the hierarchy
        - hierarchy converted to two separate vectors (one with the integer representations, and one with the names),
        - first value of vectors represents :math:`c0`, and the others are sorted alphabetically
    """
    cat_list = listified(cat_list, str)
    assert sorted(cat_list) == cat_list
    num_cat = len(cat_list)

    if isinstance(dataset, classes.DataManager):
        assert list(dataset.al) == cat_list
        dataset = dataset.bag

    if isinstance(dataset, str) or isinstance(dataset, Path):
        dataset = classes.ActivationLog(dataset)

    if isinstance(dataset, classes.ActivationLog):
        assert list(dataset.columns) == cat_list
        dataset = dataset.to_bag()

    assert isinstance(dataset, Counter)

    assert_valid_partition(set(cat_list), hierarchy.values())
    hierarchy_names = list(sorted(hierarchy.keys()))
    h_vector = [bitops.subset2int(hierarchy[k], cat_list) for k in hierarchy_names]
    h = len(hierarchy)

    def transform(r):
        """
        Generates the value of :math:`r'` from `r`

        Parameters
        ----------
        r : int
            The original receipt

        Returns
        -------
        r' : int
            The adapted receipt :math:`r'`, as discussed in Subsection \\ref{sec:compopt:ws2w}

        """

        return r + (int(''.join([str(1 if (hamming_weight(ci & r) > 0) else 0) for ci in h_vector]), 2) << num_cat)

    dataset = Counter({transform(r): v for r, v in dataset.items()})
    hierarchy_vector_prime = [((2 ** h - 1) << num_cat)] + h_vector
    hierarchy_names_prime = [ISC_WEIGHT] + hierarchy_names

    return DSX(dataset, hierarchy_vector_prime, hierarchy_names_prime, num_cat)


@njit
def precompute_stuff(h, r_values, num_cat, hierarchy_vector, dr_values):
    """
    The actual precomputation of the :math:`\delta_{c_i}(X^2|_z)` values. The return value is the
    :math:`\delta_c(X)` matrix of equation \\ref{eq:distance:dwx:tensor}. Each value of this matrix computed using
    equation \\ref{eq:distance:dcix2z:bitp}. As a result, this method also implements \\ref{eq:distance:dciab:bit}.

    Parameters
    ----------
    h : int
        The number of actual super-categories (not accounting for the special super-category)
    r_values : iterable of int
        The value of :math:`\mathcal{R'}`
    num_cat : int
        The number of actual categories
    hierarchy_vector : iterable of int
        The values of :math:`c_0, c_1, ..., c_h`
    dr_values : iterable of int
        The value of :math:`\mathcal{D'}`, ordered in the same way as r_values

    Returns
    -------
    d_c_x_matrix : np.array of size :math:`2^h` by :math:`(h+1)`
        The matrix :math:`\delta(X)`

    """
    d_c_x_matrix = np.zeros((2 ** h, h + 1))
    for ia in range(len(r_values)):
        for ib in range(ia + 1, len(r_values)):
            ap = r_values[ia]
            bp = r_values[ib]
            z = (ap >> num_cat) | (bp >> num_cat)
            for i in range(len(hierarchy_vector)):
                d = hamming_weight(hierarchy_vector[i] & (ap | bp))
                aj = (0 if d == 0 else (1 - hamming_weight(hierarchy_vector[i] & (ap & bp)) / d))
                d_c_x_matrix[z, i] += dr_values[ia] * dr_values[ib] * aj
    return d_c_x_matrix * 2


class DSX:
    def __init__(self, dataset, hierarchy_vector_prime, hierarchy_names_prime, num_cat):
        """
        This class tracks the dataset, hierarchy, and num_cat, such that all assertions are done once. It is meant to
        represent :math:`X`, though we track the hierarchy in order allow computations of :math:`s(r)`, and as such of
        :math:`X^2|_z`. The given input is based on :math:`r' \in R'`, as discussed in Subsection
        \\ref{sec:compopt:ws2w}.

        Parameters
        ----------
        dataset : Counter
            The frequency of each representation (the value of :math:`\mathcal{D'}`)
        hierarchy_vector_prime : list of int
            The hierarchy in its binary representation (the values of :math:`\mathcal{c_i}`)
        hierarchy_names_prime : list of String
            The names of each of super-categories, in the same order as hierarchy_vector_prime. The first value
            must be base.stringconstants.ISC_WEIGHT.
        num_cat : int
            The number of categories for this data (the value of :math:`|\mathcal{C}|`)

        Raises
        ------
        AssertionError
            If the largest original receipt in the dataset is not smaller than :math:`2^{|\mathcal{C}|}`.

            If the smallest original receipt in the dataset is not larger than 0.

            If the sum of hierarchy_vector_prime[1:] is not equal to :math:`2^{|\mathcal{C}|} - 1` or if any of the
            values of the hierarchy_vector_prime is 0. (i.e. if the hierarchy is not a partition over the categories).

            If the first value of the hierarchy_vector_prime is not equal to
            :math:`(2^h) \ll |\mathcal{C}|`.

            If the first value of the hierarchy_names_prime is not equal to :math:`c_0`.
        """

        # INPUT TYPE VERIFICATION #
        assert isinstance(dataset, Counter)
        assert all(isinstance(di, int) for di in dataset.keys())
        hierarchy_vector_prime = np.array(hierarchy_vector_prime, dtype=np.uint64)
        assert isinstance(hierarchy_names_prime, list)
        hierarchy_names_prime = listified(hierarchy_names_prime, str)
        assert isinstance(num_cat, int)

        h = len(hierarchy_names_prime) - 1

        # DATASET VERIFICATION #
        original_receipts = {r_prime & (2 ** num_cat - 1) for r_prime in dataset.keys()}
        # All original datapoints must be a subset of C
        assert max(original_receipts) < 2 ** num_cat, \
            'maximum category encoding is not smaller than 2 ^ number of categories'
        # Datapoints must be larger than 0 (non-empty receipts)
        assert min(original_receipts) > 0, 'Encodings should be bigger than 0'
        self.__dataset = dataset

        # HIERARCHY VERIFICATION #
        # after the first index
        assert sum(hierarchy_vector_prime[1:]) == 2 ** num_cat - 1, \
            'hierarchy should contain all categories at least once (i.e. its sum should be 2 ** num_cat - 1)'
        assert all([h > 0 for h in hierarchy_vector_prime]), \
            'hierarchy cannot contain empty sets (i.e. 0 valued integer representations)'
        # on the first index
        assert hierarchy_vector_prime[0] == (2 ** h - 1) << num_cat, \
            'the first value of the hierarchy vector needs to be' \
            ' (2 ** h - 1) << num_cat'
        assert hierarchy_names_prime[0] == ISC_WEIGHT, f'first super-category should be "{ISC_WEIGHT}"'

        # DATA IS OKAY, STORE IT
        self.__dataset = dataset
        self.hierarchy_vector = hierarchy_vector_prime
        self.__hierarchy_names = hierarchy_names_prime
        self.__num_cat = num_cat
        self.__h = h

        # OPTIMIZE : Precompute only useful z-values, not all
        # You are currently "precomputing" all values of z and z', but this might not be necessary, as some might not
        # have pairs of receipts a', b' such that z = s'(a') | s'(b').

        self.r_values = np.array(list(self.__dataset.keys()), dtype=np.uint64)
        self.dr_values = np.array([self[r] for r in self.r_values], dtype=np.uint64)
        hierarchy_vector = self.hierarchy_vector

        self.__d_ci_x_z_matrix = precompute_stuff(h=self.__h,
                                                  r_values=self.r_values,
                                                  dr_values=self.dr_values,
                                                  num_cat=num_cat,
                                                  hierarchy_vector=hierarchy_vector)
        self.__zp_values = np.arange(2 ** self.__h, 2 ** (self.__h + 1))

    def pure_distance_matrix(self, weights):
        return self.__distance_matrix(weights, False)

    def multiplicity_distance_matrix(self, weights):
        return self.__distance_matrix(weights, True)

    def __distance_matrix(self, weights, multiplicity):
        w = self.validate_and_convert_weights(weights)
        if multiplicity:
            rp_values = self.r_values
            freq_values = self.dr_values
        else:
            rp_values = self.pure_values
            freq_values = np.ones((len(rp_values),))

        return compute_distance_matrix(rp_values=rp_values, freq_values=freq_values,
                                       w=w, hierarchy_vector=self.hierarchy_vector)

    @property
    def original_df_r(self):
        """
        A DataFrame describing the original values of :math:`\\mathcal{R}`.

        Returns
        -------
        original_r: pd.DataFrame
            DataFrame with the values of :math:`r'` as index, and the respective values of :math:`\\mathcal{D}(r')`,
            :math:`r` and :math:`\\vec{s}(r)` as columns.
        """
        df = pd.DataFrame(columns=[size, cat, supercat])
        df.index.name = "r'"
        for k, v in self.__dataset.items():
            df.loc[k, size] = v
            df.loc[k, cat] = k & (2 ** self.__num_cat - 1)
            df.loc[k, supercat] = k >> self.__num_cat
        df = df.astype(int)
        return df

    @property
    def d_ci_x_z_matrix(self):
        return self.__d_ci_x_z_matrix

    def __contains__(self, item):
        return item in self.__dataset

    def __str__(self):
        return f'dsx object with {self.size_r} [{self.size_d}] receipts'

    def __repr__(self):
        return str(self)

    @property
    def hierarchy_names(self):
        return self.__hierarchy_names

    @property
    def size_d(self):
        return sum(self.__dataset.values())

    @property
    def size_r(self):
        return len(self.__dataset)

    def unit_weight_vector(self):
        """
        Generates a weight vector that is valid for this DSX, and where each weight is 1

        Returns
        -------
        weights: np.ndarray
            weights vector for this DSX, where each weight is 1.

        """
        return self.validate_and_convert_weights([1] * (self.__h + 1))

    def get_named_weight_vector(self, weights):
        """
        Converts a weight vector to mapping from this hierarchy to the weights.

        Parameters
        ----------
        weights: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.

        Returns
        -------
        w: dict from str to Number
            Mapping from each of super-category to its weight in the given weight vector.
        """
        weights = self.validate_and_convert_weights(weights)
        return {hi: wi for hi, wi in zip(self.__hierarchy_names, weights)}

    @property
    def h(self):
        return self.__h

    def __getitem__(self, rp):
        """
        Gets the number of visits represented by the receipt :math:`r'`, i.e. the value :math:`\mathcal{D'}(r')`

        Parameters
        ----------
        rp: int
            The receipt :math:`r'` for which to get the number of visits represented by it

        Returns
        -------
        count: int
            The value of :math: `$\dataset'(r')$`
        """
        return self.__dataset[rp]

    def keys(self):
        """
        Gets the receipt values of this dataset; the value of :math:`\mathcal{R}'`

        Returns
        -------
        R: Set of int
            The set :math:`\mathcal{R}'`
        """
        return self.__dataset.keys()

    def get_subset(self, k):
        """
        Generate a DSX object for the given subset :math:`k\subseteq\mathcal{R}'`

        Parameters
        ----------
        k: iterable of Int
            The representations for which to get the new DSX

        Returns
        -------
        x: DSX
            Subset of this dataset
        """
        return DSX(dataset=Counter({r: v for r, v in self.__dataset.items() if r in k}),
                   hierarchy_vector_prime=self.hierarchy_vector,
                   hierarchy_names_prime=self.__hierarchy_names,
                   num_cat=self.__num_cat)

    def assert_partition(self, clustering):
        """
        Asserts that the given clustering is a valid partition over this DSX

        Parameters
        ----------
        clustering: iterable of DSX
            collection of DSX that is the partition

        Raises
        ------
        AssertionError
            If the given collection of DSX is not a valid partition over this DSX

        """
        clustering = listified(clustering, DSX)
        assert_valid_partition(set(self.keys()), [set(dsx_i.keys()) for dsx_i in clustering])

    def get_multiplicity_distance_function(self, weights):
        """
        Generate a distance metric function for the given weights, not correcting for multiplicity. This is useful
        when making comparisons on receipts (i.e. on a basis of :math:`\\mathcal{R}`)

        Parameters
        ----------
        weights: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
        """
        return self.__get_distance_metric(weights, True)

    def get_pure_distance_metric(self, weights):
        """
        Generate a distance metric function for the given weights, not correcting for multiplicity. This is useful
        when making comparisons on two single visits (i.e. on a basis of :math:`\\mathcal{D}`)

        Parameters
        ----------
        weights: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
        """
        return self.__get_distance_metric(weights, False)

    def __get_distance_metric(self, weights, multiplicity):
        w = self.validate_and_convert_weights(weights)

        if multiplicity:
            def inner(ap, bp):
                return distance_function(ap, bp, self.hierarchy_vector, w, self[ap], self[bp])
        else:
            def inner(ap, bp):
                return distance_function(ap, bp, self.hierarchy_vector, w, 1, 1)

        return inner

    def get_sum(self, weights):
        """
        Implementation of Equation \\ref{eq:distance:dwxz:bitp:tensor}.

        Parameters
        ----------
        weights: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.

        Returns
        -------
        dwx: float
            The value of :math:`\delta_{\\vec{w}}(X)`
        """
        w = self.validate_and_convert_weights(weights)

        # Dot product for the numerator vector
        n = np.dot(self.__d_ci_x_z_matrix, w)

        # Bitwise operations for the denominator vector
        # In the text, there is a Z'-matrix, but this is less trouble
        d = np.zeros(len(self.__zp_values))
        for i in range(self.__h + 1):
            d += np.bitwise_and(self.__zp_values >> (self.__h - i), 1) * w[i]

        # If w0 is 0, then d[0] will be 0 as well. By its definition, n[0] will be 0 too, as such the following
        # operation prevents a division by 0, while maintaining correctness
        d[0] += 1

        return np.sum(n / d)

    def remove(self, receipts_to_be_removed):
        if not isinstance(receipts_to_be_removed, Iterable):
            receipts_to_be_removed = [receipts_to_be_removed]
        receipts_to_be_removed = Counter(receipts_to_be_removed)
        for receipt, multiplicity in receipts_to_be_removed.items():
            assert self.__dataset[receipt] >= multiplicity
        self.__dataset -= receipts_to_be_removed

        # TODO removing receipts from the DSX essentially invalidates all functionality.
        # The only functionality that is not affected is the original_df (which is also the function that requires
        # the removal of receipts because of experiment 2). As such, all other values are now None'd, which should
        # break all other functions.

        self.__d_ci_x_z_matrix = None
        self.r_values = None
        self.dr_values = None
        self.__zp_values = None

    def validate_and_convert_weights(self, weights):
        """
        Validates and converts the given weights

        Parameters
        ----------
        weights: dict of str to float, iterable of float
            The weights to be used.
            If dict or pd.Series, missing weights are treated as 0. The value of base.stringconstants.ISC_WEIGHT or all
            others must be non-zero. If iterable, the length must be :math:`h+1`. The first or all other values must be
            non-zero.

        Returns
        -------
        w: np.ndarray
            Vector of weights, validated to match the requirements.

        Raises
        ------
        AssertionError
            If the weights are an iterable of the wrong length. If any of the weights is negative. If the weight of
            base.stringconstants.ISC_WEIGHT (the first index) is not positive and the rest of the weights are
            non-positive.

        """
        # convert series to dict
        if isinstance(weights, pd.Series):
            weights = weights.to_dict()

        # convert dict to vector of appropriate order
        if isinstance(weights, dict):
            w = weights.copy()
            weights = [w.pop(k, 0) for k in self.__hierarchy_names]
            assert len(w) == 0, f'Found unknown weights : {w.keys()}'

        # verify vector
        assert len(weights) == len(self.hierarchy_vector), \
            f'Given weight vector is of wrong length (is {len(weights)}, should be {len(self.hierarchy_vector)}'
        assert all([wi >= 0 for wi in weights]), 'Weights should be non-negative'
        assert (weights[0] > 0 or all([wi > 0 for wi in weights[1:]])), \
            'The first or all other weights should be positive'

        return np.array(weights)

    @property
    def pure_values(self):
        return np.array(sum([[r] * self.__dataset[r] for r in self.keys()], []), dtype=np.uint64)


@njit
def distance_function(ap, bp, hierarchy_vector, w, freq_a, freq_b):
    """
    Computes the distance between ap and bp for a predefined set of weights.

    Parameters
    ----------
    ap: int
        Receipt :math:`a'`
    bp: int
        Receipt :math:`a'`
    hierarchy_vector: list of int
        :math:`c_i` values
    w: list of float
        :math:`w_i` values
    freq_a: int
        multiplicity of receipt :math:`a`
    freq_b: int
        multiplicity of receipt :math:`b`

    Returns
    -------
    d: float
        The distance :math:`\\delta_{\\vec{w}}(a',b')` between :math:`a'` and :math:`b`

    """

    distance_numerator = 0
    distance_denominator = 0

    ap_and_bp = ap & bp
    ap_or_bp = ap | bp

    for ci, wi in zip(hierarchy_vector, w):
        sc_denominator = hamming_weight(ci & ap_or_bp)
        if sc_denominator != 0:
            sc_numerator = hamming_weight(ci & ap_and_bp)
            distance_numerator += wi * (1 - sc_numerator / sc_denominator)
            distance_denominator += wi
    return freq_a * freq_b * distance_numerator / distance_denominator


@njit
def compute_distance_matrix(rp_values, freq_values, w, hierarchy_vector):
    ret = np.zeros((rp_values.shape[0], rp_values.shape[0]))

    for i in range(len(rp_values)):
        for j in range(i + 1, len(rp_values)):
            ret[i, j] = distance_function(ap=rp_values[i], bp=rp_values[j],
                                          hierarchy_vector=hierarchy_vector, w=w,
                                          freq_a=freq_values[i], freq_b=freq_values[j])
            ret[j, i] = ret[i, j]

    return ret
