from pathlib import Path

import numpy as np
import pandas as pd

from cjo.base.stringconstants import REPETITION
from functions import file_functions
from functions import dataframe_operations
from functions.general_functions import get_time_str
from cjo.weighted_adapted_jaccard.bootstrap.single_bootstrap import Bootstrap
from cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers import IEMFactory, DSXWeightsOptimizer, DSXClustering
from cjo.weighted_adapted_jaccard.distances.implementation import DSX


def run_bootstrap(dsx,
                  dataset,
                  hierarchy,
                  weight_vectors=None,
                  iem_mode=IEMFactory.LOG,
                  optimization_mode=DSXWeightsOptimizer.DIFFERENTIAL_EVOLUTION,
                  cluster_mode=DSXClustering.PYCLUSTERING_VORONOI,
                  optimization_kwargs=None,
                  experiment_name=None,
                  weight_seed=0,
                  fd=None,
                  save_start_time=True,
                  **kwargs):
    """
    Runs bootstrap for a given DSX, hierarchy, optimization strategy, iem, and initial weights. Results are saved in
    ./results/bootstrapping/x, where x is either the current time or 'experiment_name', or both. The resulting folder
    contains a file with the settings for this experiment, and a folder which contain all individual repetitions.

    Parameters
    ----------
    dsx: DSX
        Dataset to be used
        # TODO this can be loaded from the next two
    dataset: str
        Name of the dataset file
    hierarchy: str
        Name of the hierarchy file
    weight_vectors: int, dict from str to np.ndarray, or None
        Initial weights. If int, this many random weight vectors are generated. If dict, these weight vectors are used.
        If None, each super-category is initiated with prominent weight, as well as one initialization without prominent
        weights; generating a total of :math:`h+2` weight vectors.

    iem_mode: str
        The IEM mode name from :py:class:`IEMFactory
        <cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers.IEMFactory>`
    optimization_mode: str
        The optimization strategy for :py:class:`DSXWeightsOptimizer
        <cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers.DSXWeightsOptimizer>`
    cluster_mode: str
        The clustering implementation to use for :py:class:`DSXClustering
        <cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers.DSXClustering>`
    optimization_kwargs: None or dict
        Key-worded arguments for the :py:class:`DSXWeightsOptimizer
        <cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers.DSXWeightsOptimizer>`

    experiment_name: None or str
        Optional nickname for this run. If None, save_start_time must be True
    weight_seed: int
        seed for initializing weights

    fd: str or Path or None
        Location where to save the root folder of this run. If None, it will be saved in ./results
    save_start_time: bool
         Whether to include the start time in the root folder name. If False, nickname must be given.

    Other Parameters
    ----------------
    n_max: int (default = 200)
        Maximum number of iterations
    epsilon: Number (default = 0.01)
        Maximum difference between successive iterations to stop a repetition
    randomization: float (default = 0.001)
        The randomization to add each round. Before finding the new clusters, the new weight vector is normalized,
        and each weight is increased with a random value in :math:`[0,randomization)`. Must be non-negative.
    k: int (default = 5)
        Number of clusters to find

    """

    #################
    # VERIFICATIONS #
    #################

    assert isinstance(dsx, DSX)

    # Set the weights
    n = len(dsx.hierarchy_names)
    if weight_vectors is None:
        weight_vectors = dict()
        weight_vectors['equal'] = [1] * n
        for i, h in enumerate(dsx.hierarchy_names):
            weight_vectors[h] = [1] * i + [1000] + [1] * (n - 1 - i)
    elif isinstance(weight_vectors, int):
        np.random.seed(weight_seed)
        weight_vectors = {f'random_{k}': np.random.rand(n) for k in range(weight_vectors)}
    assert isinstance(weight_vectors, dict)

    ######################
    # OPTIMIZER_CREATION #
    ######################
    assert isinstance(optimization_mode, str)
    if optimization_kwargs is None:
        optimization_kwargs = dict()
    optimizer = DSXWeightsOptimizer(w_max=2000, h=dsx.h, strategy=optimization_mode, **optimization_kwargs)

    ###############
    # IEM FACTORY #
    ###############
    assert isinstance(iem_mode, str)
    iem_factory = IEMFactory(iem_mode)

    ########################
    # CLUSTERING ALGORITHM #
    ########################
    assert isinstance(cluster_mode, str)
    clustering_algorithm = DSXClustering(cluster_mode)

    ######################
    # GLOBAL SAVE FOLDER #
    ######################
    assert save_start_time or (experiment_name is not None)

    if fd is None:
        fd = Path('.') / 'results' / 'bootstrap'
    else:
        fd = Path(fd)

    folder_name = ''
    if save_start_time:
        folder_name += get_time_str()
    if experiment_name is not None:
        folder_name += f'[{experiment_name}]'

    fd /= folder_name

    fd.mkdir(parents=True, exist_ok=True)

    ###############
    # FILL KWARGS #
    ###############
    kwargs.setdefault('epsilon', 0.01)
    kwargs.setdefault('randomization', 0.001)
    kwargs.setdefault('n_max', 200)
    kwargs.setdefault('k', 5)

    ###################
    # SAVE PARAMETERS #
    ###################
    with open(fd / 'settings.txt', 'w+') as wf:
        wf.write(f'dsx_size_r;{dsx.size_r}\n')
        wf.write(f'dsx_size_d;{dsx.size_d}\n')
        wf.write(f'optimization_strategy;{optimization_mode}\n')
        wf.write(f'iem_mode;{iem_mode}\n')
        wf.write(f'clustering_mode;{cluster_mode}\n')
        wf.write(f'epsilon;{kwargs["epsilon"]}\n')
        wf.write(f'n_max;{kwargs["n_max"]}\n')
        wf.write(f'randomization;{kwargs["randomization"]}\n')
        wf.write(f'k;{kwargs["k"]}\n')
        for k, v in optimization_kwargs.items():
            wf.write(f'__os_{k};{v}\n')

        wf.write(f'dataset;{dataset}\n')
        wf.write(f'hierarchy;{hierarchy}\n')

    ##################
    # RUN EXPERIMENT #
    ##################

    for nickname, j in weight_vectors.items():
        Bootstrap(dsx=dsx,
                  optimizer=optimizer,
                  iem_factory=iem_factory,
                  clustering_algorithm=clustering_algorithm,
                  initial_weights=j,
                  fd_results=fd / 'single bootstraps' / nickname,
                  **kwargs).process()
    compress_mbr(fd)
    return fd


def compress_mbr(mbr_fd):
    mbr_fd = Path(mbr_fd)
    sbr_folder = mbr_fd / 'single bootstraps'

    cycles = pd.Series(dtype=int)
    results = pd.DataFrame()

    for sbr_fd in file_functions.list_dirs(sbr_folder, False):
        fd = mbr_fd / 'single bootstraps' / sbr_fd

        with open(fd / 'cycle.txt', 'r') as rf:
            cycles[sbr_fd] = int(rf.readline())

        df_results = dataframe_operations.import_df(fd / 'results.csv', dtype=str)
        df_results[REPETITION] = sbr_fd
        results = results.append(df_results)

    dataframe_operations.export_df(cycles, mbr_fd / 'cycles.csv')
    dataframe_operations.export_df(results, mbr_fd / 'results.csv')

    file_functions.delete(sbr_folder)
