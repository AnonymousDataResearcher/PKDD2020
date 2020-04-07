import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, SpectralClustering

from functions import file_functions
from functions import dataframe_operations
from cjo.weighted_adapted_jaccard.result_computations.bootstrap_result import MultipleBootstrapResult
from cjo.weighted_adapted_jaccard.bootstrap import bootstraphelpers
from cjo.base.stringconstants import IEM

from data import data_loader
from scripts.PKDD_EXPERIMENTS import PKDD_PARAMETERS

# computations of IEM
iem_factory = bootstraphelpers.IEMFactory(bootstraphelpers.IEMFactory.LOG)

# Algorithm parameters
HAC_linkage = ['complete', 'average', 'single']
algorithms = ['Our framework', 'Voronoi', 'Spectral'] + \
             [f'HAC - {linkage}' for linkage in HAC_linkage]


# This is the computation script of the third experiment: Competitor Analysis. It generates 6 files, one for each
# competitor (5) and the value for our framework. Each of the files contains the IEM for each dataset. If the file
# already exists, the script assumes the computation is already done, and hence skips the computation.

def sk_learn_score(clustering_instance, w, dsx, matrix):
    """
    Compute the score of a sklearn clustering algorithm on a dataset.

    Parameters
    ----------
    clustering_instance: Scikit-Learn Clustering Algorithm Object
        An object of the scikit-learn clustering algorithms
    w: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
    dsx: DSX
        The DSX object of the dataset. This is used to create clusters subsets for the IEM computation
    matrix: np.ndarray of size (n_visits, n_visits)
        The distance or similarity matrix (depending on what the clustering algorithm needs)

    Returns
    -------
    IEM: float
        The Internal Evaluation Metric of applying the given algorithm to the given dataset.
    """
    # Verify input
    assert matrix.shape[0] == len(dsx.pure_values)

    # Do the clustering
    clustering_instance.fit(X=matrix)

    # Get the cluster labels
    labels = np.unique(clustering_instance.labels_)

    # Generate DSX objects for the clusters
    clusters = [dsx.get_subset(k=set(dsx.pure_values[clustering_instance.labels_ == label])) for label in labels]

    # Return the IEM
    return iem_factory.create_function(dsx, clusters)(w)


def compute_voronoi(dsx, w):
    """
    Compute the score of a voronoi clustering algorithm on the given dataset
    (sk-learn doesn't do k-medoids)

    Parameters
    ----------
    dsx: DSX
        The DSX object of the dataset. This is used to create clusters subsets for the IEM computation
    w: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.

    Returns
    -------
    IEM: float
        The Internal Evaluation Metric of applying the given algorithm to the given dataset.
    """
    # Load the algorithm
    dsx_c = bootstraphelpers.DSXClustering(bootstraphelpers.DSXClustering.PYCLUSTERING_VORONOI)

    # Compute the clusters
    _, clusters = dsx_c.cluster(dsx, weights=w, initial_medoids=None, k=4)

    # Compute the IEM
    return iem_factory.create_function(dsx, clusters)(w)


def run():
    if not PKDD_PARAMETERS.RESULTS_3.exists():
        PKDD_PARAMETERS.RESULTS_3.mkdir(parents=True)
    results = dict()

    # Check which algorithms need to be evaluated
    for alg in algorithms:
        if (PKDD_PARAMETERS.RESULTS_3 / f'{alg}.csv').exists():
            continue
        else:
            results[alg] = pd.Series(name=IEM, dtype=np.float64)
            results[alg].index.name = 'dataset'
    if len(results) == 0:
        return
    print(f'Executing experiment 3 (Competitor Analysis) for {len(results)} algorithms')

    # Compute IEMs for each dataset
    for fd in file_functions.list_dirs(PKDD_PARAMETERS.RESULTS_BOOTSTRAP, False):
        mbr = MultipleBootstrapResult(PKDD_PARAMETERS.RESULTS_BOOTSTRAP / fd)

        # load dataset
        dsx = data_loader.generic_dsx_loader(mbr.settings['dataset'], mbr.settings['hierarchy'])
        w = dsx.unit_weight_vector()
        pure_distance_matrix = dsx.pure_distance_matrix(w)
        assert np.amax(pure_distance_matrix) <= 1.0

        # Hierarchical clustering
        for linkage in HAC_linkage:
            alg = f'HAC - {linkage}'
            if alg in results:
                a = AgglomerativeClustering(n_clusters=4, affinity='precomputed', linkage=linkage)
                results[alg].loc[fd] = sk_learn_score(a, w, dsx, pure_distance_matrix)

        # Contribution
        if 'Our framework' in results:
            results['Our framework'].loc[fd] = mbr.the_iem

        # Spectral
        if 'Spectral' in results:
            a = SpectralClustering(n_clusters=4, affinity='precomputed', random_state=0)
            results['Spectral'].loc[fd] = sk_learn_score(a, w, dsx, 1 - pure_distance_matrix)

        # k-medoids
        if 'Voronoi' in results:
            results['Voronoi'].loc[fd] = compute_voronoi(dsx, w)

    # Export the results
    for alg, sr in results.items():
        dataframe_operations.export_df(sr, PKDD_PARAMETERS.RESULTS_3 / f'{alg}.csv')


if __name__ == '__main__':
    run()
