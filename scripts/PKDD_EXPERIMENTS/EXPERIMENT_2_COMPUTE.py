from collections import Counter
import pandas as pd

from cjo.base.stringconstants import cat, size
from functions import file_functions
from functions import dataframe_operations
from functions.general_functions import listified
from cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers import DSXClustering
from cjo.weighted_adapted_jaccard.distances.implementation import data_2_dataset_x, DSX
from cjo.weighted_adapted_jaccard.result_computations import cluster_analysis
from cjo.weighted_adapted_jaccard.result_computations.bootstrap_result import MultipleBootstrapResult

from scripts.PKDD_EXPERIMENTS import PKDD_PARAMETERS
from data import data_loader


# This is the computation script of the second experiment: Cluster Analysis. It generates k+1 files, one for the cluster
# contents of each dataset for each medoid (k total), and 1 for the contents of each dataset. The names of the files
# are the abstract representations of the original medoids.

def run():
    # Make results directory
    if not PKDD_PARAMETERS.RESULTS_2.exists():
        PKDD_PARAMETERS.RESULTS_2.mkdir(parents=True)

    # The results of each cluster are gathered in a DataFrame
    cluster_dfs = dict()

    # Get the results of the first dataset
    fp_mbr_folder = PKDD_PARAMETERS.RESULTS_BOOTSTRAP / '[Multi_S3A3_0]'
    fp_mbr = MultipleBootstrapResult(fp_mbr_folder)

    # Read the hierarchy
    h = data_loader.generic_hierarchy(fp_mbr.settings['hierarchy'])

    # Load the dataset in the dedicated DSX class
    master_dsx = data_loader.generic_dsx_loader(fp_mbr.settings['dataset'], fp_mbr.settings['hierarchy'])

    # Load the fixed point: the weights, and the medoids
    fixed_point = fp_mbr.the_fixed_point
    w = master_dsx.validate_and_convert_weights(fixed_point[0])

    # The medoids need to be converted because of the way the computations are done in an abstract space
    medoids_prime = fixed_point[1].values
    medoids = [int(j) for j in master_dsx.original_df_r.loc[medoids_prime][cat].values]

    # For each cluster that we will find; save one DataFrame
    for medoid_prime in list(medoids_prime) + ['All']:
        cluster_dfs[medoid_prime] = pd.DataFrame(
            columns=[size] + list(h.keys()) + sum([list(v) for v in h.values()], []))

    # This is the method that does the clustering
    clustering_algorithm = DSXClustering(DSXClustering.PYCLUSTERING_VORONOI).cluster

    # Analyze all datasets
    for fd in file_functions.list_dirs(PKDD_PARAMETERS.RESULTS_BOOTSTRAP, False):
        # Load the original data, and add the medoids
        i = int(fd.split('_')[-1][:-1])
        bag = data_loader.al_data(3, i).to_bag() + Counter(medoids)
        dsx = data_2_dataset_x(bag, h, sorted(sum([list(v) for v in h.values()], [])))

        # Cluster the data
        _, clustering = clustering_algorithm(dsx, w, initial_medoids=medoids_prime)
        clustering = listified(clustering, DSX)

        # Here you do the check if each separate cluster has its own fixed point medoid
        res = dict()
        for im in medoids_prime:
            memberships = [im in c for c in clustering]
            if sum(memberships) != 1:
                # medoid not in any (should not be possible)
                # medoid in any multiple clusters
                raise Exception(f'something went wrong for dataset {i}')
            res[im] = clustering[memberships.index(True)]
        else:
            # Loop completes normally, so each clustering has a single medoid
            for k, v in res.items():
                # Remove the medoid from the cluster
                v.remove(k)
                # Compute the cluster statistics
                cluster_dfs[k].loc[i, :] = cluster_analysis.single_cluster_2_statistics(v, h)
                cluster_dfs[k].loc[i, size] = v.size_d

        # Compute the statistics of the entire dataset
        dsx.remove(medoids_prime)
        cluster_dfs['All'].loc[i, :] = cluster_analysis.single_cluster_2_statistics(dsx, h)
        cluster_dfs['All'].loc[i, size] = dsx.size_d

    # Export the results, one file per medoid
    for k, v in cluster_dfs.items():
        # Save statistics (dataset x hierarchy)
        dataframe_operations.export_df(v, PKDD_PARAMETERS.RESULTS_2 / f'cluster_{k}.csv', index=True)


if __name__ == '__main__':
    run()
