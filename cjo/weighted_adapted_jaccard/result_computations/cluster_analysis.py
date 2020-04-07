import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cjo.base.stringconstants import cat, supercat, size
from functions import tex_functions
from functions.general_functions import listified
from cjo.weighted_adapted_jaccard.distances.bitops import int2bitvector
from cjo.weighted_adapted_jaccard.distances.implementation import DSX
from cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers import DSXClustering
from cjo_settings.generalsettings import category_abbreviation_dict


def single_cluster_2_statistics(dsx, hierarchy):
    """
    Generates descriptive statistics on the activation of all non-root nodes of the hierarchy in the given dsx

    Parameters
    ----------
    dsx: DSX
        The dataset to compute summary statistics for.
    hierarchy: dict of str to set of str
        The hierarchy to use.

    Returns
    -------
    summary: pd.Series
        Summary statistics on the cluster in terms of fraction of :math:`\\mathcal{D}` with activations in each
        (super-)category.
    """
    # input
    assert isinstance(dsx, DSX)
    assert isinstance(hierarchy, dict)

    cat_list = sorted(sum([list(v) for v in hierarchy.values()], []))
    sc_list = sorted(list(hierarchy.keys()))
    df = dsx.original_df_r

    rp_list = list(dsx.keys())
    sizes = df[size]

    # DataFrame receipt -> activations categories
    df2 = pd.DataFrame(data=[int2bitvector(df.loc[rp, cat], len(cat_list)) for rp in rp_list], columns=cat_list,
                       index=rp_list)
    # DataFrame receipt -> activation super-categories
    df3 = pd.DataFrame(data=[int2bitvector(df.loc[rp, supercat], len(sc_list)) for rp in rp_list], columns=sc_list,
                       index=rp_list)
    # DataFrame receipt -> activation categories and super-categories
    df = pd.concat([df2, df3], axis=1)

    # Multiply transpose by sizes (you get the multiplicity-adjusted activation) and divide by the total number of
    return ((df.T * sizes).T.sum() / (sizes.sum())).sort_values(ascending=False)


def clustering_2_statistics(clusters, hierarchy):
    """
    Generates descriptive statistics on the activation of all non-root nodes of the hierarchy of all clusters.

    Parameters
    ----------
    clusters: iterable of DSX
        The clusters
    hierarchy: dict of str to set of str
        The hierarchy

    Returns
    -------
    df: pd.DataFrame
        clustering statistics DataFrame
    """
    df = pd.DataFrame()
    for m, c in enumerate(clusters):
        df[m + 1] = single_cluster_2_statistics(c, hierarchy)
    return df


def cluster_statistics_2_tex(cluster_statistics, hierarchy, fn_out, df_ci=None, inclusion_missing=True, **kwargs):
    """
    Generates a pdf from clustering activation statistics

    Parameters
    ----------
    cluster_statistics: pd.DataFrame
        hierarchy x clusters DataFrame
    hierarchy: dict of (str) -> (set of str)
        The hierarchy
    fn_out: str or Path
        Output location
    df_ci: pd.DataFrame
        As cluster_statistics, but then a standard deviation value
    inclusion_missing: bool
        If True, all values will be proceeded by True or False, and each values will at least be 50% (i.e. it gives the
        inclusion or missing percentage; whichever is higher). If False, the inclusion percentages are given.
    """
    if df_ci is None:
        f = '.0f'
    else:
        f = '.1f'
    cat_list = sum([list(v) for v in hierarchy.values()], [])
    df = pd.DataFrame()
    abbreviation_dict = dict()
    for cluster_name, stat in cluster_statistics.iteritems():
        s = f'${stat[size]:{f}}'
        if df_ci is not None:
            s += f'\\pm{df_ci.loc[size, cluster_name]:{f}}'
        s += '$'
        df.loc[cluster_name, size] = s
        for sc in hierarchy.keys():

            if inclusion_missing:
                # (adapted) average
                if stat[sc] < 0.5:
                    s = f'\\texttt{{False}} : ${100 - stat[sc] * 100:{f}}'
                else:
                    s = f'\\texttt{{True}} : ${stat[sc] * 100:{f}}'
            else:
                s = f'${stat[sc] * 100:{f}}'

            # std
            if df_ci is not None:
                s += f'\\pm{100 * df_ci.loc[sc, cluster_name]:{f}}'

            # close
            s += '$\\%'
            df.loc[cluster_name, sc] = s

        df.loc[cluster_name, ''] = ''

        for i, (k, v) in enumerate(stat[cat_list].sort_values(ascending=False).head(3).items()):

            # Make improvised multirow cells
            if df_ci is not None:
                df.loc[cluster_name, f'{cat} {i + 1}'] = f'\\texttt{{{category_abbreviation_dict[k]}}}'
                df.loc[cluster_name, f'{cat}|{i + 1}'] = \
                    f'${v * 100:{f}}\\pm{100 * df_ci.loc[k, cluster_name]:{f}}$\\%'

            else:

                df.loc[cluster_name, f'{cat} {i + 1}'] = \
                    f'\\texttt{{{category_abbreviation_dict[k]}}} : ${v * 100:{f}}$\\%'

            # save abbreviation for caption
            abbreviation_dict[k] = category_abbreviation_dict[k]

    df.index.name = 'Cluster'
    df.rename(index=lambda z: f'$K_{{{z}}}$', inplace=True)
    df.reset_index(drop=False, inplace=True)
    df.set_index(['Cluster', size], inplace=True)

    # Remove the columns that are part of the improvised MultiRow cells
    if df_ci is not None:
        df.rename(columns={f'{cat}|{i + 1}': '' for i in range(3)}, inplace=True)

    df = df.T
    abbreviations = sorted(abbreviation_dict.keys(), key=lambda z: abbreviation_dict[z])

    def fix_abb(abb):
        abb = abb.capitalize()
        if abb.endswith(' np'):
            abb = abb[:-2] + 'NP'
        elif abb.endswith(' p'):
            abb = abb[:-1] + 'P'
        return abb

    abbreviations = [fix_abb(c) for c in abbreviations]
    categories = sorted(abbreviation_dict.values())

    caption = 'Descriptive Statistics of each cluster.' \
              ' The abbreviations are ' + \
              ', '.join([f'\\texttt{{{v}}}: {k}' for k, v in zip(abbreviations, categories)]) + '.'

    return tex_functions.df_to_table(df, caption=caption, fn_out=fn_out, escape=False, add_phantom=True,
                                     **kwargs), caption


def make_cluster_and_visualization(master, weights, fn_out, initial_medoids=None):
    """
    Clusters the dataset given the weights, and visualizes the result in given output.

    Parameters
    ----------
    master: DSX
        Dataset to be clustered
    weights: dict of str to Number, iterable of Number
        The weights to be used. See :py:meth:`validate_and_convert_weights()
        <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
    fn_out: str or Path
        Output of the visualization
    initial_medoids: iterable of int or None
        Initial medoids for the clustering method.

    Returns
    -------
    The resulting medoids and clusters

    """
    assert isinstance(master, DSX)
    weights = master.validate_and_convert_weights(weights)

    # Clustering
    medoids, clustering = DSXClustering(DSXClustering.NUMBA_VORONOI).cluster(master, weights,
                                                                             initial_medoids=initial_medoids)

    # Create and sort clusters on size
    d = {m: c for m, c in zip(medoids, clustering)}
    medoids = sorted(medoids, key=lambda m: d[m].size_d)
    clustering = sorted(clustering, key=lambda c: c.size_d)

    # Visualization
    visualize_clustering(clustering, weights, fn_out)

    return medoids, clustering


def visualize_clustering(clustering, weights, fn_out):
    """
    Visualizes a clustering with given weights, and stores it in an output file.

    Parameters
    ----------
    clustering: iterable of DSX
        Clusters, as DSX objects
    weights: dict of str to Number, iterable of Number
        The weights to be used. See :py:meth:`validate_and_convert_weights()
        <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
    fn_out: str or Path
        The output of the visualization.
    """
    clustering = listified(clustering, DSX)

    sorted_points = sum([list(cluster.pure_values) for cluster in clustering], [])
    d_metric = clustering[0].get_pure_distance_metric(weights)
    n = len(sorted_points)
    d_matrix = np.empty((n, n))
    for i in range(len(sorted_points)):
        d_matrix[i, i] = 0
        for j in range(i, len(sorted_points)):
            d_matrix[i, j] = d_metric(sorted_points[i], sorted_points[j])
            d_matrix[j, i] = d_matrix[i, j]

    cluster_sizes = [c.size_d for c in clustering]

    f, ax = plt.subplots(1)
    assert isinstance(ax, plt.Axes)

    # Plot the similarities
    ax.imshow(1 - d_matrix, cmap='Greys')
    ticks = []

    # Plot the edges of the clusters
    lt = -0.5
    rb = -0.5
    for i in cluster_sizes:
        rb += i
        w = 2
        ax.plot([lt] * 2, [lt, rb], 'r', linewidth=w)
        ax.plot([rb] * 2, [lt, rb], 'r', linewidth=w)
        ax.plot([lt, rb], [lt] * 2, 'r', linewidth=w)
        ax.plot([lt, rb], [rb] * 2, 'r', linewidth=w)
        ticks.append((rb + lt) / 2)
        lt += i

    ax.xaxis.tick_top()
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f'$K_{{{i + 1}}}$' for i in range(len(ticks))])
    ax.set_yticklabels([f'$K_{{{i + 1}}}$' for i in range(len(ticks))])
    ax.set_xlim(-0.5, n + 0.5)
    ax.set_ylim(n + 0.5, -0.5)
    ax.tick_params(axis='both', which='both', length=0)

    # Save the figure
    f.set_size_inches(5, 5)
    plt.savefig(fn_out, bbox_inches='tight')
    plt.close(f)
