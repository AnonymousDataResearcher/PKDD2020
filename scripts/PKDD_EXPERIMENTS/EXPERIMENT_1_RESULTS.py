import itertools
import pandas as pd

from cjo.base.stringconstants import bootstrap_rep_count, IEM, ISC_WEIGHT, ISC_WEIGHT_LATEX
from functions import file_functions
from functions import tex_functions, confidence_interval
from cjo.weighted_adapted_jaccard.result_computations.bootstrap_result import MultipleBootstrapResult

from scripts.PKDD_EXPERIMENTS import PKDD_PARAMETERS


# This scripts computes the results for the Weights Learning experiment. It assumes the EXPERIMENT_1_COMPUTE is run

def get_results_for_one_dataset(dataset_results):
    """
    Get the results of a single dataset (of PKDD_PARAMETERS.REPETITION_COUNT repetitions)

    Parameters
    ----------
    dataset_results : str
        The foldername of the results on the datasets

    Returns
    -------
    sc : List of str
        The super-categories used (for verification)
    results : pd.DataFrame
        The results of this single dataset, containing:

            - The weights; converted to a two-digit value, separated by a ';' in a single column, in the order given by
            sc
            - The number of iterations of each repetition
            - The source; which is the value of i
            - The IEM of each repetition


    """

    # Get the results
    fd = PKDD_PARAMETERS.RESULTS_BOOTSTRAP / dataset_results
    mbr = MultipleBootstrapResult(fd)

    # Conversion of a DataFrame in order to combine them later
    def convert(r):
        return ';'.join([f'{r[sc]:.2f}' for sc in mbr.super_categories])

    # Compute the new DataFrames
    df = pd.DataFrame()
    df['weights'] = mbr.summary.apply(convert, axis=1)
    df[bootstrap_rep_count] = mbr.summary[bootstrap_rep_count]
    df['source'] = dataset_results
    df[IEM] = mbr.summary[IEM]

    # Return results
    return mbr.super_categories, df


def run():
    # Make results directory
    if not PKDD_PARAMETERS.RESULTS_1.exists():
        PKDD_PARAMETERS.RESULTS_1.mkdir(parents=True)

    # Get the names of the bootstrap results (each folder == one repetition == iterations until stability)
    repetition_folders_names = file_functions.list_dirs(PKDD_PARAMETERS.RESULTS_BOOTSTRAP, False)

    # Get all results
    super_categories = None
    df = None
    for i in repetition_folders_names:
        # Get the results
        sc, single_df = get_results_for_one_dataset(i)

        # Results on the first dataset
        if super_categories is None:
            super_categories = sc
            df = single_df

        # Merge results of subsequent datasets
        else:
            # Verify that the super_categories are in the same order (to safely combine the results)
            assert super_categories == sc, 'Something went wrong with the super-categories order'
            # Add the results
            df = pd.concat([df, single_df])

    # Fill missing values (some datasets might not have some weight vectors)
    all_sources = df['source'].unique()
    all_weights = df['weights'].unique()
    mi_all = pd.MultiIndex.from_tuples(([(a, b) for a, b in itertools.product(all_sources, all_weights)]),
                                       names=['source', 'weights'])
    df.set_index(keys=['source', 'weights'], inplace=True)
    df = df.reindex(mi_all)
    df[bootstrap_rep_count] = df[bootstrap_rep_count].fillna(0) / PKDD_PARAMETERS.REPETITION_COUNT * 100

    # Final results (these get converted to the final table)
    res = pd.DataFrame()
    n_datasets = len(repetition_folders_names)

    # Compute the mean and 95%CI of the support
    res['mean'] = df.groupby('weights').mean()[bootstrap_rep_count]
    res['std'] = df.groupby('weights').std()[bootstrap_rep_count]
    res['CI95'] = confidence_interval.std_n_to_ci(res['std'], n_datasets, 0.95)
    res['Support'] = res.apply(lambda r: f'${r["mean"]:.1f} \\pm {r["CI95"]:.1f}\\%$', axis=1)

    # Retrieve the weights
    res.index.name = 'weights'
    res.reset_index(drop=False, inplace=True)
    df_weights = pd.DataFrame(res['weights'].str.split(';', expand=True).values, columns=super_categories)
    res = pd.merge(left=res, right=df_weights, left_index=True, right_index=True)

    # Sort on support
    res.sort_values('mean', inplace=True, ascending=False)
    res.drop(columns=['mean', 'std', 'CI95', 'weights'], inplace=True)
    res.rename(columns={ISC_WEIGHT : ISC_WEIGHT_LATEX}, inplace=True)

    # Generate a tex-file
    res.set_index('Support', inplace=True)
    tex_functions.df_to_table(res.head(5).T,
                              caption='The most frequently found weights, showing the number of repetitions that found '
                                     'this combination of weights, averaged over the 100 datasets, with their '
                                     '95\\%-confidence intervals.',
                              label='tab:res:weights',
                              add_phantom=True,
                              fn_out=PKDD_PARAMETERS.RESULTS_1 / 'weights.tex', escape=False, floating='h!')


if __name__ == '__main__':
    run()
