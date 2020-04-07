import pandas as pd

from cjo.base.stringconstants import IEM
from functions import file_functions
from functions import tex_functions, dataframe_operations

from scripts.PKDD_EXPERIMENTS import PKDD_PARAMETERS


def run():
    # Setup results
    res = pd.DataFrame(columns=[IEM])
    res.index.name = 'Algorithm'

    # Load all values
    for fn in file_functions.list_files(PKDD_PARAMETERS.RESULTS_3, False):
        if not fn.endswith('.csv'):
            continue
        name = fn[:-4]
        df = dataframe_operations.import_df(PKDD_PARAMETERS.RESULTS_3 / fn)
        res.loc[name, IEM] = f'${df[IEM].mean():.3f}\\pm{df[IEM].std():.3f}$'

    # Put the results in a tex-file
    res.rename(columns={IEM: '$\Phi$'}, inplace=True)
    tex_functions.df_to_table(res.reset_index(), escape=False, index=False,
                              add_phantom=True, phantom_column_position=1,
                              fn_out=PKDD_PARAMETERS.RESULTS_3 / 'competitor_analysis.tex')


if __name__ == '__main__':
    run()
