"""
This is my 3rd attempt at handling objects in the project - part B
"""
import numpy as np

from cjo.base.stringconstants import hl_name, hl_encoding, sku
from cjo.base.verification import assert_hierarchy_has_correct_column_names
from functions import dataframe_operations


def get_n_levels(hierarchy):
    df = dataframe_operations.import_df(hierarchy)
    return len(df.columns) - (1 if sku in hierarchy.columns else 0)


def encode_hierarchy(source):
    df = dataframe_operations.import_df(source)
    assert_hierarchy_has_correct_column_names(df)
    num_levels = get_n_levels(df)

    def create_encoding(f, k):
        df.loc[:, hl_encoding(k)] = df[hl_name(k)].map(
            {n: f'{j:0{int(f)}d}' for j, n in enumerate(df[hl_name(k)].drop_duplicates().sort_values())})

    # Lower levels
    for i in range(1, num_levels + 1):
        # Note that children are not necessarily sequential, as each level is encoded separately.
        create_encoding(f=np.floor(np.log10(len(df[hl_name(i)].unique()))) + 1, k=i)
        if i != 1:
            df.loc[:, hl_encoding(i)] = df[hl_encoding(i - 1)] + '.' + df[hl_encoding(i)]

    return df
