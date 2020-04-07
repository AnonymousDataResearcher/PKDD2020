"""
This is my 3rd attempt at handling objects in the project
"""

from cjo.base.stringconstants import hl_encoding, hl_name, sku, sku_des
from functions import dataframe_operations


def assert_hierarchy_has_correct_column_names(source, require_sku=True):
    df = dataframe_operations.import_df(source)

    expected_n_levels = len(df.columns) - (1 if require_sku else 0)

    for i in range(1, expected_n_levels + 1):
        assert hl_name(i) in df.columns, f'{hl_name(i)} not in columns.'
    if require_sku:
        assert sku in df.columns, f'{sku} not in columns'


def assert_hierarchy_has_non_duplicate_encoding(source):
    df = dataframe_operations.import_df(source)
    level = 0
    while hl_encoding(level + 1) in df.columns:
        level += 1

    assert level > 0, f'{hl_encoding(1)} not found.'

    if level == 1:
        return

    for i in range(2, level + 1):
        assert len(df[hl_encoding(i)].drop_duplicates()) == \
               len(df[[hl_encoding(i - 1), hl_encoding(i)]].drop_duplicates()), f'Duplicate {hl_encoding(i)}'


def assert_is_retail_sku_info(source):
    """
    Asserts a given source satisfied the requirements of retailer sku info

    Parameters
    ----------
    source: str, Path, or pd.DataFrame
        The source to be checked

    """
    source = dataframe_operations.import_df(source)
    if source.index.name == sku:
        source.reset_index(drop=False)
    assert set(source.columns) == {sku, hl_encoding(0), sku_des}
