import pandas as pd
from cjo.base import verification
from cjo.base.stringconstants import cat, hl_encoding, supercat
from functions import dataframe_operations


def hierarchy_from_excel(fn, hierarchy_name, sheet_name='Hierarchies'):
    """
    Reads a hierarchy from excel.

    Parameters
    ----------
    fn: str or Path
        The location of the hierarchy
    hierarchy_name: str
        The name of the hierarchy in the excel
    sheet_name: str, Optional
        The name of the sheet where the hierarchies are

    Returns
    -------
    h: dict(str -> set(str))
        The hierarchy, where the keys are the super-categories, and the values are the categories in them.
    """
    df = pd.read_excel(fn, sheet_name=sheet_name)
    assert cat in df.columns, f'Sheet does not contain {cat} column'
    assert hierarchy_name in df.columns, f'Hierarchy named {hierarchy_name} not found!'
    return __hierarchy_from_sr(df.set_index(cat)[hierarchy_name])


def __hierarchy_from_sr(sr):
    """
    Gets a hierarchy from a pd.Series.

    Parameters
    ----------
    sr: pd.Series
        Series with the hierarchy with Category as index, and the super-categories as values

    Returns
    -------
    h: dict of str to set of str
        The hierarchy, with super-categories as keys and the respective categories set as values.
    """
    assert isinstance(sr, pd.Series)
    return {k: set(sr[sr == k].index) for k in sr.unique()}


def hierarchy_from_csv(fn):
    """
    Gets a hierarchy from a csv.

    Parameters
    ----------
    fn: str or Path
        File with the hierarchy

    Returns
    -------
    h: dict of str to set of str
        The hierarchy, with super-categories as keys and the respective categories set as values.
    """
    return __hierarchy_from_sr(dataframe_operations.import_df(fn).set_index(cat)[supercat])


def create_sku_map_from_labelled_haai_and_sku_data(labelled_haai_file, retailer_sku_file, fn_out):
    """
    Creates a sku_map from Haai results and the retailer sku file.

    Parameters
    ----------
    labelled_haai_file: str or Path
        The location of the labelled haai file
    retailer_sku_file: str or Path
        The location of the retailer sku file
    fn_out: str or Path
        The target location of the merged DataFrame

    """
    df_labels = dataframe_operations.import_df(labelled_haai_file)
    assert set(df_labels.columns) == {hl_encoding(0), cat}

    df_retailer_sku_info = dataframe_operations.import_df(retailer_sku_file)
    verification.assert_is_retail_sku_info(df_retailer_sku_info)

    df = pd.merge(left=df_retailer_sku_info, right=df_labels, left_on=hl_encoding(0), right_on=hl_encoding(0))
    df.drop(columns=hl_encoding(0), inplace=True)
    dataframe_operations.export_df(df, fn_out)
