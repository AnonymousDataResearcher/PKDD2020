from pathlib import Path

import numpy as np
import pandas as pd

from cjo.base.classes import RawLog, TransactionLog, UnlabelledVisitLog, UnlabelledBasketLog, features
from cjo.base.stringconstants import visit, value, sku_count, sku
from functions import dataframe_operations
from cjo.base.stringconstants import id_features

global_precision = 0.001


def matches(sr1, sr2):
    """
    Checks if the two given series match within the precision requirements
    Parameters
    ----------
    sr1 : Series
        One of the series to compare
    sr2 : Series
        One of the series to compare

    Returns
    -------
    matches : Series of Boolean
        If series are id_feature or not a numeric feature, True if values are the same, False otherwise
        If series are numeric feature, True if values are within ``global_precision`` from each other,
        False otherwise

    Raises
    ------
    AssertionError
        if index of ``sr1`` does not equal index of ``sr2``
        if name of ``sr1`` is not equal to name of ``sr2
        if name of ``sr1`` (or subsequently ``sr2``) is not an id_feature or a feature
    """
    assert isinstance(sr1, pd.Series), 'sr1 not a Series'
    assert isinstance(sr2, pd.Series), 'sr2 not a Series'
    assert set(sr1.index) == set(sr2.index), 'index does not match'
    assert sr1.name == sr2.name, 'name does not match'
    if sr1.name in id_features + [visit, value, sku, sku_count] or not features.needs_numeric_conversion(sr1.name):
        return sr1 == sr2
    else:
        return np.abs(sr1 - sr2) <= global_precision


def compare(log_true, log_computed):
    assert type(log_true) == type(log_computed)
    assert set(log_true.index) == set(log_computed.index)
    f_true = set(log_true.columns)
    f_computed = set(log_computed.columns)

    f_both = f_true.intersection(f_computed)
    f_not_implemented = f_true.difference(f_computed)
    f_not_tested = f_computed.difference(f_true)

    f_correct = set()
    f_incorrect = set()
    for f in f_both:
        if all(matches(log_true[f], log_computed[f])):
            f_correct.add(f)
        else:
            f_incorrect.add(f)

    if len(f_incorrect) + len(f_not_implemented) + len(f_not_tested) == 0:
        print('All passed')
        return

    print('Correctly implemented : {}'.format(len(f_correct)))
    if len(f_not_implemented) > 0:
        print('Not implemented : {}'.format(f_not_implemented))
    if len(f_not_tested) > 0:
        print('Not tested : {}'.format(f_not_tested))

    for f in f_incorrect:
        print('Incorrect : {}'.format(f))
        print(pd.DataFrame(data={'True': log_true[f], 'Computed': log_computed[f]}).to_string())


if __name__ == '__main__':
    fd_true = Path('./cjo') / 'resources' / 'TestFiles' / 'new_tests'
    rl = RawLog(fd_true / 'RL.csv')

    tl_true = TransactionLog(fd_true / 'TL.csv')
    tl_computed = rl.to_tl()
    print('\nTransaction Log:')
    compare(tl_true, tl_computed)

    uvl_true = UnlabelledVisitLog(fd_true / 'UVL.csv')
    uvl_computed = rl.to_uvl()
    print('\nUnlabelled Visit Log:')
    compare(uvl_true, uvl_computed)

    ubl_true = UnlabelledBasketLog(fd_true / 'UBL.csv')
    sku_map = dataframe_operations.import_sr(fd_true / 'SKU_MAP.csv')
    ubl_computed = tl_computed.to_ubl(sku_map)
    print('\nUnlabelled Basket Log:')
    compare(ubl_true, ubl_computed)
