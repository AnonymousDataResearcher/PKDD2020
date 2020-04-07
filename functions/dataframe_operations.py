from pathlib import Path

import pandas as pd
import locale

"""
Wrapper file to have easier imports/exports for DataFrames
Further contains some functionality for frequent DataFrame Operations
"""


###########################
# Pandas read csv wrapper #
###########################
def import_df(fn, skip_rows=None, dtype=None, na_values=None, use_cols=None, names=None, sep=';', header=0, index=None):
    if isinstance(fn, pd.DataFrame):
        # TODO : also use other parameters?
        return fn
    if na_values is None:
        na_values = []
    df = pd.read_csv(fn,
                     encoding=locale.getpreferredencoding(False),
                     header=header,
                     sep=sep,
                     dtype=dtype,
                     skiprows=skip_rows,
                     na_values=na_values,
                     usecols=use_cols,
                     names=names,
                     index_col=index)
    return df


############################
# Pandas write csv wrapper #
############################
def export_df(df, fn, index=None, header=True):
    assert isinstance(df, pd.DataFrame) or isinstance(df, pd.Series)

    if index is None:
        # default value for series is true, for DataFrame is false
        index = isinstance(df, pd.Series)

    Path(fn).parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path_or_buf=fn,
              sep=';',
              index=index,
              encoding=locale.getpreferredencoding(False),
              header=header
              )


###################################
# convert time difference to days #
###################################
def timedelta2days(sr):
    assert isinstance(sr, pd.Series)
    return sr.apply(lambda x: x.total_seconds() / 3600 / 24)


##################################
# Parse str dates in a DataFrame #
##################################
def fix_dates(frame, date_column_string='datum', str_format=None):
    if date_column_string is None:
        # parse Index
        frame.index = pd.to_datetime(frame.index, infer_datetime_format=True, format=str_format)
        return

    for date in [x for x in frame.columns if date_column_string in x]:
        frame[date] = pd.to_datetime(frame[date], infer_datetime_format=True, format=str_format)


def import_sr(fn, apply=None, **kwargs):
    if isinstance(fn, pd.Series):
        sr = fn
    else:
        sr = import_df(fn, **kwargs)
        sr.set_index(sr.columns[0], inplace=True)
        sr = sr[sr.columns[0]]
    if apply is not None:
        assert callable(apply), 'apply should be callable if not None'
        sr = sr.apply(apply)
    return sr
