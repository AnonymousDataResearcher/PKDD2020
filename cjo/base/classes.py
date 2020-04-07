import abc
from collections import Counter
from pathlib import Path
import pandas as pd

from cjo.base import features
from cjo.base.features import basket_feature, \
    basket_subset_feature
from cjo.base.stringconstants import visit, consumer, timestamp, sku, sku_count, sku_des, \
    value, invoice, week, label, cat, unknown, size, width, avg_value, UXDF_FEATURES, UXDF_TOTAL_PRICE, \
    UXDF_HIGHEST_QUANTITY, UXDF_NUMBER_ITEMS, UXDF_NUMBER_PRODUCTS, UXDF_PERISHABLE_FRACTION, UXDF_MEAL, \
    UXDF_EAT_TODAY, UXDF_REMEMBER, int_representation
from functions import file_functions
from functions import dataframe_operations
from functions.general_functions import listified
from cjo_settings.generalsettings import default_categories_uno, default_categories

dtypes = {consumer: str,
          timestamp: str,
          visit: str,
          invoice: str,
          week: str,
          sku: str,
          sku_count: float,
          value: float,
          basket_feature(value): float,
          basket_feature(size): float,
          basket_feature(avg_value): float,
          basket_feature(width): int,
          label: str,
          sku_des: str,
          UXDF_PERISHABLE_FRACTION: float,
          UXDF_NUMBER_ITEMS: float,
          UXDF_NUMBER_PRODUCTS: int,
          UXDF_HIGHEST_QUANTITY: float,
          UXDF_TOTAL_PRICE: float,
          UXDF_MEAL: bool,
          UXDF_EAT_TODAY: bool,
          UXDF_REMEMBER: bool,
          **{c: int for c in default_categories_uno},
          int_representation: int,
          }

rl_features = [visit, consumer, timestamp, invoice, sku, sku_count, value]
rl_optional_features = []

tl_features = [visit, sku, sku_count, value]

lbl_features = [label]

uvl_features = [visit, consumer, timestamp, invoice]
uvl_optional_features = []


class _LogBase:

    def __init__(self, source, **kwargs):
        """
        Initializes a log from a DataFrame, File, or Log

        Parameters
        ----------
        source: str, Path, pd.DataFrame, or same type
            source of the log. If str or Path, the source is read as DataFrame from this location. If same type, the df
            of the source is used

        Other Parameters
        ----------------
        categories: iterable of str, default = default_categories
            Categories to consider. Used in Activation Log

        """
        if isinstance(source, type(self)):
            self.df = source.df
            return
        elif isinstance(source, str) or isinstance(source, Path):
            assert Path(source).exists()
            source = dataframe_operations.import_df(source, dtype=dtypes)
        assert isinstance(source, pd.DataFrame), 'Source is not a file or DataFrame or same type'

        self.df = pd.DataFrame()
        for f in self.required_features():
            if f != self.id_feature():
                assert f in source.columns, 'Missing required feature {}'.format(f)
                self[f] = source[f].astype(dtypes[f])

        for f in self.optional_features():
            if f in source.columns:
                self[f] = source[f].astype(dtypes[f])

        self.add_additional_features(source, **kwargs)

        if self.id_feature() in source.columns:
            self[self.id_feature()] = source[self.id_feature()].astype(dtypes[self.id_feature()])
            self.df.set_index(keys=self.id_feature(), inplace=True)
        else:
            if self.id_feature() is not None:
                assert self.index.name == self.id_feature(), 'Missing id_feature in columns and index'
                self.df.index = self.index.astype(dtypes[self.id_feature()])

    @abc.abstractmethod
    def required_features(self):
        """
        The list of features that need to be in the log. This is called on construction, and override in subclasses

        Returns
        -------
        required_features: List of str
            List of features that need to be in the log.
        """
        pass

    @abc.abstractmethod
    def optional_features(self):
        """
        The list of features that may be in the log. This is called on construction, and override in subclasses. Any
        of these features in the source are also imported

        Returns
        -------
        optional_features: List of str
            List of features that may be in the log.
        """
        pass

    @abc.abstractmethod
    def id_feature(self):
        """
        The feature that identifies a log entry. Some logs will not have these, but others do.
        Called on construction, overridden in subclasses.

        Returns
        -------
        id_feature: str
            The feature identifying a single log entity
        """
        pass

    def add_additional_features(self, source, **kwargs):
        """
        Adds additional features, that cannot be known up front, from the source. This construction allows the source
        to be loaded once from file
        Parameters
        ----------
        source

        Returns
        -------

        """
        pass

    def assert_unique(self, f, idx=None):
        """
        Assert feature is unique for each index in the DataFrame.

        Parameters
        ----------
        f: str or Iterable(str)
            The feature(s) to be asserted
        idx: str or None
            The column to treat as index. If None, the index of the DataFrame is used.

        """
        f = listified(f, str, validation=lambda x: x in self.df.columns)
        if idx is None:
            len_vid = len(self.df)
            len_f = len(self.df.reset_index(drop=False)[[self.index.name] + f])
        else:
            # On column
            assert idx in self.df.columns, 'given idx not in df.columns'
            len_vid = len(self[idx].drop_duplicates())
            len_f = len(self[[idx] + f].drop_duplicates())

        if len_f != len_vid:
            if len(f) == 1:
                raise ValueError('Duplicate value of {}'.format(f[0]))
            else:
                raise ValueError('Duplicate combinations of {}'.format(*f))

    def assert_unique_index(self):
        """
        Assert the index of the DataFrame is unique.
        """
        assert self.index.is_unique, 'Index of source is not unique'

    @property
    def index(self):
        """
        Forward for DataFrame.index

        Returns
        -------
        index: pd.Index
            The index of the DataFrame
        """
        return self.df.index

    @property
    def columns(self):
        """
        Forward for DataFrame.columns

        Returns
        -------
        columns: pd.Index
            The columns of the DataFrame
        """
        return self.df.columns

    def export(self, fn):
        """
        Export this log to a given filename. Also takes care of index of visits.

        Parameters
        ----------
        fn: str or Path
            The location where the log should be exported to.
        """
        if self.index.name == visit:
            dataframe_operations.export_df(self.df.reset_index(drop=False), fn)
        else:
            dataframe_operations.export_df(self.df, fn)
        return self

    def __getitem__(self, k):
        """
        Forward method to getting items from the DataFrame. Equivalent to self.df[k]

        Parameters
        ----------
        k: Object
            The index to get from the DataFrame

        Returns
        -------
        v: Object
            The value of df[k]

        """
        return self.df[k]

    def __setitem__(self, k, v):
        """
        Forward method for setting items in the DataFrame. Equivalent to self.df[k] = v.

        Parameters
        ----------
        k: Object
            The index to set in the DataFrame
        v: Object
            The value to set
        """
        self.df[k] = v

    def __str__(self):
        """
        str representation of the Log.

        Returns
        -------
        str: str
            str representation of the Log.
        """
        return str(self.df)

    def to_str(self):
        """
        Forward method of to_str function of the DataFrame

        Returns
        -------
        str: str
            to_str value of the DataFrame of the Log.
        """
        return self.df.to_str()

    def subset_visit(self, visits):
        """
        Get a subset of the log, for a given iterable of visit ids

        Parameters
        ----------
        visits: str or Iterable(str)
            Visit ids to get.

        Returns
        -------
        Log: Log
            Log with only the given visits. Missing visits are ignored.
        """
        visits = listified(visits, str)
        if visit in self.columns:
            return type(self)(self.df[self.df[visit].isin(visits)])
        elif self.index.name == visit:
            visits = listified(visits, str, filtering=lambda v: v in self.df.index)
            return type(self)(self.df.loc[visits])

    def iterrows(self):
        """
        Forward method to df.iterrows()

        Yields
        ------
        index : label or tuple of label
            The index of the row. A tuple for a `MultiIndex`.
        data : Series
            The data of the row as a Series.

        it : generator
            A generator that iterates over the rows of the frame.

        """
        return self.df.iterrows()

    def __len__(self):
        """
        The length of this Log.

        Returns
        -------
        len: int
            The length of the log.
        """
        return len(self.df)


class RawLog(_LogBase):
    """
    Raw data log. This is the first step in any studies; and its source will be case-specific.
    """

    def required_features(self):
        return rl_features

    def optional_features(self):
        return rl_optional_features

    def id_feature(self):
        return None

    def __init__(self, source):
        super().__init__(source)

        self.assert_unique(f=[visit, consumer, timestamp], idx=visit)
        for f in self.optional_features():
            if f in self.columns:
                self.assert_unique(f=f, idx=visit)

    def to_tl(self):
        """
        Convert this Raw Log to a Transaction Log.

        Returns
        -------
        tl: TransactionLog
            The Transaction Log that results from this Raw log
        """
        return TransactionLog(self.df[tl_features])

    def to_uvl(self):
        """
        Convert this Raw Log to an Unlabelled Visit Log

        Returns
        -------
        uvl: UnlabelledVisitLog
            The Unlabelled Visit Log that results from this Raw Log
        """
        return UnlabelledVisitLog(
            self.df[uvl_features + [f for f in uvl_optional_features if f in self.columns]].drop_duplicates())


class TransactionLog(_LogBase):
    """
    Transaction data without date/invoice/consumer
    """

    def __init__(self, source, **kwargs):
        super().__init__(source, **kwargs)

    def required_features(self):
        return tl_features

    def optional_features(self):
        return []

    def id_feature(self):
        return None

    def to_ubl(self, sku_map):
        """
        Convert this Transaction Log to an Unlabelled Basket Log

        Parameters
        ----------
        sku_map: pd.DataFrame, str or pd.Series
            Source of the sku_map. If str, it is loaded from this location. If Series or DataFrame, it uses the
            column Category.

        Returns
        -------
        ubl: UnlabelledBasketLog
            The Unlabelled Basket Log that results from this Transaction Log and the given SKU Map.
        """
        # TODO : add discount stuff
        # TODO : what do missing SKU do here exactly?
        # Check sku_map
        sku_map = extract_sku_map(sku_map, cat)

        # Add column with categories
        tl_with_categories = self.df.merge(right=sku_map, left_on=sku, right_index=True, how='left')

        # Result
        ubl = pd.DataFrame()

        # Add to result
        tl_x = tl_with_categories.groupby([visit, cat]).agg({sku_count: sum}).unstack(level=cat, fill_value=0)
        tl_x.columns = tl_x.columns.droplevel()

        # OPTIMIZE : this copying might be crazy expensive, but it seems to be the only way to mutate vl
        # OPTIMIZE maybe let tl_x be the start of the ubl DataFrame?
        for subset_name in tl_x.columns:
            ubl[basket_subset_feature(size, cat, subset_name)] = tl_x[subset_name]

        # Note that reindexing will not do it, since that returns a new objects rather than adapting the existing df
        for subset_feature in {basket_subset_feature(size, cat, c) for c in default_categories_uno}:
            if subset_feature not in ubl.columns:
                ubl[subset_feature] = 0
            del subset_feature

        return UnlabelledBasketLog(ubl)

    def to_ntl(self, sku_map, allow_na=True):
        """
        Convert this Transaction Log to a Named Transaction Log.

        Parameters
        ----------
        sku_map: DataFrame, Series, or str
            Source of the sku_map. If str, it is loaded from this location. If DataFrame, it uses the
            column SKUDes.
        allow_na: Boolean, Optional
            Whether to allow missing SKU codes in the SKU Map (i.e. SKU codes that are in the Transaction Log, but not
            in the SKU Map). If allowed, missing names are replaced with 'unknown'. If not allowed, will raise
            ValueError if missing SKU are encountered.

        Returns
        -------
        ntl: NamedTransactionLog
            The NamedTransactionLog that results from this Transaction Log and the given SKU Map.
        """
        sku_map = extract_sku_map(sku_map, sku_des)

        # Compute ntl
        ntl = NamedTransactionLog(self.df.merge(right=sku_map, how='left', left_on=sku, right_index=True))

        # Check and handle na
        if ntl[sku_des].isna().sum() > 0:
            if allow_na:
                ntl[sku_des] = ntl[sku_des].fillna(unknown)
            else:
                raise ValueError('Some {} do not have a {}'.format(sku, sku_des))
        return ntl


class NamedTransactionLog(TransactionLog):
    """
    Transaction Data with product names.
    """

    def __init__(self, source, **kwargs):
        super().__init__(source, **kwargs)
        self.df.loc[:, sku_des] = self.df.loc[:, sku_des].replace('nan', float('nan'))

    def required_features(self):
        return [sku_des] + [i for i in super().required_features() if i != sku]


class UnlabelledBasketLog(_LogBase):
    """
    Aggregated Basket contents for each visit.
    """

    def __init__(self, source, **kwargs):
        super().__init__(source, **kwargs)
        self.assert_unique_index()

    def add_additional_features(self, source, **kwargs):
        for f in features.get_basket_subset_features(source.columns, agg=[size, avg_value], subset_type=cat):
            self.df[f] = source[f].astype(float)

    def required_features(self):
        return []

    def optional_features(self):
        return []

    def id_feature(self):
        return visit

    def to_lbl(self, labels):
        """
        Create a Labelled Basket Log from this Unlabelled Basket Log given labels.

        Parameters
        ----------
        labels: Series
            The labels.

        Returns
        -------
        lbl: LabelledBasketLog
            The Labelled Basket Log that results from combining this Unlabelled Basket Log with the given labels.

        Raises
        ------
        AssertionError
            If the index of labels is not visit.
            If there are visits in this Log that are not in the labels

        """
        assert isinstance(labels, pd.Series), 'labels should be presented as series'
        assert labels.index.name == visit, f'labels should have index named "{visit}"'
        assert labels.name == label, f'labels should have name "{label}"'
        assert set(self.index).issubset(set(labels.index)), 'labels\' index does have all of self index'

        return LabelledBasketLog(self.df.merge(right=labels, left_index=True, right_index=True))

    def to_activation(self, keep_uno=False):
        """
        Create an Activation Log from this Unlabelled Basket Log.

        Parameters
        ----------
        keep_uno: bool, Optional
            Whether to keep the categories Unknown and Not a product. If not kept, some receipts will become empty;
            these are discarded

        Returns
        -------
        al: ActivationLog
            The Activation Log created from this Unlabelled Basket Log.
        """
        sf = features.get_basket_subset_features(self.columns, size, cat)
        df = self.df[sf].applymap(lambda x: 1 if x > 0 else 0).rename(
            columns={i: features.get_subset_name(i) for i in sf})

        if keep_uno:
            df = df[sorted(default_categories_uno)]
        else:
            df = df[sorted(default_categories)]
            df = df[df.sum(axis=1) > 0]
        return ActivationLog(df)


class ActivationLog(_LogBase):
    """
    Activation of the baskets.
    """

    def __init__(self, source, **kwargs):
        super().__init__(source, **kwargs)

    def required_features(self):
        return []

    def optional_features(self):
        return []

    def add_additional_features(self, source, **kwargs):
        categories = kwargs.get('categories', default_categories)
        for c in categories:
            assert c in source.columns, f'category was not found: {c}'
            assert source[c].isin([0, 1]).all(), f'category has non 0,1 values: {c}'
            self.df.loc[:, c] = source[c]
        self.df = self.df[sorted(categories)]

    def id_feature(self):
        return visit

    def __to_ial_df(self):
        """
        Internal method to do the actual conversion for making it a DataFrame suitable for both Integer Activation Log
        and for the Bag.

        Returns
        -------
        df: DataFrame
            The integer representation of the 0/1 vectors in this Activation Log

        """
        cat_list = sorted(self.columns)
        df = self.df.apply(lambda row: int(''.join([str(row[i]) for i in cat_list]), 2), axis=1).to_frame(
            int_representation)
        return df

    def to_ial(self):
        """
        Create an Integer Activation Log from this Activation Log.


        Returns
        -------
        ial: IntegerActivationLog
            The Integer Activation Log that results from this Activation Log.

        """
        return IntegerActivationLog(self.__to_ial_df())

    def to_bag(self):
        """
        Create a Bag from this Activation Log (skipping the Integer Activation Log in between).

        Returns
        -------
        bag: Counter
            The Bag that results from this Activation Log.

        """
        return Counter(self.__to_ial_df()[int_representation])


class IntegerActivationLog(_LogBase):
    """
    Integer representation of the activation of the baskets
    """

    def required_features(self):
        return [int_representation]

    def optional_features(self):
        return []

    def id_feature(self):
        return visit

    def to_bag(self):
        """
        Create a Bag from this Integer Representation Log

        Returns
        -------
        bag: Counter
            The Bag that results from this Integer Representation Log
        """
        return Counter(self[int_representation])


class LabelledBasketLog(UnlabelledBasketLog):
    """
    Basket contents with a label.
    """

    def __init__(self, source, **kwargs):
        super().__init__(source, **kwargs)
        self.assert_unique_index()

    def required_features(self):
        return super().required_features() + [label]

    def optional_features(self):
        return []

    def id_feature(self):
        return visit


class UnlabelledVisitLog(_LogBase):
    """
    Visit descriptions (without Basket).
    """

    def __init__(self, source, **kwargs):
        super().__init__(source, **kwargs)
        self.assert_unique_index()

        self.assert_unique(f=[consumer, timestamp, invoice])
        for f in self.optional_features():
            if f in self.columns:
                self.assert_unique(f=f)

    def required_features(self):
        return uvl_features

    def optional_features(self):
        return uvl_optional_features

    def id_feature(self):
        return visit


def extract_sku_map(source, col):
    """
    Extracts the desired information from a sku_map

    Parameters
    ----------
    source: pd.Series, filename or pd.DataFrame
        source value of the sku_map. If filename, imports as DataFrame.
    col: str
        the column that defines the sku_map

    Returns
    -------
    sr: pd.Series
        Same as source if source was pd.Series. Otherwise tries to get the desired column and rename the index to sku

    Raises
    ------
    IndexError
        If the source is a DataFrame or file, and the name of the index is not SKU and none of the columns is SKU.
        If the source is a DataFrame of file, and the none of the columns is col.
    """
    if isinstance(source, pd.Series):
        return source
    else:
        source = dataframe_operations.import_df(source)
        if source.index.name == sku:
            return source[col]
        else:
            return source.set_index(sku)[col]


def data_manager_from_folder(*fds):
    """
    Creates a data manager object from a list of folders, adding all files it finds, provided they are correctly
    formatted. (xxx.csv, with xxx in fn_options).

    Parameters
    ----------
    fds: args
        list of folders to search in

    Returns
    -------
    dm: DataManager
        DataManager with all found files

    # TODO : test
    # TODO allow bag as well?
    """

    fn_options = ['rl ', 'tl ', 'sku_map ', 'ubl ', 'ntl ', 'uxdl ', 'al ', 'ial ', 'uvl', 'lbl ', 'labels']

    args = dict()

    for fd in fds:
        if fd is None:
            continue
        for fn in file_functions.list_files(fd):
            x = Path(fn).name.lower().rsplit('.', 1)
            if x[1] != 'csv':
                continue

            if x[0] in fn_options:
                args[x[0]] = fn

    return DataManager(**args)


class DataManager:

    def __init__(self, rl=None, tl=None, sku_map=None, ubl=None, ntl=None, uxfl=None, al=None, ial=None, uvl=None,
                 lbl=None, labels=None, bag=None, categories=None):
        """
        This class is used to easily access and convert between the different logs. You simply load any available data
        and then allow to easily let the class handle the conversion. Inputs can be DataFrames, actual classes and even
        filenames. This class is **not** recommended for intensive computational purposes, since all conversions are
        done with all checks that come with creating new classes. The class is fine for pre-processing and quick
        evaluation purposes though.

        Parameters
        ----------
        rl: RawLog, DataFrame or str
        tl: TransactionLog, DataFrame or str
        sku_map: DataFrame or Series
        ubl: UnlabelledBasketLog, DataFrame or str
        ntl: NamedTransactionLog, DataFrame or str
        uxfl: UXFeatureLog, DataFrame or str
        al: ActivationLog, DataFrame or str
        ial: IntegerActivationLog, DataFrame or str
        uvl: UnlabelledVisitLog, DataFrame or str
        lbl: LabelledBasketLog, DataFrame or str
        labels: Series
        bag: Counter(int)
        categories: iterable of str
        """

        self.rl = rl
        self.tl = tl
        self.sku_map = sku_map
        self.ubl = ubl
        self.ntl = ntl
        self.uxdl = uxfl
        self.al = al
        self.ial = ial
        self.uvl = uvl
        self.lbl = lbl
        self.labels = labels
        self.bag = bag
        self.categories = categories

    @property
    def rl(self):
        return self.__rl

    @rl.setter
    def rl(self, rl):
        self.__rl = None if rl is None else RawLog(rl)

    @property
    def tl(self):
        if self.__tl is None:
            if self.rl is not None:
                self.__tl = self.rl.to_tl()
        return self.__tl

    @tl.setter
    def tl(self, tl):
        self.__tl = None if tl is None else TransactionLog(tl)

    @property
    def sku_map(self):
        return self.__sku_map

    @sku_map.setter
    def sku_map(self, sku_map):
        self.__sku_map = None if sku_map is None else dataframe_operations.import_df(sku_map)

    @property
    def ubl(self):
        if self.__ubl is None:
            if self.tl is not None and self.sku_map is not None:
                self.__ubl = self.tl.to_ubl(self.sku_map)
        return self.__ubl

    @ubl.setter
    def ubl(self, ubl):
        self.__ubl = None if ubl is None else UnlabelledBasketLog(ubl)

    @property
    def ntl(self):
        if self.__ntl is None:
            if self.tl is not None and self.sku_map is not None:
                self.__ntl = self.tl.to_ntl(sku_map=self.sku_map)
        return self.__ntl

    @ntl.setter
    def ntl(self, ntl):
        self.__ntl = None if ntl is None else NamedTransactionLog(ntl)

    @property
    def uxdl(self):
        if self.__uxdl is None:
            if self.tl is not None and self.sku_map is not None:
                self.__uxdl = self.tl.to_uxdl(self.sku_map)
        return self.__uxdl

    @uxdl.setter
    def uxdl(self, uxdl):
        self.__uxdl = None if uxdl is None else UXFeatureLog(uxdl)

    @property
    def al(self):
        if self.__al is None:
            if self.ubl is not None:
                self.__al = self.ubl.to_activation(self.categories)
        return self.__al

    @al.setter
    def al(self, al):
        self.__al = None if al is None else ActivationLog(al)

    @property
    def ial(self):
        if self.__ial is None:
            if self.al is not None:
                self.__ial = self.al.to_ial()
        return self.__ial

    @ial.setter
    def ial(self, ial):
        self.__ial = None if ial is None else IntegerActivationLog(ial)

    @property
    def uvl(self):
        if self.__uvl is None:
            if self.rl is not None:
                self.__uvl = self.rl.to_uvl()
        return self.__uvl

    @uvl.setter
    def uvl(self, uvl):
        self.__uvl = None if uvl is None else UnlabelledVisitLog(uvl)

    @property
    def lbl(self):
        if self.__lbl is None:
            if self.ubl is not None and self.labels is not None:
                self.__lbl = self.ubl.to_lbl(self.labels)
        return self.__lbl

    @lbl.setter
    def lbl(self, lbl):
        self.__lbl = None if lbl is None else LabelledBasketLog(lbl)

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, labels):
        self.__labels = None if labels is None else dataframe_operations.import_sr(labels)

    @property
    def bag(self):
        if self.__bag is None:
            if self.al is not None:
                self.__bag = self.al.to_bag()
            elif self.ial is not None:
                self.__bag = self.ial.to_bag()
        return self.__bag

    @bag.setter
    def bag(self, bag):
        self.__bag = None if bag is None else Counter(bag)
