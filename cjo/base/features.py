"""
Module to handle all features
"""
from cjo.base.stringconstants import value, brand, supplier, cat, size, width, avg_value, \
    abs_discount, frac_discount, has_discount
from functions.general_functions import listified


########################################################################################################################
# Verify the existence of a feature                                                                                    #
########################################################################################################################
def is_feature(feature):
    """
    Verifies if the given argument is a feature

    Parameters
    ----------
    feature: String
        The feature to be checked

    Returns
    -------
    is_feature : Boolean
        True if the given feature is a feature, False otherwise
    """
    # features need to have pre and post
    if feature.count('.') != 1:
        return False

    pre = __pre_post(feature)[0]

    # If the pre is a historic basket feature, we treat the check as regular basket feature
    if __has_historic_basket_prefix(feature):
        pre = __basket_prefix

    # if proper prefix, we check the postfix, else return False
    return {__basket_prefix: is_basket_feature,
            __timestamp_prefix: __is_timestamp_feature,
            __visit_feature_prefix: __is_visit_feature}.get(pre, lambda x: False)(feature)


def assert_feature(feature):
    """
    Asserts that the given argument is a feature

    Parameters
    ----------
    feature: String
        Feature to be asserted. Raises AssertionError if not a feature
    """
    assert is_feature(feature), 'Not a feature : {}'.format(feature)


# ######################################## Timestamp Features ######################################################## #
__timestamp_prefix = 'ts'
ts_day_of_week = __timestamp_prefix + '.day_of_week'
ts_is_weekend = __timestamp_prefix + '.is_weekend'
ts_time_since_last = __timestamp_prefix + '.time_since_last'
ts_quarter = __timestamp_prefix + '.quarter'
ts_month = __timestamp_prefix + '.month'
ts_is_pp = __timestamp_prefix + '.is_pp'
ts_pp_week = __timestamp_prefix + '.pp_week'


def __is_timestamp_feature(feature):
    """
    Verifies if the given argument is a timestamp feature

    Parameters
    ----------
    feature: String
        The feature to be checked

    Returns
    -------
    __is_timestamp_feature : Boolean
        True if the given feature is a timestamp feature, False otherwise
    """
    return feature in [ts_day_of_week,
                       ts_is_weekend,
                       ts_time_since_last,
                       ts_is_pp,
                       ts_pp_week,
                       ts_quarter,
                       ts_month]


# ######################################## Visit Features ############################################################ #
__visit_feature_prefix = 'v'
visit_store = __visit_feature_prefix + '.store'


def __is_visit_feature(feature):
    """
    Verifies if the given argument is a visit feature

    Parameters
    ----------
    feature: String
        The feature to be checked

    Returns
    -------
    __is_visit_feature : Boolean
        True if the given feature is a visit feature, False otherwise
    """

    return feature in [visit_store]


# ######################################## Basket Features ########################################################### #
__basket_prefix = 'b'
dow = 'dow'
historic = 'bs'

__base_basket_postfixes_agg = [size, width, value, avg_value]
__base_basket_postfixes_discount = [abs_discount, frac_discount, has_discount]
__base_basket_postfixes = __base_basket_postfixes_agg + __base_basket_postfixes_discount

num = 'Num'  # the number of different Suppliers / Brands / Category_y


# Create basket feature
def basket_feature(post, history=None):
    """

    Parameters
    ----------
    post : String
        Postfix of the feature. Raises AssertionError if not a basket feature postfix
    history : String, Int or None, optional
        Determines feature prefix. If None, the prefix is on the basket. If ``Features.dow`` or ``Features.historic``,
        the respective prefix is selected. If Int, the previous ``agg`` days prefix is selected. Raises TypeError if any
        other value.

    Returns
    -------
    feature : String
        The feature, verified to be an existing basket feature
    """
    if history is None:
        pre = __basket_prefix
    elif history in [dow, historic]:
        pre = history
    elif isinstance(history, int) and history > 0:
        pre = '{}{}'.format(__basket_prefix, history)
    else:
        raise TypeError(
            'history should be one of : None, "dow" (day of week), "bs" (historic), or positive integer')

    feature = '{}.{}'.format(pre, post)
    assert is_basket_feature(feature), 'Not a feature: {}'.format(feature)
    return feature


def has_non_historic_basket_prefix(feature):
    """
    Checks whether the feature has a Non-Historic Basket Prefix

    Parameters
    ----------
    feature: String
        The Feature to be checked. Does not raise AssertionError if not a Feature.

    Returns
    -------
        True if the prefix of the feature is equal to the basket_prefix, False otherwise
    """
    if feature.count('.') != 1:
        return False
    return __pre_post(feature)[0] == __basket_prefix


def __has_historic_basket_prefix(feature):
    """
    Checks whether the feature has a Historic Basket Prefix

    Parameters
    ----------
    feature: String
        The Feature to be checked. Does not raise AssertionError if not a Feature.

    Returns
    -------
        True if the prefix of the feature is either dow, b*, or b{k}.
    """
    if feature.count('.') != 1:
        return False
    pre = __pre_post(feature)[0]
    if pre in [dow, historic]:
        return True
    return pre[0] == __basket_prefix and pre[1:].isdigit()


def is_basket_feature(feature):
    """
    Verifies if the given argument is a basket feature

    Parameters
    ----------
    feature: String
        The feature to be checked

    Returns
    -------
    __is_timestamp_feature : Boolean
        True if the given feature is a basket feature, False otherwise
    """
    if feature.count('.') != 1:
        return False
    pre, post = __pre_post(feature)

    # check the prefix
    if not (has_non_historic_basket_prefix(feature) or __has_historic_basket_prefix(feature)):
        return False

    # check postfix
    if __is_basket_superset_postfix(post):
        return True

    if __is_basket_subset_postfix(post):
        return True

    if __is_basket_num_postfix(post):
        return True

    return False


# superset (i.e. complete basket)
def __is_basket_superset_postfix(postfix):
    """
    Checks if the given postfix describes a superset feature (i.e. aggregated over the complete collection of SKU rather
    than over a subset)

    Parameters
    ----------
    postfix : String
        The postfix to be checked

    Returns
    -------
    __is_basket_superset_postfix : Boolean
        True if the postfix is a basket superset postfix, False otherwise
    """
    # {base}
    if postfix in __base_basket_postfixes:
        return True


# subset (only part of a basket)
def __is_basket_subset_postfix(postfix):
    """
    Checks if the given postfix describes a subset feature (i.e. aggregated over a valid subset of the SKU collection)

    Parameters
    ----------
    postfix : String
        The postfix to be checked

    Returns
    -------
    __is_basket_subset_postfix : Boolean
        True if the postfix is a basket subset postfix, False otherwise
    """
    # {base}_{subset}
    if postfix.count('_') >= 1 and postfix.split('_')[0] in __base_basket_postfixes:
        return __is_sku_subset(postfix.split('_', 1)[1])

    return False


def basket_subset_feature(agg, subset_type, name, history=None):
    """

    Create (and verify) a basket subset feature

    Parameters
    ----------
    agg: String
        Aggregation (``size``, ``width``, ``value``, ``avg_value``, ``abs_discount``, ``frac_discount``,
        ``has_discount``)
    subset_type: String
        Type of subset (Supplier, Brand, Category_y)
    name : String
        Name of the subset
    history : String or None, Optional
        see :meth:`Features.basket_feature`

    Returns
    -------
    f: String
        Basket Subset Feature for this combination

    Raises
    ------
    AssertionError:
        If

        - ``agg`` or ``subset_type`` not a valid value
        - ``name`` not a string

        also see :meth:`Features.basket_feature`

    """
    assert is_subset_agg(agg), 'Not a correct aggregation type : {}'.format(agg)
    assert is_subset_type(subset_type), 'Type of subset not valid {}'.format(subset_type)
    assert isinstance(name, str), 'Name of subset must be String'
    postfix = '{}_{}_{}'.format(agg, subset_type, name)
    return basket_feature(postfix, history)


def is_basket_subset_feature(feature):
    """
    Checks if the given feature is a basket subset feature (i.e. aggregated over a valid subset of the SKU collection)

    Parameters
    ----------
    feature : String
        The Feature to be checked

    Returns
    -------
    is_basket_subset_feature : Boolean
        True if the feature satisfied the requirements of a basket subset feature, False otherwise
    """
    if not is_basket_feature(feature):
        return False
    return __is_basket_subset_postfix(__pre_post(feature)[1])


def __split_basket_subset_feature(feature):
    """
    Splits a basket subset_type feature into the aggregation type, subset_type type, and subset_type name in the postfix

    Parameters
    ----------
    feature : String
        The feature to be split. Raises Assertion error if it is not a Basket Subset Feature

    Returns
    -------
    agg : String
        The aggregation type of the feature
    subset_type : String
        The subset_type type of the feature
    subset_name : String
        The subset_type name of the feature
    """
    assert is_basket_subset_feature(feature), 'Not a Basket Subset Feature'
    postfix = __pre_post(feature)[1]
    values = postfix.split('_')
    agg = values[0]
    if values[1] in [supplier, brand, cat]:  # {agg}_{supplier,brand,Category}_{name}
        subset_type = values[1]
        subset_name = '_'.join(values[2:])
    else:  # {agg}_{Category}_{y}_{name}
        subset_type = '_'.join(values[1:3])
        subset_name = '_'.join(values[3:])

    return agg, subset_type, subset_name


def get_subset_name(subset_feature):
    """
    Get the subset name of a subset feature.

    Parameters
    ----------
    subset_feature: String
        The feature for which to get the subset name.

    Returns
    -------
    subset: String
        The subset name of the feature.

    Raises
    ------
    AssertionError:
        If ``subset_feature`` is not a subset feature
    """
    assert is_basket_subset_feature(subset_feature)
    return __split_basket_subset_feature(subset_feature)[2]


def get_subset_type(subset_feature):
    """
    Get the subset type of a subset feature.

    Parameters
    ----------
    subset_feature: String
        The feature for which to get the subset type.

    Returns
    -------
    subset: String
        The subset type of the feature.

    Raises
    ------
    AssertionError:
        If ``subset_feature`` is not a subset feature
    """
    assert is_basket_subset_feature(subset_feature)
    return __split_basket_subset_feature(subset_feature)[1]


def get_subset_agg(subset_feature):
    """
    Get the aggregation of a subset feature.

    Parameters
    ----------
    subset_feature: String
        The feature for which to get the aggregation.

    Returns
    -------
    subset: String
        The aggregation of the feature.

    Raises
    ------
    AssertionError:
        If ``subset_feature`` is not a subset feature
    """
    assert is_basket_subset_feature(subset_feature)
    return __split_basket_subset_feature(subset_feature)[0]


def is_subset_type(subset_type):
    """
    Check if a given String is a correct subset type. (Brand, Supplier, or Category_y)

    Parameters
    ----------
    subset_type: String
        The String to check

    Returns
    -------
    is_subset_type: Boolean
        True if the string is a subset type, False otherwise.

    """
    if subset_type in [supplier, brand, cat]:
        return True

    if subset_type.count('_') == 1:
        return subset_type.split('_')[0] == cat and subset_type.split('_')[1].isdigit()

    return False


def is_subset_agg(subset_agg):
    """
    Check if a given String is a correct aggregation for subsets. (Size, Value, Width)

    Parameters
    ----------
    subset_agg: String
        The String to be checked

    Returns
    -------
        True if the string is any of the subset aggregation types, False otherwise
    """

    return subset_agg in [size, width, value, avg_value]


def get_basket_subset_features(v, agg=None, subset_type=None):
    """
    Given a (collection of) String(s), returns all basket subset features.

    Parameters
    ----------
    v : (Collection of) String
        The String(s) to filter
    agg: (Collection of) String or None, Optional
        If not None, only take this subset aggregation
    subset_type: (Collection of) String or None, Optional
        If not None, only take this subset type

    Returns
    -------
    subset_features: List of String
        All Strings that are a subset feature, with possible additional filters for subset agg and subset type

    Raises
    ------
    AssertionError
        If ``agg`` contains non-subset aggregations
        If ``subset_type`` contains non-subset types
    """
    ret = listified(v, str, filtering=is_feature)
    ret = listified(ret, str, filtering=is_basket_subset_feature)
    if agg is not None:
        agg = listified(agg, str, validation=is_subset_agg)
        ret = listified(ret, str, filtering=lambda x: get_subset_agg(x) in agg)
    if subset_type is not None:
        subset_type = listified(subset_type, str, validation=is_subset_type)
        ret = listified(ret, str, filtering=lambda x: get_subset_type(x) in subset_type)
    return ret


def get_categories(features):
    """
    Generates a list of all categories in the features

    Parameters
    ----------
    features : (Collection of) Str
        Features to be examined

    Returns
    -------
    categories : List of Str
        The categories for which there is *any* feature

    """
    features = listified(features, str, filtering=is_basket_subset_feature)
    return list({get_subset_name(f) for f in features if get_subset_type(f) == cat})


def __is_sku_subset(subset):
    """
    Checks if the given subset type-name combination is valid

    Parameters
    ----------
    subset : String
        subset type-name combination to be checked

    Returns
    -------
    __is_sku_subset: Boolean
        True if the given subset type-name combination is valid, False otherwise
    """
    if subset.count('_') == 1:
        # Supplier_x or Brand_x
        return subset.split('_')[0] in [supplier, brand, cat]
    if subset.count('_') >= 2:
        # Category_y_x
        return subset.split('_')[0] == cat and subset.split('_')[1].isdigit()

    return False


def __is_basket_num_postfix(postfix):
    """
    Checks if the given postfix belongs to a num feature (i.e. the size of a given subset type)

    Parameters
    ----------
    postfix : String
        The postfix to be checked

    Returns
    -------
    __is_basket_num_postfix : Boolean
        True if the postfix is a basket num postfix, False otherwise
    """
    # Num_Supplier, Num_Brand
    if postfix.count('_') == 1:
        a, b = postfix.split('_')
        return a == num and b in [supplier, brand]

    # Num_Category_y
    if postfix.count('_') == 2:
        a, b, c = postfix.split('_')
        return a == num and b == cat and c.isdigit()


# ######################################## Feature Types ############################################################# #
BOOLEAN_FEATURE = 1
CATEGORICAL_FEATURE = 2
NUMERIC_FEATURE = 3
SPECIAL_NUMERIC_FEATURE = 4


def __get_feature_type(feature):
    """
    Get encoded feature type of a given feature

    Parameters
    ----------
    feature: String
        The feature to get the type for. Raises AssertionError if not a feature

    Returns
    -------
    type : Int
        The type of the feature, encoded as

        1. BOOLEAN_FEATURE
        2. CATEGORICAL_FEATURE
        3. NUMERIC_FEATURE
        4. SPECIAL_NUMERIC_FEATURE

    """
    assert_feature(feature)
    if __is_timestamp_feature:
        if feature in [ts_is_pp, ts_is_weekend]:
            return BOOLEAN_FEATURE
        if feature in [ts_day_of_week, ts_quarter, ts_month]:
            return CATEGORICAL_FEATURE
        if feature in [ts_time_since_last, ts_pp_week]:
            return NUMERIC_FEATURE
    if __is_visit_feature(feature):
        if feature in [visit_store]:
            return CATEGORICAL_FEATURE
    if is_basket_feature(feature):
        pre, post = __pre_post(feature)
        if __is_basket_superset_postfix(post) or __is_basket_num_postfix(post):
            return NUMERIC_FEATURE
        if __is_basket_subset_postfix(post):
            return SPECIAL_NUMERIC_FEATURE

    raise NotImplementedError('Feature type not implemented : {}'.format(feature))


def needs_numeric_conversion(feature):
    """
    Check if the feature needs to be interpreted as numeric

    Parameters
    ----------
    feature: String
        The feature to check. Raises AssertionError if not an existing feature.

    Returns
    -------
    needs_numeric_conversion: Boolean
        True if the feature needs to be interpreted as numeric, False otherwise
    """
    assert_feature(feature)
    return __get_feature_type(feature) in [NUMERIC_FEATURE, SPECIAL_NUMERIC_FEATURE]


def needs_boolean_conversion(feature):
    """
    Check if the feature needs to be interpreted as boolean

    Parameters
    ----------
    feature: String
        The feature to check. Raises AssertionError if not an existing feature.

    Returns
    -------
    needs_boolean_conversion: Boolean
        True if the feature needs to be interpreted as boolean, False otherwise.
    """
    assert_feature(feature)
    return __get_feature_type(feature) == BOOLEAN_FEATURE


def is_categorical(feature):
    """
    Check fi the feature is categorical

    Parameters
    ----------
    feature
        The feature to check. Raises AssertionError if not an existing feature.

    Returns
    -------
    is_categorical : Boolean
        True if the feature is categorical, False otherwise.
    """
    assert_feature(feature)
    return __get_feature_type(feature) == CATEGORICAL_FEATURE


def get_categorical_values(feature):
    """
    Get the default categorical values of a feature. Implemented for

    - ts.day_of_week
    - ts.quarter
    - ts.month

    Parameters
    ----------
    feature
        The feature to get the values for. Raises AssertionError if the feature is not a categorical feature

    Returns
    -------
    values: list of Strings
        A list of values for the feature. Raises ValueError if the feature is any of the following:

            - v.store

        This is because the feature values depend on the dataset.
    Raises
    ------
    NotImplementedError
        if it is neither of the above discussing features
    """

    assert __get_feature_type(feature) == CATEGORICAL_FEATURE, 'Not Categorical : {}'.format(feature)
    if feature == ts_day_of_week:
        return list(range(7))
    if feature == ts_quarter:
        return list(range(1, 5))
    if feature == ts_month:
        return list(range(1, 13))
    if feature == visit_store:
        raise ValueError('{} has no default values, should be learned from real data'.format(visit_store))

    raise NotImplementedError('Categorical feature {} not implemented'.format(feature))


def __pre_post(f):
    """
    Splits ``f`` as a feature into pre and post

    Parameters
    ----------
    f : String
        Feature to be split. Raises AssertionError if not exactly one ``.`` is present.

    Returns
    -------
    pre : String
        prefix of f
    post : String
        postfix of f
    """
    assert f.count('.') == 1, 'able to split {}'.format(f)
    return f.split('.')


def __monetary_or_size_feature(f, monetary):
    assert_feature(f)

    if monetary:
        options = [avg_value, value]
    else:
        options = [size]

    if __pre_post(f)[1] in options:
        return True
    if is_basket_subset_feature(f):
        if get_subset_agg(f) in options:
            return True
    return False


def is_monetary_feature(f):
    return __monetary_or_size_feature(f, True)


def is_size_feature(f):
    return __monetary_or_size_feature(f, False)
