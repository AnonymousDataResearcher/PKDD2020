"""
Contains all string constants relevant to the project, but not inherently part of features.py
"""
consumer = 'ConsumerID'
timestamp = 'Timestamp'
visit = 'VisitID'
invoice = 'InvoiceID'
week = 'Week'
sku = 'SKU'
sku_count = '{}Count'.format(sku)
value = 'Value'
label: str = 'Label'
sku_des = '{}Des'.format(sku)
brand = 'Brand'
supplier = 'Supplier'
cat = 'Category'
enc = 'Encoding'
id_features = [consumer, timestamp, invoice, week]
size = 'Size'  # total number of SKU
width = 'Width'  # total different number of SKU
avg_value = 'Avg' + value  # average value of each product
abs_discount = 'AbsDiscount'  # total discount
frac_discount = 'FracDiscount'  # discount as fraction of total
has_discount = 'HasDiscount'  # whether there is discount
int_representation = 'IntegerRepresentation'  # Integer representation of basket

supercat = f'Super-{cat}'

human_names_dict = {sku_count: 'Qty', sku_des: 'Product', value: 'Price'}

unknown = 'Unknown'

# General ML / Classification
fm_str = 'feature_mode'
lab_str = 'labelled'
bal_str = 'balanced'
clf_str = 'classifier'
acc_str = 'acc'

# SSL Specific
ssl_str = 'ssl'
k_str = 'k'
alpha_str = 'alpha'
n_lab_str = 'n_lab'
n_unlab_str = 'n_unlab'
comment_str = 'comment'

f1_str = 'F1'
train_time_str = 'train_time'
test_time_str = 'test_time'
f1_class_base_str = 'F1_{}'

bl_context_aggregate = 'Aggregate_Features'
bl_context_cat_avg_value = 'Category_{}'.format(avg_value)
bl_context_cat_size = 'Category_{}'.format(size)
bl_context_emphasis = 'Emphasis'

RECEIPT_SHOWN = 'ReceiptShown'
END_TIME = 'EndTime'
START_TIME = 'StartTime'
GIVEN_ANSWER = 'GivenAnswer'
NOTES = 'Notes'
EXPERT = 'Expert'

UXDF_TOTAL_PRICE = 'Total Price'
UXDF_HIGHEST_QUANTITY = 'Highest Quantity'
UXDF_NUMBER_ITEMS = 'Number of Items'
UXDF_NUMBER_PRODUCTS = 'Number of Products'
UXDF_PERISHABLE_FRACTION = 'Fraction Perishable'
UXDF_REMEMBER = 'Can Remember'
UXDF_MEAL = 'Meal'
UXDF_EAT_TODAY = 'Eat Today'
UXDF_FEATURES = [UXDF_TOTAL_PRICE, UXDF_HIGHEST_QUANTITY, UXDF_NUMBER_PRODUCTS, UXDF_NUMBER_ITEMS,
                 UXDF_PERISHABLE_FRACTION, UXDF_MEAL, UXDF_EAT_TODAY, UXDF_REMEMBER]


def hl_name(i):
    """
    Returns the correct naming of a level in the hierarchy

    Parameters
    ----------
    i : int
        Hierarchy level required. Must be positive

    Returns
    -------
    hierarchy_level_name : String
        Name of the level in the hierarchy at level ``i``
    """
    assert isinstance(i, int)
    assert i > 0
    return '{}_{}'.format(cat, i)


def hl_encoding(i):
    """
    Returns the correct encoding of a level in the hierarchy

    Parameters
    ----------
    i : int
        Hierarchy level required. Must be positive

    Returns
    -------
    hierarchy_level_encoding : String
        Encoding name of the level in the hierarchy at level ``i``
    """
    assert isinstance(i, int)
    assert i >= 0
    if i == 0:
        return f'{cat}_{enc}'
    else:
        return f'{cat}_{enc}_{i}'


def is_hierarchy_level_name(hln):
    """
    Verifies whether a given string is a hierarchy level name

    Parameters
    ----------
    hln : String
        The text to be checked

    Returns
    -------
    is_hln : Boolean
        True if ``hln`` is a correctly formatted hierarchy level, False otherwise
    """
    return hln.count('_') == 1 and hln.split('_')[0] == cat and hln.split('_')[1].isdigit()


# Bootstrapping

OPTIMIZATION_TIME = 'optimization'
CLUSTERING_TIME = 'clustering'
EndIterations = 'Iterations'
EndEpsilon = 'Epsilon'
EndCycle = 'Cycle'
StartReceipts = 'Initializing bootstrap with given clustering as receipts'
StartDSX = 'Initializing bootstrap with given clustering as DSX'
StartWeights = 'Initializing bootstrap with given weights'
StartDefault = 'Initializing bootstrap with unit weights'
IEM = 'IEM'

# TODO this is actually iterations...
bootstrap_rep_count = 'Repetitions'

cluster_size = 'Cluster Size'
medoid = 'Medoid'
BootstrapFixedPoint = 'Fixed Point'
BOOTSTRAP_END = 'End'
BOOTSTRAP_START = 'Start'

REPETITION = 'Repetition'
ITERATION = 'Iteration'

ISC_WEIGHT = 'w_s'
ISC_WEIGHT_LATEX = '$w_s$'
