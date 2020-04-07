from pathlib import Path
from cjo.base import classes
from cjo.base.entities import hierarchy_from_csv
from functions import dataframe_operations
from cjo.weighted_adapted_jaccard.distances.implementation import data_2_dataset_x
from cjo_settings.generalsettings import default_categories

public_data_folder = Path('data/public')
retailer_folder = public_data_folder / 'retailer'


###########
# GENERIC #
###########

def generic_hierarchy(name):
    return hierarchy_from_csv(public_data_folder / 'generic_hierarchies' / f'{name}.csv')


def generic_dsx_loader(name, hierarchy):
    if name[0] == 'S':
        # Sample
        assert len(name) == 2, f'Invalid name : {name}'
        return dsx_sample(int(name[1]), hierarchy=hierarchy)
    elif name[0] == 'D':
        return dsx_data(int(name[1]), hierarchy=hierarchy, num=int(name.split('_')[1]))
    else:
        raise ValueError(f'Unknown name {name}')


def generic_al_loader(name, num=0):
    if name[0] == 'S':
        return al_sample(int(name[1:]))
    elif name[0] == 'D':
        return al_data(int(name[1:]), dataset_number=num)
    else:
        raise ValueError(f'Unknown name {name}')


###############
# ACTUAL DATA #
###############

def al_data(dataset_exponent, dataset_number):
    assert isinstance(dataset_exponent, int)
    assert isinstance(dataset_number, int)

    if dataset_exponent in [3, 4, 5]:
        # These are saved with multiple values in one log
        assert dataset_number < 10 ** (7 - dataset_exponent)
        size = 10 ** dataset_exponent
        number_of_datasets_per_file = 500000 // size
        file_number = dataset_number // number_of_datasets_per_file
        df = dataframe_operations.import_df(retailer_folder / f'D{dataset_exponent}' / f'{file_number}.csv')
        dataset_number_in_file = dataset_number % number_of_datasets_per_file
        df = df.iloc[dataset_number_in_file * size: (dataset_number_in_file + 1) * size]
        return classes.ActivationLog(df)
    elif dataset_exponent in [6, 7]:
        assert (dataset_exponent == 6 and dataset_number < 10) or (dataset_exponent == 7 and dataset_number == 0)
        return classes.ActivationLog(retailer_folder / f'D{dataset_exponent}' / f'{dataset_number}.csv')
    else:
        raise ValueError('Illegal dataset_exponent')


def dsx_data(exp, num, hierarchy):
    al = al_data(exp, num)
    if isinstance(hierarchy, str):
        hierarchy = generic_hierarchy(hierarchy)
    return data_2_dataset_x(al, hierarchy, sorted(default_categories))


###########
# SAMPLES #
###########


def dsx_sample(exp, hierarchy='A3'):
    al = al_sample(exp)
    if isinstance(hierarchy, str):
        hierarchy = generic_hierarchy(hierarchy)
    return data_2_dataset_x(al, hierarchy, sorted(default_categories))


def al_sample(exp):
    assert exp in range(1, 5), 'given exp must be in 1...4'
    return classes.ActivationLog(public_data_folder / 'samples' / 'real dataset' / f'S{exp}.csv')


def sample_hierarchy():
    return hierarchy_from_csv(public_data_folder / 'generic_hierarchies' / 'A3.csv')


############
# EXAMPLES #
############


def example_hierarchy():
    return hierarchy_from_csv(public_data_folder / 'samples' / 'implementation guide' / 'hierarchy.csv')


def example_cat_list():
    return sum([list(v) for v in example_hierarchy().values()], [])


def __dsx_example(loader):
    return data_2_dataset_x(loader(), example_hierarchy(), sorted(example_cat_list()))


def dsx_running_example():
    return __dsx_example(al_running_example)


def dsx_extended_example():
    return __dsx_example(al_extended_example)


def dsx_extended_example2():
    return __dsx_example(al_extended_example2)


def al_running_example():
    return classes.ActivationLog(public_data_folder / 'samples' / 'implementation guide' / 'running_example.csv',
                                 categories=example_cat_list())


def al_extended_example():
    return classes.ActivationLog(public_data_folder / 'samples' / 'implementation guide' / 'extended_example.csv',
                                 categories=example_cat_list())


def al_extended_example2():
    return classes.ActivationLog(public_data_folder / 'samples' / 'implementation guide' / 'extended_example2.csv',
                                 categories=example_cat_list())
