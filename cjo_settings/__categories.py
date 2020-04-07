# This file overrides settings. This file is supposed to be shared within the company (through git), but not for
# the research community

# -- categories set_1 --
BABY_PRODUCTS = 'BABY PRODUCTS'
PERSONAL_CARE = 'PERSONAL CARE'
NON_FOOD = 'NON FOOD'
ALCOHOLS_TOBACCO = 'ALCOHOLS AND TOBACCO'
SNACKS = 'SNACKS'
BREAD = 'BREAD'
CONDIMENTS_NP = 'CONDIMENTS NP'
MILK = 'MILK'
CONDIMENTS_P = 'CONDIMENTS P'
CHEESE = 'CHEESE'
EGGS = 'EGGS'
DESSERTS = 'DESSERTS'
SPREAD_NP = 'SPREAD NP'
SPREAD_P = 'SPREAD P'
CARBS_NP = 'CARBS NP'
CARBS_P = 'CARBS P'
PREPARED_MEALS_NP = 'PREPARED MEALS NP'
PREPARED_MEALS_P = 'PREPARED MEALS P'
FRUITS_VEGETABLES_P = 'FRUITS AND VEGETABLES P'
FRUITS_VEGETABLES_NP = 'FRUITS AND VEGETABLES NP'
HOT_DRINKS = 'HOT DRINKS'
GRAINS = 'GRAINS'
MEAT_FISH_P = 'MEAT AND FISH P'
MEAT_FISH_NP = 'MEAT AND FISH NP'
SOFT_DRINKS_NP = 'SOFT DRINKS NP'
SOFT_DRINKS_P = 'SOFT DRINKS P'
SWEETS = 'SWEETS'
WASHING_CLEANING = 'WASHING AND CLEANING'
PET_ARTICLES = 'PET ARTICLES'

# additional in set_2
BAKERY = 'BAKERY'
CONDIMENTS = 'CONDIMENTS'
DAIRY_PRODUCTS = 'DAIRY PRODUCTS'
DESSERTS_ADDITIVES = 'DESSERTS AND ADDITIVES'
FATS = 'FATS'
FROZEN_CHILLED_FOOD = 'FROZEN & CHILLED FOOD'
FRUITS_VEGETABLES = 'FRUITS & VEGETABLES'
LOOSE_ARTICLES = 'LOOSE ARTICLES'
MEAT_FISH = 'MEAT & FISH'
SOFT_DRINKS = 'SOFT DRINKS'
SOUPS_SPICES = 'SOUPS & SPICES'

# ABBREVIATIONS
all_abbreviations = {BABY_PRODUCTS: 'BAB', PERSONAL_CARE: 'PER', NON_FOOD: 'NON', ALCOHOLS_TOBACCO: 'AnT',
                     SNACKS: 'SNA', BREAD: 'BRE', CONDIMENTS_NP: 'CONNP', MILK: 'MIL', CONDIMENTS_P: 'CONP',
                     CHEESE: 'CHE', EGGS: 'EGG', DESSERTS: 'DES', SPREAD_NP: 'SPRNP', SPREAD_P: 'SPRP', CARBS_P: 'CARP',
                     CARBS_NP: 'CARNP', PREPARED_MEALS_NP: 'PREP', PREPARED_MEALS_P: 'PRENP',
                     FRUITS_VEGETABLES_P: 'FnVP', FRUITS_VEGETABLES_NP: 'FnVNP', HOT_DRINKS: 'HOT', GRAINS: 'GRA',
                     MEAT_FISH_P: 'MnFP', MEAT_FISH_NP: 'MnFNP', SOFT_DRINKS_P: 'SOFP', SOFT_DRINKS_NP: 'SOFNP',
                     SWEETS: 'SWE', WASHING_CLEANING: 'WnC', PET_ARTICLES: 'PET', BAKERY: 'BAK', CONDIMENTS: 'CON',
                     DAIRY_PRODUCTS: 'DAI', DESSERTS_ADDITIVES: 'DnA', FATS: 'FAT', FROZEN_CHILLED_FOOD: 'FRO',
                     FRUITS_VEGETABLES: 'FRU', LOOSE_ARTICLES: 'LOO', MEAT_FISH: 'MEA', SOFT_DRINKS: 'SOF',
                     SOUPS_SPICES: 'SOU'}


# ICONS
def separate(s):
    return '/freepik/Separate/' + s + '.png'


def pack(s):
    return '/freepik/Supermarket_Icons/' + s + '.png'


all_icons = {
    ALCOHOLS_TOBACCO: pack('alcoholic-drinks'),
    BABY_PRODUCTS: separate('baby_feeding-bottle'),
    BAKERY: pack('bread'),
    BREAD: pack('bread'),
    CARBS_P: separate('fast-food_noodle'),
    CARBS_NP: pack('pasta'),
    CHEESE: pack('cheese-1'),
    CONDIMENTS: separate('bbq-line-craft_sausaces'),
    DAIRY_PRODUCTS: pack('milk'),
    DESSERTS: pack('ice-creams'),
    DESSERTS_ADDITIVES: pack('baking-products'),
    EGGS: pack('eggs'),
    FATS: pack('butter'),
    FROZEN_CHILLED_FOOD: pack('frozen-food'),
    FRUITS_VEGETABLES: pack('fruits'),
    FRUITS_VEGETABLES_NP: pack('canned-food'),
    FRUITS_VEGETABLES_P: pack('fruits'),
    GRAINS: pack('oatmeal'),
    HOT_DRINKS: pack('cofee-and-tea'),  # [sic]
    LOOSE_ARTICLES: pack('pasta'),
    MEAT_FISH: pack('meat'),
    MILK: pack('milk'),
    NON_FOOD: separate('bathroom_toilet-paper'),
    PERSONAL_CARE: pack('hygienic-items'),
    PET_ARTICLES: separate('pet-shop_pet-food'),
    PREPARED_MEALS_NP: pack('pizza'),
    PREPARED_MEALS_P: pack('ready-to-eat'),
    SNACKS: pack('snack'),
    SOFT_DRINKS: pack('soft-drinks'),
    SOUPS_SPICES: pack('spices'),
    SWEETS: pack('sweets'),
    WASHING_CLEANING: pack('bleach-and-soup'),  # [sic]
    CONDIMENTS_NP: separate('bbq-line-craft_sausaces'),
    CONDIMENTS_P: separate('christmas_spices'),
    SPREAD_NP: separate('bakery_jam'),
    SPREAD_P: pack('butter'),
    MEAT_FISH_P: pack('meat'),
    MEAT_FISH_NP: pack('preserves'),
    SOFT_DRINKS_NP: separate('fast-food_soft-drink'),
    SOFT_DRINKS_P: separate('coffee-and-breakfast_box-of-juice'),
}

all_perishable = {
    ALCOHOLS_TOBACCO: False,
    BABY_PRODUCTS: False,
    BAKERY: True,
    BREAD: True,
    CARBS_P: True,
    CARBS_NP: False,
    CHEESE: True,
    DAIRY_PRODUCTS: True,
    DESSERTS: False,
    EGGS: True,
    FATS: False,
    FROZEN_CHILLED_FOOD: False,
    FRUITS_VEGETABLES_NP: False,
    FRUITS_VEGETABLES_P: True,
    GRAINS: False,
    HOT_DRINKS: False,
    MILK: True,
    NON_FOOD: False,
    PERSONAL_CARE: False,
    PET_ARTICLES: False,
    PREPARED_MEALS_NP: False,
    PREPARED_MEALS_P: True,
    SNACKS: False,
    SOUPS_SPICES: False,
    SWEETS: False,
    WASHING_CLEANING: False,
    CONDIMENTS_NP: False,
    CONDIMENTS_P: True,
    SPREAD_NP: False,
    SPREAD_P: True,
    MEAT_FISH_P: True,
    MEAT_FISH_NP: False,
    SOFT_DRINKS_NP: False,
    SOFT_DRINKS_P: True,
}

# CATEGORIES
# -- set the one used in the project below
# See key-file for difference

category_set_1 = [BABY_PRODUCTS, PERSONAL_CARE, NON_FOOD, ALCOHOLS_TOBACCO, SNACKS, BREAD, CONDIMENTS_NP, MILK,
                  CONDIMENTS_P, CHEESE, EGGS, DESSERTS, SPREAD_NP, SPREAD_P, CARBS_NP, PREPARED_MEALS_NP,
                  PREPARED_MEALS_P, CARBS_P, FRUITS_VEGETABLES_P, FRUITS_VEGETABLES_NP, HOT_DRINKS, GRAINS, MEAT_FISH_P,
                  MEAT_FISH_NP, SOFT_DRINKS_NP, SOFT_DRINKS_P, SWEETS, WASHING_CLEANING, PET_ARTICLES]

category_set_2 = [ALCOHOLS_TOBACCO, BABY_PRODUCTS, BAKERY, CONDIMENTS, DAIRY_PRODUCTS, DESSERTS_ADDITIVES, FATS,
                  FROZEN_CHILLED_FOOD, FRUITS_VEGETABLES, HOT_DRINKS, LOOSE_ARTICLES, MEAT_FISH, PERSONAL_CARE,
                  PET_ARTICLES, SNACKS, SOFT_DRINKS, SOUPS_SPICES, SWEETS, WASHING_CLEANING]

# Below are the actual ones required by generalsettings.py
default_categories = category_set_1
__category_abbreviation_dict = {k: all_abbreviations[k] for k in default_categories}
__category_icons_dict = {k: 'icons/' + all_icons[k] for k in default_categories}
default_perishable_categories = [c for c in default_categories if all_perishable[c]]
