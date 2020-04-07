from pathlib import Path

from functions import file_functions
from scripts.PKDD_EXPERIMENTS import EXPERIMENT_1_RESULTS, EXPERIMENT_2_COMPUTE, \
    EXPERIMENT_2_RESULTS, EXPERIMENT_3_COMPUTE, EXPERIMENT_3_RESULTS, PKDD_PARAMETERS

if PKDD_PARAMETERS.REPETITION_COUNT in [100, 1000]:
    file_functions.copydir(f'results/precomputed/PKDD {PKDD_PARAMETERS.REPETITION_COUNT} Reps/bootstrap',
                           PKDD_PARAMETERS.RESULTS_BOOTSTRAP)
else:
    raise Exception(f'The results for {PKDD_PARAMETERS.REPETITION_COUNT} repetitions are not precomputed. Change the '
                    f'REPETITION_COUNT to 100 or '
                    '1000 in scripts/PKDD_EXPERIMENTS/PKDD_PARAMETERS.py to use precomputed results of Algorithm 1, or'
                    'use scripts/PKDD_EXPERIMENTS/ALL_EXPERIMENTS.py to compute these results.')

EXPERIMENT_1_RESULTS.run()

EXPERIMENT_2_COMPUTE.run()
EXPERIMENT_2_RESULTS.run()

EXPERIMENT_3_COMPUTE.run()
EXPERIMENT_3_RESULTS.run()
