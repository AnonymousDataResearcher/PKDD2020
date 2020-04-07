from cjo.weighted_adapted_jaccard.bootstrap.multiple_bootstrap import run_bootstrap
from data import data_loader

# Runs the bootstraps, used in all experiments in the paper
from scripts.PKDD_EXPERIMENTS.PKDD_PARAMETERS import REPETITION_COUNT


def run():
    for i in [_ for _ in range(102) if _ not in [88, 90]]:
        run_bootstrap(dsx=data_loader.dsx_data(3, i, 'A3'), dataset=f'D3_{i}', hierarchy='A3',
                      weight_vectors=REPETITION_COUNT, k=4,
                      experiment_name=f'Multi_S3A3_{i}', save_start_time=False, make_noise=True,
                      fd=f'results/PKDD {REPETITION_COUNT} Reps/bootstrap')
        print(i)


if __name__ == '__main__':
    run()
