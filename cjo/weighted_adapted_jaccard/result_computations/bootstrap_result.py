import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import data.data_loader
from cjo.base.stringconstants import IEM, bootstrap_rep_count, supercat, medoid, CLUSTERING_TIME, OPTIMIZATION_TIME, \
    REPETITION, cluster_size, ITERATION, ISC_WEIGHT
from functions import tex_functions, dataframe_operations
from functions.general_functions import listified

fp_id = 'Fixed Point'


def id_generator(values):
    values = values.unique()
    foo = int(math.ceil(math.log(len(values), 10)))
    ids = [f'{x:0{foo}d}' for x in range(len(values))]
    return {v: i for v, i in zip(values, ids)}


class MultipleBootstrapResult:

    def __init__(self, fd, fd_out=None, figure_extension='svg'):
        """
        Object to parse the results of multiple bootstraps

        Parameters
        ----------
        fd: str or Path
            Location of the results of the multiple bootstraps
        fd_out: str, Path or None
            Location of the output. If None, fd/results is used.
        figure_extension: str
            Extension of saved figures
        """

        # Input / Output
        fd = Path(fd)
        assert fd.exists()

        if fd_out is None:
            self.__fd_out = fd / 'results'
        else:
            self.__fd_out = Path(fd_out)
        assert isinstance(figure_extension, str)

        self.ext = figure_extension

        # Read the general settings (common to all repetitions)
        fn_param = fd / 'settings.txt'
        sr = dataframe_operations.import_sr(fn_param, header=None)
        self.settings = dict()
        for s in ['k', 'n_max', 'dsx_size_d', 'dsx_size_r']:
            self.settings[s] = int(sr[s])

        for s in ['randomization', 'epsilon']:
            self.settings[s] = float(sr[s])

        for s in {'dataset', 'hierarchy', 'optimization_strategy', 'iem_mode'}:
            self.settings[s] = sr[s]

        self.hierarchy = data.data_loader.generic_hierarchy(self.settings['hierarchy'])
        self.h = len(self.hierarchy)
        self.weight_column_names = [ISC_WEIGHT] + listified(self.hierarchy.keys(), str)
        self.medoid_column_names = [f'{medoid} {i}' for i in range(self.settings['k'])]

        self.sbr = dict()
        df = dataframe_operations.import_df(fd / 'results.csv')
        for rep in df[REPETITION].unique():
            df_rep = df[df[REPETITION] == rep].drop(columns=REPETITION).set_index(ITERATION)
            self.sbr[rep] = BootstrapResult(df_rep, self.settings['k'])

        # TODO : refactor this to self.results = self.compute_results()
        self.results = None
        self.compute_results()

        self.summary = self.results.groupby(fp_id).mean()[[IEM] + self.weight_column_names]
        self.summary[IEM + '_std'] = self.results.groupby(fp_id).std()[IEM]
        self.summary[bootstrap_rep_count] = self.results[fp_id].value_counts()
        self.summary.sort_values(bootstrap_rep_count, ascending=False)

    ##############
    # PROPERTIES #
    ##############

    @property
    def fd_out(self):
        """
        Output directory for results. Created once requested.

        Returns
        -------
        fd_out: Path
            The folder where to put the output of this MBR. (This is usually used when a single MBR is analyzed, the
            PKDD2020 paper has one MBR per dataset to compute the mean and CI values).
        """
        self.__fd_out.mkdir(parents=True, exist_ok=True)
        return self.__fd_out

    @property
    def unique_fp_ids(self):
        """
        Names of all fixed points

        Returns
        -------
        fps: np.ndarray
            Array of fixed point names
        """
        return self.results[fp_id].unique()

    @property
    def total_duration(self):
        return self.results[CLUSTERING_TIME].sum() + self.results[OPTIMIZATION_TIME].sum()

    def compute_results(self):
        # Getting the info
        self.results = pd.DataFrame(index=list(self.sbr.keys()),
                                    columns=[f'{supercat}_{i}' for i in range(self.h + 1)] +
                                            [f'w_{i}' for i in range(self.h + 1)] +
                                            [f'm_{i}' for i in range(self.settings['k'])])
        for k, v in self.sbr.items():
            self.results.loc[k, :] = v.fixed_point_info
        for i in range(self.h + 1):
            self.results[f'w_{i}'] = self.results[f'w_{i}'].apply(lambda x: f'{x:0.2f}')

        # Sorting
        for i in range(self.settings['k'] - 1, -1, -1):
            self.results.sort_values(by=f'm_{i}', ascending=True, kind='mergesort', inplace=True)
        for i in range(self.h + 1 - 1, -1, -1):
            self.results.sort_values(by=f'w_{i}', ascending=False, kind='mergesort', inplace=True)
        for i in range(self.h + 1 - 1, -1, -1):
            self.results.sort_values(by=f'{supercat}_{i}', ascending=True, kind='mergesort', inplace=True)

        self.results['fp_val_a'] = self.results.apply(
            lambda r: ''.join(r[i] for i in r.index if i.startswith(supercat)), axis=1)
        self.results['fp_val_b'] = self.results.apply(
            lambda r: ''.join(r[i] for i in r.index if i.startswith('w_')), axis=1)
        self.results['fp_val_c'] = self.results.apply(
            lambda r: ','.join(str(r[i]) for i in r.index if i.startswith('m_')), axis=1)

        # Rename and combine
        for j in 'abc':
            self.results[f'{fp_id}_{j}'] = self.results[f'fp_val_{j}'].replace(
                id_generator(self.results[f'fp_val_{j}']))
            self.results.drop(columns=f'fp_val_{j}', inplace=True)
        self.results[f'{fp_id}_b'] = self.results[f'{fp_id}_a'] + '.' + self.results[f'{fp_id}_b']
        self.results[f'{fp_id}_c'] = self.results[f'{fp_id}_b'] + '.' + self.results[f'{fp_id}_c']

        self.results[fp_id] = self.results[f'{fp_id}_b'].replace(
            {v: e + 1 for e, v in enumerate(self.results[f'{fp_id}_b'].value_counts().index)}).astype(int)
        self.results = self.results[[f'{fp_id}_{j}' for j in 'abc'] + [fp_id]]

        keys = list(self.sbr.keys())
        self.results[IEM] = pd.Series(data={k: v.final_iem for k, v in self.sbr.items()})
        self.results['n*'] = pd.Series(data={k: v.n_star for k, v in self.sbr.items()})
        self.results[CLUSTERING_TIME] = pd.Series(data={k: v.total_cluster_time for k, v in self.sbr.items()})
        self.results[OPTIMIZATION_TIME] = pd.Series(data={k: v.total_optimization_time for k, v in self.sbr.items()})

        weights = pd.DataFrame(data=[self.sbr[k].final_weights for k in keys], index=keys)
        medoids = pd.DataFrame(data=[self.sbr[k].final_medoids for k in keys], index=keys)
        self.results = pd.concat([self.results, medoids, weights], axis=1, sort=False)
        self.results.sort_values(by=f'{fp_id}_c', inplace=True)

    def __len__(self):
        return len(self.results)

    def __str__(self):
        return f'MBR[{len(self)} / {self.settings["dataset"]} / {self.settings["hierarchy"]}]'

    def __repr__(self):
        return str(self)

    @property
    def super_categories(self):
        return sorted([ISC_WEIGHT] + list(self.hierarchy.keys()))

    #########################
    # (FP-SPECIFIC) GETTERS #
    #########################

    def get_repetition_names(self, fp):
        """
        Returns the repetition names for a given Fixed Point.

        Parameters
        ----------
        fp: int or None
            The fixed point to be queried. If None, all repetition_names are returned.

        Returns
        -------
        index: pd.Index
            The repetition names, optionally only the ones of the given fixed point.

        """
        if fp is None:
            return self.results.index
        else:
            return self.results[self.results[fp_id] == fp].index

    def get_final_weights(self, fp=None):
        """
        Returns all final weights (of a specific fixed point).

        Parameters
        ----------
        fp: int or None
            Fixed Point for which to get the results. If None, all final weights are returned

        Returns
        -------
        final_weights: pd.DataFrame
            DataFrame with all final weights (of the given fixed point).

        """
        return self.results.loc[self.get_repetition_names(fp)][self.weight_column_names]

    def get_average_final_weight(self, fp=None):
        """
        Gets the average final weight (of a variant).

        Parameters
        ----------
        fp: int or None
            If not None, get the average for this fixed point. Otherwise get the global average.

        Returns
        -------
        w: pd.Series
        """
        return self.get_final_weights(fp).mean()

    def get_iem(self, fp=None):
        """
        Get the IEM values (for a given fixed point)

        Parameters
        ----------
        fp: int or None
            If not None, the IEM values of this fixed point are returned. Otherwise get all IEMs

        Returns
        -------

        """
        return self.results.loc[self.get_repetition_names(fp), IEM]

    def get_a_final_medoid(self, variant):
        return self.results.loc[self.get_repetition_names(variant)].iloc[0][
            [f'Medoid {i}' for i in range(self.settings['k'])]]

    def get_sbr(self, fp=None):
        """
        Returns all Single Bootstrap Results (for the given fixed point).

        Parameters
        ----------
        fp : int
            Fixed Point for which to get the sbr. If None, all sbr are returned

        Returns
        -------
        sbr : dict of str to SingleBootstrapResult
            The repetition names with their SingleBootstrapResult (of the given fixed points)

        """
        return {k: self.sbr[k] for k in self.get_repetition_names(fp)}

    def get_results(self, fp=None):
        return self.results.loc[self.get_repetition_names(fp)]

    #########
    # PLOTS #
    #########

    def __plot_final_averages(self, ax=None, repetitions=None):
        """
        Plot and save the average and std of each weight over given repetitions.

        Parameters
        ----------
        repetitions: iterable of str or None
            The names of the repetitions to plot. If None, all are plot
        ax: plt.Axes
            The Axes object to plot in
        """
        if repetitions is None:
            repetitions = self.results.index

        assert isinstance(ax, plt.Axes)
        y = self.results.loc[repetitions].mean()[self.weight_column_names].values
        if len(repetitions) == 1:
            ax.bar(x=range(len(self.weight_column_names)), height=y)
            ax.set_ylim(0, max(y) * 1.1)
        else:
            y_err = self.results.loc[repetitions].std()[self.weight_column_names].values
            ax.bar(x=range(len(self.weight_column_names)), height=y, yerr=y_err)
            ax.set_ylim(0, max(y + y_err) * 1.1)
        ax.set_xticks(list(range(len(self.weight_column_names))))
        ax.set_xticklabels(self.weight_column_names, rotation=20)

    def plot_averages(self):
        """
        Plot and save the average and std of each weight over all repetitions.
        """
        f, ax = plt.subplots(1)
        self.__plot_final_averages(ax=ax)
        ax.set_title(f'Average weight over {len(self)} initializations')
        f.set_size_inches(10, 7)
        plt.savefig(self.fd_out / f'Averages.{self.ext}', bbox_inches='tight')

    def plot_averages_per_fp(self, separate=False):
        """
        Plot the average and std per fixed point.

        Parameters
        ----------
        separate: bool
            If True, each plot is saved separately, otherwise, one big plot is made.
        """

        if separate:
            for fp in self.unique_fp_ids:
                f, ax = plt.subplots(1)
                self.__plot_average_of_a_fp(fp, ax)
                f.set_size_inches(10, 7)
                plt.savefig(self.fd_out / f'Averages_fp_{fp}.{self.ext}', bbox_inches='tight')
                plt.close(f)
        else:
            f, axarr = plt.subplots(len(self.unique_fp_ids), 1)
            for dominant_sc, ax in zip(self.unique_fp_ids, axarr):
                self.__plot_average_of_a_fp(dominant_sc, ax)
            f.set_size_inches(10, 7 * len(self.unique_fp_ids))
            plt.savefig(self.fd_out / f'AveragesAll_fp.{self.ext}', bbox_inches='tight')
            plt.close(f)

    def plot_distributions(self):
        """
        Plot and save the distribution each weight over all repetitions.
        """
        c, r = 2, math.ceil(len(self.weight_column_names) / 2)
        f, axarr = plt.subplots(c, r)
        axarr = axarr.flatten()
        for sc, ax in zip(self.weight_column_names, axarr):
            assert isinstance(ax, plt.Axes)
            foo, _, _ = ax.hist(self.results[sc].values, bins=np.arange(0, 1.01, 0.01))
            ax.set_xticks([0, 0.5, 1.0])
            ax.text(0.5, max(foo) * 0.9, sc, ha='center')
        assert isinstance(f, plt.Figure)

        f.suptitle('Distribution weight in final iteration')
        f.set_size_inches(5 * c, 3 * r)
        plt.savefig(self.fd_out / f'Distributions.{self.ext}', bbox_inches='tight')
        plt.close(f)

    def plot_fixed_points(self):
        """
        Plots all repetitions, sorted on fixed points.
        """
        ###############
        # ACTUAL PLOT #
        ###############

        f, ax = plt.subplots(1)
        lefts = pd.Series(0, index=self.results.index)
        for sc in self.weight_column_names:
            ax.barh(range(len(self) - 1, -1, -1), width=self.results[sc], height=1.0, left=lefts, label=sc)
            lefts += self.results[sc]
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=True, ncol=(self.h + 2) // 2)
        # TODO Add short names
        # OPTIMIZE speed, picture size: use summary to make the plots, not the actual runs

        ########
        # AXES #
        ########

        down = len(self)
        pos_y = []
        str_y = []
        for fixed_point_id in self.results[fp_id].unique():
            fp_count = self.summary.loc[fixed_point_id, bootstrap_rep_count]
            up = down
            down -= fp_count

            # TODO hard coded: You are only labelling if more than 3.33% of the total repetitions is this fixed point
            if fp_count > len(self) // 33:
                m = (down + up) / 2
                pos_y.append(m)
                str_y.append(f'$\\Lambda_{{{fixed_point_id}}}$')
                ax.plot([0, 1], [up - 0.5] * 2, 'k:', lw=1)
                ax.plot([0, 1], [down - 0.5] * 2, 'k:', lw=1)

        # add boxes and names
        ax.set_yticks(pos_y)
        ax.set_yticklabels(str_y)
        ax.tick_params(axis='y', which='both', length=0)

        ax.set_ylim(-0.5, len(self) - 0.5)
        ax.set_xlim(0, 1)
        ax.set_xticks([])

        ###############
        # SAVE FIGURE #
        ###############

        ax.set_xlabel('Weight')
        f.set_size_inches(5, 10)
        plt.savefig(self.fd_out / f'FixedPoints.{self.ext}', bbox_inches='tight')
        plt.close(f)

    def __plot_average_of_a_fp(self, fp, ax):
        """
        Plot the average weights of a single fixed_point

        Parameters
        ----------
        fp: int
            fixed_point to be plot
        ax: plt.Axes
            Artist to be plot in
        """
        idx = self.get_repetition_names(fp)
        self.__plot_final_averages(repetitions=idx, ax=ax)
        ax.set_title(f'Average weight over {len(idx)} initializations, where {fp} is dominant.')

    def make_all_plots(self):
        """
        Make all plots.
        """
        # self.plot_averages_per_fp(True)
        # self.plot_averages_per_fp()
        self.plot_distributions()
        self.plot_averages()
        self.plot_fixed_points()

    ##############
    # TEX TABLES #
    ##############

    def create_fixed_point_tex_table(self, top=5, ci=None, **kwargs):
        """
        Creates a tex file with a table that summarizes the variants.

        Parameters
        ----------
        top: int or None
            Number of variants to show. If None, all are shown
        ci: float or None
            If float, take this as confidence interval value. If None, skip confidence interval

        Other Parameters
        ----------------
        kwargs: dict
            Parameters that are passed to cjo.functions.TexFunctions.df_to_table

        Raises
        ------
        AssertionError
            If the confidence level is not in [0,1]
        """
        if top is None:
            top = len(self.unique_fp_ids)
        df = self.summary.head(top).copy()
        if ci is not None:
            assert 0 <= ci <= 1, 'ci must be in [0,1]'
            z = stats.norm.interval(ci, 0, 1)[1]
            ci_value = df[IEM + '_std'] / df[bootstrap_rep_count].apply(lambda n: n ** 0.5) * z
            df[IEM] = df[IEM].apply(lambda m: f'{m:.2f}') + '$\\pm$' + ci_value.apply(lambda c: f'{c:.2f}')

        df.drop(columns=IEM + '_std', inplace=True)

        df.rename(columns={
            IEM: r'$\Phi$',
        }, inplace=True)
        df = df.reset_index(drop=False).set_index([fp_id, bootstrap_rep_count, r'$\Phi$'])

        for i, r in df.iterrows():
            max_weight = r.idxmax()
            for sc, w in r.iteritems():
                df.loc[i, sc] = f'{df.loc[i, sc]:.2f}'
                if sc == max_weight:
                    df.loc[i, sc] = f'\\textbf{{{df.loc[i, sc]}}}'
        kwargs.setdefault('label', 'tab:res:variant')
        kwargs.setdefault('caption', 'Weights of most frequent variants, showing the number of repetitions and IEM. '
                                     'The highest weight in each variant is highlighted.')
        kwargs.setdefault('escape', False)
        tex_functions.df_to_table(
            df.T, fn_out=self.fd_out / 'variants_table.tex', **kwargs)

    ############
    # PRINTERS #
    ############

    def print_variants_summary(self, top=5):
        """

        Parameters
        ----------
        top: int or None
            Number of variants to show. If None, all are shown

        Prints the summary of all variants on screen.
        """
        if top is None:
            top = len(self.unique_fp_ids)
        print(self.summary.head(top).to_string())

    @property
    def the_fixed_point(self):
        """*The* fixed point"""
        sr = self.get_results(fp=1).sort_values('IEM', ascending=True).iloc[0]
        return sr[self.weight_column_names], sr[self.medoid_column_names]

    @property
    def the_iem(self):
        """the IEM of *The* fixed point"""
        return self.get_results(fp=1)['IEM'].min()


class BootstrapResult:

    def __init__(self, df, k):
        """
        Loads the results of a *single* bootstrap repetition

        Parameters
        ----------
        df: pd.DataFrame
            Results DataFrame (generated by MultipleBootstrapResult)
        k: int
            Number of clusters (this is saved at MultipleBootstrapResult level)
        """

        medoid_cols = [f'{medoid} {i}' for i in range(0, k)]
        self.medoids = pd.DataFrame()
        for c in medoid_cols:
            self.medoids[c] = df[c].astype('int64')
        size_cols = [f'{cluster_size} {i}' for i in range(0, k)]
        self.cluster_sizes = pd.DataFrame()
        for c in size_cols:
            self.cluster_sizes[c] = df[c].astype(int)

        duration_cols = [OPTIMIZATION_TIME, CLUSTERING_TIME]
        self.durations = df[duration_cols]

        iem_cols = [IEM]
        self.iem_score = df[IEM]

        non_weight_cols = medoid_cols + size_cols + duration_cols + iem_cols
        weight_cols = [c for c in df.columns if c not in non_weight_cols]
        self.weights = df[weight_cols]

    def __str__(self):
        return f'SBR[{self.n_star} iterations]'

    def __repr__(self):
        return str(self)

    ##############
    # PROPERTIES #
    ##############

    @property
    def final_weights(self):
        """The final weights of the repetition"""
        return self.weights.iloc[-1]

    @property
    def final_iem(self):
        """The last computed value of :math:`\\Phi`"""
        return self.iem_score.iloc[-1]

    @property
    def final_medoids(self):
        """The last computed medoids"""
        return self.medoids.iloc[-1]

    @property
    def n_star(self):
        """The final iteration round"""
        return self.weights.index.max()

    @property
    def total_cluster_time(self):
        return self.durations[CLUSTERING_TIME].sum()

    @property
    def total_optimization_time(self):
        return self.durations[OPTIMIZATION_TIME].sum()

    @property
    def fixed_point_info(self):
        """All required information to describe the fixed point of this single bootstrap repetition"""
        final_medoids = listified(self.final_medoids, int, sort=True, validation=lambda x: x > 0)
        f = self.final_weights.sort_values(ascending=False)
        return pd.Series(data={**{f'{supercat}_{i}': v for i, v in enumerate(f.index)},
                               **{f'w_{i}': v for i, v in enumerate(f.values)},
                               **{f'm_{i}': v for i, v in enumerate(final_medoids)}})
