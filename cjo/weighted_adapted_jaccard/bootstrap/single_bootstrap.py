import itertools
import time
from numbers import Number
from pathlib import Path

import numpy as np
import pandas as pd

from cjo.base.stringconstants import OPTIMIZATION_TIME, CLUSTERING_TIME, StartReceipts, \
    StartDSX, StartWeights, StartDefault, IEM, medoid, cluster_size, ITERATION
from functions import dataframe_operations
from functions.general_functions import listified, large_number_formatting, assert_valid_partition
from functions.message_services import CompositeMessageService, PrintMessageService, FileMessageService
from cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers import DSXWeightsOptimizer, IEMFactory, DSXClustering
from cjo.weighted_adapted_jaccard.distances.implementation import DSX


class Bootstrap:

    def __init__(self, dsx, optimizer, iem_factory, clustering_algorithm,  # settings
                 epsilon, n_max, randomization,  # termination control
                 k, fd_results, seed=0,
                 initial_medoids=None, initial_clustering=None, initial_weights=None,  # initialization
                 make_noise=False, ):
        """
        Solves the bootstrap problem, given a DSX, optimization strategy, and iem_factory. The algorithm is executed
        by calling :meth:process

        Parameters
        ----------
        dsx: DSX
            The full dataset as DSX object
        optimizer: DSXWeightsOptimizer
            Optimization strategy.
        iem_factory: IEMFactory
            Factory for the IEM function
        fd_results: str or Path
            Location to save result

        n_max: int
            see :py:meth:`multiple bootstrap <cjo.weighted_adapted_jaccard.bootstrap.multiple_bootstrap.run_bootstrap>`
        epsilon: Number
            see :py:meth:`multiple bootstrap <cjo.weighted_adapted_jaccard.bootstrap.multiple_bootstrap.run_bootstrap>`
        randomization: float
            see :py:meth:`multiple bootstrap <cjo.weighted_adapted_jaccard.bootstrap.multiple_bootstrap.run_bootstrap>`
        k: int
            see :py:meth:`multiple bootstrap <cjo.weighted_adapted_jaccard.bootstrap.multiple_bootstrap.run_bootstrap>`

        Other Parameters
        ----------------
        initial_weights: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
        initial_clustering: iterable of DSX, iterable of iterable of int, or None
            The initial clustering, either as an iterable of DSX, or otherwise as an in iterable of receipt integers.
            See Notes.
        initial_medoids: sized of int, or None
            Receipt integers representing the initial medoids. See Notes.

        make_noise: bool
            Whether to print the output
        seed: int
            The seed for numpy. This allows a reset on every run of bootstrap.

        Notes
        -----
            If the initial_weights are not given; a weight of 1 for each super-category is assumed. If the
            initial_clustering is given; the initial_clustering and the initial_medoids are used as-is in the first
            round of the bootstrap. If the initial_medoids are given but not the initial_clustering; the initial_medoids
            are used with the given or default weight vector to compute the initial_clustering.

        Raises
        ------
        AssertionError
            If randomization is negative.
        """
        ###########################
        # MAIN INPUT VERIFICATION #
        ###########################

        assert isinstance(dsx, DSX)
        self.dsx = dsx
        assert isinstance(optimizer, DSXWeightsOptimizer)
        self.optimizer = optimizer
        assert isinstance(iem_factory, IEMFactory)
        self.iem_factory = iem_factory
        assert isinstance(clustering_algorithm, DSXClustering)
        self.clustering_algorithm = clustering_algorithm

        #####################
        # BOOTSTRAP CONTROL #
        #####################
        if isinstance(epsilon, Number):
            epsilon = np.array([epsilon] * (dsx.h + 1))
        assert isinstance(epsilon, np.ndarray)
        assert len(epsilon) == dsx.h + 1
        self.epsilon = epsilon

        assert isinstance(n_max, int)
        self.n_max = n_max

        assert randomization >= 0, 'Randomization should be non-negative'
        self.randomization = randomization

        ###############################
        # INITIALIZATION OF BOOTSTRAP #
        ###############################
        self.initial_clustering = initial_clustering
        self.initial_weights = initial_weights
        self.initial_medoids = initial_medoids

        #######################
        # ADDITIONAL SETTINGS #
        #######################
        assert isinstance(k, int)
        self.k = k

        if fd_results is None:
            self.fd = None
        else:
            self.fd = Path(fd_results)
            self.fd.mkdir(parents=True, exist_ok=True)

        assert isinstance(seed, int)
        self.seed = seed

        msg_service = CompositeMessageService()
        if make_noise:
            msg_service.add_msg_service(PrintMessageService())
        if self.fd is not None:
            msg_service.add_msg_service(FileMessageService(fn=self.fd / 'log.txt'))
        self.msg_service = msg_service

        ###########################
        # INITIALIZE SAVE OBJECTS #
        ###########################
        # create a data object
        self.weights = pd.DataFrame(columns=self.dsx.hierarchy_names)
        self.cluster_sizes = pd.DataFrame(columns=range(self.k))
        self.clusters = pd.DataFrame(columns=self.dsx.keys())
        self.clusters.index.name = ITERATION
        self.medoids = pd.DataFrame(columns=range(self.k))
        self.durations = pd.DataFrame(columns=[OPTIMIZATION_TIME, CLUSTERING_TIME])
        self.iem_score = pd.Series()
        self.cycle = None
        # Verify calls to methods
        self.expected = (None, None)

    ############################################################
    # Parsing the finishing of specific steps of the bootstrap #
    ############################################################

    def finish_initialization(self, weights, medoids, clusters, initialization_type):
        """
        Saves the results from the initialization

        Parameters
        ----------
        weights: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
        medoids: iterable of int
            The medoids after initialization
        clusters: iterable of DSX
            The clusters after initialization
        initialization_type: str
            How the initialization was done. Should be one of

            - BootstrapResults.StartReceipts
            - BootstrapResults.StartDSX
            - BootstrapResults.StartWeights
            - BootstrapResults.StartDefault

        """

        assert self.expected[0] is None, 'Already initialized'
        self.expected = (1, True)

        # Validation and conversion
        weights = self.dsx.get_named_weight_vector(weights)

        assert len(medoids) == self.k
        medoids = listified(medoids, int)

        assert len(clusters) == self.k
        clusters = listified(clusters, DSX)

        # Store information
        for k, v in weights.items():
            self.weights.loc[0, k] = v
        for lab, (med, cluster) in enumerate(zip(medoids, clusters)):
            self.medoids.loc[0, lab] = med
            self.cluster_sizes.loc[0, lab] = cluster.size_r
            self.clusters.loc[0, cluster.keys()] = lab

        # print results
        self.msg_service.send_message(initialization_type)
        self.msg_service.send_message('*' * len(initialization_type))
        self.__weights_print(weights, 'Initial weights')
        self.__cluster_print(weights, medoids, clusters, 'Initial clusters')
        s = f'Iteration {1}'
        self.msg_service.send_message(f'\n{s}\n{"-" * len(s)}')

    def finish_weights_step(self, weights, iem_score, current_iteration, duration):
        """

        Parameters
        ----------
        weights: dict of str to Number, iterable of Number
            The weights to be used. See :py:meth:`validate_and_convert_weights()
            <cjo.weighted_adapted_jaccard.distances.implementation.DSX.validate_and_convert_weights>`.
        iem_score: Number
            Value of the IEM function of the given weights on the given clusters.
        current_iteration: int
            The iteration number of this round
        duration: float
            The (process) time in seconds for this weights step

        """
        assert self.expected == (current_iteration, True)
        self.expected = (current_iteration, False)

        # Save weights and iem_score
        for previous_iteration, v in self.dsx.get_named_weight_vector(weights).items():
            self.weights.loc[current_iteration, previous_iteration] = v
        self.iem_score.loc[current_iteration] = iem_score
        assert isinstance(duration, float)
        self.durations.loc[current_iteration, OPTIMIZATION_TIME] = duration

        self.__weights_print(weights=weights, title='New weights', iem_score=iem_score, duration=duration)

        # Check if there is some cyclic pattern
        for previous_iteration in range(1, current_iteration):
            if self.check_same(current_iteration, previous_iteration):
                cycle_length = current_iteration - previous_iteration

                # Save the cluster sizes and medoids since we have repetition
                self.cluster_sizes.loc[current_iteration, :] = self.cluster_sizes.loc[previous_iteration, :]
                self.medoids.loc[current_iteration, :] = self.medoids.loc[previous_iteration, :]

                # Cycle length of 1 indicates that the weights did not change
                if cycle_length == 1:
                    self.cycle = 1
                    self.terminate()
                    return True

                # Cycle length of more than 1 might indicate a cycle
                # so we check -1, -2 ,-3, ... -(cycle_length+1)
                for j in range(1, cycle_length):
                    if not self.check_same(current_iteration - j, previous_iteration - j):
                        break
                else:
                    self.cycle = current_iteration - previous_iteration
                    self.terminate()
                    return True
        return False

    def check_same(self, iter1, iter2):
        """
        Checks whether two iterations are the same, in that they have near-same (epsilon) weights, and the same
        previous medoids
        """
        if np.all(np.abs(self.weights.loc[iter2].values - self.weights.loc[iter1].values) < self.epsilon):
            # We find a set of weights that is identical to a previous run
            # We therefore check if the medoids are the same, to see if we would get stuck in a loop
            if set(self.medoids.loc[iter2 - 1].values) == set(self.medoids.loc[iter1 - 1].values):
                return True
        return False

    def finish_cluster_step(self, medoids, clusters, current_iteration, duration):
        """
        Save the new medoids and clusters.

        Parameters
        ----------
        medoids: iterable of int
            New medoids
        clusters: iterable of DSX
            New clusters
        current_iteration: int
            The iteration number of this round
        duration: float
            The (process) time in seconds taken for the clustering.

        Notes
        -----
        Saves the medoids and the size of the clusters. Each cluster is assigned to a
        'label'. The labels are initialized in :meth:`finish_initialization` . In subsequent calls, each of the clusters
        is assigned to a label as follows. For all
        combinations of clusters and labels, the distance between the medoids of the new cluster and the medoid of the
        cluster for that label is computed. For combination (cluster, label) that has the lowest distance, we assign the
        cluster to the label. We remove the cluster and label from the options and repeat this assigning another n-1
        times, for a total of n times, such that each cluster is assigned to a label.
        """

        assert self.expected == (current_iteration, False)
        self.expected = (current_iteration + 1, True)

        medoids = listified(medoids, int)
        assert len(medoids) == self.k

        clusters = listified(clusters, DSX)
        assert len(clusters) == self.k

        assert isinstance(duration, float)
        self.durations.loc[current_iteration, CLUSTERING_TIME] = duration

        medoid_clusters = {k: v for k, v in zip(medoids, clusters)}

        weights = self.weights.loc[current_iteration].to_dict()

        # distance function between new medoids and existing labels
        def d(e):
            new_medoid = int(e[0])
            old_medoid = int(self.medoids.loc[current_iteration - 1, e[1]])
            return self.dsx.get_multiplicity_distance_function(weights)(new_medoid, old_medoid)

        # Match new medoids with previous medoids
        unlabelled_medoids = set(medoids)
        available_labels = list(range(self.k))

        while len(unlabelled_medoids) > 0:
            med, label = min(itertools.product(unlabelled_medoids, available_labels), key=d)
            self.medoids.loc[current_iteration, label] = med
            self.cluster_sizes.loc[current_iteration, label] = medoid_clusters[med].size_r
            self.clusters.loc[current_iteration, medoid_clusters[med].keys()] = int(label)
            unlabelled_medoids.remove(med)
            available_labels.remove(label)

        self.__cluster_print(weights=weights, medoids=medoids, clusters=clusters, title='New clusters',
                             duration=duration)

        # Maximum number of iterations reached?
        if current_iteration == self.n_max:
            self.cycle = -1
            self.terminate()
            return True
        else:
            s = f'Iteration {current_iteration + 1}'
            self.msg_service.send_message(f'\n{s}\n{"-" * len(s)}')
            return False

    ##############################
    # Giving updates to the user #
    ##############################

    def __weights_print(self, weights, title, iem_score=None, formatting=lambda x: f'{x:.2f}', duration=None):
        self.msg_service.send_message(title)
        weights = self.dsx.get_named_weight_vector(weights)
        for k, v in weights.items():
            self.msg_service.send_message(f'\t{k} : {formatting(v)}')
        if iem_score is not None:
            self.msg_service.send_message(f'{IEM} : {formatting(iem_score)}')
        if duration is not None:
            self.msg_service.send_message(f'Total duration = {duration:.2f} s')

    def __cluster_print(self, weights, medoids, clusters, title, formatting=large_number_formatting, duration=None):
        self.msg_service.send_message(title)
        for m, c in zip(medoids, clusters):
            assert isinstance(c, DSX)
            self.msg_service.send_message(f'\t{str(c)}. {medoid} at {m}')
        total_distance = sum([cluster_i.get_sum(weights) for cluster_i in clusters])
        self.msg_service.send_message(f'Total distance = {formatting(total_distance)}')
        if duration is not None:
            self.msg_service.send_message(f'Total duration = {duration:.2f} s')

    ############################
    # TERMINATION OF BOOTSTRAP #
    ############################

    def terminate(self):
        """
        Finish the algorithm
        """
        if self.cycle is -1:
            self.msg_service.send_message(f'Ended because of max iterations\n\n')
        elif self.cycle == 1:
            self.msg_service.send_message(f'Ended because of fixed point\n\n')
        elif self.cycle > 0:
            self.msg_service.send_message(f'Ended because of {self.cycle}-cycle\n\n')
        else:
            self.msg_service.send_message(f'Ended because of unknown reasons\n\n')

        # save iteration fixed point values
        df = pd.concat([self.weights,
                        self.medoids.rename(columns=lambda i: f'{medoid} {i}'),
                        self.cluster_sizes.rename(columns=lambda i: f'{cluster_size} {i}'),
                        self.iem_score.to_frame('IEM'),
                        self.durations],
                       axis=1)
        df.index.name = ITERATION
        dataframe_operations.export_df(df, self.fd / 'results.csv', index=True)

        # Save cycle value
        with open(self.fd / 'cycle.txt', 'w+') as wf:
            wf.write(f'{self.cycle}')

    def process(self):
        """
        Runs the bootstrap, as initialized
        """
        np.random.seed(self.seed)
        #################################################
        # Clusters, Medoids, and Weights Initialization #
        #################################################

        # Try to initialize from clustering
        if self.initial_clustering is not None:
            if not all([isinstance(dsx_i, DSX) for dsx_i in self.initial_clustering]):
                init_by = StartReceipts
                assert_valid_partition(set(self.dsx.keys()), [set(ici) for ici in self.initial_clustering])
                clusters = [self.dsx.get_subset(ici) for ici in self.initial_clustering]
                medoids = self.initial_medoids
            else:
                init_by = StartDSX
                clusters = self.initial_clustering
                medoids = self.initial_medoids
            previous_weights = self.dsx.unit_weight_vector()

        # Try to initialize from weights
        elif self.initial_weights is not None:
            init_by = StartWeights
            previous_weights = self.dsx.validate_and_convert_weights(self.initial_weights)
            previous_weights /= sum(previous_weights)
            medoids, clusters = self.clustering_algorithm.cluster(dsx=self.dsx, weights=previous_weights,
                                                                  initial_medoids=self.initial_medoids, k=self.k)
        else:
            # Fall-through: initialize with unit weights
            # both are None, so revert back to the default of equal weights and find first clustering
            init_by = StartDefault
            previous_weights = self.dsx.unit_weight_vector()
            # TODO this is not working anymore?
            previous_weights /= sum(previous_weights)
            medoids, clusters = self.clustering_algorithm.cluster(self.dsx, weights=previous_weights, k=self.k)

        # Save initialization
        self.finish_initialization(previous_weights, medoids, clusters, init_by)

        ####################
        # WEIGHT ADAPTIONS #
        ####################

        def normalize(weights):
            return weights / sum(weights)

        def randomize(weights):
            return weights + (np.random.rand(len(weights)) * self.randomization)

        ########################
        # Actual bootstrapping #
        ########################

        # Do iteration
        iteration_number = 0
        while True:
            iteration_number += 1

            # Find the weights that minimize the IEM
            iem = self.iem_factory.create_function(self.dsx, clusters)
            t_start = time.process_time()
            new_weights = normalize(self.optimizer.get_optimal_weights(iem, seed=0))
            t_end = time.process_time()
            if self.finish_weights_step(new_weights, iem(new_weights), iteration_number, t_end - t_start):
                break

            # Find the medoid-clustering for the given weights
            t_start = time.process_time()
            medoids, clusters = self.clustering_algorithm.cluster(dsx=self.dsx, weights=randomize(new_weights),
                                                                  initial_medoids=medoids)

            t_end = time.process_time()
            if self.finish_cluster_step(medoids, clusters, iteration_number, t_end - t_start):
                break

        return self
