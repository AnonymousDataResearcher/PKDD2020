Distance Metric
***************

This package implements the distance metric from the PKDD 2020 paper.

Implementation of the Distance Metric
=====================================
The module implementing the weighted adapted jaccard metric.

.. automodule:: cjo.weighted_adapted_jaccard.distances.implementation
   :members:

Bit Operations
==============
The module implementing several bit functions used by, amongst others, the
:mod:`cjo.weighted_adapted_jaccard.distances.implementation` module.

.. automodule:: cjo.weighted_adapted_jaccard.distances.bitops
   :members:

Naive Implementation
====================
The module with a naive implementation of the weighted adapted jaccard metric. In fact, this is the closest we have to
an implementation without optimizations. It is meant for verification, and it is not recommended to use it for large
(more than 1000 datapoints) datasets.

.. automodule:: cjo.weighted_adapted_jaccard.distances.naive_implementation
   :members:

Bootstrap Helpers
=================
This module contains several functions that help in the bootstrapping.

.. automodule:: cjo.weighted_adapted_jaccard.bootstrap.bootstraphelpers
    :members:

Single Bootstrap
================
This module is responsible for executing a single repetition in the bootstrap

.. automodule:: cjo.weighted_adapted_jaccard.bootstrap.single_bootstrap
    :members:

Multiple Bootstrap
==================
This module is responsible for executing multiple repetitions in the bootstrap

.. automodule:: cjo.weighted_adapted_jaccard.bootstrap.multiple_bootstrap
    :members:

Bootstrap Result
================
This module is responsible for combining the results of multiple repetitions in the bootstrap

.. automodule:: cjo.weighted_adapted_jaccard.result_computations.bootstrap_result
    :members:


Cluster Analysis
================
This module is responsible for analyzing clusters

.. automodule:: cjo.weighted_adapted_jaccard.result_computations.cluster_analysis
    :members:
