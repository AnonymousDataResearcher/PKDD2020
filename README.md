PKDD 2020 Submission
====================
This is a temporary repo that provides the data and code that accompanies the paper "Why did my Consumer Shop? Learning an
Efficient Distance Metric for RetailerTransaction Data" submitted to PKDD 2020 in the Applied Data Science track.

Reproduction
============
The results can be reproduced in full, from scratch by executing "scripts/PKDD_EXPERIMENTS/ALL_EXPERIMENTS.py" from the root directory. The lion's share of the total execution time comes 100000 executions of algorithm 1; you can alternatively use "scripts/PKDD_EXPERIMENTS/POST_BOOTSTRAP.py". Doing so will load the pre-computed results of this first step.

Both scripts will produce, amongst others, produce the following files:

results/PKDD 1000 Reps/EXPERIMENT_1/weights.tex
results/PKDD 1000 Reps/EXPERIMENT_2/cluster_analysis.tex
results/PKDD 1000 Reps/EXPERIMENT_3/competitor_analysis.tex

which are the LaTeX source of tables 5 through 7 in the paper.

Code
====
The paper does not discuss the implementation, nor several speed-ups applied to the computation. The mathematics behind this can be found as a stand-alone document including several example calculations in the document "implementation_guide.tex". This material is meant as explanation of how the paper is implemented: showing the code computes what is discussed in the paper.

Data
====
The transaction data used in the experiments is located in "data/public/retailer/D3". It contains three files, each containing 50000 transactions, using the representation discussed in the paper. The 100 datasets of 1000 datapoints each are read from these files; this is done by the script "data/data_loader.py" (to prevent 100 separate files). Each row contains one visit of a consumer to the store; the first column is a unique anonymous identifier for the visit, each subsequent column is one of the categories, indicated by the first row.

Parameters
==========
The most important parameter that we have varied in the initial experiments is the number of repetitions of Algorithm 1 per datasets (i.e. the values of 'r' in Figure 5. You can change this value in "scripts/PKDD_EXPERIMENTS/PKDD_PARAMETERS.py". It is set to 1000 as default (as per the paper), but can be changed to any other value. The POST_BOOTSTRAP.py script that uses precomputed results will however only work if it is set to 1000 or 100. The results will match with the paper if it is set to 1000.

There are other parameters in the paper and code; please see "cjo/weighted_adapted_jaccard/bootstrap/multiple_bootstrap.py" for this.

Documentation
=============
The documentation of the code can be generated using the command "make html" from the docs directory, using the Sphinx package.

Dependencies
============
The software is build on Python 3.7.5, the external dependencies are indicated in requirements.txt

Known Issues
============
The code uses the string 'c0' for the inter-super-category weight, instead of 'w_s'. This is a legacy issue that will be fixed once the code goes public on a non-anonymous github. This is corrected in the paper; so that will be different in Table 5.

The Spectral Clustering competitor in the paper was implemented without random_state. This issue was discovered (and corrected) after the paper was handed in; as such the spectral clustering results might be slightly different, though the diffrence is generally smaller than the presented precision in the Table 7.

As this code is part of a project that is being executed; there are several improvement ideas scattered in the code as TODO's. However, none of these invalidate the results presented in the paper.
