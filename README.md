# Post-hoc Stochastic Concept Bottleneck Models
This repository contains the code for the thesis "*Post-hoc Stochastic Concept Bottleneck Models*" (SCBM).
(not yet available online)

**Abstract**: Recent studies have shown that modeling concept dependencies in Concept Bottle-
neck Models (CBM) can lead to significant improvements in performance, espe-
cially when concept interventions are made. In this paper, we introduce Post-hoc
Stochastic Concept Bottleneck Models (PSCBM), a method to augment an existing
CBM that treats concepts as independent with a multivariate normal distribution,
which can model concept correlations. Importantly, the existing model doesn’t
need to be retrained. It is only necessary to learn the covariance matrix of the
normal distribution. For this, we propose 2 methods: 1. minimizing the regular
CBM loss without interventions, 2. minimizing the same loss after an intervention
on a random subset of concepts. In addition, we propose the usage of empirical
covariance of the training dataset, which doesn’t require any training. We evaluate
our methods on the CUB-200-2011 dataset with various intervention strategies,
showing that they can achieve an improvement over a simple CBM in terms of
concept and target accuracy at test time and with interventions. We also discuss
some weaknesses of our methods.

## Instructions

1. Install the packages and dependencies in the file `environment.yml`. 
2. Download the datasets described in the manuscript and update the `data_path` variable in `./configs/data/data_defaults.yaml`. For CUB, we use the original Concept Bottleneck Model's CUB version.

3. For Weights & Biases support, set mode to 'online' and adjust entity in `./configs/config.yaml`.
4. Run the script `train.py` with the desired configuration of dataset and model from the `./configs/` folder. We provide a description of all arguments in the config files.

## Running Experiments

We provide scripts in the `./scripts/` directory to run experiments on a SLURM cluster and reproduce our results. 
