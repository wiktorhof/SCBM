# Post-hoc Stochastic Concept Bottleneck Models
This repository contains the code for the thesis "*Post-hoc Stochastic Concept Bottleneck Models*" (SCBM).
(not yet available online)

**Abstract**: Concept Bottleneck Models (CBMs) are interpretable models that predict the target variable through high-level human-understandable concepts, allowing users to intervene on mispredicted concepts to adjust the final output. While recent work has shown that modeling dependencies between concepts can improve CBM performance—especially under interventions—such approaches typically require retraining the entire model, which may be infeasible when access to the original data or compute is limited. In this paper, we introduce Post-hoc Stochastic Concept Bottleneck Models (PSCBMs), a lightweight method that augments any pre-trained CBM with a multivariate normal distribution over concepts by adding only a small covariance-prediction module, without retraining the backbone model. We propose two training strategies and show on real-world data that PSCBMs consistently match or improve both concept and target accuracy over standard CBMs at test time. Furthermore, we show that due to the modeling of concept dependencies, PSCBMs perform much better than CBMs under interventions, while remaining far more efficient than retraining a similar stochastic model from scratch.

## Instructions

1. Install the packages and dependencies in the file `environment.yml`. 
2. Download the datasets described in the manuscript and update the `data_path` variable in `./configs/data/data_defaults.yaml`. For CUB, we use the original Concept Bottleneck Model's CUB version.

3. For Weights & Biases support, set mode to 'online' and adjust entity in `./configs/config.yaml`.
4. Run the script `train.py` with the desired configuration of dataset and model from the `./configs/` folder. We provide a description of all arguments in the config files.

## Running Experiments

We provide scripts in the `./scripts/` directory to run experiments on a SLURM cluster and reproduce our results. 
