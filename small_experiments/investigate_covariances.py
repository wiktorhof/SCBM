"""
Several utilities to compare covariance matrices. Specifically, to compare empirical covariances calculated with different data ratios and scalings.
"""

import torch
import pandas as pd
from tqdm import tqdm
import hydra

from utils.data import get_data, get_empirical_covariance, get_empirical_covariance_of_predictions, get_concept_groups
from utils.utils import numerical_stability_check, reset_random_seeds

def cond_and_rank(train_loader):
    ### Condition numbers and matrix ranks ###
    ratios = []
    scalings = []
    conditions = []
    ranks = []
    print(f"{'Data ratio':<12}|{'Scaling:':<8}|{'Condition number':<20}|{'Matrix rank':<12}")
    for data_ratio in tqdm((0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1)):
        for covariance_scaling in (1, 2, 4, 8, 16, 32, 64):
            _, cov = get_empirical_covariance(train_loader, ratio=data_ratio, scaling_factor=covariance_scaling)
            condition = torch.linalg.cond(cov)
            conditions.append(condition)
            rank = torch.linalg.matrix_rank(cov)
            ranks.append(rank)
            ratios.append(data_ratio)
            scalings.append(covariance_scaling)
            try:
                print(f"{data_ratio:<12}|{covariance_scaling:<8}|{condition:<20}|{rank:<12}")
            except:
                pass

    df = pd.DataFrame(data={
        "data_ratio": ratios,
        "scaling": scalings,
        "covariance_condition_number": conditions,
        "covariance_rank": ranks,
    })

    print(df)
    df.to_csv("debug_logs/covariance_investigation.csv")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config):
    # Setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Additional info when using cuda
    if device.type == "cuda":
        print("Using", torch.cuda.get_device_name(0))
    else:
        print("No GPU available")

    gen = reset_random_seeds(42)
    train_loader, val_loader, test_loader = get_data(
                config,
                config.data,
                gen,
            )
    
    ### Divergence from true covariance ###
    # Using incomplete data makes the covariances diverge, but this still doesn't affect the performance.
    covariances = dict()
    for data_ratio in tqdm((0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8, 1)):
        _, cov = get_empirical_covariance(train_loader, ratio=data_ratio)
        covariances[data_ratio] = cov
    differences = {data_ratio: torch.linalg.matrix_norm(covariance - covariances[1], 'fro') for (data_ratio, covariance) in covariances.items()} # {0.01: tensor(1693.5321), 0.02: tensor(1252.2047), 0.05: tensor(789.0886), 0.1: tensor(546.1036), 0.2: tensor(269.3203), 0.5: tensor(158.7943), 0.8: tensor(82.8465), 1: tensor(0.)}
    diagonal_differences = {data_ratio: torch.linalg.matrix_norm((covariance - covariances[1]) * (~torch.eye(covariance.shape[0], dtype=bool, device=covariance.device)), 'fro') for (data_ratio, covariance) in covariances.items()}
    relative_differences = {data_ratio: torch.linalg.matrix_norm(covariance - covariances[1], 'fro')/torch.linalg.matrix_norm(covariances[1], 'fro') for (data_ratio, covariance) in covariances.items()} # {0.01: tensor(0.1350), 0.02: tensor(0.0998), 0.05: tensor(0.0629), 0.1: tensor(0.0435), 0.2: tensor(0.0215), 0.5: tensor(0.0127), 0.8: tensor(0.0066), 1: tensor(0.)}
    print(differences)
    print(relative_differences)
    print(diagonal_differences)

if __name__ == "__main__":
    main()