import torch
import pandas as pd
from tqdm import tqdm
import hydra

from utils.data import get_data, get_empirical_covariance, get_empirical_covariance_of_predictions, get_concept_groups
from utils.utils import numerical_stability_check, reset_random_seeds

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

if __name__ == "__main__":
    main()