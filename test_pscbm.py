"""
Run this file to test pscbm models using a Hydra configuration, e.g.:
    python train.py +model=PSCBM +data=CUB
"""

import os
from os.path import join
from pathlib import Path
import time
import uuid

import torch
import torch.optim as optim
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from models.losses import create_loss
from models.models import create_model

from utils.data import get_data, get_empirical_covariance, get_empirical_covariance_of_predictions, get_concept_groups
from utils.intervention import intervene_cbm, intervene_scbm, intervene_pscbm
from utils.training import (
    freeze_module,
    unfreeze_module,
    create_optimizer,
    train_one_epoch_cbm,
    train_one_epoch_scbm,
    train_one_epoch_pscbm,
    validate_one_epoch_cbm,
    validate_one_epoch_scbm,
    validate_one_epoch_pscbm,
    Custom_Metrics,
)
from utils.utils import reset_random_seeds

def run(config: DictConfig):
    pass


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    run(config)


if __name__ == "__main__":
    main()