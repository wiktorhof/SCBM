"""
Run this file to train models using a Hydra configuration, e.g.:
    python train.py +model=SCBM +data=CUB
"""

import os
from os.path import join
from pathlib import Path
import time
import uuid

import torch
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from models.losses import create_loss
from models.models import create_model

from utils.data import get_data, get_empirical_covariance, get_empirical_covariance_of_predictions, get_concept_groups
from utils.intervention import intervene_cbm, intervene_scbm, intervene_pscbm, define_strategy
from utils.training import (
    freeze_module,
    unfreeze_module,
    create_optimizer,
    create_lr_scheduler,
    train_one_epoch_cbm,
    train_one_epoch_scbm,
    train_one_epoch_pscbm_with_loss,
    train_one_epoch_pscbm_with_interventions,
    validate_one_epoch_cbm,
    validate_one_epoch_scbm,
    generate_pscbm_training_dataloader,
    create_pscbm_validation_dataloader,
    validate_one_epoch_pscbm_with_loss,
    validate_one_epoch_pscbm_with_interventions,
    Custom_Metrics,
)
from utils.utils import reset_random_seeds, save_trainable_params
from high_level_utils import setup, training_loop

def train_and_evaluate(config: DictConfig):
    # General setup
    device, gen = setup(config)
    # Load data
    train_loader, val_loader, test_loader = get_data(
        config,
        config.data,
        gen,
    )
    # Get concept names for plotting
    concept_names_graph = get_concept_groups(config.data)
    model = create_model(config)
    model.to(device)
    metrics = Custom_Metrics(config.data.num_concepts, device).to(device)
    experiment_type = config.get("experiment_type") # empirical_covariance, train_cbm, train_scbm, ...
    # Now I hardcode for empirical_covariance
    