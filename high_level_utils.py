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

def setup(config: DictConfig):
    # To use as many workers for loading data as there are CPUs available
    config.workers = len(os.sched_getaffinity(0))
    # Setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Additional info when using cuda
    if device.type == "cuda":
        print("Using", torch.cuda.get_device_name(0))
    else:
        print("No GPU available")

    # Wandb setup - only done once
    os.environ["WANDB_CACHE_DIR"] = os.path.join(
        Path(__file__).absolute().parent, "wandb", ".cache", "wandb"
    )  # S.t. on slurm, artifacts are logged to the right place
    if config.logging.mode == "online":
        wandb.login(key=os.environ["WANDB_API_KEY"], host=config.logging.host)
        print ("Successfully logged in to WandB!")
    print("Cache dir:", os.environ["WANDB_CACHE_DIR"])
    gen = reset_random_seeds(config.seed)
    return device, gen
    

def data_setup(config: DictConfig):
    pass






def training_loop(config, model, metrics, epochs, train_one_epoch, train_loader, validate_one_epoch, val_loader, validate_per_epoch, intervene=None, intervene_per_epoch=None):
    """
    There are following types of training loops in the project:
    model training:
        concept pre-training for AR (not implemented here)
        joint
        sequential or independent (concept loop, target loop)
    PSCBM covariance training
    interventions training (where trainable modules should be specified)
    Possibility to make interventions inside the training loop will be added later
    as I don't use it anyway for this moment.
    This method 1st creates the proper optimizer, then it optimizes the model for a given
    number of epochs
    """    
    device = model.device
    optimizer = create_optimizer(config.model, model)
    lr_scheduler = create_lr_scheduler(config, optimizer, interventions=False)
    print("Using the following optimizer:", optimizer.__class__.__name__,
                      "\nUsing the following learning rate scheduler:", lr_scheduler.__class__.__name__,)
                

    for epoch in range(epochs):
        if epoch % validate_per_epoch == 0:
            print("\nEVALUATION ON THE VALIDATION SET:\n")
            validate_one_epoch(
                val_loader, model, metrics, epoch, config, loss_fn, device, run,
            )
        train_one_epoch(
                        train_loader,
                        model,
                        optimizer,
                        mode,
                        metrics,
                        epoch,
                        config,
                        loss_fn,
                        device,
                        run,
        )
        lr_scheduler.step()