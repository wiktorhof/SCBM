# pylint: disable=not-callable
"""
Utility functions for training.
"""

import numpy as np
from sklearn.metrics import jaccard_score
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import wandb
import time
from time import perf_counter

from utils.metrics import calc_target_metrics, calc_concept_metrics
from utils.plotting import compute_and_plot_heatmap


def generate_training_dataloader_pscbm(
    train_loader, model, epoch, config, device, run
):
    model.eval()
    training_dataset = []
    with torch.no_grad():
        for batch in tqdm(
            train_loader, desc="Generating dataset for interventions training", position=0, leave=True
            ):
            batch_features, target_true, concepts_true = batch["features"].to(device), batch["labels"].to(device), batch["concepts"].to(device)
            if config.model.cov_type == "global" and not config.model.get("pretrain_cov", False):
                concepts_pred_probs, _, _, _ = model(batch_features, epoch)
                training_dataset.append(
                    [ batch_features.cpu(), target_true.cpu(), concepts_true.cpu(), concepts_pred_probs.cpu(), ]
                    )
            # If
            elif config.model.cov_type == "amortized":
                concepts_pred_probs, _, _, _, intermediate = model(batch_features, epoch, return_intermediate=True)
                training_dataset.append(
                    [ batch_features.cpu(), target_true.cpu(), concepts_true.cpu(), concepts_pred_probs.cpu(), intermediate.cpu(), ]
                    )
            else:
                raise ValueError(f"Covariance training is only possible in the amortized and global variants. The passed argument is {config.model.cov_type}.")
    training_dataset = [
        torch.cat(
            [sublist[i] for sublist in training_dataset], dim=0
        )
        for i in range(len(training_dataset[0]))
    ]
    training_dataset = TensorDataset(*training_dataset)
    training_dataloader = DataLoader(training_dataset, batch_size=config.model.train_batch_size, num_workers=config.workers, shuffle=True, pin_memory=True, drop_last=True)
    return training_dataloader

def pretrain_one_epoch_pscbm(
    train_loader, model, optimizer, metrics, epoch, config, loss_fn, device, run):
    """
    Pretrain the Post-hoc Stochastic Concept Bottleneck Model (PSCBM) for one epoch. This method doesn't use interventions. Instead, the 
    covariance matrix is trained in a standard way, i.e. the model is trained to predict concepts and target labels. All other elements of the model
    (encoder, concept head, target head) are frozen.
    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data. It provides precomputed intermediate representations
        of the training dataset to the end of accelerating the training.
        model (torch.nn.Module): The PSCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
    """
    model.train()
    metrics.reset()

    start = time.perf_counter()

    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch_features, target_true, concepts_true, _, intermediate = (item.to(device) for item in batch)
        concepts_pred_probs, target_pred_logits, concepts, concepts_cov = model(batch_features, epoch, intermediate=intermediate, use_covariance=True)
        target_loss, concepts_loss, prec_loss, total_loss = loss_fn(concepts, concepts_true, target_pred_logits, target_true, concepts_cov, cov_not_triang=True)
        optimizer.zero_grad()
        total_loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                p_norm = p.grad.data.norm(2)
                run.log({f"train_cov/{name}_gradient_norm": p_norm})
                print(f"train_cov/{name}_gradient_norm: {p_norm}")
        optimizer.step()
        c_norm = torch.norm(concepts_cov) / (concepts_cov.numel() ** 0.5)
        
        # Store predictions
        metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_pred_probs,
                prec_loss=prec_loss,
                cov_norm=c_norm,
            )
    end = time.perf_counter()
    metrics_dict = metrics.compute(config=config)
    if epoch == 0:
        for (k,v) in metrics_dict.items():
            run.define_metric(f"train_cov/{k}", step_metric="epoch")
            run.define_metric(f"train_cov/epoch_time", step_metric="epoch", summary="mean")
    log_dict = {f"train_cov/{k}": v for (k, v) in metrics_dict.items()}
    log_dict.update({"epoch": epoch + 1, "train_cov/epoch_time": end-start})
    
    run.log(log_dict)    

    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()

    return metrics_dict['total_loss']

def train_one_epoch_pscbm(
    train_loader, model, optimizer, metrics, epoch, config, 
    intervention_strategy, loss_fn, device, run, num_masks=10, mask_density=0.15, num_ones=None):
    """
    Train the Probabilistic Stochastic Concept Bottleneck Model (PSCBM) for one epoch. The training only involves the covariance matrix.
    It can be trained in 2 modes:
    - "global": where the covariance is a global parameter of the model
    - "amortized": there the covariance is predicted for each sample from the CBMs features
    In both cases, the CBMs parameters aren't modified by the training.
    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The PSCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        num_ones: deprecated
    """
    model.train()
    metrics.reset()
    # run.log({"epoch": epoch+1})

    start = time.perf_counter()
    # Calculate the number of ones in an intervention mask. It is a list of 1 or 2 elements.
    # If len(num_ones) == 1, it is the constant number of ones for every mask
    # If len(num_ones) == 2, it contains the lower and upper bounds for the number of ones per mask
    num_concepts = config.data.num_concepts
    if type(mask_density) == float:
        if 0.0 >= mask_density or 1.0 < mask_density:
            raise ValueError(f"mask_density must be positive and less than 1.")
        num_ones = [int(num_concepts*mask_density)]
    elif hasattr(mask_density, '__len__') and len(mask_density) == 2: # Check if mask density is an iterable of length 2, e.g. a list, tuple or an omegaconf.listconfig.ListConfig
        if 0.0 >= mask_density[0] or 1.0 < mask_density[1] or mask_density[0] > mask_density[1]:
            raise ValueError(f"mask_density must be between 0 and 1 and the upper bound cannot be smaller than the lower bound.")
        num_ones = [int(num_concepts * limit) for limit in mask_density]
    else:
        raise TypeError(f"mask_density should be a float or a list of 2 floats.")

    for k, batch in enumerate(
    tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
):
        if config.model.cov_type == "global":
            batch_features, target_true, concepts_true, concepts_pred_probs = (item.to(device) for item in batch)
            # timestamp1 = perf_counter()
            _, _, _, concepts_cov = model(batch_features, epoch, cov_only=True)
            # timestamp2 = perf_counter()
            # model_time += (timestamp2-timestamp1)
        else: # amortized
            batch_features, target_true, concepts_true, concepts_pred_probs, intermediate = (item.to(device) for item in batch)
            # timestamp1 = perf_counter()
            _, _, _, concepts_cov = model(batch_features, epoch, cov_only=True, intermediate=intermediate)
            # timestamp2 = perf_counter()
            # model_time += (timestamp2-timestamp1)

        batch_size = concepts_true.shape[0]
        scores = torch.rand(num_masks, batch_size, num_concepts)
        masks = torch.zeros_like(scores, dtype=torch.int8) #num_masks, batch_size, num_concepts
        if len(num_ones) == 2:
            num_ones_per_mask = torch.randint(num_ones[0], num_ones[1], (num_masks,), dtype=torch.int64)
            indices = [torch.topk(scores[i], num_ones_per_mask[i], dim=-1).indices for i in range(num_masks)]
            for i in range(num_masks):
                masks[i].scatter_(1, indices[i], 1)
        else:
            num_ones_per_mask = num_ones[0]
            indices = torch.topk(scores, num_ones_per_mask, dim=2).indices
            masks.scatter_(2,indices,1) #dim, idx, src
        # assert (
        # (masks.sum(dim=2)==num_ones_per_mask).all()
        # )
        masks = masks.to(device)
        # timestamp3 = perf_counter()
        # mask_time += (timestamp3-timestamp2)
        concepts_pred_mu = torch.logit(concepts_pred_probs, eps=1e-6)
        accumulated_loss = 0
        for concepts_mask in masks:
            concepts_mu_interv, concepts_cov_interv, c_mcmc_probs, c_mcmc_logits = intervention_strategy.compute_intervention(
                                    concepts_pred_mu,
                                    concepts_cov,
                                    concepts_true,
                                    concepts_mask,
                                )
            concepts_pred_probs_interv = c_mcmc_probs.mean(-1)
            concepts_pred_logits_interv = c_mcmc_logits.mean(-1)
            y_pred_intervened = model.intervene(
                concepts_pred_logits_interv,
                concepts_mask,
                batch_features,
                concepts_true,
                )
            c_norm = torch.norm(concepts_cov) / (concepts_cov.numel() ** 0.5)
            target_loss, concepts_loss, prec_loss, total_loss = loss_fn(c_mcmc_probs, concepts_true, y_pred_intervened, target_true, concepts_cov, cov_not_triang=True)
            accumulated_loss += total_loss
            # total_loss.backward(retain_graph=True)
            # optimizer.step()
            # Store predictions
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                y_pred_intervened,
                concepts_true,
                concepts_pred_probs_interv,
                prec_loss=prec_loss,
                cov_norm=c_norm,
            )     
        accumulated_loss /= masks.shape[0]
        accumulated_loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                p_norm = p.grad.data.norm(2)
                run.log({f"train_cov_int/{name}_gradient_norm": p_norm})
                #print(f"train_cov_int/{name}_gradient_norm: {p_norm}")


        optimizer.step()
        optimizer.zero_grad()
        # timestamp4 = perf_counter()
        # interventions_time += (timestamp4-timestamp3)
    end = time.perf_counter
    # Calculate and log metrics
    metrics_dict = metrics.compute(config=config)
    if epoch == 0:
        for (k,v) in metrics_dict.items():
            run.define_metric(f"train_cov_int/{k}", step_metric="epoch")
            run.define_metric(f"train_cov_int/epoch_time", step_metric="epoch", summary="mean")
    log_dict = {f"train_cov_int/{k}": v for (k, v) in metrics_dict.items()}
    log_dict.update({"epoch": epoch + 1, "train_cov_int/epoch_time": end_time})
    run.log(log_dict)    

    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()
    # print(f"""Time spend on specific activities in the training loop:
    # model evaluation: {model_time:.2f}s
    # masks creation: {mask_time:.2f}s
    # interventions: {interventions_time:.2f}s
    # """)   
    # run.log({
    #     "train/model_evaluation_time": model_time,
    #     "train/masks_creation_time": mask_time,
    #     "train/interventions_time": interventions_time,
    #     })           
    return metrics_dict['total_loss']

def train_one_epoch_scbm(
    train_loader, model, optimizer, mode, metrics, epoch, config, loss_fn, device, run
):
    """
    Train the Stochastic Concept Bottleneck Model (SCBM) for one epoch.

    This function trains the SCBM for one epoch using the provided training data loader, model, optimizer, and loss function.
    It supports different training modes and updates the model parameters accordingly. The function also computes and logs
    various metrics during the training process.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The SCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mode (str): The training mode. Supported modes are:
                    - "j": Joint training of the model.
                    - "c": Training the concept head only.
                    - "t": Training the classifier head only.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None

    Notes:
        - Depending on the training mode, certain parts of the model are set to evaluation mode.
        - The function iterates over the training data, performs forward and backward passes, and updates the model parameters.
        - Metrics are computed and logged at the end of each epoch.
    """

    model.train()
    metrics.reset()

    start = time.perf_counter()
    if (
        config.model.training_mode == "sequential"
        or config.model.training_mode == "independent"
    ):
        if mode == "c":
            model.head.eval()
        elif mode == "t":
            model.encoder.eval()

    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch_features, target_true = batch["features"].to(device), batch["labels"].to(
            device
        )
        concepts_true = batch["concepts"].to(device)

        # Forward pass
        concepts_mcmc_probs, triang_cov, target_pred_logits_interv = model(
            batch_features, epoch, c_true=concepts_true
        )

        # Backward pass depends on the training mode of the model
        optimizer.zero_grad()

        # Compute the loss
        target_loss, concepts_loss, prec_loss, total_loss = loss_fn(
            concepts_mcmc_probs,
            concepts_true,
            target_pred_logits_interv,
            target_true,
            triang_cov,
        )

        if mode == "j":
            total_loss.backward()
        elif mode == "c":
            (concepts_loss + prec_loss).backward()
        else:
            target_loss.backward()
        optimizer.step()  # perform an update

        # Store predictions
        concepts_pred_probs = concepts_mcmc_probs.mean(-1)
        metrics.update(
            target_loss,
            concepts_loss,
            total_loss,
            target_true,
            target_pred_logits_interv,
            concepts_true,
            concepts_pred_probs,
            prec_loss=prec_loss,
        )
    
    end = time.perf_counter()
    # Calculate and log metrics
    metrics_dict = metrics.compute()
    if epoch == 0:
        for i, (k, v) in enumerate(metrics_dict.items()):
            run.define_metric(f"train/{k}", step_metric="epoch")
            run.define_metric(f"train/epoch_time", step_metric="epoch", summary="mean")
    log_dict = {f"train/{k}": v for (k, v) in metrics_dict.items()}
    log_dict.update({"epoch": epoch + 1, "train/epoch_time": end-start})
    run.log(log_dict)
    # run.log({f"train/{k}": v for k, v in metrics_dict.items()})
    # run.log({"epoch": epoch + 1})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()
    return


def train_one_epoch_cbm(
    train_loader, model, optimizer, mode, metrics, epoch, config, loss_fn, device, run
):
    """
    Train a baseline method for one epoch.

    This function trains the CEM/AR/CBM for one epoch using the provided training data loader, model, optimizer, and loss function.
    It supports different training modes and updates the model parameters accordingly. The function also computes and logs
    various metrics during the training process.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        model (torch.nn.Module): The SCBM model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer for training the model.
        mode (str): The training mode. Supported modes are:
                    - "j": Joint training of the model.
                    - "c": Training the concept head only.
                    - "t": Training the classifier head only.
        metrics (object): An object to track and compute metrics during training.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and training settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None

    Notes:
        - Depending on the training mode, certain parts of the model are set to evaluation mode.
        - The function iterates over the training data, performs forward and backward passes, and updates the model parameters.
        - Metrics are computed and logged at the end of each epoch.
    """

    model.train()
    metrics.reset()

    start = time.perf_counter()
    if config.model.training_mode in ("sequential", "independent"):
        if mode == "c":
            model.head.eval()
        elif mode == "t":
            model.encoder.eval()

    for k, batch in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch_features, target_true = batch["features"].to(device), batch["labels"].to(
            device
        )
        concepts_true = batch["concepts"].to(device)

        # Forward pass
        if config.model.training_mode == "independent" and mode == "t":
            concepts_pred_probs, target_pred_logits_interv, concepts_hard = model(
                batch_features, epoch, concepts_true
            )
        elif config.model.concept_learning == "autoregressive" and mode == "c":
            concepts_pred_probs, target_pred_logits_interv, concepts_hard = model(
                batch_features, epoch, concepts_train_ar=concepts_true
            )
        else:
            concepts_pred_probs, target_pred_logits_interv, concepts_hard = model(
                batch_features, epoch
            )
        # Backward pass depends on the training mode of the model
        optimizer.zero_grad()
        # Compute the loss
        target_loss, concepts_loss, total_loss = loss_fn(
            concepts_pred_probs, concepts_true, target_pred_logits_interv, target_true
        )

        if mode == "j":
            total_loss.backward()
        elif mode == "c":
            concepts_loss.backward()
        else:
            target_loss.backward()
        optimizer.step()  # perform an update

        # Store predictions
        metrics.update(
            target_loss,
            concepts_loss,
            total_loss,
            target_true,
            target_pred_logits_interv,
            concepts_true,
            concepts_pred_probs,
        )

    end = time.perf_counter()
    # Calculate and log metrics
    metrics_dict = metrics.compute()
    if epoch == 0:
        for i, (k, v) in enumerate(metrics_dict.items()):
            run.define_metric(f"train/{k}", step_metric="epoch")
            run.define_metric(f"train/epoch_time", step_metric="epoch", summary="mean")
    log_dict = {f"train/{k}": v for (k, v) in metrics_dict.items()}
    log_dict.update({"epoch": epoch + 1, "train/epoch_time": end-start})
    run.log(log_dict)
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()
    return

def create_validation_dataloader_pscbm(
    dataloader,
    model,
    metrics,
    config,
    intervention_strategy,
    loss_fn,
    device,
    run,
    concept_names_graph=None,
    num_masks=10, # Number of random concept masks per data point
    mask_density=0.15, # Average ratio of concepts which are known in interventions.
    num_ones=16,
):
    """
    Create a validation loader for training the PSCBM for interventions.

    This function generates a validation dataset for interventions based on the provided dataloader. For each data point, it computes
    initial losses without interventions and generates a tensor of random masks to validate interventions. This allows to perform interventions during validation pass without
    reevaluating the model.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the validation data. Each batch should be a dictionary containing:
            - "features" (torch.Tensor): Input features of shape (batch_size, feature_dim).
            - "labels" (torch.Tensor): Target labels of shape (batch_size,).
            - "concepts" (torch.Tensor): Concept labels of shape (batch_size, num_concepts).
        model (torch.nn.Module): The PSCBM model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        run (wandb.Run): WandB run object for logging metrics.
        concept_names_graph (list, optional): List of concept names for plotting the heatmap. Default is None.
        num_masks (int, optional): Number of random concept masks per data point. Default is 10.
        mask_density (float, optional): Average ratio of concepts which are known in interventions. Default is 0.15.

    Returns:
        torch.utils.data.DataLoader: A DataLoader containing the intervention validation dataset. Each batch is a tuple of:
            - target_loss (torch.Tensor): Loss for the target predictions.
            - concepts_loss (torch.Tensor): Loss for the concept predictions.
            - precision_loss (torch.Tensor): Loss for the precision of predictions.
            - total_loss (torch.Tensor): Total loss combining all components.
            - concepts_pred_mu (torch.Tensor): Predicted concept logits.
            - concepts_cov (torch.Tensor): Covariance matrix of predicted concepts.
            - concepts_true (torch.Tensor): Ground truth concept labels.
            - target_true (torch.Tensor): Ground truth target labels.
            - masks (torch.Tensor): Random masks for interventions.
    """
    # Ensure mask_density is within the valid range [0, 1]
    model.eval()
    metrics.reset()
    # Determine the number of ones per mask (either constant or a range from which it is uniformly sampled)
    # Verify that the passed arguments are correct
    num_concepts = config.data.num_concepts
    if type(mask_density) == float:
        if 0.0 >= mask_density or 1.0 < mask_density:
            raise ValueError(f"mask_density must be positive and less than 1.")
        num_ones = [int(num_concepts*mask_density)]
    elif hasattr(mask_density, '__len__') and len(mask_density) == 2: # Check if mask density is an iterable of length 2, e.g. a list, tuple or an omegaconf.listconfig.ListConfig
        if 0.0 >= mask_density[0] or 1.0 < mask_density[1] or mask_density[0] > mask_density[1]:
            raise ValueError(f"mask_density must be between 0 and 1 and the upper bound cannot be smaller than the lower bound.")
        num_ones = [int(num_concepts * limit) for limit in mask_density]
    else:
        raise TypeError(f"mask_density should be a float or a list of 2 floats.")

    with torch.no_grad():
        intervention_validation_dataset = []
        masks_dataset = []
        for k, batch in enumerate(tqdm(dataloader, position=0, leave=True)):
            batch_features, target_true, concepts_true = batch["features"].to(device), batch["labels"].to(device), batch["concepts"].to(device)

            # Masks computation seems valid.
            batch_size = concepts_true.shape[0]
            scores = torch.rand(num_masks, batch_size, num_concepts)
            masks = torch.zeros_like(scores, dtype=torch.int8) #num_masks, batch_size, num_concepts
            # Generate a batch of masks with shape (num_masks, batch_size, num_concepts) where each mask contains exactly num_ones 1s.
            # Code created with the help of GitHub Copilot.
            if len(num_ones) == 2:
                num_ones_per_mask = torch.randint(num_ones[0], num_ones[1], (num_masks,), dtype=torch.int64)
                indices = [torch.topk(scores[i], num_ones_per_mask[i], dim=-1).indices for i in range(num_masks)]
                for i in range(num_masks):
                    masks[i].scatter_(1, indices[i], 1)
            else:
                num_ones_per_mask = num_ones[0]
                indices = torch.topk(scores, num_ones_per_mask, dim=2).indices
                masks.scatter_(2,indices,1) #dim, idx, src
            # TODO Reintroduce the assertion.
            # assert (
            # (masks.sum(dim=2)==num_ones).all()
            # )
            masks_dataset.append(masks)
            if config.model.cov_type == "global" and not config.model.pretrain_covariance:
                concepts_pred_probs, _, _, _ = model(batch_features, epoch=0, validation=True)
            elif config.model.cov_type == "amortized" or config.model.pretrain_covariance:
                concepts_pred_probs, _, _, _, intermediate = model(batch_features, epoch=0, validation=True, return_intermediate=True)
            else:
                raise ValueError()
            concepts_pred_mu = torch.logit(concepts_pred_probs,eps=1e-6)
            # If the underlying CBM is hard, it returns a tensor of concept samples. In soft case it doesn't, so we just use concepts_pred_probs. I am not sure, how it
            # Behaves in CEM case.
            concepts_mcmc_probs = concepts_pred_probs.unsqueeze(-1)
            # target_loss, concepts_loss, precision_loss, total_loss = loss_fn(concepts_mcmc_probs, concepts_true, target_pred_logits_interv, target_true, concepts_cov, cov_not_triang=True)
            if config.model.cov_type == "global" and not config.model.pretrain_covariance:
                intervention_validation_dataset.append(
                    [
                        # target_loss.cpu().unsqueeze(0).expand(target_true.shape[0]),
                        # concepts_loss.cpu().unsqueeze(0).expand(target_true.shape[0]),
                        # precision_loss.cpu().unsqueeze(0).expand(target_true.shape[0]),
                        # total_loss.cpu().unsqueeze(0).expand(target_true.shape[0]), 
                        concepts_pred_mu.cpu(),
                        concepts_true.cpu(),
                        target_true.cpu(),
                        batch_features.cpu(),
                    ]
                )
            else: # amortized
                intervention_validation_dataset.append(
                    [
                        # target_loss.cpu().unsqueeze(0).expand(target_true.shape[0]),
                        # concepts_loss.cpu().unsqueeze(0).expand(target_true.shape[0]),
                        # precision_loss.cpu().unsqueeze(0).expand(target_true.shape[0]),
                        # total_loss.cpu().unsqueeze(0).expand(target_true.shape[0]), 
                        concepts_pred_mu.cpu(),
                        concepts_true.cpu(),
                        target_true.cpu(),
                        batch_features.cpu(),
                        intermediate.cpu(),
                    ]
                )


        intervention_validation_dataset = [
            torch.cat(
                [sublist[i] for sublist in intervention_validation_dataset], dim=0
            )
            for i in range(len(intervention_validation_dataset[0]))
        ]
        masks_dataset = torch.cat(masks_dataset, dim=1).swapdims_(0,1)
        # masks_dataset = torch.swapdims(masks_dataset, 0, 1)
        intervention_dataset = TensorDataset(
            *intervention_validation_dataset,
            # intervention_validation_dataset[0],  # target_loss
            # intervention_validation_dataset[1],  # concepts_loss
            # intervention_validation_dataset[2],  # precision_loss
            # intervention_validation_dataset[3],  # total_loss
            # intervention_validation_dataset[4],  # concepts_pred_mu
            # intervention_validation_dataset[5],  # concepts_cov
            # intervention_validation_dataset[6],  # concepts_true
            # intervention_validation_dataset[7],  # target_true
            # intervention_validation_dataset[8],  # batch_features
            masks_dataset,                       # masks
        )
        intervention_validation_loader = DataLoader(
            intervention_dataset,
            batch_size=config.model.val_batch_size,
            shuffle=False,
            num_workers=config.workers,
            pin_memory=True,
            drop_last=False
        )

        return intervention_validation_loader


def validate_one_epoch_pscbm_pretraining(loader, model, metrics, epoch, config, loss_fn, device, run, test=False, precomputed_dataset=True):
    model.eval()
    metrics.reset()
    start = time.perf_counter()
    for k,batch in enumerate(loader):
        if precomputed_dataset:
            (
                concepts_pred_mu, concepts_true, target_true, batch_features, intermediate, masks
            ) = (item.to(device) for item in batch)
            concepts_pred_probs, target_pred_logits, concepts, concepts_cov = model(batch_features, epoch, intermediate=intermediate, use_covariance=True)
        else: #For test we don't generate a precomputed dataset as it would only be used once in the very end anyway.
            batch_features, concepts_true, target_true = batch["features"].to(device), batch["concepts"].to(device), batch["labels"].to(device)
            concepts_pred_probs, target_pred_logits, concepts, concepts_cov = model(batch_features, epoch, intermediate=None, use_covariance=True)
        target_loss, concepts_loss, prec_loss, total_loss = loss_fn(concepts, concepts_true, target_pred_logits, target_true, concepts_cov, cov_not_triang=True)
        c_norm = torch.norm(concepts_cov) / (concepts_cov.numel() ** 0.5)

        # Store predictions
        metrics.update(
            target_loss,
            concepts_loss,
            total_loss,
            target_true,
            target_pred_logits,
            concepts_true,
            concepts_pred_probs,
            prec_loss=prec_loss,
            cov_norm=c_norm,
        )
    end = time.perf_counter()
    metrics_dict = metrics.compute(config=config)
    if not test:
        if epoch == 0:
            for (k,v) in metrics_dict.items():
                run.define_metric(f"validation_cov/{k}", step_metric="epoch")
                run.define_metric(f"validation_cov/epoch_time", step_metric="epoch", summary="mean")
        log_dict = {f"validation_cov/{k}": v for (k, v) in metrics_dict.items()}
        log_dict.update({"epoch": epoch, "validation_cov/epoch_time": end-start})
        run.log(log_dict)
        prints = f"Epoch {epoch}, Validation: "
    else:
        # if epoch == 0:
        for (k,v) in metrics_dict.items():
            run.define_metric(f"test_cov/{k}", step_metric="epoch")
            run.define_metric(f"test_cov/epoch_time", step_metric="epoch", summary="mean")
        log_dict = {f"test_cov/{k}": v for (k, v) in metrics_dict.items()}
        log_dict.update({"epoch": epoch, "test_cov/epoch_time": end-time})
        run.log(log_dict)
        prints = f"Test: "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    metrics.reset()
    return metrics_dict["total_loss"]    

def validate_one_epoch_pscbm(
    loader,
    model,
    metrics,
    epoch,
    config,
    strategy,
    loss_fn,
    device,
    run,
    test=False,
    concept_names_graph=None
):
    """
    Args:
        loader 
    """
    model.eval()
    metrics.reset()
    start = time.perf_counter()
    with torch.no_grad():
        for k, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}", position=0, leave=True)):
            if config.model.cov_type == "global":
                (
                    concepts_pred_mu, concepts_true, target_true, batch_features, masks
                ) = (item.to(device) for item in batch)
            else: # amortized
                (
                    concepts_pred_mu, concepts_true, target_true, batch_features, intermediate, masks
                ) = (item.to(device) for item in batch)
            # At this point dimension swapping is necessary and here only. The reason: inside the validation dataset the leading dimension
            # is necessarily the batch dimension.
            # However, since I have multiple masks, I swap these 2 dimensions, because my functions can process a batch datapoints including masks
            # But they are unable to handle multiple masks for a single datapoint.
            masks = torch.swapdims(masks, 0, 1) # masks.size: (num_masks,batch_size,num_concepts)
            # Concepts mu is the same before and after covariance training, but concepts_cov differs.
            if config.model.cov_type == "global":
                _, _, _, concepts_cov = model(batch_features, epoch=epoch, cov_only=True)
            else: # amortized
                _, _, _, concepts_cov = model(batch_features, epoch=epoch, cov_only=True, intermediate=intermediate)
            for concepts_mask in masks:
                concepts_mu_interv, concepts_cov_interv, c_mcmc_probs, c_mcmc_logits = strategy.compute_intervention(
                    concepts_pred_mu, concepts_cov, concepts_true, concepts_mask
                )
                concepts_pred_logits_interv = c_mcmc_logits.mean(-1)
                concepts_pred_probs_interv = c_mcmc_probs.mean(-1)
                c_norm = torch.norm(concepts_cov_interv) / (concepts_cov_interv.numel()**0.5)
                target_pred_logits_interv = model.intervene(
                    concepts_pred_logits_interv, concepts_mask, batch_features, concepts_true
                    )
                concepts_mcmc_probs = concepts_pred_probs_interv.unsqueeze(-1)
                target_loss, concepts_loss, precision_loss, total_loss = loss_fn(
                    concepts_mcmc_probs, concepts_true, target_pred_logits_interv, target_true, concepts_cov, cov_not_triang=True
                )
                metrics.update(
                    target_loss,
                    concepts_loss,
                    total_loss,
                    target_true,
                    target_pred_logits_interv,
                    concepts_true,
                    concepts_pred_probs_interv,
                    cov_norm=c_norm,
                    prec_loss=precision_loss
                )
        end = time.perf_counter()
        metrics_dict = metrics.compute(validation=True, config=config)
        if not test:
            if epoch == 0:
                for (k,v) in metrics_dict.items():
                    run.define_metric(f"validation_cov_int/{k}", step_metric="epoch")
                    run.define_metric(f"validation_cov_int/epoch_time", step_metric="epoch", summary="mean")
            log_dict = {f"validation_cov_int/{k}": v for (k, v) in metrics_dict.items()}
            log_dict.update({"epoch": epoch, "validation_cov_int/epoch_time": end-start})
            run.log(log_dict)
            prints = f"Epoch {epoch}, Validation: "
        else:
            if epoch == 0:
                for (k,v) in metrics_dict.items():
                    run.define_metric(f"test_cov_int/{k}", step_metric="epoch")
                    run.define_metric("test_cov_int/epoch_time", step_metric="epoch", summary="mean")
            log_dict = {f"test_cov_int/{k}": v for (k, v) in metrics_dict.items()}
            log_dict.update({"epoch": epoch, "test_cov_int/epoch_time": end-start})
            run.log(log_dict)
            prints = f"Test: "
        for key, value in metrics_dict.items():
            prints += f"{key}: {value:.3f} "
        print(prints)
        print()
        metrics.reset()
        return metrics_dict['total_loss']





def validate_one_epoch_scbm(
    loader,
    model,
    metrics,
    epoch,
    config,
    loss_fn,
    device,
    run,
    test=False,
    concept_names_graph=None,
):
    """
    Validate the Stochastic Concept Bottleneck Model (SCBM) for one epoch.

    This function evaluates the SCBM for one epoch using the provided data loader, model, and loss function.
    It computes and logs various metrics during the validation process. It also generates
    and plots a heatmap of the learned concept correlation matrix on the final test set.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation or test data.
        model (torch.nn.Module): The SCBM model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        test (bool, optional): Flag indicating whether this is the final evaluation on the test set. Default is False.
        concept_names_graph (list, optional): List of concept names for plotting the heatmap.
                                              Default is None for which range(n_concepts) is used.

    Returns:
        None

    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - It iterates over the validation data, performs forward passes, and computes the losses.
        - Metrics are computed and logged at the end of the validation epoch.
        - During testing, the function generates and plots a heatmap of the concept correlation matrix.
    """
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():

        for k, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch_features, target_true = batch["features"].to(device), batch[
                "labels"
            ].to(device)
            concepts_true = batch["concepts"].to(device)

            concepts_mcmc_probs, triang_cov, target_pred_logits_interv = model(
                batch_features, epoch, validation=True, c_true=concepts_true
            )
            # Compute covariance matrix of concepts
            cov = torch.matmul(triang_cov, torch.transpose(triang_cov, dim0=1, dim1=2))

            if test and k % (len(loader) // 10) == 0:
                try:
                    corr = (cov[0] / cov[0].diag().sqrt()).transpose(
                        dim0=0, dim1=1
                    ) / cov[0].diag().sqrt()
                    matrix = corr.cpu().numpy()

                    compute_and_plot_heatmap(
                        matrix, concepts_true, concept_names_graph, config
                    )

                except:
                    pass

            target_loss, concepts_loss, prec_loss, total_loss = loss_fn(
                concepts_mcmc_probs,
                concepts_true,
                target_pred_logits_interv,
                target_true,
                triang_cov,
            )

            # Store predictions
            concepts_pred_probs = concepts_mcmc_probs.mean(-1)
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits_interv,
                concepts_true,
                concepts_pred_probs,
                prec_loss=prec_loss,
            )

    end = time.perf_counter()
    # Calculate and log metrics
    metrics_dict = metrics.compute(validation=True, config=config)

    if not test:
        if epoch == 0:
            for (k,v) in metrics_dict.items():
                run.define_metric(f"validation/{k}", step_metric="epoch")
                run.define_metric(f"validation/epoch_time", step_metric="epoch", summary="mean")
        log_dict = {f"validation/{k}": v for (k, v) in metrics_dict.items()}
        log_dict.update({"epoch": epoch, "validation/epoch_time": end-start})
        run.log(log_dict)
        prints = f"Epoch {epoch}, Validation: "
    else:
        if epoch == 0:
            for (k,v) in metrics_dict.items():
                run.define_metric(f"test/{k}", step_metric="epoch")
                run.define_metric(f"test/epoch_time", step_metric="epoch", summary="mean")
        log_dict = {f"test/{k}": v for (k, v) in metrics_dict.items()}
        log_dict.update({"epoch": epoch, "test/epoch_time": end-start})
        run.log(log_dict)
        prints = f"Test: "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    metrics.reset()
    return


def validate_one_epoch_cbm(
    loader,
    model,
    metrics,
    epoch,
    config,
    loss_fn,
    device,
    run,
    test=False,
    concept_names_graph=None,
):
    """
    Validate a baseline method for one epoch.

    This function evaluates the CEM/AR/CBM for one epoch using the provided data loader, model, and loss function.
    It computes and logs various metrics during the validation process.

    Args:
        loader (torch.utils.data.DataLoader): DataLoader for the validation or test data.
        model (torch.nn.Module): The model to be validated.
        metrics (object): An object to track and compute metrics during validation.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and validation settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.
        test (bool, optional): Flag indicating whether this is the final evaluation on the test set. Default is False.

    Returns:
        None

    Notes:
        - The function sets the model to evaluation mode and disables gradient computation.
        - It iterates over the validation data, performs forward passes, and computes the losses.
        - Metrics are computed and logged at the end of the validation epoch.
    """
    model.eval()

    start = time.perf_counter()
    with torch.no_grad():
        for k, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch_features, target_true = batch["features"].to(device), batch[
                "labels"
            ].to(device)
            concepts_true = batch["concepts"].to(device)

            concepts_pred_probs, target_pred_logits_interv, concepts_hard = model(
                batch_features, epoch, validation=True
            )
            if config.model.concept_learning == "autoregressive":
                concepts_input = concepts_hard
            elif config.model.concept_learning == "hard":
                concepts_input = concepts_hard
            else:
                concepts_input = concepts_pred_probs
            if config.model.concept_learning == "autoregressive":
                concepts_pred_probs = torch.mean(
                    concepts_pred_probs, dim=-1
                )  # Calculating the metrics on the average probabilities from MCMC

            target_loss, concepts_loss, total_loss = loss_fn(
                concepts_pred_probs, concepts_true, target_pred_logits_interv, target_true
            )

            # Store predictions
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits_interv,
                concepts_true,
                concepts_pred_probs,
            )

    end = time.perf_counter()
    # Calculate and log metrics
    metrics_dict = metrics.compute(validation=True, config=config)
    if not test:
        if epoch == 0:
            for (k, v) in metrics_dict.items():
                run.define_metric(f"validation/{k}", step_metric="epoch")
                run.define_metric(f"validation/epoch_time", step_metric="epoch", summary="mean")
        log_dict = {f"validation/{k}": v for (k, v) in metrics_dict.items()}
        log_dict.update({"epoch": epoch, "validation/epoch_time": end-start})
        run.log(log_dict)
        prints = f"Epoch {epoch}, Validation: "
    else:
        if epoch == 0:
            for (k, v) in metrics_dict.items():
                run.define_metric(f"test/{k}", step_metric="epoch")
                run.define_metric(f"test/epoch_time", step_metric="epoch", summary="mean")
        log_dict = {f"test/{k}": v for (k, v) in metrics_dict.items()}
        log_dict.update({"epoch": epoch, "test/epoch_time": end-start})
        run.log(log_dict)
        prints = f"Test: "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    metrics.reset()
    return


def create_optimizer(config, model):
    """
    Parse the configuration file and return a optimizer object to update the model parameters.
    """
    assert config.optimizer in [
        "sgd",
        "adam",
    ], "Only SGD and Adam optimizers are available!"

    optim_params = [
        {
            "params": filter(lambda p: p.requires_grad, model.parameters()),
            "lr": config.learning_rate,
            "weight_decay": config.weight_decay,
        }
    ]

    if config.optimizer == "sgd":
        return torch.optim.SGD(optim_params)
    elif config.optimizer == "adam":
        if optim_params[0]["weight_decay"] == 0.0:
            return torch.optim.Adam(optim_params)
        else:
            return torch.optim.AdamW(optim_params)

def create_lr_scheduler(config, optimizer, interventions=False):
	if interventions:
		epochs = config.model.i_epochs
	else:
		epochs = config.model.p_epochs
	scheduler_type = config.model.get("lr_scheduler" "step")
	if scheduler_type == "step":
		lr_scheduler = torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=0.34*epochs,
			gamma=1/config.model.get("lr_divisor", 10),
			)
	else: #"cosine":
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			epochs,
			last_epoch=-1,
        )
	return lr_scheduler


class Custom_Metrics(Metric):
    """
    Custom metrics class for tracking and computing various metrics during training and validation.

    This class extends the PyTorch Metric class and provides methods to update and compute metrics such as
    target loss, concept loss, total loss, accuracy, and Jaccard index for both target and concepts.
    It is being updated for each batch. At the end of each epoch, the compute function is called to compute
    the final metrics and return them as a dictionary.

    Args:
        n_concepts (int): The number of concepts in the model.
        device (torch.device): The device to run the computations on.

    Attributes:
        n_concepts (int): The number of concepts in the model.
        target_loss (torch.Tensor): The accumulated target loss.
        concepts_loss (torch.Tensor): The accumulated concepts loss.
        total_loss (torch.Tensor): The accumulated total loss.
        y_true (list): List of true target labels.
        y_pred_logits (list): List of predicted target logits.
        c_true (list): List of true concept labels.
        c_pred_probs (list): List of predicted concept probabilities.
        batch_features (list): List of batch features.
        cov_norm (torch.Tensor): The accumulated covariance norm.
        n_samples (torch.Tensor): The number of samples processed.
        prec_loss (torch.Tensor): The accumulated precision loss.
    """

    def __init__(self, n_concepts, device):
        super().__init__()
        self.n_concepts = n_concepts
        self.add_state("target_loss", default=torch.tensor(0.0, device=device))
        self.add_state("concepts_loss", default=torch.tensor(0.0, device=device))
        self.add_state("total_loss", default=torch.tensor(0.0, device=device))
        self.add_state("y_true", default=[])
        self.add_state("y_pred_logits", default=[])
        self.add_state("c_true", default=[])
        (
            self.add_state("c_pred_probs", default=[]),
            self.add_state("concepts_input", default=[]),
        ),
        self.add_state("batch_features", default=[])
        self.add_state("cov_norm", default=torch.tensor(0.0, device=device))
        self.add_state(
            "n_samples", default=torch.tensor(0, dtype=torch.int, device=device)
        )
        self.add_state("prec_loss", default=torch.tensor(0.0, device=device))

    def update(
        self,
        target_loss: torch.Tensor,
        concepts_loss: torch.Tensor,
        total_loss: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_logits: torch.Tensor,
        c_true: torch.Tensor,
        c_pred_probs: torch.Tensor,
        cov_norm: torch.Tensor = None,
        prec_loss: torch.Tensor = None,
    ):
        assert c_true.shape == c_pred_probs.shape

        n_samples = y_true.size(0)
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCELoss()
        self.n_samples += n_samples
        self.target_loss += target_loss * n_samples
        self.concepts_loss += concepts_loss * n_samples
        self.total_loss += total_loss * n_samples
        self.y_true.append(y_true)
        self.y_pred_logits.append(y_pred_logits.detach())
        self.c_true.append(c_true)
        self.c_pred_probs.append(c_pred_probs.detach())
        if cov_norm:
            self.cov_norm += cov_norm * n_samples
        if prec_loss:
            self.prec_loss += prec_loss * n_samples

    def compute(self, validation=False, config=None):
        y_true = dim_zero_cat(self.y_true).cpu()
        c_true = dim_zero_cat(self.c_true).cpu()
        c_pred_probs = dim_zero_cat(self.c_pred_probs).cpu()
        y_pred_logits = dim_zero_cat(self.y_pred_logits).cpu()
        c_true = c_true.numpy()
        c_pred_probs = c_pred_probs.numpy()
        c_pred = c_pred_probs > 0.5
        if y_pred_logits.size(1) == 1:
            y_pred_probs = nn.Sigmoid()(y_pred_logits.squeeze())
            y_pred = y_pred_probs > 0.5
        else:
            y_pred_probs = nn.Softmax(dim=1)(y_pred_logits)
            y_pred = y_pred_logits.argmax(dim=-1)

        target_acc = (y_true == y_pred).sum() / self.n_samples
        concept_acc = (c_true == c_pred).sum() / (self.n_samples * self.n_concepts)
        complete_concept_acc = (
            (c_true == c_pred).sum(1) == self.n_concepts
        ).sum() / self.n_samples
        target_jaccard = jaccard_score(y_true, y_pred, average="micro")
        concept_jaccard = jaccard_score(c_true, c_pred, average="micro")
        metrics = dict(
            {
                "target_loss": self.target_loss / self.n_samples,
                "prec_loss": self.prec_loss / self.n_samples,
                "concepts_loss": self.concepts_loss / self.n_samples,
                "total_loss": self.total_loss / self.n_samples,
                "y_accuracy": target_acc,
                "c_accuracy": concept_acc,
                "complete_c_accuracy": complete_concept_acc,
                "target_jaccard": target_jaccard,
                "concept_jaccard": concept_jaccard,
            }
        )

        if self.cov_norm != 0:
            metrics = metrics | {"covariance_norm": self.cov_norm / self.n_samples}

        if validation is True:
            c_pred_probs_buffer = []
            for j in range(self.n_concepts):
                c_pred_probs_buffer.append(
                    np.hstack(
                        (
                            np.expand_dims(1 - c_pred_probs[:, j], 1),
                            np.expand_dims(c_pred_probs[:, j], 1),
                        )
                    )
                )

            y_metrics = calc_target_metrics(
                y_true.numpy(), y_pred_probs.numpy(), config.data
            )
            c_metrics, c_metrics_per_concept = calc_concept_metrics(
                c_true, c_pred_probs_buffer, config.data
            )
            metrics = (
                metrics
                | {f"y_{k}": v for k, v in y_metrics.items()}
                | {f"c_{k}": v for k, v in c_metrics.items()}
            )  # | c_metrics_per_concept # Update dict

        return metrics


def freeze_module(m):
    m.eval()
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    m.train()
    for param in m.parameters():
        param.requires_grad = True
