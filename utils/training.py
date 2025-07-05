# pylint: disable=not-callable
"""
Utility functions for training.
"""

import numpy as np
from sklearn.metrics import jaccard_score
import torch
import torch.optim as optim
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
    Pretrain the Probabilistic Stochastic Concept Bottleneck Model (PSCBM) for one epoch. This method doesn't use interventions. Instead, the 
    covariance matrix is trained in a standard way, i.e. the model is trained to predict concepts and target labels. All other elements of the model
    (encoder, concept head, target head) are frozen.
"
    if interventions:
        epochs=config.model.i_epochs
    else:
        epochs=config.model.p_epochs
    scheduler_type = config.model.get("lr_scheduler", "step")
    if scheduler_type == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=0.34*epochs,
                gamma= 1/config.model.get("lr_divisor", 10),
        )
    else: #"cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
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
