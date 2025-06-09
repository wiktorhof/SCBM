# pylint: disable=not-callable
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
    train_one_epoch_cbm,
    train_one_epoch_scbm,
    train_one_epoch_pscbm,
    validate_one_epoch_cbm,
    validate_one_epoch_scbm,
    create_validation_dataset_pscbm,
    validate_one_epoch_pscbm,
    Custom_Metrics,
)
from utils.utils import reset_random_seeds


def train(config):
    """
    Run the experiments for (P)SCBMs or baselines as defined in the config setting. This method will set up the device, the correct
    experimental paths, initialize Wandb for tracking, generate the dataset, train the model, evaluate the test set performance, and
    finally it will evaluate the intervention performance based on the policies and strategies defined in the config.
    All final results and validations will be stored in Wandb, while the most important ones will be also printed out in the terminal.
    If specified, the model can also be saved for further exploration.

    Parameters
    ----------
    config: DictConfig
        The config settings for training and validating as defined in configs or in the command line.
    """
    # ---------------------------------
    #       Setup
    # ---------------------------------
    # To use as many workers for loading data as there are CPUs available
    config.workers = len(os.sched_getaffinity(0))
    # Reproducibility
    gen = reset_random_seeds(config.seed)

    # Setting device on GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Additional info when using cuda
    if device.type == "cuda":
        print("Using", torch.cuda.get_device_name(0))
    else:
        print("No GPU available")

    # Set paths
    timestr = time.strftime("%Y%m%d-%H%M%S")
    ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
    experiment_path = (
        Path(config.experiment_dir) / config.model.model / config.model.concept_learning / config.data.dataset / ex_name
    )
    experiment_path.mkdir(parents=True)
    config.experiment_dir = str(experiment_path)
    print("Experiment path: ", experiment_path)

    # Wandb
    os.environ["WANDB_CACHE_DIR"] = os.path.join(
        Path(__file__).absolute().parent, "wandb", ".cache", "wandb"
    )  # S.t. on slurm, artifacts are logged to the right place
    if config.logging.mode == "online":
        wandb.login(key=os.environ["WANDB_API_KEY"], host=config.logging.host)
        print ("Successfully logged in!")
    print("Cache dir:", os.environ["WANDB_CACHE_DIR"])
    with wandb.init(
        project=config.logging.project,
        reinit=True,
        entity=config.logging.entity,
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.logging.mode,
        tags=[config.model.tag, config.model.concept_learning, config.model.get("cov_type"), config.model.training_mode, config.data.dataset],
    ) as run:
        print ("Run initialized")
        if config.logging.mode in ["online", "disabled"]:
            run.name = run.name.split("-")[-1] + "-" + config.experiment_name
        elif config.logging.mode == "offline":
            run.name = config.experiment_name
        else:
            raise ValueError("wandb needs to be set to online, offline or disabled.")

        # ---------------------------------
        #       Prepare data and model
        # ---------------------------------
        train_loader, val_loader, test_loader = get_data(
            config,
            config.data,
            gen,
        )

        # Get concept names for plotting
        concept_names_graph = get_concept_groups(config.data)

        # Numbers of training epochs
        if config.model.training_mode == "joint":
            t_epochs = config.model.j_epochs
        elif config.model.training_mode in ("sequential", "independent"):
            c_epochs = config.model.c_epochs
            t_epochs = config.model.t_epochs
        if config.model.get("p_epochs") is not None:
            p_epochs = config.model.p_epochs

        # Initialize model and training objects
        model = create_model(config)

        # Initialize covariance with empirical covariance
        cov_type = config.model.get("cov_type", "")
        if cov_type.startswith("empirical") or cov_type=="global":
            data_ratio = config.model.get("data_ratio", 1)
            covariance_scaling = config.model.get("covariance_scaling", None)
        if cov_type in ("empirical", "empirical_true"): # empirical_true in PSCBM is equivalent to empirical in SCBM (preserved for backward compatibility)
            (model.sigma_concepts, model.covariance) = [t.to(device) for t in get_empirical_covariance(train_loader, ratio=data_ratio, scaling_factor=covariance_scaling)]
        elif cov_type == "empirical_predicted": # only in the case of PSCBM
            # I suppose, the .to(device) was redundant. If the entire model is moved to device, its attributes should be as well, no?
            (model.sigma_concepts, model.covariance) = [t.to(device) for t in get_empirical_covariance_of_predictions(model,train_loader, ratio=data_ratio, scaling_factor=covariance_scaling)]
        # Identity matrix as covariance as a baseline and for debugging purposes - it should behave the same way are a regular CBM
        elif cov_type == "identity":
            model.sigma_concepts = model.covariance = torch.eye(config.data.num_concepts).to(device)
        elif cov_type == "global":
            lower_triangle, _ = get_empirical_covariance(train_loader, ratio=data_ratio, scaling_factor=covariance_scaling)
            lower_triangle.to(device)
            rows, cols = torch.tril_indices(
                row=config.data.num_concepts, col=config.data.num_concepts, offset=0
            )
            model.sigma_concepts = torch.nn.Parameter(lower_triangle[rows, cols])
            # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
            diag_idx = rows == cols
            with torch.no_grad():
                model.sigma_concepts[diag_idx] = (
                    lower_triangle[rows, cols][diag_idx].expm1().clamp_min(1e-6).log()
                )  # softplus inverse of diag

        model.to(device)
        loss_fn = create_loss(config)

        metrics = Custom_Metrics(config.data.num_concepts, device).to(device)

        # ---------------------------------
        #            Training
        # ---------------------------------
        if config.model.model == "cbm":
            validate_one_epoch = validate_one_epoch_cbm
            train_one_epoch = train_one_epoch_cbm
            intervene = intervene_cbm
        elif config.model.model == "scbm":
            validate_one_epoch = validate_one_epoch_scbm
            train_one_epoch = train_one_epoch_scbm
            intervene = intervene_scbm
        elif config.model.model == "pscbm":
            validate_one_epoch = validate_one_epoch_pscbm
            train_one_epoch = train_one_epoch_pscbm
            intervene = intervene_pscbm

        # if config.model.model == "pscbm" and config.model.load_CBM and config.model.cov_type in ("empirical_true", "empirical_predicted"):
        #     print(
        #         "USING a pretrained CBM. No training is performed."
        #     )
        message = f"Using the following model type: {model.__class__.__name__} with {config.model.concept_learning} concept learning"
        message += f" and {config.model.cov_type} covariance." if "cov_type" in config.model.keys() else "."
        message += f"""Empirical covariance has been computed with {config.model.data_ratio*100}% of all samples
        and off-diagonal elements of the covariance matrix were scaled down by a factor of {config.model.covariance_scaling}.
        The condition of the empirical covariance is {torch.linalg.cond(model.covariance)}.
        """ if cov_type.startswith("empirical") else ""
        print(message)
        if config.model.get("load_weights", False):
            print(
                "Pretrained weights have been loaded. The model's parameters aren't trained."
            )
        else:
            print(
                "TRAINING "
                + str(config.model.model)
                + ": "
                + str(config.model.concept_learning + "\n")
            )
            wandb.define_metric("epoch")

            # Pretraining autoregressive concept structure for AR baseline
            if (
                config.model.get("pretrain_concepts")
                and config.model.concept_learning == "autoregressive"
            ):
                print("\nStarting concepts pre-training!\n")
                mode = "c"

                # Freeze the target prediction part
                model.freeze_c()
                model.encoder.apply(freeze_module)  # Freezing the encoder

                c_optimizer = create_optimizer(config.model, model)
                lr_scheduler = optim.lr_scheduler.StepLR(
                    c_optimizer,
                    step_size=config.model.decrease_every,
                    gamma=1 / config.model.lr_divisor,
                )
                for epoch in range(p_epochs):
                    # Validate the model periodically
                    if epoch % config.model.validate_per_epoch == 0:
                        print("\nEVALUATION ON THE VALIDATION SET:\n")
                        validate_one_epoch(
                            val_loader, model, metrics, epoch, config, loss_fn, device, run,
                        )
                    train_one_epoch(
                        train_loader,
                        model,
                        c_optimizer,
                        mode,
                        metrics,
                        epoch,
                        config,
                        loss_fn,
                        device,
                        run,
                    )
                    lr_scheduler.step()

                model.encoder.apply(unfreeze_module)  # Unfreezing the encoder

            # For sequential & independent training: first stage is training of concept encoder
            if config.model.training_mode in ("sequential", "independent"):
                print("\nStarting concepts training!\n")
                mode = "c"

                # Freeze the target prediction part
                model.freeze_c()

                c_optimizer = create_optimizer(config.model, model)
                lr_scheduler = optim.lr_scheduler.StepLR(
                    c_optimizer,
                    step_size=config.model.decrease_every,
                    gamma=1 / config.model.lr_divisor,
                )
                for epoch in range(c_epochs): # pylint: disable
                    # Validate the model periodically
                    if epoch % config.model.validate_per_epoch == 0:
                        print("\nEVALUATION ON THE VALIDATION SET:\n")
                        validate_one_epoch(
                            val_loader, model, metrics, epoch, config, loss_fn, device, run,
                        )
                    train_one_epoch(
                        train_loader,
                        model,
                        c_optimizer,
                        mode,
                        metrics,
                        epoch,
                        config,
                        loss_fn,
                        device,
                        run,
                    )
                    lr_scheduler.step()

                # Prepare parameters for target training by unfreezing the target prediction part and freezing the concept encoder
                model.freeze_t()

            # Sequential vs. joint optimisation
            if config.model.training_mode in ("sequential", "independent"):
                print("\nStarting target training!\n")
                mode = "t"
            else:
                print("\nStarting joint training!\n")
                mode = "j"

            optimizer = create_optimizer(config.model, model)
            lr_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.model.decrease_every,
                gamma=1 / config.model.lr_divisor,
            )

            # If sequential & independent training: second stage is training of target predictor
            # If joint training: training of both concept encoder and target predictor
            for epoch in range(0, t_epochs):
                if epoch % config.model.validate_per_epoch == 0:
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

            model.apply(freeze_module)
            if config.save_model:
                torch.save(model.state_dict(), join(experiment_path, "model.pth"))
                OmegaConf.save(config=config, f=join(experiment_path, "config.yaml"))
                print("\nTRAINING FINISHED, MODEL SAVED!", flush=True)
            else:
                print("\nTRAINING FINISHED", flush=True)

            print("\nEVALUATION ON THE TEST SET:\n")
            validate_one_epoch(
                test_loader,
                model,
                metrics,
                t_epochs,
                config,
                loss_fn,
                device,
                run,
                test=True,
                concept_names_graph=concept_names_graph,
            )
        if config.model.model == "pscbm" and config.model.cov_type in ("global"):
            print("TRAINING THE PSCBM FOR INTERVENTIONS")
            wandb.define_metric("epoch")
            wandb.define_metric("train/lr", step_metric="epoch")
            wandb.define_metric("train/epoch_time", step_metric="epoch")
            wandb.define_metric("val/epoch_time", step_metric="epoch")
            model.CBM.apply(freeze_module)
            
            #TODO paramters for optimizer and scheduler should be optimized :-) Don't I exaggerate?
            optimizer = create_optimizer(config.model, model)
            # Reduce lr if training loss stops improving
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=1 / config.model.lr_divisor,
                    patience=20
                )
            intervention_strategy = define_strategy(
                "simple_perc", train_loader, model, device, config
            ) if model.concept_learning == 'soft' else define_strategy(
                "hard", train_loader, model, device, config
            )
            print("Generating a dataset for interventions validation...")
            start_time = time.perf_counter()
            interventions_validation_dataset = create_validation_dataset_pscbm(
                val_loader,
                model,
                metrics,
                config,
                intervention_strategy,
                loss_fn,
                device,
                run,
            )
            end_time = time.perf_counter()
            best_validation_loss = torch.inf
            print(f"Dataset has been generated in {(end_time - start_time):.2f} seconds.")
            for epoch in range(config.model.i_epochs):
                # Validate the model periodically
                if epoch % config.model.validate_per_epoch == 0:
                    print("\nEVALUATION ON THE VALIDATION SET:\n")
                    # TODO Implement this function (mimicking training epoch)
                    val_start_time = time.perf_counter()
                    validation_loss = validate_one_epoch_pscbm(
                        interventions_validation_dataset, model, metrics, epoch, config, intervention_strategy, loss_fn, device, run,
                    )
                    val_end_time = time.perf_counter()
                    val_time = val_end_time - val_start_time
                    print(f"Validating the model took {val_time} s.")
                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        torch.save(model.sigma_concepts, join(experiment_path, "best_covariance.pth"))

                train_start_time = time.perf_counter()
                training_loss = train_one_epoch_pscbm(
                    train_loader, 
                    model, 
                    optimizer, 
                    metrics, 
                    epoch, 
                    config, 
                    intervention_strategy, 
                    loss_fn, 
                    device, 
                    run,
                    )
                lr_scheduler.step()
                # model2 = create_model(config)
                # for (name1, param1), (name2, param2) in zip(model.named_parameters(), model2.named_parameters()):
                #     print(f"{name1}: equal" if param1.data.equal(param2.data) else f"{name1}: different") 
                    # The output after 1 epoch is that all parameters are equal except sigma_concepts. This is what I expected.
            if config.save_model:
                torch.save(model.state_dict(), join(experiment_path, "model.pth"))
                OmegaConf.save(config=config, f=join(experiment_path, "config.yaml"))
                print("\nTRAINING FINISHED, MODEL SAVED!\n Path to model parameters: {join(experiment_path, 'model.pth')}", flush=True)
            else:
                print("\nTRAINING FINISHED", flush=True)
        # Intervention curves
        print("\nPERFORMING INTERVENTIONS:\n")
        intervene(
            train_loader, test_loader, model, metrics, t_epochs, config, loss_fn, device, run
        )
    return None


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    train(config)


if __name__ == "__main__":
    main()
