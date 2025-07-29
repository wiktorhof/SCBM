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
from itertools import product

import torch
import torch.optim as optim
# from torch.utils.data import Subset, DataLoader
import math
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from models.losses import create_loss
from models.models import create_model, initialize_covariance

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


def find_hyperparameters(config):
    """
    Find the best set of hyperparameters for a given version of PSCBM using grid search.
    Outline:
    1. Get data
    2. Create model
    3. Create pscbm_datasets
    4. Train the model with all choices of hyperparameters.
        a. Begin be setting random seed
    Parameters
    ----------
    config: DictConfig
        The config settings for training and validating as defined in configs or in the command line.
    """
    # ---------------------------------
    #       Hyperparameters
    # ---------------------------------
    learning_rates = [0.0001, 0.00001]
    weight_decays = [0, 0.001, 0.01]
    lr_schedulers = ["step", "cosine"]
    covariances = ["global", "amortized"]
    training_methods = ["SCBM_loss"]

    all_model_types = list(product(
        covariances,
        training_methods
    ))

    all_hyperparams = list(product(
    lr_schedulers,
    learning_rates,
    weight_decays,
))
    # ---------------------------------
    #       Setup
    # ---------------------------------
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
    # ---------------------------------
    #       Prepare data
    # ---------------------------------
    # Reproducibility - when loading data
    gen = reset_random_seeds(config.seed)
    train_loader, val_loader, test_loader = get_data(
        config,
        config.data,
        gen,
    )
    # Get concept names for plotting
    concept_names_graph = get_concept_groups(config.data)

    # ---------------------------------
    #      Create a model for each covariance type
    # ---------------------------------
    for cov_type in covariances:
        # Initialize model and training objects
        config.model["cov_type"] = cov_type
        model = create_model(config)
        model.to(device)
        metrics = Custom_Metrics(config.data.num_concepts, device).to(device)
        
        loss_fn = create_loss(config)
        # For interventions_validation_dataloader always use the same seed s.t.
        # All models are evaluated on the same validation dataset
        # Intervention strategy defining how to set values for intervened concepts could also be
        # Specified by the configuration file
        # In case that we are dealing with a PSCBM, generate the special loaders.
        if config.model.model == "pscbm":

            intervention_strategy = define_strategy(
                config.model.get("training_strategy", "simple_perc"), train_loader, model, device, config
            )
            reset_random_seeds(42)
            generation_start_time = time.perf_counter()
            interventions_validation_dataloader = create_pscbm_validation_dataloader(
                                val_loader,
                                model,
                                metrics,
                                config,
                                intervention_strategy,
                                loss_fn,
                                device,
                                num_masks=config.model.num_masks_val,
                                mask_density=config.model.mask_density_val,
                            )
            generation_end_time = time.perf_counter()
            print(f"Validation dataset has been generated in {(generation_end_time - generation_start_time):.2f} seconds.")
            generation_start_time = time.perf_counter()
            interventions_training_dataloader = generate_pscbm_training_dataloader(train_loader, model, 0, config, device)
            generation_end_time = time.perf_counter()
            print(f"Training dataset has been generated in {(generation_end_time - generation_start_time):.2f} seconds.")
          # Reset random seed for the next model
        # ---------------------------------
        #       For every training method, run independent hyperparameter checks
        #       And create a separate experiment directory
        # ---------------------------------
        for training_method in training_methods:

            # Set paths
            timestr = time.strftime("%Y%m%d-%H%M%S")
            ex_name = f"{timestr}_{uuid.uuid4().hex[:5]}"
            experiment_path = (
                Path(config.experiment_dir) / config.model.model / config.model.concept_learning / cov_type / training_method / config.data.dataset / "hyperparameters" / ex_name
            )
            experiment_path.mkdir(parents=True)
            config.experiment_dir = str(experiment_path)
            print("Experiment path: ", experiment_path)

            # ----------------------------
            # Now iterate over all hyperparameter combinations
            # Each of them will have its own WandB run
            # ----------------------------
            for scheduler, lr, weight_decay in all_hyperparams:
                # Reset random seed.
                # TODO Instead of using the same random seed again and again, I might generate a sequence with numpy
                gen = reset_random_seeds(config.seed)

                # For every hyperparameter combination, reinitialize the covariance matrix
                # Initialize covariance with empirical covariance
                initialize_covariance(config, model, train_loader, device)
                config.model["lr_scheduler"] = scheduler
                config.model["learning_rate"] = lr
                config.model["weight_decay"] = weight_decay
                tags = [config.model.tag, config.model.concept_learning, 
                        config.model.get("cov_type"), config.model.training_mode, 
                        config.data.dataset]
                if training_method == "SCBM_loss":
                    tags.append("hyperparams_SCBM_loss")
                elif training_method == "interventions":
                    tags.append("hyperparams_interventions")
                additional_tags = config.model.get("additional_tags", [])
                tags.extend(additional_tags)
                with wandb.init(
                    project=config.logging.project,
                    reinit="create_new",
                    entity=config.logging.entity,
                    config=OmegaConf.to_container(config, resolve=True),
                    mode=config.logging.mode,
                    tags=tags,
                ) as run:
                    print (f"""Run initialized with the following parameters:
                           training_method: {training_method},
                           covariance_type: {cov_type},
                           optimizer: {config.model.get("optimizer")},
                           learning_rate_scheduler: {scheduler},
                           initial learning rate: {lr},
                           weight decay factor: {weight_decay}
                           """)
                    experiment_name = config.experiment_name + f"_{cov_type}_{training_method}_{scheduler}_lr_{lr}_decay_{weight_decay}"
                    if config.logging.mode in ["online", "disabled"]:
                        run.name = run.name.split("-")[-1] + "-" + experiment_name
                    elif config.logging.mode == "offline":
                        run.name = experiment_name
                    else:
                        raise ValueError("wandb needs to be set to online, offline or disabled.")


                    # Numbers of training epochs
                    if config.model.training_mode == "joint":
                        t_epochs = config.model.j_epochs
                    elif config.model.training_mode in ("sequential", "independent"):
                        c_epochs = config.model.c_epochs
                        t_epochs = config.model.t_epochs
                    if config.model.get("p_epochs") is not None:
                        p_epochs = config.model.p_epochs

                    

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
                        if training_method == "SCBM_loss":
                            validate_one_epoch = validate_one_epoch_pscbm_with_loss
                            train_one_epoch = train_one_epoch_pscbm_with_loss
                        elif training_method == "interventions":
                            validate_one_epoch = validate_one_epoch_pscbm_with_interventions
                            train_one_epoch = train_one_epoch_pscbm_with_interventions
                        intervene = intervene_pscbm
                    
                    if config.model.get("load_weights", False):
                        print(
                            "Pretrained weights have been loaded. The CBM's parameters aren't trained."
                        )
                    else:
                        print(
                            "TRAINING "
                            + str(config.model.model)
                            + ": "
                            + str(config.model.concept_learning + "\n")
                        )
                        run.define_metric("epoch")
                        run.define_metric("train/epoch_time", step_metric="epoch")
                        run.define_metric("validation/epoch_time", step_metric="epoch")
                        
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
                            print("Using the following optimizer:", c_optimizer.__class__.__name__,
                                "\nUsing the following learning rate scheduler:", lr_scheduler.__class__.__name__,)
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
                            print("Using the following optimizer:", c_optimizer.__class__.__name__,
                                "\nUsing the following learning rate scheduler:", lr_scheduler.__class__.__name__,)
                            
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
                        print("Using the following optimizer:", optimizer.__class__.__name__,
                                "\nUsing the following learning rate scheduler:", lr_scheduler.__class__.__name__,)

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


                    ### PSCBM Training:
                    # 1. Covariance pre-training on regular inference
                    # 2. Interventions training with random policy and a proper policy 
                    # (currently: hard / simple_perc but it could be set up to follow configurations file)
                    try:
                        if config.model.model == "pscbm" and config.model.cov_type in ("global", "amortized"):
                            # Define epoch metric
                            run.define_metric("epoch")
                            best_validation_loss = torch.inf
                            
                            # Perform initial covariance training for p_epochs by only inferring concepts and target using model's covariance.
                            # Structure:
                            # 1. Define wandb metrics
                            # 2. Define which modules are trainable (only the covariance)
                            # TODO I might want to also train the head or concept predictor - at least for the interventions training
                            # 3. Having defined the trainable modules, create optimizer and lr_scheduler - separate for regular and separate
                            # for interventions training (!)
                            # 4. Train & validate. Every training epoch should be benchmarked for its runtime.
                            if training_method == "SCBM_loss":
                                            
                                # Define wandb metrics
                                run.define_metric("train_cov/lr", step_metric="epoch")

                                # Freeze the CBM & report trainable parameters
                                model.CBM.apply(freeze_module)
                                learnable_parameters = []
                                for name, param in model.named_parameters():
                                    if param.requires_grad:
                                        learnable_parameters.append(name)
                                        run.define_metric(f"train_cov/{name}_gradient_norm", step_metric="epoch")
                                print(f"Learnable parameters:\n{learnable_parameters}")
                                # Create optimizer and lr_scheduler
                                optimizer = create_optimizer(config.model, model)
                                lr_scheduler = create_lr_scheduler(config, optimizer, interventions=False)
                                print("Using the following optimizer:", optimizer.__class__.__name__,
                                    "\nUsing the following learning rate scheduler:", lr_scheduler.__class__.__name__,)
                                
                                print("TRAINING THE PSCBM COVARIANCE ON INFERENCE")
                                start_time = time.perf_counter()
                                for epoch in range(config.model.p_epochs):
                                    # Training
                                    run.log({"train_cov/lr": lr_scheduler.get_last_lr()[0]})
                                    train_one_epoch_pscbm_with_loss(interventions_training_dataloader, model, optimizer, metrics, epoch, config, loss_fn, device, run)
                                    lr_scheduler.step()
                                    # Validation
                                    if epoch % config.model.validate_per_epoch == 0:
                                        validation_loss = validate_one_epoch_pscbm_with_loss(interventions_validation_dataloader, model, metrics, epoch, config, loss_fn, device, run)
                                        if validation_loss < best_validation_loss:
                                            save_trainable_params(model, join (experiment_path, "model_covariance_best.pth"))
                                            best_validation_loss = validation_loss
                                                
                                end_time = time.perf_counter()
                                print(f"Training the covariance for {config.model.p_epochs} with validation every {config.model.validate_per_epoch} epochs took {(end_time-start_time):.2f}s.")
                                print("Final evaluation of trained covariance on the validation set:")
                                # Final evaluation of trained covariance on the validation set - test set shouldn't be used for hyperparameter tuning
                                validate_one_epoch_pscbm_with_loss(interventions_validation_dataloader, model, metrics, config.model.p_epochs, config, loss_fn, device, run, test=True, precomputed_dataset=True)
                                
                                if config.save_model:
                                    torch.save(model.state_dict(), join(experiment_path, "model_covariance_pretrained.pth"))
                                    OmegaConf.save(config=config, f=join(experiment_path, "config.yaml"))
                                    print(f"\nTRAINING FINISHED, MODEL SAVED!\n Path to model parameters: {join(experiment_path, 'model_pretrained.pth')}", flush=True)
                                else:
                                    print("\nTRAINING FINISHED, pre-trained model not saved", flush=True)



                            # Train the model with interventions
                            elif training_method == "interventions":
                                print("TRAINING THE PSCBM COVARIANCE ON INTERVENTIONS")
                                
                                # Define wandb metrics
                                run.define_metric("train_cov_int/lr", step_metric="epoch")
                                # Freeze the CBM & report trainable parameters
                                model.CBM.apply(freeze_module)
                                learnable_parameters = []
                                for name, param in model.named_parameters():
                                    if param.requires_grad:
                                        learnable_parameters.append(name)
                                        run.define_metric(f"train_cov_int/{name}_gradient_norm", step_metric="epoch")
                                
                                print(f"Learnable parameters for interventions training:\n{learnable_parameters}")

                                optimizer = create_optimizer(config.model, model)
                                lr_scheduler = create_lr_scheduler(config, optimizer, interventions=True)
                                print("Using the following optimizer:", optimizer.__class__.__name__,
                                    "\nUsing the following learning rate scheduler:", lr_scheduler.__class__.__name__,)

                                start_time = time.perf_counter()
                                for epoch in range(config.model.i_epochs):
                                    # Calculate intervention curves during training and log them in a separate run:
                                    run.log({"train_cov_int/lr": lr_scheduler.get_last_lr()[0]})
                                    train_one_epoch_pscbm_with_interventions(
                                        interventions_training_dataloader, 
                                        model, 
                                        optimizer, 
                                        metrics, 
                                        epoch, 
                                        config, 
                                        intervention_strategy, 
                                        loss_fn, 
                                        device, 
                                        run,
                                        num_masks=config.model.num_masks_train,
                                        mask_density=config.model.mask_density_train,
                                        )
                                    lr_scheduler.step()
                                        
                                    # Validate the model periodically
                                    if epoch % config.model.validate_per_epoch == 0: # In the initial phase of training validation seems crucial to me.
                                        print(f"\nEVALUATION ON THE VALIDATION SET at epoch {epoch}:\n")
                                        validation_loss = validate_one_epoch_pscbm_with_interventions(
                                            interventions_validation_dataloader, model, metrics, epoch, config, intervention_strategy, loss_fn, device, run,
                                        )
                                        if validation_loss < best_validation_loss:
                                            save_trainable_params(model, join (experiment_path, "model_covariance_best.pth"))
                                            best_validation_loss = validation_loss
                                end_time=time.perf_counter()
                                print(f"Training the model on interventions for {config.model.i_epochs} epochs took {time.strftime('%H:%M:%S', time.gmtime(end_time-start_time))}")
                                print("Final evaluation of trained covariance on the validation set:")
                                validate_one_epoch_pscbm_with_loss(interventions_validation_dataloader, model, metrics, config.model.p_epochs, config, loss_fn, device, run, test=True, precomputed_dataset=True)
                                if config.save_model:
                                    torch.save(model.state_dict(), join(experiment_path, "model_trained_int.pth"))
                                    OmegaConf.save(config=config, f=join(experiment_path, "config.yaml"))
                                    print(f"\nTRAINING ON INTERVENTIONS FINISHED, MODEL SAVED!\n Path to model parameters: {join(experiment_path, 'model_trained_int.pth')}", flush=True)
                                else:
                                    print("\nTRAINING ON INTERVENTIONS FINISHED", flush=True)
                    except Exception as e:
                        # print(f"""When training a model {model.__class__.__name__} with:
                        #       {cov_type} covariance,
                        #       learning rate scheduler: {scheduler.__class__.__name__},
                        #       initial learning rate: {lr},
                        #       weight decay factor: {weight_decay},
                        #       the following exception occured:
                        #       {e}. 
                        #       Continuing training with the next configuration.""")
                        print (f"An exception occured: \n{e}\nContinuing training with the next configuration.")
                    # Intervention curves. - Not calculted during hyperparameter tuning.
                    # When tuning hyperparameters for SCBM-loss PSCBM, I want to be able not to
                    # calculate these intervention curves in order to save computaions.
                    # if config.model.get("calculate_interventions", True):
                    #     print("\nPERFORMING INTERVENTIONS ON THE FINAL TRAINED MODEL:\n")
                    #     intervene(
                    #         train_loader, test_loader, model, metrics, t_epochs, config, loss_fn, device, run
                    #     )
    return None


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    find_hyperparameters(config)


if __name__ == "__main__":
    main()
