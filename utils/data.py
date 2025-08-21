"""
Utility functions for data loading.
"""

import os
import torch
from torch.utils.data import DataLoader

from datasets.CUB_dataset import get_CUB_dataloaders
from utils.utils import numerical_stability_check


def get_data(config_base, config, gen):
    """
    Parse the configuration file and return the relevant dataset loaders.

    This function parses the provided configuration file and returns the appropriate dataset loaders based on the
    specified dataset type. It also sets the data path based on the hostname or the configuration file if working
    locally and on a cluster. The function supports synthetic datasets, CUB, CIFAR-10, and CIFAR-100 datasets.

    Args:
        config_base (dict): The base configuration dictionary.
        config (dict): The data configuration dictionary containing dataset and data path information.
        gen (object): A generator object to control the randomness of the data loader.

    Returns:
        tuple: A tuple containing the training data loader, validation data loader, and test data loader.
    """
    if "data_path" not in config:
        # Local Datafolder if not already specified in yaml
        config.data_path = "../datasets/"
    elif config.data_path is None:
        config.data_path = "../datasets/"
    else:
        pass
    if config.dataset == "CUB":
        print("CUB DATASET")
        trainset, validset, testset = get_CUB_dataloaders(
            config,
        )
    else:
        NotImplementedError("ERROR: Dataset not supported!")

    config = config_base
    train_loader = DataLoader(
        trainset,
        batch_size=config.model.train_batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        generator=gen,
        drop_last=True,
    )
    val_loader = DataLoader(
        validset,
        batch_size=config.model.val_batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        generator=gen,
    )
    test_loader = DataLoader(
        testset,
        batch_size=config.model.val_batch_size,
        num_workers=config.workers,
        generator=gen,
    )

    return train_loader, val_loader, test_loader

def get_empirical_covariance(dataloader, ratio=1, scaling_factor=None):
    """
    Compute the empirical covariance matrix of the concepts in the given dataloader.

    This function computes the empirical covariance matrix of the concepts in the given dataloader.
    It first concatenates all the concept data into a single tensor, then applies a logit transformation
    to the data to work in the correct space. The covariance matrix is computed from the transformed data
    and brought into a lower triangular form using Cholesky decomposition. In comments, an alternative
    covariance computation is provided that can be used if the dataset is too large to fit into memory.

    Args:
        dataloader (torch.utils.data.DataLoader): A dataloader containing batches of data with a "concepts" key.

    Returns:
        torch.Tensor: The lower triangular form of the empirical covariance matrix.
    """
    print("Computing empirical covariance")
    data = []
    tmp_dataloader = DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers,
        pin_memory=True,
        generator=dataloader.generator,
        drop_last=False, # That's the actual parameter that I care about
    )
    #print("Temporary dataloader created")
    data_to_load = int(ratio * len(tmp_dataloader.dataset))
    loaded_data = 0
    for batch in tmp_dataloader:
        concepts = batch["concepts"]
        loaded_data += concepts.shape[0]
        #print(f"{loaded_data}/{data_to_load}")
        if loaded_data > data_to_load:
            excess = loaded_data-data_to_load
            data.append(concepts[:-excess])
            print (f"Computing empirical covariance with {loaded_data-excess} out of total {len(tmp_dataloader.dataset)} samples.")
            break
        data.append(concepts)
    data = torch.cat(data)  # Concatenate all data into a single tensor
    #print("Loaded all data")
    data_logits = torch.logit(data, eps=1e-6)
    covariance = torch.cov(data_logits.transpose(0, 1))

    # Bringing it into lower triangular form
    if scaling_factor:
        rows, cols = torch.tril_indices(row=covariance.shape[1], col=covariance.shape[1], offset=-1)
        covariance[rows, cols] /= scaling_factor
        covariance[cols, rows] /= scaling_factor
    covariance = numerical_stability_check(covariance, device="cpu")
    lower_triangle = torch.linalg.cholesky(covariance)

    ####### Alternative cov computation if dataset was too large for memory
    # num_samples = 0
    # for i, batch in enumerate(dataloader):
    #     concepts = batch["concepts"]
    #     if i == 0:
    #         logits = torch.logit(0.05 + 0.9 * concepts).sum(0)
    #     else:
    #         logits += torch.logit(0.05 + 0.9 * concepts).sum(0)
    #     num_samples += concepts.shape[0]
    # logits_mean = logits / num_samples

    # for i, batch in enumerate(dataloader):
    #     concepts = batch["concepts"]
    #     temp = (torch.logit(0.05 + 0.9 * concepts) - logits_mean).unsqueeze(-1)
    #     if i == 0:
    #         cov = torch.matmul(temp, temp.transpose(-2, -1)).sum(0)
    #     else:
    #         cov += torch.matmul(temp, temp.transpose(-2, -1)).sum(0)
    # cov = cov / num_samples

    ########
    return lower_triangle, covariance

def get_empirical_covariance_of_predictions(model, dataloader, ratio=1, scaling_factor=None):
    """
    Compute the empirical covariance matrix of the concept logits predicted by CBM_model from features in dataloader.

    This function first concatenates all the concept data into a single tensor, then applies a logit transformation
    to the data to work in the correct space. The covariance matrix is computed from the transformed data
    and brought into a lower triangular form using Cholesky decomposition. In comments, an alternative
    covariance computation is provided that can be used if the dataset is too large to fit into memory.

    Args:
        dataloader (torch.utils.data.DataLoader): A dataloader containing batches of data with a "concepts" key.

    Returns:
        torch.Tensor: The lower triangular form of the empirical covariance matrix.
    """
    model.eval()
    with torch.no_grad():
        data = []
        tmp_dataloader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=True,
            num_workers=dataloader.num_workers,
            pin_memory=True,
            generator=dataloader.generator,
            drop_last=False, # That's the actual parameter that I care about
        )
        data_to_load = int(ratio * len(tmp_dataloader.dataset))
        loaded_data = 0

        for batch in tmp_dataloader:
            features = batch["features"]
            # Calculate concept logits with CBM_model
            c_logits,_,_ = model(features, 300, validation=True)
            loaded_data += c_logits.shape[0]
            if loaded_data > data_to_load:
                excess = loaded_data - data_to_load
                data.append(c_logits[:-excess])
                print(
                    f"Computing empirical covariance with {loaded_data-excess} out of total {len(tmp_dataloader.dataset)} samples.")
                break
            data.append(c_logits)
        data = torch.cat(data)  # Concatenate all data into a single tensor
        covariance = torch.cov(data.transpose(0, 1))

        # Bringing it into lower triangular form
        if scaling_factor:
            rows, cols = torch.tril_indices(row=covariance.shape[1], col=covariance.shape[1], offset=-1)
            covariance[rows, cols] /= scaling_factor
            covariance[cols, rows] /= scaling_factor
        covariance = numerical_stability_check(covariance, device="cpu")
        lower_triangle = torch.linalg.cholesky(covariance)

    # ###### Alternative cov computation if dataset was too large for memory
    # num_samples = 0
    #
    # # Calculate concept mean
    # for i, batch in enumerate(dataloader):
    #     features = batch["features"]
    #     if i == 0:
    #         c_logits = CBM_model.concept_predictor(CBM_model.encoder(features)).sum(0)
    #     else:
    #         c_logits += CBM_model.concept_predictor(CBM_model.encoder(features)).sum(0)
    #     num_samples += c_logits.shape[0]
    # logits_mean = c_logits / num_samples
    #
    # # Calculate covariance matrix
    # for i, batch in enumerate(dataloader):
    #     features = batch["features"]
    #     c_logits = CBM_model.concept_predictor(CBM_model.encoder(features)).sum(0)
    #     temp = (c_logits - logits_mean).unsqueeze(-1)
    #     if i == 0:
    #         cov = torch.matmul(temp, temp.transpose(-2, -1)).sum(0)
    #     else:
    #         cov += torch.matmul(temp, temp.transpose(-2, -1)).sum(0)
    # cov = cov / num_samples
    # lower_triangle = torch.linalg.cholesky(cov)

    ########
    return lower_triangle, covariance


def get_concept_groups(config):
    """
    Retrieve the concept groups based on the dataset specified in the configuration.

    This function retrieves the concept groups based on the dataset specified in the configuration.
    This is used for plotting the heatmap of the correlation matrix with the correct concept names.

    Args:
        config (DictConfig): The configuration dictionary.

    Returns:
        list: A list of concept names.
    """
    if config.dataset == "CUB":
        # Oracle grouping based on concept type for CUB
        with open(
            os.path.join(config.data_path, "CUB/CUB_200_2011/concept_names.txt"),
            "r",
        ) as f:
            concept_names = []
            for line in f:
                concept_names.append(line.replace("\n", "").split("::"))
        concept_names_graph = [": ".join(name) for name in concept_names]

    else:
        concept_names_graph = [str(i) for i in range(config.num_concepts)]

    return concept_names_graph
