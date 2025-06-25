# pylint: disable=not-callable
"""
SCBM and baseline models.
"""

import os
from pathlib import Path
import math
import torch
from torch import nn
from torch.distributions import RelaxedBernoulli, MultivariateNormal
import torch.nn.functional as F
from torchvision import models
from omegaconf import DictConfig, OmegaConf
import yaml

from models.networks import FCNNEncoder
from utils.training import freeze_module, unfreeze_module
from utils.data import get_empirical_covariance
from utils.utils import numerical_stability_check

def create_model(config: DictConfig):
    """
    Parse the configuration file and return a relevant model
    """
    if config.model.model == "cbm":
        return CBM(config)
    elif config.model.model == "scbm":
        return SCBM(config)
    elif config.model.model == "pscbm": # Post-Hoc SCBM
        return PSCBM(config)
    else:
        print("Could not create model with name ", config.model, "!")
        quit()

def load_weights(model: nn.Module, config: DictConfig):
    """

    Args:
        model: the model for which weights should be loaded
        config: configuration file

    Returns:
        If a model_dir is explicitly specified in the configurations file, load weights from this file.
        If no model_dir is specified, iterate through all weights files which correspond to the same
        model, concept learning and dataset configuration. If there are many such files, preferably load
        weights from one that also has a config file associated with it - one can then make sure that also
        cov_type and learning_mode are the same.
    """
    # If some exact path is specified, and it corresponds to an existing file, and
    # it has the correct pytorch extension, load it
    weights_loaded = False
    model_dir=None
    if 'model_dir' in config.model.keys() and Path(config.model['model_dir']).is_file() and Path(
            config.model['model_dir']).suffix in ('.pth', '.pt'):
        model_dir = config.model.model_dir
        model.load_state_dict(torch.load(model_dir, weights_only=True, map_location=torch.device('cpu')))
        print(f"Model weights have been loaded from the specified file: {model_dir}.")
    # Otherwise, infer the path from model, concept learning and dataset information
    else:
        # experiment_type records information about the model, concept encoding and dataset
        experiment_type = Path(config.experiment_dir).parent
        # Get the first file that matches experiment_type and is a PyTorch file (we assume, it contains proper model weights)
        # try:
        #     model_dir = experiment_type.glob("**/*.pth").__next__()
        # except StopIteration:
        #     raise FileNotFoundError("No file to load CBM weights!")
        for model_dir in experiment_type.glob("**/*.pth"):
            """
            I want to add additional checks here, to make sure that weights are loaded
            from a model with the same configuration.
            If there is a configuration file in addition to the weights file, we can compare some configurations.
            The list might need to be expanded in the future.
            If no configuration file is present, the file will be recorded as a last resort.
            """
            if model_dir.parent.joinpath('config.yaml').is_file():
                with open(model_dir.parent.joinpath('config.yaml'), 'r') as file:
                    loaded_models_config = yaml.safe_load(file)
                if (
                        loaded_models_config.get('model').get('cov_type') == config.get('model').get('cov_type') and
                        loaded_models_config.get('model').get('training_mode') == config.get('model').get('training_mode')
                ):
                    model.load_state_dict(torch.load(model_dir, weights_only=True, map_location=torch.device('cpu')))
                    print(f"Loaded model weights from {model_dir}. cov_type and training_mode have been checked "
                          f"for concordance.\n")
                    weights_loaded = True
                    break
        if not weights_loaded:
            if model_dir:
                model.load_state_dict(torch.load(model_dir, weights_only=True, map_location=torch.device('cpu')))
                print(f"Loaded model weights from {model_dir}. cov_type and training_mode have NOT been checked "
                        f"for concordance.\n")
            else:
                raise FileNotFoundError("No model with corresponding configuration to load weights from it.")

class PSCBM(nn.Module):
    """
    This class implements a Post-Hoc Stochastic Concept Bottleneck Model (PSCBM), which adapts the Stochastic Concept Bottleneck Model (SCBM) to work in a post-hoc setting. The covariance matrix for the concepts is computed for a trained Concept Bottleneck Model (CBM), allowing for stochastic modeling of concept predictions, especially during interventions.
    Key Features:
    - Supports different types of covariance matrices for the concepts:
        - "identity": Identity covariance matrix.
        - "empirical_true": Empirical covariance computed over ground truth concept values.
        - "empirical_predicted": Empirical covariance computed over model-predicted concept values.
        - "empirical": equivalent to "empirical_true"
        - "global": Learnable global covariance matrix. I.e. the same indepentently of the sample
        - "amortized": Covariance matrix predicted by a neural network conditioned on input features. That is, it can adapt to a specific sample.
    - Loads and wraps a trained CBM loading pretrained weights. Training the CBM from scratch is not implemented
    - Provides a versatile forward method that can:
        - Return only the covariance matrix if requested.
        - Calculate concept probabilities and target logits either by using the underlying CBM or by sampling from a normal distribution defined by the concept prediction and the covariance matrix.
        - Optionally it returns or takes as argument intermediate encoder representations. This can significantly speed up the evaluation in case we need to reevaluate it multiple times for the same inputs and gradients for the encoder are not needed. Use cases for this: validation/test pass or training only the covariance predictor.
    - Supports concepts interventions.
    - Includes methods to freeze and unfreeze parts of the model for different training regimes.
        config: Configuration object containing model, data, and training parameters.
    Attributes:
        num_concepts (int): Number of concepts.
        num_classes (int): Number of target classes.
        encoder_arch (str): Architecture of the encoder.
        head_arch (str): Architecture of the head.
        training_mode (str): Training mode (e.g., "joint", "sequential").
        concept_learning (str): Concept learning mode (e.g., "hard", "soft", "embedding").
        num_monte_carlo (int): Number of Monte Carlo samples for stochastic inference.
        num_samples (int): Number of samples for concept sampling.
        straight_through (bool): Whether to use straight-through estimator for relaxed Bernoulli sampling.
        curr_temp (float): Current temperature for relaxed Bernoulli.
        num_epochs (int): Number of training epochs.
        cov_type (str): Type of covariance matrix used.
        CBM (nn.Module): Wrapped Concept Bottleneck Model.
        head (nn.Module): Head module for final prediction.
        encoder (nn.Module): Encoder module for feature extraction.
        concept_predictor (nn.Module): Module for predicting concepts.
        act_c (callable): Activation function for concept outputs.
        n_features (int): Number of features output by the encoder.
        pred_dim (int): Output dimension of the prediction head.
        sigma_concepts (Tensor or nn.Module): Covariance parameters (fixed, learnable, or amortized).
    Methods:
        forward(x, epoch, c_true=None, validation=False, return_full=True, intermediate=None, cov_only=False, return_intermediate=False, use_covariance=False):
            The forward method can return predictions, sampled concepts, and/or the covariance matrix.
            The intervene method performs intervention on the concepts and returns the model's prediction after intervention.
            The freeze_c method freezes the head module for concept learning.
            The freeze_t method freezes the encoder and concept predictor modules for target learning, and unfreezes the head.
    """

    def __init__(self, config):
        super(PSCBM, self).__init__()

        config_model = config.model
        self.num_concepts = config.data.num_concepts
        self.num_classes = config.data.num_classes
        self.encoder_arch = config_model.encoder_arch
        self.head_arch = config_model.head_arch
        # self.training_mode = config_model.training_mode
        self.training_mode = 'joint' # In PSCBM if we train covariance, we will need to backpropagate target error through concept sampling
        self.concept_learning = config_model.concept_learning
        self.num_monte_carlo = config_model.num_monte_carlo
        self.num_samples = config_model.get("num_samples", self.num_monte_carlo)
        self.straight_through = config_model.straight_through
        self.curr_temp = torch.tensor(0.5) # Final temperature of SCBM
        if self.training_mode == "joint":
            self.num_epochs = config_model.j_epochs
        else:
            self.num_epochs = config_model.t_epochs
        self.cov_type = config_model.cov_type

        #Architecture is exported to a sub-class:
        self.CBM = CBM(config)
        if config_model.get('load_weights', False):
            # If some exact path is specified and it corresponds to an existing file and
            # it has the correct pytorch extension, load it
            CBM_dir = None
            message = ""
            if 'CBM_dir' in config_model.keys() and Path(config_model['CBM_dir']).is_file() and Path(
            config_model['CBM_dir']).suffix in ('.pth', '.pt'):
                CBM_dir = config_model.CBM_dir
                message = f"Loaded CBM weights from the file specified in configurations: {config_model['CBM_dir']}"
            # Otherwise, infer the path from model, concept learning and dataset information
            else:
                #experiment_type records information about the model, concept encoding and dataset
                experiment_type = Path(config.experiment_dir).parent
                path_parts = list(experiment_type.parts)
                # Replace 'pscbm' with 'cbm'
                path_parts = ['cbm' if part == 'pscbm' else part for part in path_parts]
                experiment_type = Path(*path_parts)

                for CBM_dir in experiment_type.glob("**/*.pth"):
                    if CBM_dir.parent.joinpath('config.yaml').is_file():
                        with open(CBM_dir.parent.joinpath('config.yaml'), 'r') as file:
                            loaded_models_config = yaml.safe_load(file)
                        if (
                                # loaded_models_config.model.get('cov_type') == config.model.get('cov_type') and
                                loaded_models_config.get('model').get('training_mode') == config.model.get('training_mode')
                        ):
                            message = f"""Loaded model weights from {CBM_dir}. training_mode has been checked
                                for concordance.\n"""
                            break

                # Check whether a message has ben generated - equivalent to finding a verified weights file
                if not message:
                    if CBM_dir:
                        message = f"""Loaded model weights from {CBM_dir}. training_mode has NOT
                         been checked for concordance.\n"""
                    else: # No CBM_dir has been found
                        raise FileNotFoundError("No model with corresponding configuration to load weights from it.")

            self.CBM.load_state_dict(torch.load(CBM_dir, weights_only=True, map_location=torch.device('cpu')))
            print(message)

        self.head = self.CBM.head
        self.encoder = self.CBM.encoder
        self.concept_predictor = self.CBM.concept_predictor
        self.act_c = self.CBM.act_c

        self.n_features = self.CBM.n_features
        self.pred_dim = self.CBM.pred_dim

        # Compute covariance if it is constant
        if self.cov_type in ("identity", "empirical_true", "empirical_predicted", "empirical"):
            self.sigma_concepts = torch.zeros(
                int(self.num_concepts * (self.num_concepts + 1) / 2)
            )
        # Initialize covariance if it is learnable
        elif self.cov_type == "global":
            self.sigma_concepts = nn.Parameter(
                torch.zeros(int(self.num_concepts *(self.num_concepts + 1) / 2))
                )
        elif self.cov_type == "amortized":
            self.sigma_concepts = nn.Linear(
                self.n_features,
                int(self.num_concepts * (self.num_concepts + 1) / 2),
                bias=True,
            )
            self.sigma_concepts.weight.data *= (
                0.01  # To prevent exploding covariance matrix at initialization
            )

        # In a latter step I want to implement training the covariance matrix for intervention efficacy.
        else:
            raise NotImplementedError("Other covariance types are not implemented.")


    def _compute_covariance(self, x, intermediate):
        
        if self.cov_type.startswith("empirical") or self.cov_type == "identity":
            concepts_cov = self.covariance.repeat(x.shape[0],1)
        elif self.cov_type == "global":
            c_sigma_triang = self.sigma_concepts.repeat(x.shape[0], 1)
        elif self.cov_type == "amortized":
            # This condition is only triggered if cov_only=True and intermediate=None
            if intermediate is None:
                intermediate = self.encoder(x)
            c_sigma_triang = self.sigma_concepts(intermediate)
        if not (self.cov_type.startswith("empirical") or self.cov_type == "identity"):
            # Create a lower-triangular matrix
            c_triang_cov = torch.zeros(
                (c_sigma_triang.shape[0], self.num_concepts, self.num_concepts),
                device=c_sigma_triang.device,
            )
            rows, cols = torch.tril_indices(
                row=self.num_concepts, col=self.num_concepts, offset=0
            )
            diag_idx = rows == cols
            c_triang_cov[:, rows, cols] = c_sigma_triang
            # Make the diagonal positive
            c_triang_cov[:, range(self.num_concepts), range(self.num_concepts)] = (
                F.softplus(c_sigma_triang[:, diag_idx]) + 1e-6
            )
            concepts_cov = c_triang_cov @ c_triang_cov.transpose(dim0=-2, dim1=-1)

        return concepts_cov

    def forward(self, x, epoch, c_true=None, validation=False, return_full=True, intermediate=None, cov_only=False, return_intermediate=False, use_covariance=False):
        """
        args:
        intermediate: if we are only interested in calculating the covariance, we can pass the encoder's features s.t. they don't have to be reevaluated.
        Returns:
        concepts_pred_probs
        target_pred_logits
        concepts
        concepts_cov: in full form, not triangular - This is my design choice
        use_covariance: if yes concepts logits are sampled from a normal distribution given by covariance and mu. It has no effect if cov_only==True

        Structure of this function:
        
        1. Create the covariance matrix. In "amortized" case evaluating the encoder may be necessary
        2. if cov_only, set other return variables to None
        3. in not cov_only, calculate other return variables:
            if use_covariance:
                a) Calculate intermediate = encoder(x) if intermediate is None
                b) Sample concepts
                c) Calculate target from concepts
            else:
                Get concept & target prediction directly from the CBM
        4. Return the appropriate tuple (including intermediate or not)
        
        This flow is a bit complicated but the advantage is that it allows various use cases like returning only the covariance matrix, using the covariance matrix to sample
        concepts or passing the intermediate encoder representation in order to speed up computations.
        """
        # Step 1
        concepts_cov = self._compute_covariance(x, intermediate)
        # Step 2
        if cov_only:
            concepts_pred_probs, target_pred_logits, concepts = None, None, None
            return_intermediate = False # Just in case someone passed incorrect arguments
        # Step 3
        else:
            if use_covariance:
                intermediate = self.encoder(x) if intermediate is None else intermediate
                c_mu = self.concept_predictor(intermediate)
                # concepts_cov = numerical_stability_check(concepts_cov, concepts_cov.device)
                c_dist = MultivariateNormal(c_mu, covariance_matrix=concepts_cov)
                c_mcmc_logit = c_dist.rsample([self.num_samples]).movedim(0, -1) # [batch_size, num_concepts, num_samples]
                c_mcmc_prob = self.act_c(c_mcmc_logit)
                # At this point original SCBM implementation kicks in
                # START COPY-PASTE
                # For all MCMC samples simultaneously sample from Bernoulli
                if validation or self.training_mode == "sequential":
                    # No backpropagation necessary - we only train the head after concept predictor has been trained
                    c_mcmc = torch.bernoulli(c_mcmc_prob)
                elif self.training_mode == "independent":
                    c_mcmc = c_true.unsqueeze(-1).repeat(1, 1, self.num_monte_carlo).float() # True labels as input for head
                else:
                    # Backpropagation necessary
                    curr_temp = self.curr_temp.to(c_mcmc_prob.device)
                    dist = RelaxedBernoulli(temperature=curr_temp, probs=c_mcmc_prob)

                    # Bernoulli relaxation
                    mcmc_relaxed = dist.rsample()
                    if self.straight_through:
                        # Straight-Through Gumbel Softmax
                        mcmc_hard = (mcmc_relaxed > 0.5) * 1
                        c_mcmc = mcmc_hard - mcmc_relaxed.detach() + mcmc_relaxed
                    else:
                        c_mcmc = mcmc_relaxed

                # I replaced the for loop from the original code with a vectorized version. The result is the same but it takes roughly 30 times less with 100 MCMC samples and batch size 384
                if self.concept_learning == "hard":
                    c = c_mcmc
                elif self.concept_learning == "soft":
                    c = c_mcmc_logit
                else:
                    # I could implement CEM here if time is
                    raise NotImplementedError
                # Move concepts dimension to the front. Pass everything at once through the head
                c = c.movedim(-1, 0)
                y_pred_logits = self.head(c)  # [num_monte_carlo, batch_size, num_classes]
                if self.pred_dim == 1:
                    y_pred_probs = torch.sigmoid(y_pred_logits).mean(0)
                else:
                    y_pred_probs = torch.softmax(y_pred_logits, dim=-1).mean(0)
                # Again I ask: Why do we calculate logits and then probabilities from them is obvious. But why do we move back to the logit space then?
                if self.pred_dim == 1:
                    y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
                else:
                    y_pred_logits = torch.log(y_pred_probs + 1e-6)

                concepts_pred_probs = c_mcmc_prob.mean(-1)
                target_pred_logits = y_pred_logits
                concepts = mcmc_relaxed # Concept probabilities sampled from the Relaxed Bernoulli
            
            # Don't use covariance - use vanilla CBM
            else:
                if return_intermediate:
                    concepts_pred_probs, target_pred_logits, concepts, intermediate = self.CBM(x, epoch, c_true=c_true, validation=validation, intermediate=intermediate, return_intermediate=True)
                else:
                    concepts_pred_probs, target_pred_logits, concepts = self.CBM(x, epoch, c_true=c_true, validation=validation, intermediate=intermediate)
                
        # Step 4
        if return_intermediate:
            return concepts_pred_probs, target_pred_logits, concepts, concepts_cov, intermediate
        else:
            return concepts_pred_probs, target_pred_logits, concepts, concepts_cov

    def intervene(self, concepts_intervened_logits, c_mask, input_features, c_true):
        """
        This function does de facto the same as the corresponding function in regular CBM. It is however simplified,
        because autoregressive mode is not supported. 
        It is of course assumed that concept correlations have already been applied to the passed probabilities.
        Args:
            concepts_mu_intervened: concepts LOGITS after intervention
            c_mask:
            input_features:
            c_true:

        Returns:
            y_pred_logits: the model's prediction after correcting the concepts.
        """
        concepts_intervened_probs=self.act_c(concepts_intervened_logits)

        if self.concept_learning in ("hard", "autoregressive", "embedding"):
            # Set intervened-on hard concepts to 0/1
            concepts_intervened_probs = (c_true * c_mask) + concepts_intervened_probs * (1 - c_mask)

        return self.CBM.intervene(concepts_intervened_probs, c_mask, input_features, None)

    def compute_temperature(self, epoch, device):
        final_temp = torch.tensor([0.5], device=device)
        init_temp = torch.tensor([1.0], device=device)
        rate = (math.log(final_temp) - math.log(init_temp)) / float(self.num_epochs)
        curr_temp = max(init_temp * math.exp(rate * epoch), final_temp)
        self.curr_temp = curr_temp
        return curr_temp

    def freeze_c(self):
        self.CBM.head.apply(freeze_module)

    def freeze_t(self):
        self.CBM.head.apply(unfreeze_module)
        self.CBM.encoder.apply(freeze_module)
        self.CBM.concept_predictor.apply(freeze_module)


class SCBM(nn.Module):
    """
    Stochastic Concept Bottleneck Model (SCBM) with Learned Covariance Matrix.

    This class implements a Stochastic Concept Bottleneck Model (SCBM) that extends concept prediction by incorporating
    a learned covariance matrix. The SCBM aims to capture the uncertainty and dependencies between concepts, providing
    a more robust and interpretable model for concept-based learning tasks.

    Key Features:
    - Predicts concepts along with a learned covariance matrix to model the relationships and uncertainties between concepts.
    - Supports various training modes and intervention strategies to improve model performance and interpretability.

    Args:
        config (dict): Configuration dictionary containing model and data settings.

    Noteworthy Attributes:
        training_mode (str): The training mode (e.g., "joint", "sequential", "independent").
        num_monte_carlo (int): The number of Monte Carlo samples for uncertainty estimation.
        straight_through (bool): Flag indicating whether to use straight-through gradients.
        curr_temp (float): The current temperature for the Gumbel-Softmax distribution.
        cov_type (str): The type of covariance matrix ("empirical", "global", or "amortized", where "empirical is fixed at start").

    Methods:
        forward(x, epoch, validation=False, c_true=None):
            Perform a forward pass through the model.
        intervene(c_mcmc_probs, c_mcmc_logits):
            Perform an intervention on the model's concept predictions.
    """

    def __init__(self, config: DictConfig):
        super(SCBM, self).__init__()

        # Configuration arguments
        config_model = config.model
        self.num_concepts = config.data.num_concepts
        self.num_classes = config.data.num_classes
        self.encoder_arch = config_model.encoder_arch
        self.head_arch = config_model.head_arch
        self.training_mode = config_model.training_mode
        self.concept_learning = config_model.concept_learning
        self.num_monte_carlo = config_model.num_monte_carlo
        self.straight_through = config_model.straight_through
        self.curr_temp = 1.0
        if self.training_mode == "joint":
            self.num_epochs = config_model.j_epochs
        else:
            self.num_epochs = config_model.t_epochs
        self.cov_type = config_model.cov_type

        # Architectures
        # Encoder h(.)
        if self.encoder_arch == "FCNN":
            self.n_features = 256
            self.encoder = FCNNEncoder(
                num_inputs=config.data.num_covariates, num_hidden=self.n_features, num_deep=2
            )
        elif self.encoder_arch == "resnet18":
            self.encoder_res = models.resnet18(weights=None)
            self.encoder_res.load_state_dict(
                torch.load(
                    os.path.join(
                        config_model.model_directory, "resnet/resnet18-5c106cde.pth"
                    )
                )
            )

            self.n_features = self.encoder_res.fc.in_features
            self.encoder_res.fc = Identity()
            self.encoder = nn.Sequential(self.encoder_res)

        elif self.encoder_arch == "simple_CNN":
            self.n_features = 256
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(9216, self.n_features),
                nn.ReLU(),
            )

        else:
            raise NotImplementedError("ERROR: architecture not supported!")

        self.mu_concepts = nn.Linear(self.n_features, self.num_concepts, bias=True)

        if self.cov_type == "global":
            self.sigma_concepts = nn.Parameter(
                torch.zeros(int(self.num_concepts * (self.num_concepts + 1) / 2))
            )  # Predict lower triangle of concept covariance
        elif self.cov_type in ("empirical", "empirical_true"):
            self.sigma_concepts = torch.zeros(
                int(self.num_concepts * (self.num_concepts + 1) / 2)
            )
        else: # "amortized"
            self.sigma_concepts = nn.Linear(
                self.n_features,
                int(self.num_concepts * (self.num_concepts + 1) / 2),
                bias=True,
            )
            self.sigma_concepts.weight.data *= (
                0.01  # To prevent exploding covariance matrix at initialization
            )

        # Assume binary concepts
        self.act_c = nn.Sigmoid()

        # Link function g(.)
        if self.num_classes == 2:
            self.pred_dim = 1
        elif self.num_classes > 2:
            self.pred_dim = self.num_classes

        if self.head_arch == "linear":
            fc_y = nn.Linear(self.num_concepts, self.pred_dim)
            self.head = nn.Sequential(fc_y)
        else:
            fc1_y = nn.Linear(self.num_concepts, 256)
            fc2_y = nn.Linear(256, self.pred_dim)
            self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)
        if config_model.get("load_weights", False):
            load_weights(self, config)

    def forward(self, x, epoch, validation=False, return_full=False, c_true=None):
        """
        Perform a forward pass through the Stochastic Concept Bottleneck Model (SCBM).

        This method performs a forward pass through the SCBM, predicting concept probabilities and logits for the target variable.

        Args:
            x (torch.Tensor): The input covariates. Shape: (batch_size, input_dims)
            epoch (int): The current epoch number.
            validation (bool, optional): Flag indicating whether this is a validation pass. Default is False.
            return_full (bool, optional): Flag indicating whether to also return mu of concept. Default is False.
            c_true (torch.Tensor, optional): The ground-truth concept values. Required for "independent" training mode. Default is None.

        Returns:
            tuple: A tuple containing:
                - c_mcmc_prob (torch.Tensor): MCMC samples for predicted concept probabilities. Shape: (batch_size, num_concepts, num_monte_carlo)
                - c_triang_cov (torch.Tensor): Cholesky decomposition of the concept logit covariance matrix. Shape: (batch_size, num_concepts, num_concepts)
                - y_pred_logits (torch.Tensor): Logits for the target variable. Shape: (batch_size, num_classes)
                - c_mu (torch.Tensor, optional): Predicted concept means. Shape: (batch_size, num_concepts). Returned if `return_full` is True.
        Notes:
            - The method first obtains intermediate representations from the encoder.
            - It then predicts the concept means and the Cholesky decomposition of the covariance matrix in the logit space.
            - The method samples from the predicted normal distribution to obtain concept logits and probabilities.
            - Depending on the training mode, it handles different strategies for sampling and backpropagation.
            - Finally, it predicts the target variable logits by averaging over multiple Monte Carlo samples.
        """

        # Get intermediate representations
        intermediate = self.encoder(x)

        # Get mu and cholesky decomposition of covariance
        c_mu = self.mu_concepts(intermediate)
        if self.cov_type == "global":
            c_sigma = self.sigma_concepts.repeat(c_mu.size(0), 1)
        elif self.cov_type in ("empirical", "empirical_true"):
            c_sigma = self.sigma_concepts.unsqueeze(0).repeat(c_mu.size(0), 1, 1)
        else: # "amortized"
            c_sigma = self.sigma_concepts(intermediate)

        if self.cov_type in ("empirical", "empirical_true"):
            c_triang_cov = c_sigma
        else:
            # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
            c_triang_cov = torch.zeros(
                (c_sigma.shape[0], self.num_concepts, self.num_concepts),
                device=c_sigma.device,
            )
            rows, cols = torch.tril_indices(
                row=self.num_concepts, col=self.num_concepts, offset=0
            )
            diag_idx = rows == cols
            c_triang_cov[:, rows, cols] = c_sigma
            c_triang_cov[:, range(self.num_concepts), range(self.num_concepts)] = (
                F.softplus(c_sigma[:, diag_idx]) + 1e-6
            )

        # Sample from predicted normal distribution
        c_dist = MultivariateNormal(c_mu, scale_tril=c_triang_cov)
        c_mcmc_logit = c_dist.rsample([self.num_monte_carlo]).movedim(
            0, -1
        )  # [batch_size,num_concepts,mcmc_size]
        c_mcmc_prob = self.act_c(c_mcmc_logit)

        # For all MCMC samples simultaneously sample from Bernoulli
        if validation or self.training_mode == "sequential":
            # No backpropagation necessary
            c_mcmc = torch.bernoulli(c_mcmc_prob)
        elif self.training_mode == "independent":
            c_mcmc = c_true.unsqueeze(-1).repeat(1, 1, self.num_monte_carlo).float()
        else:
            # Backpropagation necessary
            curr_temp = self.compute_temperature(epoch, device=c_mcmc_prob.device)
            dist = RelaxedBernoulli(temperature=curr_temp, probs=c_mcmc_prob)

            # Bernoulli relaxation
            mcmc_relaxed = dist.rsample()
            if self.straight_through:
                # Straight-Through Gumbel Softmax
                mcmc_hard = (mcmc_relaxed > 0.5) * 1
                c_mcmc = mcmc_hard - mcmc_relaxed.detach() + mcmc_relaxed
            else:
                c_mcmc = mcmc_relaxed

        # MCMC loop for predicting label has been replaced by a vectorized version. Same result but much faster.
        if self.concept_learning == "hard":
            c = c_mcmc
        elif self.concept_learning == "soft":
            c = c_mcmc_logit
        else:
            # I could implement CEM here if time is
            raise NotImplementedError
        # Move concepts dimension to the front. Pass everything at once through the head
        c = c.movedim(-1, 0)
        y_pred_logits = self.head(c)  # [num_monte_carlo, batch_size, num_classes]
        if self.pred_dim == 1:
            y_pred_probs = torch.sigmoid(y_pred_logits).mean(0)
        else:
            y_pred_probs = torch.softmax(y_pred_logits, dim=-1).mean(0)
        # Again I ask: Why do we calculate logits and then probabilities from them is obvious. But why do we move back to the logit space then?
        if self.pred_dim == 1:
            y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
        else:
            y_pred_logits = torch.log(y_pred_probs + 1e-6)

        # Return concept mu for interventions
        if return_full:
            return c_mcmc_prob, c_mu, c_triang_cov, y_pred_logits
        else:
            return c_mcmc_prob, c_triang_cov, y_pred_logits

    def intervene(self, c_mcmc_probs, c_mcmc_logits):
        y_pred_probs_i = 0
        c_hard = torch.bernoulli(c_mcmc_probs)
        for i in range(self.num_monte_carlo):
            if self.concept_learning == "soft":
                c_i = c_mcmc_logits[:, :, i]
            else:
                c_i = c_hard[:, :, i]

            y_pred_logits_i = self.head(c_i)
            if self.pred_dim == 1:
                y_pred_probs_i += torch.sigmoid(y_pred_logits_i)
            else:
                y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)

        y_pred_probs = y_pred_probs_i / self.num_monte_carlo
        if self.pred_dim == 1:
            y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
        else:
            y_pred_logits = torch.log(y_pred_probs + 1e-6)

        return y_pred_logits

    def compute_temperature(self, epoch, device):
        final_temp = torch.tensor([0.5], device=device)
        init_temp = torch.tensor([1.0], device=device)
        rate = (math.log(final_temp) - math.log(init_temp)) / float(self.num_epochs)
        curr_temp = max(init_temp * math.exp(rate * epoch), final_temp)
        self.curr_temp = curr_temp
        return curr_temp

    def freeze_c(self):
        self.head.apply(freeze_module)

    def freeze_t(self):
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)
        self.mu_concepts.apply(freeze_module)
        if isinstance(self.sigma_concepts, nn.Linear):
            self.sigma_concepts.apply(freeze_module)
        else:
            self.sigma_concepts.requires_grad = False


class CBM(nn.Module):
    """
    Model class encompassing all baselines: Hard & Soft Concept Bottleneck Model (CBM),
                                            Concept Embedding Model (CEM), and Autoregressive CBM (AR).

    This class implements the baselines. Depending on the choice of model, only a small part of the full code is used.
    Check the if statements in the forward method to see which part of the code is used for which model.

    Args:
        config (dict): Configuration dictionary containing model and data settings.

    Noteworthy Attributes:
        training_mode (str): The training mode (e.g., "joint", "sequential", "independent").
        concept_learning (str): The concept learning method ("hard", "soft", "embedding", or "autoregressive").
                                This determines the type of method to use
        num_monte_carlo (int): The number of Monte Carlo samples for sampling Gumbel Softmax in AR.
        straight_through (bool): Flag indicating whether to use straight-through gradients.
        curr_temp (float): The current temperature for the Gumbel-Softmax distribution.
    """

    def __init__(self, config: DictConfig):
        super(CBM, self).__init__()

        # Configuration arguments
        config_model = config.model
        self.num_concepts = config.data.num_concepts
        self.num_classes = config.data.num_classes
        self.encoder_arch = config_model.encoder_arch
        self.head_arch = config_model.head_arch
        self.training_mode = config_model.training_mode
        self.concept_learning = config_model.concept_learning
        if self.concept_learning in ("hard", "autoregressive"):
            self.num_monte_carlo = config_model.num_monte_carlo
            self.straight_through = config_model.straight_through
            self.curr_temp = 1.0
            if self.training_mode == "joint":
                self.num_epochs = config_model.j_epochs
            else:
                self.num_epochs = config_model.t_epochs
        elif self.concept_learning == "embedding":
            self.CEM_embedding = config_model.embedding_size

        # Architectures
        # Encoder h(.)
        if self.encoder_arch == "FCNN":
            self.n_features = 256
            self.encoder = FCNNEncoder(
                num_inputs=config.data.num_covariates, num_hidden=self.n_features, num_deep=2
            )
        elif self.encoder_arch == "resnet18":
            self.encoder_res = models.resnet18(weights=None)
            self.encoder_res.load_state_dict(
                torch.load(
                    os.path.join(
                        config_model.model_directory, "resnet/resnet18-5c106cde.pth"
                    ), 
                    weights_only=True
                )
            )
            self.n_features = self.encoder_res.fc.in_features
            self.encoder_res.fc = Identity()
            self.encoder = nn.Sequential(self.encoder_res)

        elif self.encoder_arch == "simple_CNN":
            self.n_features = 256
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 32, 5, 3),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5, 3),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25),
                nn.Flatten(),
                nn.Linear(9216, self.n_features),
                nn.ReLU(),
            )

        else:
            raise NotImplementedError("ERROR: architecture not supported!")
        if self.concept_learning == "embedding":
            print(
                "Please be aware that our implementation of CEMs is without training on interventions! This is because we would deem this an unfair comparison to our method that is also not trained on interventions. Still, be careful when using this CEM code for derivative works"
            )
            self.positive_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.n_features, self.CEM_embedding, bias=True),
                        nn.LeakyReLU(),
                    )
                    for _ in range(self.num_concepts)
                ]
            )
            self.negative_embeddings = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.n_features, self.CEM_embedding, bias=True),
                        nn.LeakyReLU(),
                    )
                    for _ in range(self.num_concepts)
                ]
            )
            self.scoring_function = nn.Sequential(
                nn.Linear(self.CEM_embedding * 2, 1, bias=True), nn.Sigmoid()
            )
            self.concept_dim = self.CEM_embedding * self.num_concepts
        else:
            if self.concept_learning == "autoregressive":
                self.concept_predictor = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(self.n_features + i, 50, bias=True),
                            nn.LeakyReLU(),
                            nn.Linear(50, 1, bias=True),
                        )
                        for i in range(self.num_concepts)
                    ]
                )

            else:
                self.concept_predictor = nn.Linear(
                    self.n_features, self.num_concepts, bias=True
                )
            self.concept_dim = self.num_concepts

        # Assume binary concepts
        self.act_c = nn.Sigmoid()

        # Link function g(.)
        if self.num_classes == 2:
            self.pred_dim = 1
        elif self.num_classes > 2:
            self.pred_dim = self.num_classes

        if self.head_arch == "linear":
            fc_y = nn.Linear(self.concept_dim, self.pred_dim)
            self.head = nn.Sequential(fc_y)
        else:
            fc1_y = nn.Linear(self.concept_dim, 256)
            fc2_y = nn.Linear(256, self.pred_dim)
            self.head = nn.Sequential(fc1_y, nn.ReLU(), fc2_y)
        if config.model.get("load_weights", False) and config.model.model != 'pscbm':
            load_weights(self, config)

    def forward(
        self,
        x,
        epoch,
        c_true=None,
        validation=False,
        concepts_train_ar=False,
        intermediate=None,
        return_intermediate=False,
    ):
        """
        Perform a forward pass through one of the baselines.

        This method performs a forward pass predicting concept probabilities and logits for the target variable.
        It handles different concept learning strategies and training modes, including hard, soft, autoregressive, and embedding-based concepts.

        Args:
            x (torch.Tensor): The input covariates. Shape: (batch_size, input_dims)
            epoch (int): The current epoch number.
            c_true (torch.Tensor, optional): The ground-truth concept values. Required for "independent" training mode. Default is None.
            validation (bool, optional): Flag indicating whether this is a validation pass. Default is False.
            concepts_train_ar (torch.Tensor, optional): Ground-truth concept values for autoregressive training. Default is False.
            return_intermediate (bool, optional): Flag indicating whether to also return the encoder's intermediate values.
            This can be useful if we need to calculate amortized covariance based on this CBM.

        Returns:
            tuple: A tuple containing:
                - c_prob (torch.Tensor): Predicted concept probabilities. Shape: (batch_size, num_concepts)
                - y_pred_logits (torch.Tensor): Logits for the target variable. Shape: (batch_size, label_dim)
                - c (torch.Tensor): Predicted hard concept values (if method permits, otherwise the concept representation). Shape: (batch_size, num_concepts, num_monte_carlo) for MCMC sampling or (batch_size, num_concepts) otherwise.
        """

        # Get intermediate representations
        if intermediate is None:
            intermediate = self.encoder(x)

        # Get concept predictions
        if self.concept_learning in ("hard", "soft"):
            # CBM
            c_logit = self.concept_predictor(intermediate)
            c_prob = self.act_c(c_logit)

            if self.concept_learning in ("hard"):
                # Hard CBM
                if self.training_mode == "sequential" or validation:
                    # Sample from Bernoulli M times, as we don't need to backprop
                    c_prob_mcmc = c_prob.unsqueeze(-1).expand(
                        -1, -1, self.num_monte_carlo
                    )
                    c = torch.bernoulli(c_prob_mcmc)

                # Relax bernoulli sampling with Gumbel Softmax to allow for backpropagation
                elif self.training_mode == "joint":
                    curr_temp = self.compute_temperature(epoch, device=c_prob.device)
                    dist = RelaxedBernoulli(temperature=curr_temp, probs=c_prob)
                    c_relaxed = dist.rsample([self.num_monte_carlo]).movedim(0, -1)
                    if self.straight_through:
                        # Straight-Through Gumbel Softmax
                        c_hard = (c_relaxed > 0.5) * 1
                        c = c_hard - c_relaxed.detach() + c_relaxed
                    else:
                        # Reparametrization trick.
                        c = c_relaxed

                else:
                    raise NotImplementedError

        elif self.concept_learning == "autoregressive":
            # AR
            if validation:
                c_prob, c_hard = [], []
                for predictor in self.concept_predictor:
                    if c_prob:
                        concept = []
                        for i in range(
                            self.num_monte_carlo
                        ):  # MCMC samples for evaluation and interventions, but not for training
                            concept_input_i = torch.cat(
                                [intermediate, torch.cat(c_hard, dim=1)[..., i]], dim=1
                            )
                            concept.append(self.act_c(predictor(concept_input_i)))
                        concept = torch.cat(concept, dim=-1)
                        c_relaxed = torch.bernoulli(concept)[:, None, :]
                        concept = concept[:, None, :]
                        concept_hard = c_relaxed

                    else:
                        concept_input = intermediate
                        concept = self.act_c(predictor(concept_input))
                        concept = concept.unsqueeze(-1).expand(
                            -1, -1, self.num_monte_carlo
                        )
                        c_relaxed = torch.bernoulli(concept)
                        concept_hard = c_relaxed
                    c_prob.append(concept)
                    c_hard.append(concept_hard)
                c_prob = torch.cat([c_prob[i] for i in range(self.num_concepts)], dim=1)
                c = torch.cat([c_hard[i] for i in range(self.num_concepts)], dim=1)

            elif self.training_mode == "independent":
                # Training
                if c_true is None and concepts_train_ar is not False:
                    c_prob, c_hard = [], []
                    for c_idx, predictor in enumerate(self.concept_predictor):
                        if c_hard:
                            concept_input = torch.cat(
                                [intermediate, concepts_train_ar[:, :c_idx]], dim=1
                            )
                        else:
                            concept_input = intermediate
                        concept = self.act_c(predictor(concept_input))

                        # No Gumbel softmax because backprop can happen through the input connection
                        c_relaxed = torch.bernoulli(concept)
                        concept_hard = c_relaxed

                        # NOTE that the following train-time variables are overly good because they are taking ground truth as input. At validation time, we sample
                        c_prob.append(concept)
                        c_hard.append(concept_hard)
                    c_prob = torch.cat(
                        [c_prob[i] for i in range(self.num_concepts)], dim=1
                    )
                    c = torch.cat([c_hard[i] for i in range(self.num_concepts)], dim=1)

                else:  # Training the head with the GT concepts as input
                    c_prob = c_true.float()
                    c = c_true.float()

            else:
                raise NotImplementedError

        elif self.concept_learning == "embedding":
            # CEM
            if self.training_mode == "joint":
                # Obtaining concept embeddings
                c_p = [p(intermediate) for p in self.positive_embeddings]
                c_n = [n(intermediate) for n in self.negative_embeddings]

                # Concept probabilities from scoring function
                c_prob = [
                    self.scoring_function(torch.cat((c_p[i], c_n[i]), dim=1))
                    for i in range(self.num_concepts)
                ]

                # Final concept embedding
                z_prob = [
                    c_prob[i] * c_p[i] + (1 - c_prob[i]) * c_n[i]
                    for i in range(self.num_concepts)
                ]
                z_prob = torch.cat([z_prob[i] for i in range(self.num_concepts)], dim=1)
                c_prob = torch.cat([c_prob[i] for i in range(self.num_concepts)], dim=1)
                c = z_prob
            else:
                raise Exception("CEMs are trained jointly, change training mode")

        # Get predicted targets
        if self.concept_learning == "hard" or (
            self.concept_learning == "autoregressive" and validation
        ):
            # Hard CBM or validation of AR. Takes MCMC samples.
            # MCMC loop for predicting label
            y_pred_probs_i = 0
            for i in range(self.num_monte_carlo):
                c_i = c[:, :, i]
                y_pred_logits_i = self.head(c_i)
                if self.pred_dim == 1:
                    y_pred_probs_i += torch.sigmoid(y_pred_logits_i)
                else:
                    y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)
            y_pred_probs = y_pred_probs_i / self.num_monte_carlo

            if self.pred_dim == 1:
                y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
            else:
                y_pred_logits = torch.log(y_pred_probs + 1e-6)

        elif self.concept_learning == "soft":
            # Soft CBM
            y_pred_logits = self.head(
                c_logit
            )  # NOTE that we're passing logits not probs in soft case as is also done by Koh et al.
            c = torch.empty_like(c_prob)

        elif self.concept_learning == "embedding" or (
            self.concept_learning == "autoregressive" and not validation
        ):
            # CEM or training of AR. Takes ground truth concepts.
            # If CEM: c are predicte embeddings, if AR: c are ground truth concepts
            y_pred_logits = self.head(c)

        if return_intermediate:
            return c_prob, y_pred_logits, c, intermediate
        else:
            return c_prob, y_pred_logits, c

    def intervene(
        self,
        concepts_interv_probs,
        concepts_mask,
        input_features,
        concepts_pred_probs,
    ):
        if self.concept_learning == "soft":
            # Soft CBM
            c_logit = torch.logit(concepts_interv_probs, eps=1e-6)
            y_pred_logits = self.head(c_logit)

        elif self.concept_learning in ("hard", "autoregressive"):
            # Hard CBM or AR
            y_pred_probs_i = 0

            if self.concept_learning == "hard":
                c_prob_mcmc = concepts_interv_probs.unsqueeze(-1).expand(
                    -1, -1, self.num_monte_carlo
                )
                c = torch.bernoulli(c_prob_mcmc)

                # Fix intervened-on concepts to ground truth
                c[concepts_mask == 1] = (
                    concepts_interv_probs[concepts_mask == 1]
                    .unsqueeze(-1)
                    .expand(-1, self.num_monte_carlo)
                )
                weight = torch.ones((c.shape[0], self.num_monte_carlo), device=c.device)

            elif self.concept_learning == "autoregressive":
                # Note: Here, concepts_interv_probs are already the hard, MCMC sampled concepts as determined by the intervene_ar function
                id = torch.nonzero(
                    concepts_interv_probs * concepts_mask == 1, as_tuple=False
                )
                weight_k = torch.log(
                    1 - concepts_pred_probs + 1e-6
                )  # If intervened-on concepts have value 0
                weight_k.index_put_(
                    list(id.t()),
                    torch.log(concepts_pred_probs + 1e-6)[id[:, 0], id[:, 1], id[:, 2]],
                    accumulate=False,
                )  # If intervened-on concepts have value 1
                weight_k = (
                    weight_k * concepts_mask
                )  # Only compute weight for intervened-on concepts
                weight = torch.sum(weight_k, dim=(1))  # Sum over concepts
                weight = torch.softmax(
                    weight, dim=-1
                )  # Replicating their implementation (from log to prob space)
                c = concepts_interv_probs

            for i in range(self.num_monte_carlo):
                c_i = c[:, :, i]
                y_pred_logits_i = self.head(c_i)
                if self.pred_dim == 1:
                    y_pred_probs_i += weight[:, i].unsqueeze(1) * torch.sigmoid(
                        y_pred_logits_i
                    )
                else:
                    y_pred_probs_i += weight[:, i].unsqueeze(1) * torch.softmax(
                        y_pred_logits_i, dim=1
                    )
            y_pred_probs = y_pred_probs_i / torch.sum(weight, dim=1).unsqueeze(1)
            if self.pred_dim == 1:
                y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
            else:
                y_pred_logits = torch.log(y_pred_probs + 1e-6)

        elif self.concept_learning == "embedding":
            # CEM
            # Get intermediate representations
            intermediate = self.encoder(input_features)
            # Obtaining concept embeddings
            c_p = [p(intermediate) for p in self.positive_embeddings]
            c_n = [n(intermediate) for n in self.negative_embeddings]
            # Final concept embedding
            z_prob = [
                concepts_interv_probs[:, i].unsqueeze(1) * c_p[i]
                + (1 - concepts_interv_probs[:, i].unsqueeze(1)) * c_n[i]
                for i in range(self.num_concepts)
            ]
            z_prob = torch.cat([z_prob[i] for i in range(self.num_concepts)], dim=1)
            y_pred_logits = self.head(z_prob)

        return y_pred_logits

    def intervene_ar(self, concepts_true, concepts_mask, input_features):
        """
        Perform an intervention on the Autoregressive CBM.

        This method performs an intervention on the Autoregressive CBM by fixing the intervened-on concepts
        to their ground-truth values and MCMC sampling the remaining concepts.
        The predicted probabilities of the intervened-on concepts are stored nevertheless to compute the reweighting.
        The reweighting is computed afterwards using the intervene function.

        Args:
            concepts_true (torch.Tensor): The ground-truth concept values. Shape: (batch_size, num_concepts, num_monte_carlo)
            concepts_mask (torch.Tensor): A mask indicating which concepts are intervened. Shape: (batch_size, num_concepts, num_monte_carlo)
            input_features (torch.Tensor): The input features for the encoder. Shape: (batch_size, input_dims)

        Returns:
            tuple: A tuple containing:
                - c_prob (torch.Tensor): Predicted concept probabilities. Shape: (batch_size, num_concepts, num_monte_carlo)
                - c (torch.Tensor): Hard predicted concept values with interventions applied. Shape: (batch_size, num_concepts, num_monte_carlo)
        """
        # Concept predictions for autoregressive model. Intervened-on concepts are fixed to ground truth
        intermediate = self.encoder(input_features)
        c_prob, c_hard = [], []
        for j, (predictor) in enumerate(self.concept_predictor):
            if c_prob:
                concept = []
                for i in range(
                    self.num_monte_carlo
                ):  # MCMC samples for evaluation and interventions, but not for joint training
                    concept_input_i = torch.cat(
                        [intermediate, torch.cat(c_hard, dim=1)[..., i]], dim=1
                    )
                    concept.append(self.act_c(predictor(concept_input_i)))
                concept = torch.cat(concept, dim=-1)
                concept_hard = torch.bernoulli(concept)[:, None, :]
                concept = concept[:, None, :]
            else:
                concept_input = intermediate
                concept = self.act_c(predictor(concept_input))
                concept = concept.unsqueeze(-1).expand(-1, -1, self.num_monte_carlo)
                concept_hard = torch.bernoulli(concept)

            concept_hard = (
                concept_hard * (1 - concepts_mask[:, j, :])[:, None, :]
                + concepts_mask[:, j, :][:, None, :]
                * concepts_true[:, j, :][:, None, :]
            )  # Only update if it is not an intervened on
            concept = (
                concept * (1 - concepts_mask[:, j, :][:, None, :])
                + concepts_mask[:, j, :][:, None, :]
                * concepts_true[:, j, :][:, None, :]
            )

            c_prob.append(concept)
            c_hard.append(concept_hard)
        c_prob = torch.cat([c_prob[i] for i in range(self.num_concepts)], dim=1)
        c = torch.cat([c_hard[i] for i in range(self.num_concepts)], dim=1)
        return c_prob, c

    def compute_temperature(self, epoch, device):
        final_temp = torch.tensor([0.5], device=device)
        init_temp = torch.tensor([1.0], device=device)
        rate = (math.log(final_temp) - math.log(init_temp)) / float(self.num_epochs)
        curr_temp = max(init_temp * math.exp(rate * epoch), final_temp)
        self.curr_temp = curr_temp
        return curr_temp

    def freeze_c(self):
        self.head.apply(freeze_module)

    def freeze_t(self):
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)
        self.concept_predictor.apply(freeze_module)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
