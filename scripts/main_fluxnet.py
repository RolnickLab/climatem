# Here we have a quick main where we are testing data loading with different ensemble members and ideally with different climate models.
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from climatem.data_loader.fluxnet_dataset import FluxnetDataModule
from climatem.model.train_control_model import TrainingFluxnet
from climatem.model.tsdcd_control import TSDCD

torch.set_warn_always(False)

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")

class Bunch:
    """A class that has one variable for each entry of a dictionary."""

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def to_dict(self):
        return self.__dict__


def main(
    experiment_params, data_params, train_params, model_params, optim_params, plot_params
):
    
    """
    :param hp: object containing hyperparameter values
    """
    t0 = time.time()

    # Control as much randomness as possible
    torch.manual_seed(experiment_params.random_seed)
    np.random.seed(experiment_params.random_seed)

    if experiment_params.gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # generate data and split train/test
    device = torch.device("cuda" if (torch.cuda.is_available() and experiment_params.gpu) else "cpu")

    datamodule = FluxnetDataModule(
        data_path=data_params.csv_path,

        # rolling_mean=data_params.rolling_mean,
        growing_season_filter_ndaysabove=data_params.growing_season_filter_ndaysabove,
        deseasonalize=data_params.deseasonalize,

        val_split=data_params.val_split,

        batch_size=data_params.batch_size,
        num_workers=experiment_params.num_workers,
        pin_memory=experiment_params.pin_memory,
    )
    datamodule.setup(accelerator)

    # set the model
    model = TSDCD(
        num_layers=model_params.num_layers,
        num_hidden=model_params.num_hidden,
        position_embedding_dim=model_params.position_embedding_dim,
        distr_x="gaussian",
        d_x=experiment_params.d_x,
        d_z=experiment_params.d_z,
        hard_gumbel=model_params.hard_gumbel,
    )


    # Make folder to save run results
    exp_path = Path(experiment_params.exp_path)
    os.makedirs(exp_path, exist_ok=True)
    name = f"spars_low_acy_only_lr_{train_params.lr}_bs_{data_params.batch_size}_spmuin_{optim_params.sparsity_mu_init}_spth_{optim_params.sparsity_upper_threshold}_numhid_{model_params.num_hidden}_embdim_{model_params.position_embedding_dim}"
    exp_path = exp_path / name
    os.makedirs(exp_path, exist_ok=True)

    # create path to exp and save hyperparameters
    save_path = exp_path / "training_results"
    os.makedirs(save_path, exist_ok=True)
    plots_path = exp_path / "plots"
    os.makedirs(plots_path, exist_ok=True)
    # Here could maybe implement a "save()" function inside each class
    hp = {}
    hp["exp_params"] = experiment_params.__dict__
    hp["data_params"] = data_params.__dict__
    hp["train_params"] = train_params.__dict__
    hp["model_params"] = model_params.__dict__
    hp["optim_params"] = optim_params.__dict__
    with open(exp_path / "params.json", "w") as file:
        json.dump(hp, file, indent=4)

    # train, always with the latent version
    trainer = TrainingFluxnet(
        model,
        datamodule,
        train_params,
        optim_params,
        plot_params,
        save_path,
        plots_path,
        accelerator,
        wandbname=name,
    )

    # where is the model at this point?
    print("Where is my model?", next(trainer.model.parameters()).device)

    valid_loss = trainer.train_with_QPM()

    print(f"valid loss {valid_loss}")

class expParams():
    def __init__(
        self,
        exp_path = "/network/scratch/j/julien.boussard/results/FLUXNET/",
        d_z = 2,
        d_x = 5,
        random_seed = 1,
        gpu = True,
        num_workers = 0,
        pin_memory = False,
        verbose = True
    ):
        self.exp_path = exp_path
        self.d_z = d_z
        self.d_x = d_x
        self.random_seed = random_seed
        self.gpu = gpu
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose

class dataParams():
    def __init__(
        self,
        csv_path = "/network/scratch/j/julien.boussard/FLUXNET_data.csv",
        growing_season_filter_ndaysabove = 10,
        deseasonalize = True,
        val_split = 0.1,
        batch_size = 128,
    ):
        self.csv_path = csv_path
        self.growing_season_filter_ndaysabove = growing_season_filter_ndaysabove
        self.deseasonalize = deseasonalize
        self.val_split = val_split
        self.batch_size = batch_size

class trainParams():
    def __init__(
        self,
        lr = 0.001,
        lr_scheduler_epochs = [10000, 25000, 50000],
        lr_scheduler_gamma = 1,
        max_iteration = 200000,
        patience = 5000,
        patience_post_thresh = 50,
        valid_freq = 5
    ):
        self.lr = lr
        self.lr_scheduler_epochs = lr_scheduler_epochs
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.patience = patience
        self.max_iteration = max_iteration
        self.patience_post_thresh = patience_post_thresh
        self.valid_freq = valid_freq

class modelParams():
    def __init__(
        self,
        num_hidden = 0,
        num_layers = 0,
        position_embedding_dim = 0,
        hard_gumbel = False
    ):
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.position_embedding_dim = position_embedding_dim
        self.hard_gumbel = hard_gumbel

class plotParams():
    def __init__(
        self,
        plot_freq = 1000,
        print_freq = 1000,
    ):
        self.plot_freq = plot_freq
        self.print_freq = print_freq

class optimParams():
    def __init__(
        self,
        optimizer = "rmsprop",
        use_sparsity_constraint = True,
        binarize_transition = True,
        update_sparsity_after_acyclicity = False,

        reg_coeff = 0.12801,
        reg_coeff_connect = 0,

        crps_coeff = 1,
    
        schedule_reg = 0,
        schedule_ortho = 0,
        schedule_sparsity = 0,    

        sparsity_mu_init = 1e-5,
        sparsity_mu_mult_factor = 1.2,
        sparsity_omega_gamma = 0.0001,
        sparsity_omega_mu = 0.9,
        sparsity_h_threshold = 1e-2,
        sparsity_min_iter_convergence = 1000,
        sparsity_upper_threshold = 0.25,
        sparsity_first_threshold = 0.5,

        acyclic_mu_init = 1e-5,
        acyclic_mu_mult_factor = 1.2,
        acyclic_omega_gamma = 0.01,
        acyclic_omega_mu = 0.9,
        acyclic_h_threshold = 1e-2,
        acyclic_min_iter_convergence = 1000,
    ):
        self.optimizer = optimizer
        self.use_sparsity_constraint = use_sparsity_constraint
        self.binarize_transition = binarize_transition
        self.update_sparsity_after_acyclicity = update_sparsity_after_acyclicity

        self.reg_coeff = reg_coeff
        self.reg_coeff_connect = reg_coeff_connect

        self.crps_coeff = crps_coeff

        self.schedule_reg = schedule_reg
        self.schedule_ortho = schedule_ortho
        self.schedule_sparsity = schedule_sparsity

        self.sparsity_mu_init = sparsity_mu_init
        self.sparsity_mu_mult_factor = sparsity_mu_mult_factor
        self.sparsity_omega_gamma = sparsity_omega_gamma
        self.sparsity_omega_mu = sparsity_omega_mu
        self.sparsity_h_threshold = sparsity_h_threshold
        self.sparsity_min_iter_convergence = sparsity_min_iter_convergence
        self.sparsity_upper_threshold = sparsity_upper_threshold
        self.sparsity_first_threshold = sparsity_first_threshold

        self.acyclic_mu_init = acyclic_mu_init
        self.acyclic_mu_mult_factor = acyclic_mu_mult_factor
        self.acyclic_omega_gamma = acyclic_omega_gamma
        self.acyclic_omega_mu = acyclic_omega_mu
        self.acyclic_h_threshold = acyclic_h_threshold
        self.acyclic_min_iter_convergence = acyclic_min_iter_convergence

if __name__ == "__main__":

    experiment_params = expParams()
    data_params = dataParams()
    train_params = trainParams()
    model_params = modelParams()
    plot_params = plotParams()
    optim_params = optimParams()


    main(experiment_params, data_params, train_params, model_params, optim_params, plot_params)



