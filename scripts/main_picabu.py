# Here we have a quick main where we are testing data loading with different ensemble members and ideally with different climate models.

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
torch.set_warn_always(False)
from accelerate import Accelerator

from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from climatem.model.metrics import edge_errors, mcc_latent, precision_recall, shd, w_mae
from climatem.model.tsdcd_latent import LatentTSDCD
from climatem.model.train_model import TrainingLatent
from climatem.config import *

accelerator = Accelerator()


class Bunch:
    """A class that has one variable for each entry of a dictionary."""

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def to_dict(self):
        return self.__dict__

    # def fancy_print(self, prefix=''):
    #     str_list = []
    #     for key, val in self.__dict__.items():
    #         str_list.append(prefix + f"{key} = {val}")
    #     return '\n'.join(str_list)


def main(        
        experiment_params,
        data_params,
        gt_params,
        train_params, 
        model_params, 
        optim_params, 
        plot_params,
        savar_params
):
    """
    :param hp: object containing hyperparameter values
    """
    t0 = time.time()

    # Control as much randomness as possible
    torch.manual_seed(experiment_params.random_seed)
    np.random.seed(experiment_params.random_seed)

    # Use GPU
    # TODO: Make everything Double instead of FLoat on GPU
    if experiment_params.gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    # Create folder
    # args.exp_path = os.path.join(args.exp_path, f"exp{args.exp_id}")
    # if not os.path.exists(args.exp_path):
    #     os.makedirs(args.exp_path)

    # generate data and split train/test
    if experiment_params.gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if data_params.data_format == "hdf5":
        print("IS HDF5")
        return
    else:
        datamodule = CausalClimateDataModule(
            tau=experiment_params.tau, 
            num_months_aggregated=data_params.num_months_aggregated, 
            train_val_interval_length=data_params.train_val_interval_length,
            in_var_ids = data_params.in_var_ids,
            out_var_ids = data_params.out_var_ids,
            train_years = data_params.train_years,
            train_historical_years = data_params.train_historical_years,
            test_years = data_params.test_years,  # do we want to implement keeping only certain years for testing?
            val_split = 1-train_params.ratio_train,  # fraction of testing to split for valdation
            seq_to_seq = data_params.seq_to_seq,  # if true maps from T->T else from T->1
            channels_last = data_params.channels_last,  # wheather variables come last our after sequence lenght
            train_scenarios = data_params.train_scenarios,
            test_scenarios = data_params.test_scenarios,
            train_models = data_params.train_models,
            # test_models = data_params.test_models,
            batch_size = data_params.batch_size,
            eval_batch_size = data_params.eval_batch_size,
            num_workers = experiment_params.num_workers,
            pin_memory = experiment_params.pin_memory,
            load_train_into_mem = data_params.load_train_into_mem,
            load_test_into_mem = data_params.load_test_into_mem,
            verbose = experiment_params.verbose,
            seed = experiment_params.random_seed,
            seq_len = data_params.seq_len,
            data_dir = data_params.climateset_data,
            output_save_dir = data_params.data_dir,
            num_ensembles = data_params.num_ensembles,  # 1 for first ensemble, -1 for all
            lon = experiment_params.lon,
            lat = experiment_params.lon,
            num_levels = data_params.num_levels,
            global_normalization = data_params.global_normalization,
            seasonality_removal = data_params.seasonality_removal,
            reload_climate_set_data = data_params.reload_climate_set_data,
            #Below SAVAR data arguments
            time_len = savar_params.time_len,
            comp_size = savar_params.comp_size,
            noise_val = savar_params.noise_val,
            n_per_col = savar_params.n_per_col,
            difficulty = savar_params.difficulty,
            seasonality = savar_params.seasonality,
            overlap = savar_params.overlap,
            is_forced = savar_params.is_forced,
            plot_original_data = savar_params.plot_original_data,
        ) 
        datamodule.setup()

    train_dataloader = iter(datamodule.train_dataloader())
    # val_dataloader = iter(datamodule.val_dataloader())

    # WE SHOULD REMOVE THIS, and initialize with params
    x, y = next(train_dataloader)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    # initialize model
    d = x.shape[2]
    print("d:", d)

    if model_params.instantaneous:
        print("Using instantaneous connections")
        num_input = d * (experiment_params.tau + 1) * (model_params.tau_neigh * 2 + 1)
    else:
        num_input = d * experiment_params.tau * (model_params.tau_neigh * 2 + 1)

    # set the model
    model = LatentTSDCD(
        num_layers=model_params.num_layers,
        num_hidden=model_params.num_hidden,
        num_input=num_input,
        num_output=2, #This should be parameterized somewhere? 
        num_layers_mixing=model_params.num_layers_mixing,
        num_hidden_mixing=model_params.num_hidden_mixing,
        coeff_kl=optim_params.coeff_kl,
        d=d,
        distr_z0="gaussian",
        distr_encoder="gaussian",
        distr_transition="gaussian",
        distr_decoder="gaussian",
        d_x=experiment_params.d_x,
        d_z=experiment_params.d_z,
        tau=experiment_params.tau,
        instantaneous=model_params.instantaneous,
        nonlinear_mixing=model_params.nonlinear_mixing,
        hard_gumbel=model_params.hard_gumbel,
        no_gt=gt_params.no_gt,
        debug_gt_graph=gt_params.debug_gt_graph,
        debug_gt_z=gt_params.debug_gt_z,
        debug_gt_w=gt_params.debug_gt_w,
        # gt_w=data_loader.gt_w,
        # gt_graph=data_loader.gt_graph,
        tied_w=model_params.tied_w,
        # also
        fixed=model_params.fixed,
        fixed_output_fraction=model_params.fixed_output_fraction,
    )

    # Make folder to save run results
    exp_path = Path(experiment_params.exp_path)
    os.makedirs(exp_path, exist_ok = True)
    data_var_ids_str = str(data_params.in_var_ids)[1:-1].translate({ord('\''):None}).translate({ord(','):None}).translate({ord(' '):None})
    name = f"var_{data_var_ids_str}_scenarios_{data_params.train_scenarios[0]}_nonlinear_{model_params.nonlinear_mixing}_tau_{experiment_params.tau}_z_{experiment_params.d_z}_lr_{train_params.lr}_spreg_{optim_params.reg_coeff}_ormuinit_{optim_params.ortho_mu_init}_spmuinit_{optim_params.sparsity_mu_init}_spthres_{optim_params.sparsity_upper_threshold}_fixed_{model_params.fixed}_num_ensembles_{data_params.num_ensembles}_instantaneous_{model_params.instantaneous}_crpscoef_{optim_params.crps_coeff}_spcoef_{optim_params.spectral_coeff}_tempspcoef_{optim_params.temporal_spectral_coeff}"
    exp_path = exp_path / name
    os.makedirs(exp_path, exist_ok = True)

    # create path to exp and save hyperparameters
    save_path = exp_path / "training_results"
    os.makedirs(save_path, exist_ok=True)
    plots_path = exp_path / "plots"
    os.makedirs(plots_path, exist_ok=True)
    # Here could maybe implement a "save()" function inside each class 
    hp = {}
    hp["exp_params"] = experiment_params.__dict__
    hp["data_params"] = data_params.__dict__
    hp["gt_params"] = gt_params.__dict__
    hp["train_params"] = train_params.__dict__
    hp["model_params"] = model_params.__dict__
    hp["optim_params"] = optim_params.__dict__
    with open(exp_path / "params.json", "w") as file:
        json.dump(hp, file, indent=4)

    # # load the best metrics
    # with open(os.path.join(hp.data_path, "best_metrics.json"), 'r') as f:
    #     best_metrics = json.load(f)
    best_metrics = {"recons": 0, "kl": 0, "mcc": 0, "elbo": 0}

    # train, always with the latent version
    trainer = TrainingLatent(model, datamodule, 
                             experiment_params, gt_params, model_params, train_params, optim_params, plot_params, 
                             save_path, plots_path, 
                             best_metrics, d, wandbname=name)

    # where is the model at this point?
    print("Where is my model?", next(trainer.model.parameters()).device)

    valid_loss = trainer.train_with_QPM()

    # save final results, (MSE)
    metrics = {"shd": 0.0, "precision": 0.0, "recall": 0.0, "train_mse": 0.0, "val_mse": 0.0, "mcc": 0.0}
    # if we have the GT, also compute (SHD, Pr, Re, MCC)
    if not hp.no_gt:
        # Here can remove this ---
        if hp.instantaneous:
            gt_graph = trainer.gt_dag
        else:
            gt_graph = trainer.gt_dag[:-1]  # remove the graph G_t

        learned_graph = trainer.model.get_adj().detach().numpy().reshape(gt_graph.shape[0], gt_graph.shape[1], -1)

        score, cc_program_perm, assignments, z, z_hat, _ = mcc_latent(trainer.model, trainer.data)
        permutation = np.zeros((gt_graph.shape[1], gt_graph.shape[1]))
        permutation[np.arange(gt_graph.shape[1]), assignments[1]] = 1
        gt_graph = permutation.T @ gt_graph @ permutation

        metrics["mcc"] = score
        metrics["w_mse"] = w_mae(
            trainer.model.autoencoder.get_w_decoder().detach().numpy()[:, :, assignments[1]], datamodule.gt_w
        )
        metrics["shd"] = shd(learned_graph, gt_graph)
        metrics["precision"], metrics["recall"] = precision_recall(learned_graph, gt_graph)
        errors = edge_errors(learned_graph, gt_graph)
        metrics["tp"] = errors["tp"]
        metrics["fp"] = errors["fp"]
        metrics["tn"] = errors["tn"]
        metrics["fn"] = errors["fn"]
        metrics["n_edge_gt_graph"] = np.sum(gt_graph)
        metrics["n_edge_learned_graph"] = np.sum(learned_graph)
        metrics["execution_time"] = time.time() - t0

        for key, val in valid_loss.items():
            metrics[key] = val

    # assert that trainer.model is in eval mode
    if trainer.model.training:
        print("Model is in train mode")
    else:
        print("Model is in eval mode")

    # NOTE: just dummies here for now
    train_mse, train_smape, val_mse, val_smape = 10.0, 10.0, 10.0, 10.0

    # save the metrics
    metrics["train_mse"] = train_mse
    metrics["train_smape"] = train_smape
    metrics["val_mse"] = val_mse
    metrics["val_smape"] = val_smape

    # save the metrics
    with open(os.path.join(hp.exp_path, "results.json"), "w") as file:
        json.dump(metrics, file, indent=4)

    # finally, save the model
    torch.save(trainer.model.state_dict(), os.path.join(hp.exp_path, "model.pth"))


def assert_args(
        experiment_params,
        data_params,
        gt_params,
        optim_params, 
):
    
    """Raise errors or warnings if some args should not take some combination of values."""
    # raise errors if some args should not take some combination of values
    if gt_params.no_gt and (gt_params.debug_gt_graph or gt_params.debug_gt_z or gt_params.debug_gt_w):
        raise ValueError("Since no_gt==True, all other args should not use ground-truth values")

    if experiment_params.latent and (experiment_params.d_z is None or experiment_params.d_x is None or experiment_params.d_z <= 0 or experiment_params.d_x <= 0):
        raise ValueError("When using latent model, you need to define d_z and d_x with integer values greater than 0")

    # string input with limited possible values
    supported_dataformat = ["numpy", "hdf5"]
    if data_params.data_format not in supported_dataformat:
        raise ValueError(
            f"This file format ({data_params.data_format}) is not \
                         supported. Supported types are: {supported_dataformat}"
        )
    supported_optimizer = ["sgd", "rmsprop"]
    if optim_params.optimizer not in supported_optimizer:
        raise ValueError(
            f"This optimizer type ({optim_params.optimizer}) is not \
                         supported. Supported types are: {supported_optimizer}"
        )
    
    # warnings, strange choice of args combination
    if not experiment_params.latent and gt_params.debug_gt_z:
        warnings.warn("Are you sure you want to use gt_z even if you don't have latents")
    if experiment_params.latent and (experiment_params.d_z > experiment_params.d_x):
        warnings.warn("Are you sure you want to have a higher dimension for d_z than d_x")

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Causal models for climate data")
    # for the default values, check default_params.json

    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/single_param_file.json",
        help="Path to a json file with values for all parameters",
    )
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        params = json.load(f)

    experiment_params = expParams(**params["exp_params"])
    data_params = dataParams(**params["data_params"])
    gt_params = gtParams(**params["gt_params"])
    train_params = trainParams(**params["train_params"])
    model_params = modelParams(**params["model_params"])
    optim_params = optimParams(**params["optim_params"])
    plot_params = plotParams(**params["plot_params"])
    savar_params =savarParams(**params["savar_params"])

    if "savar" in data_params.in_var_ids:
        experiment_params.lat = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.lon = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.d_x = int(experiment_params.lat * experiment_params.lon)
        plot_params.savar = True
    else:
        plot_params.savar = False

    assert_args(
            experiment_params,
            data_params,
            gt_params,
            optim_params, 
    )

    main(
        experiment_params,
        data_params,
        gt_params,
        train_params, 
        model_params, 
        optim_params, 
        plot_params,
        savar_params
    )

    # Experiment parameters
    # parser.add_argument("--exp-path", type=str, default="../../causal_climate_exp/", help="Path to experiments")
    # parser.add_argument(
    #     "--config-exp-path",
    #     type=str,
    #     default="../emulator/configs/datamodule/climate.json",
    #     help="Path to a json file with specifics of experiments",
    # )
    # parser.add_argument(
    #     "--config-path",
    #     type=str,
    #     default="default_params.json",
    #     help="Path to a json file with values for all hyperparameters",
    # )
    # parser.add_argument(
    #     "--use-data-config",
    #     action="store_true",
    #     help="If true, overwrite some parameters to fit \
    #                     parameters that have been used to generate data",
    # )
    # parser.add_argument("--exp-id", type=int, help="ID specific to the experiment")

    # # For synthetic datasets, can use the ground-truth values to do ablation studies
    # parser.add_argument(
    #     "--debug-gt-z", action="store_true", help="If true, use the ground truth value of Z (use only to debug)"
    # )
    # parser.add_argument(
    #     "--debug-gt-w", action="store_true", help="If true, use the ground truth value of W (use only to debug)"
    # )
    # parser.add_argument(
    #     "--debug-gt-graph", action="store_true", help="If true, use the ground truth graph (use only to debug)"
    # )
    # parser.add_argument(
    #     "--no-w-constraint",
    #     action="store_true",
    #     help="If True, does not apply constraint on W (non-negativity and ortho)",
    # )

    # # Dataset properties
    # parser.add_argument("--data-path", type=str, help="Path to the dataset")
    # parser.add_argument("--data-format", type=str, help="numpy|hdf5")
    # parser.add_argument(
    #     "--no-gt", action="store_true", help="If True, does not use any ground-truth for plotting and metrics"
    # )

    # # dataset transformation
    # parser.add_argument("--seasonality-removal", action="store_true", help="Deseasonalise the data")

    # # specific to model with latent variables
    # parser.add_argument("--latent", action="store_true", help="Use the model that assumes latent variables")
    # parser.add_argument("--tied-w", action="store_true", help="Use the same matrix W, as the decoder, for the encoder")
    # parser.add_argument("--nonlinear-mixing", action="store_true", help="The encoder/decoder use NN")
    # parser.add_argument("--coeff-kl", type=float, help="coefficient that is multiplied to the KL term ")
    # parser.add_argument("--d-z", type=int, help="if latent, d_z is the number of cluster z")
    # parser.add_argument("--d-x", type=int, help="if latent, d_x is the number of gridcells")

    # parser.add_argument("--fixed", action="store_true", help="Keep transition matrix fixed as all ones")
    # parser.add_argument("--fixed-output-fraction", type=float, help="Fraction of 1s and 0s in fixed output matrix, ")

    # parser.add_argument("--instantaneous", action="store_true", help="Use instantaneous connections")
    # parser.add_argument("--tau", type=int, help="Number of past timesteps to consider")
    # parser.add_argument("--tau-neigh", type=int, help="Radius of neighbor cells to consider")
    # parser.add_argument("--ratio-train", type=int, help="Proportion of the data used for the training set")
    # parser.add_argument("--ratio-valid", type=int, help="Proportion of the data used for the validation set")
    # parser.add_argument("--batch-size", type=int, help="Number of samples per minibatch")

    # # Model hyperparameters: architecture
    # parser.add_argument("--num-hidden", type=int, help="Number of hidden units")
    # parser.add_argument("--num-layers", type=int, help="Number of hidden layers")
    # parser.add_argument("--num-output", type=int, help="Number of output units")

    # parser.add_argument(
    #     "--num-hidden-mixing",
    #     type=int,
    #     help="Number of hidden \
    #                     units for the encoder/decoder learning the mixing function",
    # )
    # parser.add_argument(
    #     "--num-layers-mixing",
    #     type=int,
    #     help="Number of hidden \
    #                     layers for the encoder/decoder learning the mixing function",
    # )

    # # Model hyperparameters: optimization
    # parser.add_argument("--optimizer", type=str, help="sgd|rmsprop")
    # parser.add_argument("--reg-coeff", type=float, help="Coefficient for the sparsity regularisation term")
    # parser.add_argument("--reg-coeff-connect", type=float, help="Coefficient for the connectivity regularisation term")
    # parser.add_argument("--lr", type=float, help="Initial learning rate")
    # parser.add_argument(
    #     "--lr-scheduler-epochs",
    #     type=lambda x: list(map(int, x.split(","))),
    #     help="Number of iterations to decrease lr by lr-scheduler-gamma",
    # )
    # parser.add_argument(
    #     "--lr-scheduler-gamma",
    #     type=int,
    #     help="Value by which to multiply LR at each iteration lr-schedule, 1 does not decrease LR, 0.1 does by 10",
    # )
    # parser.add_argument("--random-seed", type=int, help="Random seed for torch and numpy")
    # parser.add_argument(
    #     "--schedule-reg", type=int, help="Start applying the sparsity regularization only after X number of steps"
    # )
    # parser.add_argument(
    #     "--schedule-ortho", type=int, help="Start applying the orthogonality constraint only after X number of steps"
    # )
    # parser.add_argument(
    #     "--schedule-sparsity", type=int, help="Start applying the sparsity constraint only after X number of steps"
    # )
    # parser.add_argument(
    #     "--hard-gumbel", action="store_true", help="If true, use the hard version when sampling the masks"
    # )

    # # ALM/QPM options
    # # orthogonality constraint
    # parser.add_argument("--ortho-mu-init", type=float, help="initial value of mu for the constraint")
    # parser.add_argument(
    #     "--ortho-mu-mult-factor",
    #     type=float,
    #     help="Multiply mu by this amount when constraint not sufficiently decreasing",
    # )
    # parser.add_argument("--ortho-omega-gamma", type=float, help="Precision to declare convergence of subproblems")
    # parser.add_argument(
    #     "--ortho-omega-mu", type=float, help="After subproblem solved, h should have reduced by this ratio"
    # )
    # parser.add_argument("--ortho-h-threshold", type=float, help="Can stop if h smaller than h-threshold")
    # parser.add_argument(
    #     "--ortho-min-iter-convergence", type=int, help="Minimal number of iteration before checking if has converged"
    # )

    # parser.add_argument("--sparsity-mu-init", type=float, help="initial value of mu for the constraint")
    # parser.add_argument(
    #     "--sparsity-mu-mult-factor",
    #     type=float,
    #     help="Multiply mu by this amount when constraint not sufficiently decreasing",
    # )
    # parser.add_argument("--sparsity-omega-gamma", type=float, help="Precision to declare convergence of subproblems")
    # parser.add_argument(
    #     "--sparsity-omega-mu", type=float, help="After subproblem solved, h should have reduced by this ratio"
    # )
    # parser.add_argument("--sparsity-h-threshold", type=float, help="Can stop if h smaller than h-threshold")
    # parser.add_argument(
    #     "--sparsity-min-iter-convergence", type=int, help="Minimal number of iteration before checking if has converged"
    # )

    # parser.add_argument("--sparsity-upper-threshold", type=float, help="Upper threshold for the sparsity constraint")

    # # acyclicity constraint
    # parser.add_argument("--acyclic-mu-init", type=float, help="initial value of mu for the constraint")
    # parser.add_argument(
    #     "--acyclic-mu-mult-factor",
    #     type=float,
    #     help="Multiply mu by this amount when constraint not sufficiently decreasing",
    # )
    # parser.add_argument("--acyclic-omega-gamma", type=float, help="Precision to declare convergence of subproblems")
    # parser.add_argument(
    #     "--acyclic-omega-mu", type=float, help="After subproblem solved, h should have reduced by this ratio"
    # )
    # parser.add_argument("--acyclic-h-threshold", type=float, help="Can stop if h smaller than h-threshold")
    # parser.add_argument(
    #     "--acyclic-min-iter-convergence", type=int, help="Minimal number of iteration before checking if has converged"
    # )

    # parser.add_argument("--mu-acyclic-init", type=float, help="initial value of mu for the acyclicity constraint")
    # parser.add_argument("--h-acyclic-threshold", type=float, help="Can stop if h smaller than h-threshold")

    # parser.add_argument("--max-iteration", type=int, help="Maximal number of iteration before stopping")
    # parser.add_argument("--patience", type=int, help="Patience used after the acyclicity constraint is respected")
    # parser.add_argument(
    #     "--patience-post-thresh", type=int, help="Patience used after the thresholding of the adjacency matrix"
    # )

    # # adding loss coefficients
    # parser.add_argument("--crps-coeff", type=float, help="Coefficient for the CRPS term of the loss")
    # parser.add_argument("--spectral-coeff", type=float, help="Coefficient for the spectral term of the loss")
    # parser.add_argument(
    #     "--temporal-spectral-coeff", type=float, help="Coefficient for the temporal spectral term of the loss"
    # )

    # # logging
    # parser.add_argument("--valid-freq", type=int, help="Frequency of evaluating the loss on the validation set")
    # parser.add_argument("--plot-freq", type=int, help="Plotting frequency")
    # parser.add_argument(
    #     "--plot-through-time",
    #     action="store_true",
    #     help="If true, save each plot in a \
    #                     different file with a name depending on the iteration",
    # )
    # parser.add_argument("--print-freq", type=int, help="Printing frequency")

    # # device and numerical precision
    # parser.add_argument("--gpu", action="store_true", help="Use GPU")
    # parser.add_argument("--float", action="store_true", help="Use Float precision")

    # parser.add_argument("--ishdf5", action="store_true", help="Use GPU")
    # args = parser.parse_args()

    # # nasty thing going on where we have two config files.
    # # if a json file with params is given, update params accordingly
    # if args.config_path != "":
    #     print(f"using config file: {args.config_path}")
    #     default_params = vars(args)
    #     with open(args.config_path, "r") as f:
    #         params = json.load(f)

    #     for key, val in params.items():
    #         if default_params[key] is None or not default_params[key]:
    #             default_params[key] = val
    #     args = Bunch(**default_params)

    # # use some parameters from the data generating process';;
    # if args.use_data_config != "":
    #     with open(args.config_exp_path, "r") as f:
    #         params = json.load(f)
    #     args.d_x = params["d_x"]
    #     if "latent" in params:
    #         args.latent = params["latent"]
    #         if args.latent:
    #             args.d_z = params["d_z"]
    #     if "tau" in params:
    #         args.tau = params["tau"]
    #     if "neighborhood" in params:
    #         args.tau_neigh = params["neighborhood"]

    # # args.nonlinear_mixing = True
    # args.latent = True
    # print(args.no_gt)

    # args = assert_args(args)

    # main(args)
