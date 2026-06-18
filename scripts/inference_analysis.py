# Inference results on test set (one ssp), please not this is only possible for multi-scenario setting!


import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import healpy as hp
import pandas as pd
from climatem.data_loader.causal_datamodule import CausalClimateDataMultiScenarioModule
# from climatem.model.tsdcd_latent_hvae import LatentTSDCD as LatentTSDCDWithGLobalZ
# from climatem.model.tsdcd_latent import LatentTSDCD
from climatem.config import *
from climatem.utils import parse_args, update_config_withparse

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")
def plot_compare_predictions_icosahedral(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    sample: int,
    coordinates: np.ndarray,
    path,
    ssp
):
    """Plot a prediction from the method, the last time step and the ground-truth."""

    if y_true.shape[1] > 1:
        fig, axs = plt.subplots(
            y_true.shape[1], 2, subplot_kw={"projection": ccrs.Robinson()}, layout="constrained", figsize=(2*4, 4*y_true.shape[1])
        )
    else:
        fig, axs = plt.subplots(
            1, 2, subplot_kw={"projection": ccrs.Robinson()}, layout="constrained", figsize=(32, 8)
        )
        axs = [axs]
    labels = ["ts", "mmrbc", "so2", "co2mass", "ch4global"]
    for j, ax_row in enumerate(axs):

        for i, ax in enumerate(ax_row):

            ax.set_global()
            ax.coastlines()
            # Add some map features for context
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, edgecolor="black")
            ax.gridlines(draw_labels=False)

            # Unpack coordinates for vectorized scatter plot
            # something like lonlat_vertex_mapping.txt
            lon = coordinates[:, 0]
            lat = coordinates[:, 1]

            # Vectorized scatter plot with color array
 
            if i == 0:
                # print('y shape:', y_true.shape)
                s = ax.scatter(
                    x=lon,
                    y=lat,
                    c=y_true[sample, j, :],
                    alpha=1,
                    s=30,
                    vmin=np.amin(y_true[sample, j, :]),
                    vmax=np.amax(y_true[sample, j, :]),
                    cmap="RdBu_r",
                    transform=ccrs.PlateCarree(),
                )
                ax.set_title("Ground truth")
 
            elif i == 1:
                # print('y_recons shape:', y_recons.shape)
                s = ax.scatter(
                    x=lon,
                    y=lat,
                    c=y_hat[sample, j, :],
                    alpha=1,
                    s=30,
                    vmin=np.amin(y_true[sample, j, :]),
                    vmax=np.amax(y_true[sample, j, :]),
                    cmap="RdBu_r",
                    transform=ccrs.PlateCarree(),
                )
                ax.set_title("Prediction")
            fig.colorbar(
            s,
            ax=ax,
            label=labels[j],
            orientation="vertical",
            shrink=0.6,
        )

        # add one colorbar for all subplots
        # fig.colorbar(s, ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)
        # if j == 0:
        #     fig.colorbar(
        #         s, ax=ax_row[1], label=f"ts", orientation="vertical", shrink=0.6
        #     )  # adjust shrink
        # elif j == 1:
        #     fig.colorbar(s, ax=ax_row[1], label="mmrbc", orientation="vertical", shrink=0.6)
        # elif j == 2:
        #     fig.colorbar(s, ax=ax_row[1], label="so2", orientation="vertical", shrink=0.6)
        # elif j == 3:
        #     fig.colorbar(s, ax=ax_row[1], label="co2mass", orientation="vertical", shrink=0.6)
        # elif j == 4:
        #     fig.colorbar(s, ax=ax_row[1], label="ch4global", orientation="vertical", shrink=0.6)

    fname = f"compare_predictions_{sample}.png"

    plt.suptitle(f"Ground truth and Prediction {ssp}", fontsize=24)
    # plt.legend()
    plt.savefig(path / fname, format="png")
    plt.close()

def get_all_temp_and_gt(dataloader, model, device):
           # Start again at the beginning of the dataloader.

    # iterate through the data and append all the y values together
    pred_all = []
    gt_all = []
    domain_names = []


    for i in range(len(dataloader)):
        batch = next(dataloader)
        x = batch["x"]
        y = batch["y"]
        domain = batch["domain"]

        x = torch.nan_to_num(x).to(device)
        y = torch.nan_to_num(y).to(device)
        y = y[:, 0]        
       
        with torch.no_grad():
           pred, _,_,_,_,_  = model.predict(x, y) 

        pred_all.append(pred)
        gt_all.append(y)
        domain_names.extend(domain)
       

    pred_all = torch.cat(pred_all, dim=0)
    pred_all = torch.nan_to_num(pred_all) 

    gt_all = torch.cat(gt_all, dim=0)
    gt_all = torch.nan_to_num(gt_all)

    return pred_all, gt_all, domain_names     

def subplot_ssp_line(pred, gt, unique_domains, domain_names,  variable, save_path):

    plt.figure(figsize=(14, 7))
    n_domains = len(unique_domains)

    fig, axes = plt.subplots(
        n_domains,
        1,
        figsize=(14, 4 * n_domains),
        sharex=True
    )

    if n_domains == 1:
        axes = [axes]

    for idx, domain in enumerate(unique_domains):
        ax = axes[idx]

        mask = domain_names == domain

        pred_domain = pred[mask]
        gt_domain = gt[mask]
        time_steps = pd.date_range(start="2015-01-01", periods=len(pred_domain), freq="MS" ) 

        ax.plot(time_steps, gt_domain, label="GT")
        ax.plot(time_steps, pred_domain, "--", label="Pred")

        ax.set_title(f"{domain} {variable}")
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel("Time")

    plt.savefig(save_path / f"per_ssp_pred_gt_line_{variable}.png", dpi=150)
    plt.close()

def subplot_ssp_scatter(pred, gt, unique_domains, domain_names, variable, save_path):
    fig, axes = plt.subplots(1,len(unique_domains),figsize=(6 * len(unique_domains), 5))
    if len(unique_domains) == 1:
        axes = [axes]

    for idx, domain in enumerate(unique_domains):
        ax = axes[idx]

        mask = domain_names == domain

        pred_domain = pred[mask]
        gt_domain = gt[mask]

        ax.scatter(
            gt_domain,
            pred_domain,
            alpha=0.6,
            s=20
        )

        # 1:1 line
        min_val = min(gt_domain.min(), pred_domain.min())
        max_val = max(gt_domain.max(), pred_domain.max())

        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            'k--',
            linewidth=1,
        )

        # metrics
        rmse = np.sqrt(np.mean((pred_domain - gt_domain) ** 2))
        corr = np.corrcoef(gt_domain, pred_domain)[0, 1]

        ax.set_title(
            f'{variable}-{domain}\nRMSE={rmse:.3f}, R={corr:.3f}'
        )
        ax.set_xlabel('Ground Truth')
        ax.set_ylabel('Prediction')
        # ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        save_path / f"per_ssp_pred_gt_scatter_{variable}.png",
        dpi=150,
    )
    plt.close()
def main(
    experiment_params, 
    data_params, 
    # gt_params, 
    train_params, 
    model_params, 
    optim_params, 
    plot_params, 
    savar_params,
    rollout_params,
    exp_id,
    iter_id,
):
    """
    :param hp: object containing hyperparameter values
    """

    # Control as much randomness as possible
    torch.manual_seed(experiment_params.random_seed)
    np.random.seed(experiment_params.random_seed)
    
    device = torch.device("cuda" if (torch.cuda.is_available() and experiment_params.gpu) else "cpu")

    if experiment_params.gpu and torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    if data_params.data_format == "hdf5":
        print("IS HDF5")
        return
    else:
        datamodule = CausalClimateDataMultiScenarioModule(
            tau=experiment_params.tau,
            num_months_aggregated=data_params.num_months_aggregated,
            train_val_interval_length=data_params.train_val_interval_length,
            in_var_ids=data_params.in_var_ids,
            out_var_ids=data_params.out_var_ids,
            train_years=data_params.train_years,
            train_historical_years=data_params.train_historical_years,
            test_years=data_params.test_years,  # do we want to implement keeping only certain years for testing?
            val_split=1 - train_params.ratio_train,  # fraction of testing to split for valdation
            seq_to_seq=data_params.seq_to_seq,  # if true maps from T->T else from T->1
            channels_last=data_params.channels_last,  # wheather variables come last our after sequence lenght
            train_scenarios=["historical", "ssp126"],
            test_scenarios=["ssp126", "ssp245","ssp370"],
            train_models=data_params.train_models,
            # test_models = data_params.test_models,
            batch_size=data_params.batch_size,
            eval_batch_size=data_params.eval_batch_size,
            num_workers=experiment_params.num_workers,
            pin_memory=experiment_params.pin_memory,
            load_train_into_mem=data_params.load_train_into_mem,
            load_test_into_mem=data_params.load_test_into_mem,
            verbose=experiment_params.verbose,
            seed=experiment_params.random_seed,
            seq_len=data_params.seq_len,
            data_dir=data_params.climateset_data,
            output_save_dir=data_params.data_dir,
            num_ensembles=data_params.num_ensembles,  # 1 for first ensemble, -1 for all
            lon=experiment_params.lon,
            lat=experiment_params.lon,
            num_levels=data_params.num_levels,
            global_normalization=data_params.global_normalization,
            seasonality_removal=data_params.seasonality_removal,
            reload_climate_set_data=data_params.reload_climate_set_data,
            icosahedral_coordinates_path=data_params.icosahedral_coordinates_path,
        )
        datamodule.setup()

    d = len(data_params.in_var_ids)
    print(f"Using {d} variables")

    if model_params.instantaneous:
        print("Using instantaneous connections")
        num_input = d * (experiment_params.tau + 1) 
    else:
        num_input = d * experiment_params.tau 

    # set the model
    if experiment_params.d_z_global >= 1:
        from climatem.model.tsdcd_latent_hvae import LatentTSDCD as LatentTSDCDWithGLobalZ
        print(f"Using Hierarchical Model")
    
        class LatentTSDCD:
            def __init__(self, *args, **kwargs):
                self._model = LatentTSDCDWithGLobalZ(*args, d_z_global=experiment_params.global_z, **kwargs)
        
            def __getattr__(self, name):
                return getattr(self._model, name)
    else:
        print(f"Using SDCD (non-Hierarchical) Model")
        from climatem.model.tsdcd_latent import LatentTSDCD

    model = LatentTSDCD(
        num_layers=model_params.num_layers,
        num_hidden=model_params.num_hidden,
        num_input=num_input,
        num_output=2,  # This should be parameterized somewhere?
        num_layers_mixing=model_params.num_layers_mixing,
        num_hidden_mixing=model_params.num_hidden_mixing,
        position_embedding_dim=model_params.position_embedding_dim,
        transition_param_sharing=model_params.transition_param_sharing,
        position_embedding_transition=model_params.position_embedding_transition,
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
        nonlinear_dynamics=model_params.nonlinear_dynamics,
        nonlinear_mixing=model_params.nonlinear_mixing,
        tied_w=model_params.tied_w,
        fixed=model_params.fixed,
        fixed_output_fraction=model_params.fixed_output_fraction,
    )
    
    # read paths 
    coordinates = np.load(data_params.icosahedral_coordinates_path)

    exp_path = Path(experiment_params.exp_path)
    if not os.path.exists(exp_path): 
        raise ValueError(f"Results path {exp_path} doesn't exist. Model should be saved in this folder")
 

    name = exp_id

    exp_path = exp_path / name
    print("exp_path experiment_params.exp_path:", exp_path)
    if not os.path.exists(exp_path): 
        raise ValueError(f"Results path {exp_path} does not exist. Are you using the same parameters?")

    # create path to exp and save hyperparameters
    save_path = exp_path / "inference"
    os.makedirs(save_path, exist_ok=True)

    # seed = 1
    save_path = save_path / f"inf_iter{iter_id}_spherical{str(optim_params.take_spherical_harmonics)}"
    os.makedirs(save_path, exist_ok=True)

    model_path = exp_path #/ "training_results"

    model_file = "model.pth"
    print(f"The model being teseted is under: {model_path / model_file}")
    state_dict_vae_final = torch.load(
        model_path / model_file, 
        map_location=device
    )
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict_vae_final.items()})

    # Move the model to the GPU
    model = model.to(device)
    print("Where is the model?", next(model.parameters()).device)
    model.eval()

    

    test_dataloader = iter(datamodule.test_dataloader(accelerator))
    pred_all_test, gt_all_test, domain_names = get_all_temp_and_gt(dataloader=test_dataloader, model=model, device=device)
    # reset the dataloader
    test_dataloader = iter(datamodule.test_dataloader(accelerator))
    
    print("pred_all_test",pred_all_test.shape)
    print("gt_all_test",gt_all_test.shape)  
    print("domain_names len", len(domain_names))  

    y_true = gt_all_test.detach().cpu().numpy() 
    y_hat = pred_all_test.detach().cpu().numpy() 

    samples = [1, y_true.shape[0]//2, y_true.shape[0]-1]
    for sample in samples:
        plot_compare_predictions_icosahedral(y_true=y_true,y_hat=y_hat,sample=sample,coordinates=coordinates,path=save_path, ssp=domain_names[sample])
    

    # ----------- Line plot of prediction and gt mean over space ----------------- #
    pred_all_test_np = pred_all_test.detach().cpu().numpy()   # [B,d, d_x]
    pred_all_test_global_mean = pred_all_test_np.mean(-1)
   
    gt_all_test_np = gt_all_test.detach().cpu().numpy()   # [B, d,d_x]
    gt_all_test_global_mean = gt_all_test_np.mean(-1)
    domain_names = np.asarray(domain_names)

    variables = ["ts","mmrbc","so2","co2mass","ch4global"]

    unique_domains = np.unique(domain_names)
    print(f"Unique domains: {unique_domains}")

    # Plot 1: plot all ssp in a single figure
    # plt.figure(figsize=(14, 7)) 
    colors = ['blue', 'red', 'green', 'orange', 'purple'] 

    num_vars = pred_all_test_global_mean.shape[1]
    
    fig, axes = plt.subplots(num_vars, 1, figsize=(10, 4*num_vars), sharex=True)

    if num_vars == 1:
        axes = [axes]

    for v, ax in enumerate(axes):
        for idx, domain in enumerate(unique_domains): 
            mask = domain_names == domain
            pred_domain = pred_all_test_global_mean[:,v][mask].flatten() 
            gt_domain = gt_all_test_global_mean[:,v][mask].flatten()
            time_steps = pd.date_range(start="2015-01-01", periods=len(pred_domain), freq="MS" ) 
            ax.plot(time_steps, gt_domain, label=f'GT-{domain}', color=colors[idx], alpha=0.7)
            ax.plot(time_steps, pred_domain, label=f'Pred-{domain}', color=colors[idx], linestyle='--', alpha=0.7)
        ax.set_ylabel(f'{variables[v]}')
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time Steps")

    # plt.tight_layout()
    plt.savefig(save_path / "all_ssp_pred_gt_line.png")
    plt.close()


    # Plot 2: plot each ssp into a subplot
    for i in range(pred_all_test_global_mean.shape[1]):
        variable = variables[i]
        pred = pred_all_test_global_mean[:,i]
        gt = gt_all_test_global_mean[:,i]
        subplot_ssp_line(pred, gt, unique_domains, domain_names, variable, save_path)
        subplot_ssp_scatter(pred, gt, unique_domains, domain_names, variable, save_path)

def load_rollouts(traj_path):
    trajs = []

    for i in range(1020):
        traj = np.load(traj_path/ f"trajectory_iteration_{i}.npy")  # shape (50, 1, 5, 3072)
        # print(f"a signle traj has the shape of {traj.shape}")
        trajs.append(traj.mean((0,1)))     # -> (5, 3072)

    trajs = np.stack(trajs, axis=0) 
    print("trajs",trajs.shape) 
    variables = ["ts","mmrbc","so2","co2mass","ch4global"]

    ssps = ["ssp126", "ssp245", "ssp370"]

    # Plot 1: plot all ssp in a single figure
    # plt.figure(figsize=(14, 7)) 
    colors = ['blue', 'red', 'green', 'orange', 'purple'] 

    num_vars = len(variables)
    
    # ---------------- load GT ----------------
    gt_data = {}
    for ssp in ssps:
        gt = np.load(traj_path / f"gt_all_y_{ssp}.npy")  # (1027,1,5,3072)
        gt = gt.squeeze(1)  # (1027,5,3072)
        gt_data[ssp] = gt

    # ---------------- align time ----------------
    T = min(1020, 1027)

    trajs = trajs[:T]
    for ssp in ssps:
        gt_data[ssp] = gt_data[ssp][:T]

    time_steps = pd.date_range(start="2015-01-01", periods=T, freq="MS")

    # ---------------- plotting ----------------
    fig, axes = plt.subplots(len(variables), 1, figsize=(12, 4 * len(variables)), sharex=True)

    if len(variables) == 1:
        axes = [axes]

    for v, ax in enumerate(axes):

        # ===== prediction =====
        pred = trajs[:, v, :].mean(axis=-1)  # (T,)
        ax.plot(time_steps, pred, color="black", label="Prediction")

        # ===== GT per SSP =====
        for i, ssp in enumerate(ssps):
            gt = gt_data[ssp][:, v, :].mean(axis=-1)  # (T,)

            ax.plot(time_steps, gt,
                    color=colors[i],
                    label=f"GT-{ssp}",
                    alpha=0.7)

        ax.set_ylabel(variables[v])
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("Time")

    plt.tight_layout()
    plt.savefig(traj_path / "rollouts_gt_vs_pred_370.png")
    plt.close()


if __name__ == "__main__":

    # args = parse_args()
    
    # cwd = Path.cwd()
    # root_path = cwd.parent
    # # config_path = root_path / f"configs"
    # exp_id = args.exp_id
    # iter_id = args.iter_id
    # folder = exp_id.split("/")[0] 
    # exp_id = exp_id.split("/")[-1]
    # config_path = Path("/home/mila/s/shanz/scratch/results") / folder / exp_id
    # json_path = config_path / args.config_path
    
    # with open(json_path, "r") as f:
    #     params = json.load(f)
    # params = update_config_withparse(params, args)

    # # get user's scratch directory:
    # scratch_path = os.getenv("SCRATCH")
    # params["data_params"]["data_dir"] = params["data_params"]["data_dir"].replace("$SCRATCH", scratch_path)
    # print ("new data path:", params["data_params"]["data_dir"])

    # params["exp_params"]["exp_path"] = params["exp_params"]["exp_path"].replace("$SCRATCH", scratch_path)
    # print ("new exp path:", params["exp_params"]["exp_path"])

    # # get directory of project via current file (aka .../climatem/scripts/main_picabu.py)
    # params["data_params"]["icosahedral_coordinates_path"] = params["data_params"]["icosahedral_coordinates_path"].replace("$CLIMATEMDIR", root_path.absolute().as_posix())
    # print ("new icosahedron path:", params["data_params"]["icosahedral_coordinates_path"])
    
    # # For rollout, most cases we already have the climate dataset during training
    # params["data_params"]["reload_climate_set_data"] = True
    # print ("new reload_climate_set_data:", params["data_params"]["reload_climate_set_data"])

    # experiment_params = expParams(**params["exp_params"])
    # data_params = dataParams(**params["data_params"])
    # # gt_params = gtParams(**params["gt_params"])
    # train_params = trainParams(**params["train_params"])
    # model_params = modelParams(**params["model_params"])
    # optim_params = optimParams(**params["optim_params"])
    # plot_params = plotParams(**params["plot_params"])
    # savar_params = savarParams(**params["savar_params"])
    # rollout_params = rolloutParams(**params["rollout_params"])

    # #Overwrite arguments if using savar
    # if "savar" in data_params.in_var_ids:
    #     experiment_params.lat = int(savar_params.comp_size * savar_params.n_per_col)
    #     experiment_params.lon = int(savar_params.comp_size * savar_params.n_per_col)
    #     experiment_params.d_x = int(experiment_params.lat * experiment_params.lon)
    #     plot_params.savar = True
    # else:
    #     plot_params.savar = False

    # main(experiment_params, data_params, train_params, model_params, optim_params, plot_params, savar_params, rollout_params, exp_id, iter_id)
    load_rollouts(Path("/home/mila/s/shanz/scratch/results/small_debug_multiscenario_test/FALSE_AUG_200_var_tsco2massmmrbcso2ch4global_nlinmix_True_nlindyn_True_tau_5_z_60_lr_0.0001_bs_128_20260609_075641/rollouts/bs_1_np_50_npp_10_t_1020_sc_log_bayesian_temp_True_iter100000"))
