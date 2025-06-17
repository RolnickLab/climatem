# Here we have a quick main where we are testing data loading with different ensemble members and ideally with different climate models.
import json
import os
import time
import shutil

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from climatem.config import *
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from climatem.utils import parse_args, update_config_withparse

# TODO ADD SEED TO FILE PATH

torch.set_warn_always(False)

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")

args = parse_args()

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
config_path = os.path.join(root_path, "configs", args.config_path)

with open(config_path, "r") as f:
    params = json.load(f)
config_obj_list = update_config_withparse(params, args)

# get user's scratch directory on Mila cluster:
scratch_path = os.getenv("SCRATCH")
params["data_params"]["data_dir"] = params["data_params"]["data_dir"].replace("$SCRATCH", scratch_path)
print("new data path:", params["data_params"]["data_dir"])

params["exp_params"]["exp_path"] = params["exp_params"]["exp_path"].replace("$SCRATCH", scratch_path)
print("new exp path:", params["exp_params"]["exp_path"])

# get directory of project via current file (aka .../climatem/scripts/main_picabu.py)
params["data_params"]["icosahedral_coordinates_path"] = params["data_params"]["icosahedral_coordinates_path"].replace(
    "$CLIMATEMDIR", root_path
)
print("new icosahedron path:", params["data_params"]["icosahedral_coordinates_path"])

experiment_params = expParams(**params["exp_params"])
data_params = dataParams(**params["data_params"])
gt_params = gtParams(**params["gt_params"])
train_params = trainParams(**params["train_params"])
model_params = modelParams(**params["model_params"])
optim_params = optimParams(**params["optim_params"])
plot_params = plotParams(**params["plot_params"])
savar_params = savarParams(**params["savar_params"])

torch.manual_seed(experiment_params.random_seed)
np.random.seed(experiment_params.random_seed)

data_dir = params["data_params"]["data_dir"]
os.makedirs(data_dir, exist_ok=True)

datamodule = CausalClimateDataModule(
    tau=experiment_params.tau,
    future_timesteps=experiment_params.future_timesteps,
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
    train_scenarios=data_params.train_scenarios,
    test_scenarios=data_params.test_scenarios,
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
    lat=experiment_params.lat,
    num_levels=data_params.num_levels,
    global_normalization=data_params.global_normalization,
    seasonality_removal=data_params.seasonality_removal,
    reload_climate_set_data=data_params.reload_climate_set_data,
    icosahedral_coordinates_path=data_params.icosahedral_coordinates_path,
    # Below SAVAR data arguments
    time_len=savar_params.time_len,
    comp_size=savar_params.comp_size,
    noise_val=savar_params.noise_val,
    n_per_col=savar_params.n_per_col,
    difficulty=savar_params.difficulty,
    seasonality=savar_params.seasonality,
    overlap=savar_params.overlap,
    is_forced=savar_params.is_forced,
    plot_original_data=savar_params.plot_original_data,
)
datamodule.setup()

# save config file to data directory
savar_name = datamodule.train_val_input4mips.savar_name
shutil.copy2(json_path, os.path.join(data_dir, f"{savar_name}_config.json"))
