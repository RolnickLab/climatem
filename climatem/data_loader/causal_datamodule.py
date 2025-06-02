# Here we try to modify the climate_data_loader so that we can use data from multiple ensemble members of a climate model, and indeed across climate models.

import os
from typing import Optional

import numpy as np
import torch

from climatem.constants import AVAILABLE_MODELS_FIRETYPE, OPENBURNING_MODEL_MAPPING

# import relevant data loading modules
from climatem.data_loader.climate_datamodule import ClimateDataModule
from climatem.data_loader.cmip6_dataset import CMIP6Dataset
from climatem.data_loader.era5_dataset import ERA5Dataset
from climatem.data_loader.input4mip_dataset import Input4MipsDataset
from climatem.data_loader.savar_dataset import SavarDataset


class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index: int):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class CausalClimateDataModule(ClimateDataModule):
    """
    This class inherits from the ClimateDataModule class and uses the same initialization parameters.

    The setup method is overwritten and performs data preprocessing for causal discovery models.
    """

    def __init__(self, tau=5, future_timesteps=1, num_months_aggregated=1, train_val_interval_length=100, **kwargs):
        super().__init__(self)

        # kwargs are initialized as self.hparams by the Lightning module
        # WHat is this line? We cannot have different test vs train models
        # self.hparams.test_models = None if self.hparams.test_models else self.hparams.train_models
        self.hparams.test_models = self.hparams.train_models
        self.tau = tau
        self.future_timesteps = future_timesteps
        self.num_months_aggregated = num_months_aggregated
        self.train_val_interval_length = train_val_interval_length
        self.shuffle_train = False  # need to keep order for causal train / val splits

    @staticmethod
    def years_to_list(years_str):
        """Convert years input to list of years."""
        if years_str is None:
            return []
        elif isinstance(years_str, int):
            return [years_str]
        elif isinstance(years_str, str):
            if len(years_str) != 9:
                raise ValueError("Years string must be in the format xxxx-yyyy (eg. 2015-2100).")

            years = years_str.split("-")
            min_year, max_year = int(years[0]), int(years[1])
            return np.arange(min_year, max_year + 1)
        else:
            raise ValueError(f"years_str must be int, str, or None, not {type(years_str)}")

    def setup(self, stage: Optional[str] = None):
        if stage in ["fit", "validate", None]:
            openburning_specs = (
                OPENBURNING_MODEL_MAPPING[self.hparams.train_models]
                if self.hparams.train_models in AVAILABLE_MODELS_FIRETYPE
                else OPENBURNING_MODEL_MAPPING["other"]
            )

            train_years = self.years_to_list(self.hparams.train_years)
            train_historical_years = self.years_to_list(self.hparams.train_historical_years)

            os.makedirs(self.hparams.output_save_dir, exist_ok=True)
            # Here add an option for SAVAR dataset
            # TODO: propagate "reload argument here"
            # TODO: make sure all arguments are propagated i.e. seasonality_removal, output_save_dir
            datasets = []
            input_sources = self.hparams.in_var_ids  # e.g. {"era5": ["t2m"], "cmip6": ["ts"]}

            for source, vars in input_sources.items():
                if source == "savar":
                    train_val_input4mips = SavarDataset(
                        # Make sure these arguments are propagated
                        output_save_dir=self.hparams.output_save_dir,
                        lat=self.hparams.lat,
                        lon=self.hparams.lon,
                        tau=self.tau,
                        global_normalization=self.hparams.global_normalization,
                        seasonality_removal=self.hparams.seasonality_removal,
                        reload_climate_set_data=self.hparams.reload_climate_set_data,
                        time_len=self.hparams.time_len,
                        comp_size=self.hparams.comp_size,
                        noise_val=self.hparams.noise_val,
                        n_per_col=self.hparams.n_per_col,
                        difficulty=self.hparams.difficulty,
                        seasonality=self.hparams.seasonality,
                        overlap=self.hparams.overlap,
                        is_forced=self.hparams.is_forced,
                        plot_original_data=self.hparams.plot_original_data,
                    )
                elif source == "era5":
                    train_val_input4mips = ERA5Dataset(
                        years=self.hparams.train_years,
                        historical_years=self.hparams.train_historical_years,
                        data_dir=self.hparams.data_dir,
                        climate_model=self.hparams.train_models,
                        num_ensembles=self.hparams.num_ensembles,
                        variables=list(vars),
                        scenarios=self.hparams.train_scenarios,
                        channels_last=self.hparams.channels_last,
                        mode="train+val",
                        output_save_dir=self.hparams.output_save_dir,
                        lon=self.hparams.lon,
                        lat=self.hparams.lat,
                        icosahedral_coordinates_path=self.hparams.icosahedral_coordinates_path,
                        global_normalization=self.hparams.global_normalization,
                        seasonality_removal=self.hparams.seasonality_removal,
                        reload_climate_set_data=self.hparams.reload_climate_set_data,
                    )
                elif source == "cmip6":
                    train_val_input4mips = CMIP6Dataset(
                        years=train_years,
                        historical_years=train_historical_years,
                        data_dir=self.hparams.data_dir,
                        climate_model=self.hparams.train_models,
                        num_ensembles=self.hparams.num_ensembles,
                        variables=list(vars),
                        scenarios=self.hparams.train_scenarios,
                        channels_last=self.hparams.channels_last,
                        openburning_specs=openburning_specs,
                        mode="train+val",
                        output_save_dir=self.hparams.output_save_dir,
                        lon=self.hparams.lon,
                        lat=self.hparams.lat,
                        icosahedral_coordinates_path=self.hparams.icosahedral_coordinates_path,
                        global_normalization=self.hparams.global_normalization,
                        seasonality_removal=self.hparams.seasonality_removal,
                        reload_climate_set_data=self.hparams.reload_climate_set_data,
                    )
                else:
                    train_val_input4mips = Input4MipsDataset(
                        years=train_years,
                        historical_years=train_historical_years,
                        data_dir=self.hparams.data_dir,
                        variables=self.hparams.in_var_ids,
                        scenarios=self.hparams.train_scenarios,
                        channels_last=self.hparams.channels_last,
                        openburning_specs=openburning_specs,
                        mode="train+val",
                        output_save_dir=self.hparams.output_save_dir,
                        lon=self.hparams.lon,
                        lat=self.hparams.lat,
                        icosahedral_coordinates_path=self.hparams.icosahedral_coordinates_path,
                        global_normalization=self.hparams.global_normalization,
                        seasonality_removal=self.hparams.seasonality_removal,
                        reload_climate_set_data=self.hparams.reload_climate_set_data,
                    )

            ratio_train = 1 - self.hparams.val_split

            datasets.append(train_val_input4mips)

            # Construct input_var_shapes and offsets for mask metadata
            for var, spatial_dim in train_val_input4mips.input_var_shapes.items():
                self.input_var_shapes[var] = spatial_dim

            num_vars = len(self.input_var_shapes)
            d_x = 1  # variables * latents
            d_z = num_vars  # num_vars * dims (one latent per var per time step)

            obs_to_latent_mask = np.zeros((d_z, d_x), dtype=np.float32)
            for var, spatial_dim in train_val_input4mips.input_var_shapes.items():
                dim = self.input_var_shapes[var]
                obs_to_latent_mask[dim:dim] = 1.0
            self.obs_to_latent_mask = obs_to_latent_mask
            train, val = train_val_input4mips.get_causal_data(
                tau=self.tau,
                future_timesteps=self.future_timesteps,
                channels_last=self.hparams.channels_last,
                num_vars=num_vars,  # total num vars
                num_scenarios=1,
                num_ensembles=1,
                num_years=train_val_input4mips.length,
                ratio_train=ratio_train,
                num_months_aggregated=self.num_months_aggregated,
                interval_length=self.train_val_interval_length,
                obs_to_latent_mask=obs_to_latent_mask,
                mode="train+val",
            )
            if "savar" in self.hparams.in_var_ids:
                self.savar_gt_modes = train_val_input4mips.gt_modes
                self.savar_gt_noise = train_val_input4mips.gt_noise
                self.savar_gt_adj = train_val_input4mips.gt_adj

            train_x, train_y = train
            train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], -1))
            train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], train_y.shape[2], -1))

            self.d = train_x.shape[2]
            self._data_train = CausalDataset(train_x, train_y)
            self.n_train = train_x.shape[0]

            if val is not None:
                val_x, val_y = val
                val_x = val_x.reshape((val_x.shape[0], val_x.shape[1], val_x.shape[2], -1))
                val_y = val_y.reshape((val_y.shape[0], val_y.shape[1], val_y.shape[2], -1))
                self._data_val = CausalDataset(val_x, val_y)

            self.coordinates = train_val_input4mips.coordinates

        if stage in ["test", None]:
            openburning_specs = {
                test_model: (
                    OPENBURNING_MODEL_MAPPING[test_model]
                    if test_model in AVAILABLE_MODELS_FIRETYPE
                    else OPENBURNING_MODEL_MAPPING["other"]
                )
                for test_model in self.hparams.test_models
            }
