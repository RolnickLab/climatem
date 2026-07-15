# Here we try to modify the climate_data_loader so that we can use data from multiple ensemble members of a climate model, and indeed across climate models.

import os
from typing import Optional

import numpy as np
import torch

from climatem.constants import AVAILABLE_MODELS_FIRETYPE, OPENBURNING_MODEL_MAPPING

# import relevant data loading modules
from climatem.data_loader.climate_datamodule import ClimateDataModule
from climatem.data_loader.cmip6_dataset import CMIP6Dataset, MultiscenarioDataset
from climatem.data_loader.era5_dataset import ERA5Dataset
from climatem.data_loader.input4mip_dataset import Input4MipsDataset
from climatem.data_loader.savar_dataset import SavarDataset


class CausalDataset(torch.utils.data.Dataset):
    def __init__(
        self, x, y, co2_forcing=None, aerosol_forcing=None, gt_co2_latent=None, gt_aerosol_latent=None, domain=None
    ):
        #  Unsqueeze the variable dimension for savar data
        if co2_forcing.ndim == 3:
            co2_forcing = co2_forcing[:, :, None, :]
        if aerosol_forcing.ndim == 3:
            aerosol_forcing = aerosol_forcing[:, :, None, :]
        self.x = x
        self.y = y
        self.global_forcings = co2_forcing
        self.spatial_forcings = aerosol_forcing
        self.gt_co2_latent = gt_co2_latent
        self.gt_aerosol_latent = gt_aerosol_latent
        self.domain = domain

    def __getitem__(self, index: int):
        """
        Return batch as dictionary if forcings are available, otherwise as tuple.

        Returns:
            dict: {'x': x, 'y': y, 'co2_forcing': co2, 'aerosol_forcing': aerosol,
                   'gt_co2_latent': gt_co2, 'gt_aerosol_latent': gt_aerosol} if forcings present
            tuple: (x, y) if forcings not present
            # x (3069, 5, 1, 3072) y (3069, 1, 1, 3072) global_forcing (3069, 6, n, 1) spatial_forcing(3069, 6, n, 3072)-> I now squeeze var dim for forcings to stay consistency
        """

        if self.global_forcings is not None and self.spatial_forcings is not None:
            result = {
                "x": self.x[index],
                "y": self.y[index],
                "global_forcings": self.global_forcings[index],
                "spatial_forcings": self.spatial_forcings[index],
            }
            if self.domain is not None:
                result["domain"] = self.domain[index]
            # Add ground truth forcing latents if available
            if self.gt_co2_latent is not None:
                result["gt_co2_latent"] = self.gt_co2_latent[index]
            if self.gt_aerosol_latent is not None:
                result["gt_aerosol_latent"] = self.gt_aerosol_latent[index]
            return result
        else:
            # Backward compatibility: return tuple if no forcings
            return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class CausalClimateDataModule(ClimateDataModule):
    """
    This class inherits from the ClimateDataModule class and uses the same initialization parameters.

    The setup method is overwritten and performs data preprocessing for causal discovery models.
    """

    def __init__(self, tau=5, future_timesteps=1, num_months_aggregated=1, train_val_interval_length=100, **kwargs):
        super().__init__(**kwargs)

        # kwargs are initialized as self.hparams by the Lightning module
        # WHat is this line? We cannot have different test vs train models
        # self.hparams.test_models = None if self.hparams.test_models else self.hparams.train_models
        self.hparams.test_models = self.hparams.train_models
        self.tau = tau
        self.future_timesteps = future_timesteps
        self.num_months_aggregated = num_months_aggregated
        self.train_val_interval_length = train_val_interval_length
        self.shuffle_train = False  # need to keep order for causal train / val splits
        self.savar_name = None

    @staticmethod
    def years_to_list(years_str):
        """Convert years input to list of years."""
        if years_str is None:
            return []
        elif isinstance(years_str, int):
            return [years_str]
        elif isinstance(years_str, str):
            print(years_str)
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
            if "savar" in self.hparams.in_var_ids:
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
                    periods=self.hparams.periods,
                    amplitudes=self.hparams.amplitudes,
                    phases=self.hparams.phases,
                    yearly_jitter_amp=self.hparams.yearly_jitter_amp,
                    yearly_jitter_phase=self.hparams.yearly_jitter_phase,
                    overlap=self.hparams.overlap,
                    is_forced=self.hparams.is_forced,
                    f_1=self.hparams.f_1,
                    f_2=self.hparams.f_2,
                    f_time_1=self.hparams.f_time_1,
                    f_time_2=self.hparams.f_time_2,
                    ramp_type=self.hparams.ramp_type,
                    linearity=self.hparams.linearity,
                    poly_degrees=self.hparams.poly_degrees,
                    plot_original_data=self.hparams.plot_original_data,
                    use_separate_forcings=self.hparams.use_separate_forcings,
                    forcing_amplification=self.hparams.forcing_amplification,
                    forcing_conditioning=self.hparams.forcing_conditioning,
                    aerosol_scale=self.hparams.aerosol_scale,
                    aerosol_spatial_contrast=self.hparams.aerosol_spatial_contrast,
                    aerosol_ramp_up_time=self.hparams.aerosol_ramp_up_time,
                    aerosol_peak_time=self.hparams.aerosol_peak_time,
                    aerosol_decline_time=self.hparams.aerosol_decline_time,
                    n_co2_latents=self.hparams.n_co2_latents,
                    n_aerosol_latents=self.hparams.n_aerosol_latents,
                    co2_effect_strength=self.hparams.co2_effect_strength,
                    aerosol_effect_strength=self.hparams.aerosol_effect_strength,
                    noise_ar1=self.hparams.noise_ar1,
                    noise_ar1_rho=self.hparams.noise_ar1_rho,
                    enable_background=self.hparams.enable_background,
                    background_strength=self.hparams.background_strength,
                    background_strength_mode=self.hparams.background_strength_mode,
                    background_smoothness=self.hparams.background_smoothness,
                    background_timescale_rho=self.hparams.background_timescale_rho,
                    background_n_modes=self.hparams.background_n_modes,
                )
                self.savar_name = train_val_input4mips.savar_name

            elif (
                "tas" in self.hparams.in_var_ids
                or "pr" in self.hparams.in_var_ids
                or "psl" in self.hparams.in_var_ids
                or "ts" in self.hparams.in_var_ids
            ):

                print(
                    f"Causal datamodule self.hparams.icosahedral_coordinates_path {self.hparams.icosahedral_coordinates_path}"
                )
                train_val_input4mips = CMIP6Dataset(
                    years=train_years,
                    historical_years=train_historical_years,
                    data_dir=self.hparams.data_dir,
                    climate_model=self.hparams.train_models,
                    num_ensembles=self.hparams.num_ensembles,
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
            elif "t2m" in self.hparams.in_var_ids:
                train_val_input4mips = ERA5Dataset(
                    years=train_years,
                    historical_years=train_historical_years,
                    data_dir=self.hparams.data_dir,
                    climate_model=self.hparams.train_models,
                    num_ensembles=self.hparams.num_ensembles,
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

            train, val = train_val_input4mips.get_causal_data(
                tau=self.tau,
                future_timesteps=self.future_timesteps,
                channels_last=self.hparams.channels_last,
                num_vars=len(self.hparams.in_var_ids),
                num_scenarios=len(self.hparams.train_scenarios),
                num_ensembles=self.hparams.num_ensembles,
                num_years=len(train_years),
                ratio_train=ratio_train,
                num_months_aggregated=self.num_months_aggregated,
                interval_length=self.train_val_interval_length,
                mode="train+val",
            )
            if "savar" in self.hparams.in_var_ids:
                self.savar_gt_modes = train_val_input4mips.gt_modes
                self.savar_gt_noise = train_val_input4mips.gt_noise
                self.savar_gt_adj = train_val_input4mips.gt_adj
                self.forcing_indices = getattr(train_val_input4mips, "forcing_indices", None)
                # Store reference to SAVAR instance for later plotting
                self.savar = train_val_input4mips

            train_x, train_y = train
            train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], -1))
            train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], train_y.shape[2], -1))

            # Get forcing data if available (only for SAVAR with dual exogenous forcings)
            # For other datasets, getattr returns None and CausalDataset falls back to tuple mode
            co2_forcing_train = getattr(train_val_input4mips, "co2_forcing_train", None)
            aerosol_forcing_train = getattr(train_val_input4mips, "aerosol_forcing_train", None)
            co2_forcing_valid = getattr(train_val_input4mips, "co2_forcing_valid", None)
            aerosol_forcing_valid = getattr(train_val_input4mips, "aerosol_forcing_valid", None)

            # Get ground truth forcing latents if available (for SAVAR with forcing latent supervision)
            gt_co2_latent_train = getattr(train_val_input4mips, "gt_co2_latent_train", None)
            gt_aerosol_latent_train = getattr(train_val_input4mips, "gt_aerosol_latent_train", None)
            gt_co2_latent_valid = getattr(train_val_input4mips, "gt_co2_latent_valid", None)
            gt_aerosol_latent_valid = getattr(train_val_input4mips, "gt_aerosol_latent_valid", None)

            self.d = train_x.shape[2]
            self._data_train = CausalDataset(
                x=train_x,
                y=train_y,
                co2_forcing=co2_forcing_train,
                aerosol_forcing=aerosol_forcing_train,
                gt_co2_latent=gt_co2_latent_train,
                gt_aerosol_latent=gt_aerosol_latent_train,
            )
            self.n_train = train_x.shape[0]

            if val is not None:
                val_x, val_y = val
                val_x = val_x.reshape((val_x.shape[0], val_x.shape[1], val_x.shape[2], -1))
                val_y = val_y.reshape((val_y.shape[0], val_y.shape[1], val_y.shape[2], -1))
                self._data_val = CausalDataset(
                    x=val_x,
                    y=val_y,
                    co2_forcing=co2_forcing_valid,
                    aerosol_forcing=aerosol_forcing_valid,
                    gt_co2_latent=gt_co2_latent_valid,
                    gt_aerosol_latent=gt_aerosol_latent_valid,
                )

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


class CausalClimateDataMultiScenarioModule(ClimateDataModule):
    """
    DataModule for causal discovery using multi-scenario climate data.

    It loads spatial variables and global forcing variables separately, then converts them into CausalDataset inputs.
    """

    def __init__(
        self,
        tau=5,
        future_timesteps=1,
        num_months_aggregated=1,
        train_val_interval_length=100,
        **kwargs,
    ):
        super().__init__(self)

        self.hparams.test_models = self.hparams.train_models
        self.tau = tau
        self.future_timesteps = future_timesteps
        self.num_months_aggregated = num_months_aggregated
        self.train_val_interval_length = train_val_interval_length
        self.shuffle_train = False
        self.savar_name = None

    @staticmethod
    def years_to_list(years_str):
        """Convert years input to list of years."""
        if years_str is None:
            return []
        elif isinstance(years_str, int):
            return [years_str]
        elif isinstance(years_str, str):
            print(years_str)
            if len(years_str) != 9:
                raise ValueError("Years string must be in the format xxxx-yyyy (eg. 2015-2100).")

            years = years_str.split("-")
            min_year, max_year = int(years[0]), int(years[1])
            return np.arange(min_year, max_year + 1)
        else:
            raise ValueError(f"years_str must be int, str, or None, not {type(years_str)}")

    def setup(self, stage: Optional[str] = None):
        train_years = self.years_to_list(self.hparams.train_years)  # list 1600-2100

        train_historical_years = self.years_to_list(self.hparams.train_historical_years)

        os.makedirs(self.hparams.output_save_dir, exist_ok=True)

        if stage in ["fit", "validate", None]:
            train_val_dataset = self._build_multiscenario_dataset(
                years=train_years,
                historical_years=train_historical_years,
                scenarios=self.hparams.train_scenarios,
                mode="train+val",
            )

            self._setup_train_val(train_val_dataset, train_years)
            self.coordinates = train_val_dataset.coordinates

        if stage in ["test", None]:
            test_dataset = self._build_multiscenario_dataset(
                years=train_years,
                historical_years=train_historical_years,
                scenarios=self.hparams.test_scenarios,
                mode="test",
            )

            self._setup_test(test_dataset)

    def _build_multiscenario_dataset(self, years, historical_years, scenarios, mode):
        openburning_specs = (
            OPENBURNING_MODEL_MAPPING[self.hparams.train_models]
            if self.hparams.train_models in AVAILABLE_MODELS_FIRETYPE
            else OPENBURNING_MODEL_MAPPING["other"]
        )

        return MultiscenarioDataset(
            years=years,
            historical_years=historical_years,
            data_dir=self.hparams.data_dir,
            climate_model=self.hparams.train_models,
            num_ensembles=self.hparams.num_ensembles,
            variables=self.hparams.in_var_ids,
            scenarios=scenarios,
            channels_last=self.hparams.channels_last,
            openburning_specs=openburning_specs,
            mode=mode,
            output_save_dir=self.hparams.output_save_dir,
            lon=self.hparams.lon,
            lat=self.hparams.lat,
            icosahedral_coordinates_path=self.hparams.icosahedral_coordinates_path,
            global_normalization=self.hparams.global_normalization,
            seasonality_removal=self.hparams.seasonality_removal,
            reload_climate_set_data=self.hparams.reload_climate_set_data,
        )

    def _get_causal_data(self, dataset, data, mode, domain_names, num_years=None, ratio_train=None):
        return dataset.get_causal_data(
            tau=self.tau,
            data=data,
            future_timesteps=self.future_timesteps,
            ratio_train=ratio_train,
            num_months_aggregated=self.num_months_aggregated,
            interval_length=self.train_val_interval_length,
            mode=mode,
            domain_names=domain_names,
        )

    def _setup_train_val(self, dataset, train_years):
        ratio_train = 1 - self.hparams.val_split
        #  for SSP
        train_spatial_forcing_ssp, val_spatial_forcing_ssp = self._get_causal_data(
            dataset,
            dataset.norm_spatial_forcing_data_ssp,
            mode="train+val",
            num_years=len(train_years),
            ratio_train=ratio_train,
            domain_names=dataset.scenarios,
        )
        train_global_forcing_ssp, val_global_forcing_ssp = self._get_causal_data(
            dataset,
            dataset.norm_global_forcing_data_ssp,
            mode="train+val",
            num_years=len(train_years),
            ratio_train=ratio_train,
            domain_names=dataset.scenarios,
        )
        train_target_ssp, val_target_ssp = self._get_causal_data(
            dataset,
            dataset.norm_target_data_ssp,
            mode="train+val",
            num_years=len(train_years),
            ratio_train=ratio_train,
            domain_names=dataset.scenarios,
        )

        #  for HIST
        train_target_hist, val_target_hist = self._get_causal_data(
            dataset,
            dataset.norm_target_data_hist,
            mode="train+val",
            num_years=len(train_years),
            ratio_train=ratio_train,
            domain_names=["historical"],
        )

        train_spatial_forcing_hist, val_spatial_forcing_hist = self._get_causal_data(
            dataset,
            dataset.norm_spatial_forcing_data_hist,
            mode="train+val",
            num_years=len(train_years),
            ratio_train=ratio_train,
            domain_names=["historical"],
        )
        train_global_forcing_hist, val_global_forcing_hist = self._get_causal_data(
            dataset,
            dataset.norm_global_forcing_data_hist,
            mode="train+val",
            num_years=len(train_years),
            ratio_train=ratio_train,
            domain_names=["historical"],
        )

        train = self._merge_historical_and_ssp(
            target_ssp=train_target_ssp,
            target_hist=train_target_hist,
            spatial_forcing_ssp=train_spatial_forcing_ssp,
            global_forcing_ssp=train_global_forcing_ssp,
            spatial_forcing_hist=train_spatial_forcing_hist,
            global_forcing_hist=train_global_forcing_hist,
        )

        self._data_train = train
        self.n_train = train.x.shape[0] if hasattr(train, "x") else len(train)

        if val_spatial_forcing_ssp is not None:
            self._data_val = self._merge_historical_and_ssp(
                target_ssp=val_target_ssp,
                target_hist=val_target_hist,
                spatial_forcing_ssp=val_spatial_forcing_ssp,
                global_forcing_ssp=val_global_forcing_ssp,
                spatial_forcing_hist=val_spatial_forcing_hist,
                global_forcing_hist=val_global_forcing_hist,
            )

    def _setup_test(self, dataset):
        test_spatial_forcing, _ = self._get_causal_data(
            dataset, dataset.norm_spatial_forcing_data_ssp, mode="test", domain_names=dataset.scenarios
        )
        print("in test, scenarios are ", dataset.scenarios)
        test_global_forcing, _ = self._get_causal_data(
            dataset, dataset.norm_global_forcing_data_ssp, mode="test", domain_names=dataset.scenarios
        )
        test_target, test_domain = self._get_causal_data(
            dataset, dataset.norm_target_data_ssp, mode="test", domain_names=dataset.scenarios
        )

        self._data_test = self._build_causal_dataset(
            test_target, test_spatial_forcing, test_global_forcing, test_domain
        )

    def _merge_historical_and_ssp(
        self,
        target_ssp,
        target_hist,
        spatial_forcing_ssp,
        global_forcing_ssp,
        spatial_forcing_hist,
        global_forcing_hist,
    ):
        spatial_forcing_ssp_x, spatial_forcing_ssp_y, _ = (
            spatial_forcing_ssp  # (1550, 5, 2var, 3072)  #(1550, 1, 2, 3072)
        )
        global_forcing_ssp_x, global_forcing_ssp_y, _ = global_forcing_ssp  # (1550, 5, 2, 1)  #(1550, 1, 2, 1)

        spatial_forcing_hist_x, spatial_forcing_hist_y, _ = (
            spatial_forcing_hist  # (1519, 5, 2, 3072) # (1519, 1, 2, 3072)
        )
        global_forcing_hist_x, global_forcing_hist_y, _ = global_forcing_hist  # (1519, 5, 2, 1)  # (1519, 1, 2, 1)

        target_ssp_x, target_ssp_y, target_ssp_name = target_ssp  # (1550, 5, 1, 3072)  # (1550, 1, 1, 3072)
        target_hist_x, target_hist_y, target_hist_name = target_hist  # (1519, 5, 1, 3072) # (1519, 1, 1, 3072)
        spatial_forcing = (
            np.concatenate((spatial_forcing_ssp_x, spatial_forcing_hist_x), axis=0),
            np.concatenate((spatial_forcing_ssp_y, spatial_forcing_hist_y), axis=0),
        )
        global_forcing = (
            np.concatenate((global_forcing_ssp_x, global_forcing_hist_x), axis=0),
            np.concatenate((global_forcing_ssp_y, global_forcing_hist_y), axis=0),
        )
        target = (
            np.concatenate((target_ssp_x, target_hist_x), axis=0),
            np.concatenate((target_ssp_y, target_hist_y), axis=0),
        )
        domain = np.concatenate((target_ssp_name, target_hist_name), axis=0)
        return self._build_causal_dataset(target, spatial_forcing, global_forcing, domain)

    def _build_causal_dataset(self, target_data, spatial_forcing_data, global_forcing_data, domain):
        spatial_forcing_x, spatial_forcing_y = spatial_forcing_data
        global_forcing_x, global_forcing_y = global_forcing_data
        target_x, target_y = target_data
        # For forcings, we directly input tau+1 steps

        spatial_forcing = np.concatenate((spatial_forcing_x, spatial_forcing_y), axis=1)
        global_forcing = np.concatenate((global_forcing_x, global_forcing_y), axis=1)

        target_x = self._flatten_space(target_x)
        target_y = self._flatten_space(target_y)
        spatial_forcing = self._flatten_space(spatial_forcing)
        global_forcing = self._flatten_space(global_forcing)
        self.d = target_x.shape[2]

        return CausalDataset(
            x=target_x,
            y=target_y,
            co2_forcing=global_forcing,
            aerosol_forcing=spatial_forcing,
            domain=domain,
        )

    @staticmethod
    def _flatten_space(data):
        return data.reshape(data.shape[0], data.shape[1], data.shape[2], -1)


if __name__ == "__main__":
    datamodule = CausalClimateDataMultiScenarioModule()
