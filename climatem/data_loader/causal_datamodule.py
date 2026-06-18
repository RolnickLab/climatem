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

# class CausalDataset(torch.utils.data.Dataset):
#     def __init__(self, x, y, global_forcing=None, spatial_forcing=None, domain=None):
#         self.x = x
#         self.y = y
#         self.global_forcing = global_forcing
#         self.spatial_forcing = spatial_forcing
#         self.domain=domain

#     def __getitem__(self, index: int):
#         """
#         Return batch as dictionary if forcings are available, otherwise as tuple.

#         Returns:
#             dict: {'x': x, 'y': y, 'co2_forcing': co2, 'aerosol_forcing': aerosol,
#                    'gt_co2_latent': gt_co2, 'gt_aerosol_latent': gt_aerosol} if forcings present
#             tuple: (x, y) if forcings not present
#         """
#         if self.global_forcing is not None and self.spatial_forcing is not None:
#             result = {
#                 "x": self.x[index],
#                 "y": self.y[index],
#                 "global_forcing": self.global_forcing[index],
#                 "spatial_forcing": self.spatial_forcing[index],
#             }
#             if self.domain is not None:
#                 result["domain"] = self.domain[index]
#             return result
#         else:
#             # Backward compatibility: return tuple if no forcings
#             return self.x[index], self.y[index]


#     def __len__(self):
#         return len(self.x)
class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, global_forcing=None, spatial_forcing=None, domain=None):
        # x (3069, 5, 1, 3072) y (3069, 1, 1, 3072) global_forcing (3069, 6, 2, 1) spatial_forcing(3069, 6, 2, 3072)
        self.global_forcing = global_forcing
        self.spatial_forcing = spatial_forcing
        self.domain = domain
        global_forcing_expanded = np.repeat(global_forcing, x.shape[-1], axis=-1)
        self.x = np.concatenate([x, spatial_forcing[:, :-1], global_forcing_expanded[:, :-1]], axis=2)
        self.y = np.concatenate([y, spatial_forcing[:, -1:], global_forcing_expanded[:, -1:]], axis=2)
        # print("self.x",self.x.shape)
        # print("self.y",self.y.shape)

    def __getitem__(self, index: int):
        """
        Return batch as dictionary if forcings are available, otherwise as tuple.

        Returns:
            dict: {'x': x, 'y': y, 'co2_forcing': co2, 'aerosol_forcing': aerosol,
                   'gt_co2_latent': gt_co2, 'gt_aerosol_latent': gt_aerosol} if forcings present
            tuple: (x, y) if forcings not present
        """
        if self.global_forcing is not None and self.spatial_forcing is not None:
            result = {
                "x": self.x[index],
                "y": self.y[index],
                "global_forcing": self.global_forcing[index],
                "spatial_forcing": self.spatial_forcing[index],
            }
            if self.domain is not None:
                result["domain"] = self.domain[index]
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
            # channels_last=self.hparams.channels_last,
            # num_vars=len(self.hparams.in_var_ids),
            # num_scenarios=len(self.hparams.train_scenarios),
            # num_ensembles=self.hparams.num_ensembles,
            # num_years=num_years,
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
        # print("test_domain",test_domain.dtype, test_domain)
        # print("test_domain",test_domain.shape)

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
        spatial_forcing_ssp_x, spatial_forcing_ssp_y, _ = spatial_forcing_ssp
        # print("spatial_forcing_ssp_x",spatial_forcing_ssp_x.shape) #(1550, 5, 2var, 3072)
        # print("spatial_forcing_ssp_y",spatial_forcing_ssp_y.shape) #(1550, 1, 2, 3072)
        global_forcing_ssp_x, global_forcing_ssp_y, _ = global_forcing_ssp
        # print("global_forcing_ssp_x",global_forcing_ssp_x.shape)  # (1550, 5, 2, 1)
        # print("global_forcing_ssp_y",global_forcing_ssp_y.shape) #(1550, 1, 2, 1)

        spatial_forcing_hist_x, spatial_forcing_hist_y, _ = spatial_forcing_hist
        # print("spatial_forcing_hist_x",spatial_forcing_hist_x.shape) #(1519, 5, 2, 3072)
        # print("spatial_forcing_hist_y",spatial_forcing_hist_y.shape) # (1519, 1, 2, 3072)
        global_forcing_hist_x, global_forcing_hist_y, _ = global_forcing_hist
        # print("global_forcing_hist_x",global_forcing_hist_x.shape) # (1519, 5, 2, 1)
        # print("global_forcing_hist_y",global_forcing_hist_y.shape) # (1519, 1, 2, 1)

        target_ssp_x, target_ssp_y, target_ssp_name = target_ssp
        # print("target_ssp_x",target_ssp_x.shape) #(1550, 5, 1, 3072)
        # print("target_ssp_y",target_ssp_y.shape) # (1550, 1, 1, 3072)
        target_hist_x, target_hist_y, target_hist_name = target_hist
        # print("target_hist_x",target_hist_x.shape) #(1519, 5, 1, 3072)
        # print("target_hist_y",target_hist_y.shape) # (1519, 1, 1, 3072)
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
        # print("spatial_forcing_x",spatial_forcing_x.shape)
        # print("spatial_forcing_y",spatial_forcing_y.shape)

        spatial_forcing = np.concatenate((spatial_forcing_x, spatial_forcing_y), axis=1)
        global_forcing = np.concatenate((global_forcing_x, global_forcing_y), axis=1)

        target_x = self._flatten_space(target_x)
        target_y = self._flatten_space(target_y)
        spatial_forcing = self._flatten_space(spatial_forcing)
        global_forcing = self._flatten_space(global_forcing)
        self.d = target_x.shape[2]

        return CausalDataset(target_x, target_y, global_forcing, spatial_forcing, domain)

    @staticmethod
    def _flatten_space(data):
        return data.reshape(data.shape[0], data.shape[1], data.shape[2], -1)


if __name__ == "__main__":
    datamodule = CausalClimateDataMultiScenarioModule()


# class CausalClimateDataMultiScenarioModule(ClimateDataModule):
#     """
#     This class inherits from the ClimateDataModule class and uses the same initialization parameters.

#     The setup method is overwritten and performs data preprocessing for causal discovery models.
#     """

#     def __init__(self, tau=5, future_timesteps=1, num_months_aggregated=1, train_val_interval_length=100, **kwargs):
#         super().__init__(self)

#         # kwargs are initialized as self.hparams by the Lightning module
#         # WHat is this line? We cannot have different test vs train models
#         # self.hparams.test_models = None if self.hparams.test_models else self.hparams.train_models
#         self.hparams.test_models = self.hparams.train_models
#         self.tau = tau
#         self.future_timesteps = future_timesteps
#         self.num_months_aggregated = num_months_aggregated
#         self.train_val_interval_length = train_val_interval_length
#         self.shuffle_train = False  # need to keep order for causal train / val splits
#         self.savar_name = None

#     @staticmethod
#     def years_to_list(years_str):
#         """Convert years input to list of years."""
#         if years_str is None:
#             return []
#         elif isinstance(years_str, int):
#             return [years_str]
#         elif isinstance(years_str, str):
#             print(years_str)
#             if len(years_str) != 9:
#                 raise ValueError("Years string must be in the format xxxx-yyyy (eg. 2015-2100).")

#             years = years_str.split("-")
#             min_year, max_year = int(years[0]), int(years[1])
#             return np.arange(min_year, max_year + 1)
#         else:
#             raise ValueError(f"years_str must be int, str, or None, not {type(years_str)}")

#     def setup(self, stage: Optional[str] = None):
#         if stage in ["fit", "validate", None]:
#             openburning_specs = (
#                 OPENBURNING_MODEL_MAPPING[self.hparams.train_models]
#                 if self.hparams.train_models in AVAILABLE_MODELS_FIRETYPE
#                 else OPENBURNING_MODEL_MAPPING["other"]
#             )

#             train_years = self.years_to_list(self.hparams.train_years)
#             train_historical_years = self.years_to_list(self.hparams.train_historical_years)

#             os.makedirs(self.hparams.output_save_dir, exist_ok=True)
#             # Here add an option for SAVAR dataset
#             # TODO: propagate "reload argument here"
#             # TODO: make sure all arguments are propagated i.e. seasonality_removal, output_save_dir

#             train_val_input4mips = MultiscenarioDataset(
#                     years=train_years,
#                     historical_years=train_historical_years,
#                     data_dir=self.hparams.data_dir,
#                     climate_model=self.hparams.train_models,
#                     num_ensembles=self.hparams.num_ensembles,
#                     variables=self.hparams.in_var_ids,
#                     scenarios=self.hparams.train_scenarios,
#                     channels_last=self.hparams.channels_last,
#                     openburning_specs=openburning_specs,
#                     mode="train+val",
#                     output_save_dir=self.hparams.output_save_dir,
#                     lon=self.hparams.lon,
#                     lat=self.hparams.lat,
#                     icosahedral_coordinates_path=self.hparams.icosahedral_coordinates_path,
#                     global_normalization=self.hparams.global_normalization,
#                     seasonality_removal=self.hparams.seasonality_removal,
#                     reload_climate_set_data=self.hparams.reload_climate_set_data,
#                 )


#             ratio_train = 1 - self.hparams.val_split

#             train_spatial_ssp, val_spatial_ssp = train_val_input4mips.get_causal_data(
#                 tau=self.tau,
#                 data = train_val_input4mips.spatial_ssp_data,
#                 future_timesteps=self.future_timesteps,
#                 channels_last=self.hparams.channels_last,
#                 num_vars=len(self.hparams.in_var_ids),
#                 num_scenarios=len(self.hparams.train_scenarios),
#                 num_ensembles=self.hparams.num_ensembles,
#                 num_years=len(train_years),
#                 ratio_train=ratio_train,
#                 num_months_aggregated=self.num_months_aggregated,
#                 interval_length=self.train_val_interval_length,
#                 mode="train+val",
#             )
#             train_global_ssp, val_global_ssp = train_val_input4mips.get_causal_data(
#                 tau=self.tau,
#                 data = train_val_input4mips.global_ssp_norm_data,
#                 future_timesteps=self.future_timesteps,
#                 channels_last=self.hparams.channels_last,
#                 num_vars=len(self.hparams.in_var_ids),
#                 num_scenarios=len(self.hparams.train_scenarios),
#                 num_ensembles=self.hparams.num_ensembles,
#                 num_years=len(train_years),
#                 ratio_train=ratio_train,
#                 num_months_aggregated=self.num_months_aggregated,
#                 interval_length=self.train_val_interval_length,
#                 mode="train+val",
#             )
#             train_spatial_historical, val_spatial_historical = train_val_input4mips.get_causal_data(
#                 tau=self.tau,
#                 data = train_val_input4mips.spatial_historical_data,
#                 future_timesteps=self.future_timesteps,
#                 channels_last=self.hparams.channels_last,
#                 num_vars=len(self.hparams.in_var_ids),
#                 num_scenarios=len(self.hparams.train_scenarios),
#                 num_ensembles=self.hparams.num_ensembles,
#                 num_years=len(train_years),
#                 ratio_train=ratio_train,
#                 num_months_aggregated=self.num_months_aggregated,
#                 interval_length=self.train_val_interval_length,
#                 mode="train+val",
#             )
#             train_global_historical, val_global_historical= train_val_input4mips.get_causal_data(
#                 tau=self.tau,
#                 data = train_val_input4mips.global_historical_norm_data,
#                 future_timesteps=self.future_timesteps,
#                 channels_last=self.hparams.channels_last,
#                 num_vars=len(self.hparams.in_var_ids),
#                 num_scenarios=len(self.hparams.train_scenarios),
#                 num_ensembles=self.hparams.num_ensembles,
#                 num_years=len(train_years),
#                 ratio_train=ratio_train,
#                 num_months_aggregated=self.num_months_aggregated,
#                 interval_length=self.train_val_interval_length,
#                 mode="train+val",
#             )
#             # for spatial ssp data:
#             train_spatial_ssp_x, train_spatial_ssp_y = train_spatial_ssp
#             # for spatial historical data:
#             train_spatial_historical_x, train_spatial_historical_y = train_spatial_historical

#             # for global ssp data:
#             train_global_ssp_x, train_global_ssp_y = train_global_ssp
#             # for global historical data:
#             train_global_historical_x, train_global_historical_y = train_global_historical

#             train_spatial_x = torch.cat((train_spatial_ssp_x + train_spatial_historical_x),dim=0)
#             train_global_x = torch.cat((train_global_ssp_x + train_global_historical_x),dim=0)

#             train_spatial_y = train_spatial_ssp_y + train_spatial_historical_y
#             train_global_y = train_global_ssp_y + train_global_historical_y


#             train_x, train_y = train_spatial_x[:,:,0],  train_spatial_y[:,:,0]
#             train_spatial_forcing_x, train_spatial_forcing_y = train_spatial_x[:,:,1:],  train_spatial_y[:,:,1:]
#             train_globel_forcing_x,train_globel_forcing_y  = train_global_x[:,:,1:], train_global_y[:,:,1:]
#             train_spatial_forcing = torch.cat((train_spatial_forcing_x, train_spatial_forcing_y), dim=-3) #dim=time
#             train_globel_forcing = torch.cat((train_globel_forcing_x, train_globel_forcing_y), dim=-3) #dim=time


#             train_x = train_x.reshape((train_x.shape[0], train_x.shape[1], train_x.shape[2], -1))
#             train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], train_y.shape[2], -1))
#             train_spatial_forcing = train_spatial_forcing.reshape((train_spatial_forcing.shape[0], train_spatial_forcing.shape[1], train_spatial_forcing.shape[2], -1))
#             train_globel_forcing = train_globel_forcing.reshape((train_globel_forcing.shape[0], train_globel_forcing.shape[1], train_globel_forcing.shape[2], -1))

#             self.d = train_x.shape[2]
#             self._data_train = CausalDataset(train_x, train_y, train_globel_forcing, train_spatial_forcing)
#             self.n_train = train_x.shape[0]

#             if val_spatial_ssp is not None:
#                 val_spatial_ssp_x, val_spatial_ssp_y = val_spatial_ssp
#                 # for spatial historical data:
#                 val_spatial_historical_x, val_spatial_historical_y =val_spatial_historical

#                 # for global ssp data:
#                 val_global_ssp_x, val_global_ssp_y = val_global_ssp
#                 # for global historical data:
#                 val_global_historical_x, val_global_historical_y = val_global_historical

#                 val_spatial_x = torch.cat((val_spatial_ssp_x , val_spatial_historical_x),dim=0)
#                 val_global_x = torch.cat((val_global_ssp_x, val_global_historical_x),dim=0)

#                 val_spatial_y = val_spatial_ssp_y + val_spatial_historical_y
#                 val_global_y = val_global_ssp_y + val_global_historical_y


#                 val_x, val_y = val_spatial_x[:,:,0], val_spatial_y[:,:,0]
#                 val_spatial_forcing_x, val_spatial_forcing_y = val_spatial_x[:,:,1:], val_spatial_y[:,:,1:]
#                 val_globel_forcing_x,val_globel_forcing_y  = val_global_x[:,:,1:], val_global_y[:,:,1:]
#                 val_spatial_forcing = torch.cat((val_spatial_forcing_x, val_spatial_forcing_y), dim=-3) #dim=time
#                 val_globel_forcing = torch.cat((val_globel_forcing_x, val_globel_forcing_y), dim=-3) #dim=time


#                 val_x = val_x.reshape((val_x.shape[0], val_x.shape[1], val_x.shape[2], -1))
#                 val_y = val_y.reshape((val_y.shape[0], val_y.shape[1], val_y.shape[2], -1))
#                 val_spatial_forcing = val_spatial_forcing.reshape((val_spatial_forcing.shape[0], val_spatial_forcing.shape[1], val_spatial_forcing.shape[2], -1))
#                 val_globel_forcing = val_globel_forcing.reshape((val_globel_forcing.shape[0], val_globel_forcing.shape[1], val_globel_forcing.shape[2], -1))
#                 self._data_val = CausalDataset(val_x, val_y, val_globel_forcing, val_spatial_forcing)

#             self.coordinates = train_val_input4mips.coordinates

#         if stage in ["test", None]:
#             openburning_specs = {
#                 test_model: (
#                     OPENBURNING_MODEL_MAPPING[test_model]
#                     if test_model in AVAILABLE_MODELS_FIRETYPE
#                     else OPENBURNING_MODEL_MAPPING["other"]
#                 )
#                 for test_model in self.hparams.test_models
#             }
#             test_input4mips = MultiscenarioDataset(
#                     years=train_years,
#                     historical_years=train_historical_years,
#                     data_dir=self.hparams.data_dir,
#                     climate_model=self.hparams.train_models,
#                     num_ensembles=self.hparams.num_ensembles,
#                     variables=self.hparams.in_var_ids,
#                     scenarios=self.hparams.test_scenarios, # to do: add test senarios
#                     channels_last=self.hparams.channels_last,
#                     openburning_specs=openburning_specs,
#                     mode="test",
#                     output_save_dir=self.hparams.output_save_dir,
#                     lon=self.hparams.lon,
#                     lat=self.hparams.lat,
#                     icosahedral_coordinates_path=self.hparams.icosahedral_coordinates_path,
#                     global_normalization=self.hparams.global_normalization,
#                     seasonality_removal=self.hparams.seasonality_removal,
#                     reload_climate_set_data=self.hparams.reload_climate_set_data,
#                 )

#             test_spatial_ssp = test_input4mips.get_causal_data(
#                 tau=self.tau,
#                 data = test_input4mips.spatial_ssp_data,
#                 future_timesteps=self.future_timesteps,
#                 num_months_aggregated=self.num_months_aggregated,
#                 mode="test",
#             )
#             test_global_ssp = test_input4mips.get_causal_data(
#                 tau=self.tau,
#                 data = test_input4mips.global_ssp_norm_data,
#                 future_timesteps=self.future_timesteps,
#                 num_months_aggregated=self.num_months_aggregated,
#                 mode="test",
#             )
#         test_spatial_x, test_spatial_y = test_spatial_ssp

#         # for global ssp data:
#         test_global_x, test_global_y = test_global_ssp


#         test_x, test_y = test_spatial_x[:, :, 0], test_spatial_y[:, :, 0]
#         test_spatial_forcing_x, test_spatial_forcing_y = test_spatial_x[:, :, 1:], test_spatial_y[:, :, 1:]
#         test_global_forcing_x, test_global_forcing_y = test_global_x[:, :, 1:], test_global_y[:, :, 1:]

#         test_spatial_forcing = torch.cat((test_spatial_forcing_x, test_spatial_forcing_y), dim=-3)  # dim=time
#         test_global_forcing = torch.cat((test_global_forcing_x, test_global_forcing_y), dim=-3)  # dim=time


#         test_x = test_x.reshape((test_x.shape[0], test_x.shape[1], test_x.shape[2], -1))
#         test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], test_y.shape[2], -1))
#         test_spatial_forcing = test_spatial_forcing.reshape(
#             (test_spatial_forcing.shape[0], test_spatial_forcing.shape[1], test_spatial_forcing.shape[2], -1)
#         )
#         test_global_forcing = test_global_forcing.reshape(
#             (test_global_forcing.shape[0], test_global_forcing.shape[1], test_global_forcing.shape[2], -1)
#         )

#         self._data_test = CausalDataset(
#             test_x,
#             test_y,
#             test_global_forcing,
#             test_spatial_forcing
#         )
