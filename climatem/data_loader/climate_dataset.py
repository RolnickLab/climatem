# import glob
# import os
# from datetime import datetime, timedelta
import itertools
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import xarray as xr

# from climatem.plotting.plot_data import plot_species, plot_species_anomaly
from climatem.utils import get_logger

# from climatem.constants import (  # INPUT4MIPS_NOM_RES,; INPUT4MIPS_TEMP_RES,; CMIP6_NOM_RES,; CMIP6_TEMP_RES,; NO_OPENBURNING_VARS,
#     AVAILABLE_MODELS_FIRETYPE,
#     OPENBURNING_MODEL_MAPPING,
# )


log = get_logger()


# base data set: implements copy to slurm, get item etc pp
# cmip6 data set: model wise
# input4mips data set: same per model
# from datamodule create one of these per train/test/val


class ClimateDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        years: Union[int, str] = "2015-2020",
        mode: str = "train",  # Train or test maybe # deprecated
        output_save_dir: Optional[str] = "Climateset_DATA",
        reload_climate_set_data: Optional[bool] = True,
        climate_model: str = "NorESM2-LM",  # implementing single model only for now
        num_ensembles: int = 1,  # 1 for first ensemble, -1 for all
        scenarios: Union[List[str], str] = ["ssp126", "ssp370", "ssp585"],
        historical_years: Union[Union[int, str], None] = "1850-1900",
        # NOTE:() here we are trying to implement multiple variables
        variables: List[str] = ["pr"],
        seq_to_seq: bool = True,  # TODO: implement if false
        channels_last: bool = False,
        load_data_into_mem: bool = True,  # Keeping this true be default for now
        seq_len: int = 12,
        lat: int = 96,
        lon: int = 144,
        icosahedral_coordinates_path: Optional[str] = "../../mappings/vertex_lonlat_mapping.npy",
        # input_transform=None,  # TODO: implement
        # input_normalization="z-norm",  # TODO: implement
        # output_transform=None,
        # output_normalization="z-norm",
        global_normalization: bool = True,
        seasonality_removal: bool = True,
        *args,
        **kwargs,
    ):
        """
        Args:
            years (Union[int,str], optional): test years. Defaults to "2015-2020".
            mode (str, optional): _description_. Defaults to "train".
            climate_model (str, optional): climate model from options "NorESM2-LM", "CESM2", "GISS-??". Defaults to "NorESM2-LM".
            historical_years (Union[Union[int, str], None], optional): meaningless parameter TODO. Defaults to "1850-1900".
            variables (Dict[str, List[str]], optional): map of sources and variables e.g. precipitation "pr", temperature "tas". Defaults to {"cmip6": "pr"}.
            seq_to_seq (bool, optional): _description_. Defaults to True.
            load_data_into_mem (bool, optional): _description_. Defaults to True.
            output_normalization (str, optional): _description_. Defaults to "z-norm".
            seasonality_removal (bool, optional): remove season through monthly normalisation. Defaults to True.
        """

        super().__init__()
        self.output_save_dir = Path(output_save_dir)
        self.reload_climate_set_data = reload_climate_set_data
        # Here need to propagate argument data_params.reload_climate_set_data

        self.channels_last = channels_last
        self.load_data_into_mem = load_data_into_mem
        self.variables = variables
        if isinstance(scenarios, str):
            scenarios = [scenarios]

        self.scenarios = scenarios
        if isinstance(years, int):
            self.years = years
        else:
            self.years = self.get_years_list(
                years, give_list=True
            )  # Can use this to split data into train/val eg. 2015-2080 train. 2080-2100 val.
        if historical_years is None:
            self.historical_years = []
        elif isinstance(historical_years, int):
            self.historical_years = historical_years
        else:
            self.historical_years = self.get_years_list(
                historical_years, give_list=True
            )  # Can use this to split data into train/val eg. 2015-2080 train. 2080-2100 val.
        self.n_years = len(self.years) + len(self.historical_years)

        self.global_normalization = global_normalization
        self.seasonality_removal = seasonality_removal

        # if climate_model in AVAILABLE_MODELS_FIRETYPE:
        #     openburning_specs = OPENBURNING_MODEL_MAPPING[climate_model]
        # else:
        #     openburning_specs = OPENBURNING_MODEL_MAPPING["other"]

        self.seq_len = seq_len
        self.lat = lat
        self.lon = lon
        self.icosahedral_coordinates_path = icosahedral_coordinates_path

    # NOTE:() changing this so it can deal with with grib files and netcdf files
    # this operates variable wise now.... #TODO: sizes for input4mips / adapt to mulitple vars
    def load_into_mem(
        self,
        paths: List[str],
        variables: List[str],
        channels_last: bool = True,
        seq_to_seq: bool = True,
        get_years: Optional[List[int]] = None,
    ):
        """
        Load multiple variables from single-file datasets (e.g., NetCDF or GRIB).

        Args:
            paths (List[List[str]]): absolute to filepath
            variables (List[str]): list of input variables e.g. pr, tas, etc.
            channels_last (bool, optional): reshape data to channels. Defaults to True.
            seq_to_seq (bool, optional): TBC. Defaults to True. #TODO
        """

        array_list = []
        input_var_shapes = {}
        input_var_offsets = [0]

        for vlist in paths:
            if vlist[0][-3:] == ".nc":
                temp_data = xr.open_mfdataset(vlist, concat_dim="time", combine="nested").compute()
                temp_data = temp_data.drop_dims("bnds", errors="ignore")

            elif vlist[0].endswith(".grib"):
                temp_data = xr.open_mfdataset(vlist, engine="cfgrib", concat_dim="time", combine="nested").compute()
            # TODO : handle gribs together
            elif vlist[0].endswith(".grib2"):
                # TODO: not all data will have this name to remove leap days + we should remove feb 29?
                filtered_vlist = list(itertools.chain(*vlist))
                filtered_vlist = [item for item in vlist if "000366.grib2" not in item]
                temp_data = xr.open_mfdataset(
                    filtered_vlist, engine="cfgrib", concat_dim="time", combine="nested"
                ).compute()

            else:
                print("File extension not recognized, please use either .nc or .grib")
                continue

            temp_data = temp_data.to_array().to_numpy()  # Should be of shape (vars, time, lat, lon)
            array_list.append(temp_data)
            var_name = list(temp_data.data_vars.keys())[0]
            t, h, w = array_list.shape
            years = t // self.seq_len
            input_var_shapes[var_name] = h * w
            input_var_offsets.append(self.input_var_offsets[-1] + h * w)

        temp_data = np.concatenate(array_list, axis=0)
        num_vars = len(variables)
        if paths[0][0].endswith(".grib"):
            years = len(paths[0])
            temp_data = temp_data.reshape(num_vars, years, self.seq_len, -1)

        elif paths[0][0].endswith(".grib2"):
            # Use self.seq_len = 365 (post-leap-day-removal)
            filtered_vlist = [f for f in vlist if int(f[-10:-6]) <= 365]
            vlist = filtered_vlist
            years = len(vlist) // self.seq_len
            temp_data = temp_data.reshape(num_vars, years, self.seq_len, -1)

        else:
            years = len(paths[0])
            temp_data = temp_data.reshape(num_vars, years, self.seq_len, self.lon, self.lat)

        if seq_to_seq is False:
            temp_data = temp_data[:, :, -1, :, :]  # only take last time step
            temp_data = np.expand_dims(temp_data, axis=2)

        if channels_last:
            temp_data = temp_data.transpose((1, 2, 3, 4, 0))
        elif paths[0][0][-5:] in [".grib", "grib2"]:
            temp_data = temp_data.transpose((1, 2, 0, 3))
        else:
            temp_data = temp_data.transpose((1, 2, 0, 3, 4))

        return temp_data, input_var_shapes, input_var_offsets

        # (86*num_scenarios!, 12, vars, 96, 144). Desired shape where 86*num_scenaiors can be the batch dimension. Can get items of shape (batch_size, 12, 96, 144) -> #TODO: confirm that one item should be one year of one scenario
        # or maybe without being split into lats and lons...if we are working on the icosahedral? (years, months, no. of vars, no. of unique coords)

    # NOTE:() rewriting this currently to try to use icosahedral code...

    def load_coordinates_into_mem(self, paths: List[List[str]]) -> np.ndarray:
        first_file = paths[0][0]

        if first_file.endswith(".grib") or first_file.endswith(".grib2"):
            coordinates = np.load(self.icosahedral_coordinates_path)
        elif paths[0][0][-5:] == "grib2":
            coordinates = np.loadtxt(self.icosahedral_coordinates_path, skiprows=1, usecols=(1, 2))
        else:
            temp_data = xr.open_mfdataset(paths[0], concat_dim="time", combine="nested").compute()
            # Try to load `lat` and `lon` directly and fall back to error if not found
            try:
                lat = temp_data.lat.to_numpy()
                lon = temp_data.lon.to_numpy()
            except AttributeError:
                raise ValueError("Latitude and longitude not found in NetCDF file structure.")

            coordinates = np.meshgrid(lon, lat)
            coordinates = np.c_[coordinates[1].flatten(), coordinates[0].flatten()]

        return coordinates

    @staticmethod
    def create_multi_res_data(data, num_months_aggregated):
        num_months_aggregated = np.asarray(num_months_aggregated)
        num_months_aggregated_total = num_months_aggregated.sum()
        if data.shape[1] % num_months_aggregated_total != 0:
            print("WARNING:num_months_aggregated does not divide time dimension. Dropping last few months.")
            end_idx = (data.shape[1] // num_months_aggregated_total) * num_months_aggregated_total
            data = data[:, :end_idx]

    @staticmethod
    def aggregate_months(data, num_months_aggregated):
        """Divide the data into chunks of size num_months_aggregated and use the average of each chunk."""
        # check if time dim is divisible by num_months_aggregated
        # if not print warning and drop the last few months
        if data.shape[1] % num_months_aggregated != 0:
            print("WARNING:num_months_aggregated does not divide time dimension. Dropping last few months.")
            end_idx = (data.shape[1] // num_months_aggregated) * num_months_aggregated
            data = data[:, :end_idx]

        # introduce a new dimension of size num_months_aggregated

        # print("Inside aggregate_months, and the data before reshaping is:", data.shape)
        reshaped_data = data.reshape(data.shape[0], -1, num_months_aggregated, *data.shape[2:])
        # print("Still inside aggregate months, reshaped_data shape:", reshaped_data.shape)

        # average over the new dimension
        aggregated_data = np.nanmean(reshaped_data, axis=2)
        # print("Shape of the aggregated data?:", aggregated_data.shape)
        return aggregated_data

    def split_data_by_interval(self, data, tau, ratio_train, interval_length=100):
        """Given a dataset and interval length, divide the data into intervals, then splits each interval into training
        and validation indices based on ratio."""
        assert interval_length <= data.shape[0], "interval length is longer than the data"

        idx_train, idx_valid = [], []
        t_max = data.shape[0]
        n_intervals = t_max // interval_length

        # split each interval into train and validation
        for i in range(n_intervals):
            start = i * interval_length
            n_train_interval = int(interval_length * ratio_train)
            idx_train.extend(range(start + tau, start + n_train_interval))
            idx_valid.extend(range(start + n_train_interval, start + interval_length))

        idx_train, idx_valid = np.array(idx_train), np.array(idx_valid)
        return idx_train, idx_valid

    def get_overlapping_sequences(self, data, idxs, tau, future_timesteps):
        """
        Given a dataset, time indices, and lag, generate sequences.

        Return input sequences and next step labels.
        """
        x_list, y_list = [], []
        for idx in idxs:
            x_idx = data[idx - tau : idx]  # input includes tau lagged time steps
            y_idx = data[idx : idx + future_timesteps]  # labels are the next time step
            x_list.append(x_idx)
            y_list.append(y_idx)

        return x_list, y_list

    def get_causal_data(
        self,
        tau,
        future_timesteps,
        channels_last,
        num_vars,
        num_scenarios,
        num_ensembles,
        num_years,
        mode,
        num_months_aggregated=1,
        ratio_train=None,
        interval_length=100,
        obs_to_latent_mask: Optional[np.ndarray] = None,
    ):
        ...
        # Once x_train and x_valid are constructed:
        if obs_to_latent_mask is None:
            self.obs_to_latent_mask = obs_to_latent_mask

        """
        Constructs dataset for causal discovery model.

        Splits each scenario into training and validation sets, then generates overlapping sequences.
        """

        num_years = self.length

        data = self.Data

        if channels_last:
            data = data.transpose((0, 1, 4, 2, 3))

        # TODO: breaks if not same number of years in each scenario i.e. historical vs ssp

        try:
            data = data.reshape(num_scenarios, num_years * self.seq_len, num_vars, self.lon, self.lat)

        except ValueError:
            print(
                "I saw a ValueError and now I am reshaping the data differently, probably as I have icosahedral data!"
            )

            data = data.reshape(1, num_years * self.seq_len, num_vars, -1)

        if isinstance(num_months_aggregated, (int, np.integer)) and num_months_aggregated > 1:
            data = self.aggregate_months(data, num_months_aggregated)
            # for each scenario in data, generate overlapping sequences
            if mode == "train" or mode == "train+val":
                # print("IN IF")
                x_train_list, y_train_list = [], []
                x_valid_list, y_valid_list = [], []

                for scenario in data:
                    idx_train, idx_valid = self.split_data_by_interval(scenario, tau, ratio_train, interval_length)

                    x_train, y_train = self.get_overlapping_sequences(scenario, idx_train, tau, future_timesteps)
                    x_train_list.extend(x_train)
                    y_train_list.extend(y_train)
                    x_valid, y_valid = self.get_overlapping_sequences(scenario, idx_valid, tau, future_timesteps)
                    x_valid_list.extend(x_valid)
                    y_valid_list.extend(y_valid)

                train_x, train_y = np.stack(x_train_list), np.stack(y_train_list)
                print("train_x shape:", train_x.shape)
                print("train_y shape:", train_y.shape)
                if ratio_train == 1:
                    valid_x, valid_y = np.array(x_valid_list), np.array(y_valid_list)
                else:
                    valid_x, valid_y = np.stack(x_valid_list), np.stack(y_valid_list)
                # z-score normalization
                mean_x, std_x = self.get_mean_std(train_x)
                stats_x = {"mean": mean_x, "std": std_x}

                mean_y, std_y = self.get_mean_std(train_y)
                stats_y = {"mean": mean_y, "std": std_y}

                train = train_x, train_y
                valid = valid_x, valid_y

                return train, valid
            else:
                x_test_list, y_test_list = [], []
                for scenario in data:
                    idx_test = np.arange(tau, scenario.shape[0])
                    x_test, y_test = self.get_overlapping_sequences(scenario, idx_test, tau, future_timesteps)
                    x_test_list.extend(x_test)
                    y_test_list.extend(y_test)

                test_x, test_y = np.stack(x_test_list), np.stack(y_test_list)
                test_y = np.expand_dims(test_y, axis=1)

                # z-score normalization
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode="train+val", file="statistics", kwargs=self.fname_kwargs, causal=True
                )
                stats_fname = self.output_save_dir / stats_fname
                stats = np.load(stats_fname, allow_pickle=True)
                stats_x, stats_y = stats
                test = test_x, test_y

                return test

        else:

            if mode == "train" or mode == "train+val":
                x_train_list, y_train_list = [], []
                x_valid_list, y_valid_list = [], []
                for scenario in data:
                    idx_train, idx_valid = self.split_data_by_interval(scenario, tau, ratio_train, interval_length)

                    x_train, y_train = self.get_overlapping_sequences(scenario, idx_train, tau, future_timesteps)
                    x_train_list.extend(x_train)
                    y_train_list.extend(y_train)

                    x_valid, y_valid = self.get_overlapping_sequences(scenario, idx_valid, tau, future_timesteps)
                    x_valid_list.extend(x_valid)
                    y_valid_list.extend(y_valid)

                train_x, train_y = np.stack(x_train_list), np.stack(y_train_list)
                if ratio_train == 1:
                    valid_x, valid_y = np.array(x_valid_list), np.array(y_valid_list)
                else:
                    valid_x, valid_y = np.stack(x_valid_list), np.stack(y_valid_list)

                train = train_x, train_y
                valid = valid_x, valid_y
                return train, valid
            else:
                x_test_list, y_test_list = [], []
                for scenario in data:
                    idx_test = np.arange(tau, scenario.shape[0])
                    x_test, y_test = self.get_overlapping_sequences(scenario, idx_test, tau, future_timesteps)
                    x_test_list.extend(x_test)
                    y_test_list.extend(y_test)

                test_x, test_y = np.stack(x_test_list), np.stack(y_test_list)
                test_y = np.expand_dims(test_y, axis=1)
                test = test_x, test_y
                return test

    def save_data_into_disk(self, data: np.ndarray, fname: str, output_save_dir: str) -> str:

        np.savez(output_save_dir / fname, data=data)
        return output_save_dir / fname

    def get_save_name_from_kwargs(self, mode: str, file: str, kwargs: Dict, causal: Optional[bool] = False):
        fname = ""
        coordinates_fname = ""
        # print("KWARGs:", kwargs)

        # if file == "statistics":
        #     # only cmip 6
        #     if "climate_model" in kwargs:
        #         fname += f"{kwargs['climate_model']}_"
        #         coordinates_fname += f"{kwargs['climate_model']}_"
        #     if "num_ensembles" in kwargs:
        #         fname += f"{str(kwargs['num_ensembles'])}_"
        #         coordinates_fname += f"{str(kwargs['num_ensembles'])}_"  # all
        #     fname += f"{'_'.join(kwargs['variables'])}_"
        #     coordinates_fname += f"{'_'.join(kwargs['variables'])}_"
        #     if causal:
        #         fname += "causal_"
        #         coordinates_fname += "causal_"
        # else:

        for k in kwargs:
            if isinstance(kwargs[k], List):
                fname += f"{k}_{'_'.join(kwargs[k])}_"
                coordinates_fname += f"{k}_{'_'.join(kwargs[k])}_"
            else:
                fname += f"{k}_{kwargs[k]}_"
                coordinates_fname += f"{k}_{kwargs[k]}_"
        if causal:
            fname += "causal_"
            coordinates_fname += "causal_"
        if file == "statistics":
            fname += f"{mode}_{file}.npy"
            coordinates_fname += f"{mode}_coordinates.npy"
        else:
            fname += f"{mode}_{file}.npz"
            coordinates_fname += f"{mode}_coordinates.npy"

        # print(fname)
        return fname, coordinates_fname

    def copy_to_slurm(self, fname):
        pass

    def _reload_data(self, fname):
        try:
            in_data = np.load(fname, allow_pickle=True)
        except zipfile.BadZipFile as e:
            log.warning(f"{fname} was not properly saved or has been corrupted.")
            raise e
        try:
            in_files = in_data.files
        except AttributeError:
            return in_data

        if len(in_files) == 1:
            return in_data[in_files[0]]
        else:
            return {k: in_data[k] for k in in_files}

    def get_years_list(self, years: str, give_list: Optional[bool] = False):
        """
        Get a string of type 20xx-21xx.

        Split by - and return min and max years.
        Can be used to split train and val.
        """
        if len(years) != 9:
            log.warn(
                "Years string must be in the format xxxx-yyyy eg. 2015-2100 with string length 9. Please check the year string."
            )
            raise ValueError
        splits = years.split("-")
        min_year, max_year = int(splits[0]), int(splits[1])

        if give_list:
            return np.arange(min_year, max_year + 1, step=1)
        return min_year, max_year

    def get_dataset_statistics(self, data, mode, type="z-norm", mips="cmip6"):
        if mode == "train" or mode == "train+val":
            if type == "z-norm":
                mean, std = self.get_mean_std(data)
                return mean, std
            elif type == "minmax":
                min_val, max_val = self.get_min_max(data)
                return min_val, max_val
            else:
                print(f"Normalizing of type {type} has not been implemented!")
        else:
            print("In testing mode, skipping statistics calculations.")

    # make sure we are normalising correctly...
    # loading the coordinates and statistics - make sure these are loaded sensibly!

    def get_mean_std(self, data):
        # DATA shape (258, 12, 4, 96, 144) or DATA shape (258, 12, 2, 96, 144)

        # Here we are working with ClimateSet data on a regular grid
        if data.ndim == 5:
            data = np.moveaxis(
                data, 2, 0
            )  # DATA shape (258, 12, 4, 96, 144) -> (4, 258, 12, 96, 144) easier to calulate statistics
            vars_mean = np.nanmean(data, axis=(1, 2, 3, 4))  # sDATA shape (258, 12, 4, 96, 144)
            vars_std = np.nanstd(data, axis=(1, 2, 3, 4))
            vars_mean = np.expand_dims(vars_mean, (1, 2, 3, 4))  # Shape of mean & std (4, 1, 1, 1, 1)
            vars_std = np.expand_dims(vars_std, (1, 2, 3, 4))

        # Here we work with the icosahedral data, so we do not have separate lat and lon dimensions
        elif data.ndim == 4:
            data = np.moveaxis(data, 2, 0)
            vars_mean = np.nanmean(data, axis=(1, 2, 3))
            vars_std = np.nanstd(data, axis=(1, 2, 3))
            vars_mean = np.expand_dims(vars_mean, (1, 2, 3))
            vars_std = np.expand_dims(vars_std, (1, 2, 3))
        else:
            print("Data dimension not recognized. Please check the dimensions of the data.")
            raise ValueError

        return vars_mean, vars_std

    def get_min_max(self, data):

        if data.ndim == 5:
            data = np.moveaxis(
                data, 2, 0
            )  # DATA shape (258, 12, 4, 96, 144) -> (4, 258, 12, 96, 144) easier to calulate statistics
            vars_max = np.nanmax(data, axis=(1, 2, 3, 4))  # sDATA shape (258, 12, 4, 96, 144)
            vars_min = np.nanmin(data, axis=(1, 2, 3, 4))
            vars_max = np.expand_dims(vars_max, (1, 2, 3, 4))  # Shape of mean & std (4, 1, 1, 1, 1)
            vars_min = np.expand_dims(vars_min, (1, 2, 3, 4))
        elif data.ndim == 4:
            data = np.moveaxis(data, 2, 0)
            vars_max = np.nanmax(data, axis=(1, 2, 3))
            vars_min = np.nanmin(data, axis=(1, 2, 3))
            vars_max = np.expand_dims(vars_max, (1, 2, 3))
            vars_min = np.expand_dims(vars_min, (1, 2, 3))
        else:
            print("Data dimension not recognized. Please check the dimensions of the data.")
            raise ValueError

        return vars_min, vars_max

    def normalize_data(self, data, stats, type="z-norm"):

        # Only implementing z-norm for now
        # z-norm: (data-mean)/(std + eps); eps=1e-9
        # min-max = (v - v.min()) / (v.max() - v.min())

        # print("Normalizing data...")
        data = np.moveaxis(data, 2, 0)  # DATA shape (258, 12, 4, 96, 144) -> (4, 258, 12, 96, 144)
        norm_data = (data - stats["mean"]) / (stats["std"])
        print("I completed the normalisation of the data.")

        norm_data = np.moveaxis(norm_data, 0, 2)  # Switch back to (258, 12, 4, 96, 144)

        # Replace NaNs with 0s
        norm_data = np.nan_to_num(norm_data)

        # print("Really, I completed the normalisation of the data, just about to return.")
        return norm_data

    def remove_seasonality(self, data):
        """
        Function to remove seasonality from the data There are various different options to do this These are just
        different methods of removing seasonality.

        e.g.
        monthly - remove seasonality on a per month basis
        rolling monthly - remove seasonality on a per month basis but using a rolling window,
        removing only the average from the months that have preceded this month
        linear - remove seasonality using a linear model to predict seasonality

        or trend removal
        emissions - remove the trend using the emissions data, such as cumulative CO2
        """

        # print("Removing seasonality from the data.")
        median = np.nanmedian(data, axis=0)
        abs_dev = np.abs(data - median[None])
        mad = np.nanmedian(abs_dev, axis=0)

        # Consistency factor so that MAD ~= std for Gaussian data
        mad_scaled = mad * 1.4826

        # make a numpy array containing the mean and std for each month:
        remove_season_stats = np.array([median, mad_scaled], dtype=data.dtype)
        np.save(self.output_save_dir / "remove_season_stats", remove_season_stats, allow_pickle=True)

        print("Just about to return the data after removing seasonality (robust median/MAD).")
        mad_safe = np.where(mad_scaled == 0, 1, mad_scaled)
        deseasonalized = (data - median[None]) / mad_safe[None]
        return deseasonalized

    def write_dataset_statistics(self, fname, stats):
        #            fname = fname.replace('.npz.npy', '.npy')
        np.save(self.output_save_dir / fname, stats, allow_pickle=True)

    def load_dataset_statistics(self, fname, mode, mips):
        if "train_" in fname:
            fname = fname.replace("train", "train+val")
        elif "test" in fname:
            fname = fname.replace("test", "train+val")

        stats_data = np.load(self.output_save_dir / fname, allow_pickle=True).item()

        return stats_data

    def load_dataset_coordinates(self, fname, mode, mips):
        if "train_" in fname:
            fname = fname.replace("train", "train+val")
        elif "test" in fname:
            fname = fname.replace("test", "train+val")

        coordinates_data = np.load(self.output_save_dir / fname, allow_pickle=True)

        return coordinates_data

    def __str__(self):
        return f"ClimateDataset: {self.n_years} years used."

    def __len__(self):
        raise NotImplementedError("Subclasses should implement __len__ method.")

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses should implement __getitem__ method.")

    def get_mask_metadata(self) -> Tuple[List[Tuple[int, ...]], List[str]]:
        return list(self.input_var_shapes.values()), list(self.input_var_shapes.keys())
