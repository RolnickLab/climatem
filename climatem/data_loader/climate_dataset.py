# NOTE: as of 14th Oct, I am also trying to get this to work for multiple variables.

import glob
import os
import zipfile
from typing import Dict, List, Optional, Tuple, Union

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

# JK: consider using the __init__.py for path handling
# feel free to move this to the constants instead for consistency!
from climatem import MAPPINGS_DIR

from climatem.constants import (  # INPUT4MIPS_NOM_RES,; INPUT4MIPS_TEMP_RES,
    AVAILABLE_MODELS_FIRETYPE,
    CMIP6_NOM_RES,
    CMIP6_TEMP_RES,
    NO_OPENBURNING_VARS,
    OPENBURNING_MODEL_MAPPING,
)
# from climatem.plotting.plot_data import plot_species, plot_species_anomaly
from climatem.utils import get_logger

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
        out_variables: Union[str, List[str]] = "pr",
        in_variables: Union[str, List[str]] = ["BC_sum", "SO2_sum", "CH4_sum", "CO2_sum"],
        seq_to_seq: bool = True,  # TODO: implement if false
        channels_last: bool = False,
        load_data_into_mem: bool = True,  # Keeping this true be default for now
        seq_len: int = 12,
        lat: int = 96,
        lon: int = 144,
        # input_transform=None,  # TODO: implement
        # input_normalization="z-norm",  # TODO: implement
        # output_transform=None,
        # output_normalization="z-norm",
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
            out_variables (Union[str, List[str]], optional): output variable of precipitation "pr", temperature "tas". Defaults to "pr".
            in_variables (Union[str, List[str]], optional): TBC. Defaults to ["BC_sum","SO2_sum", "CH4_sum", "CO2_sum"].
            seq_to_seq (bool, optional): _description_. Defaults to True.
            load_data_into_mem (bool, optional): _description_. Defaults to True.
            output_normalization (str, optional): _description_. Defaults to "z-norm".
            seasonality_removal (bool, optional): remove season through monthly normalisation. Defaults to True.
        """

        super().__init__()
        self.test_dir = output_save_dir
        self.output_save_dir = output_save_dir
        self.reload_climate_set_data = reload_climate_set_data
        #Here need to propagate argument data_params.reload_climate_set_data

        self.channels_last = channels_last
        self.load_data_into_mem = load_data_into_mem

        if isinstance(in_variables, str):
            in_variables = [in_variables]
        if isinstance(out_variables, str):
            out_variables = [out_variables]
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

        self.seasonality_removal = seasonality_removal

        if climate_model in AVAILABLE_MODELS_FIRETYPE:
            openburning_specs = OPENBURNING_MODEL_MAPPING[climate_model]
        else:
            openburning_specs = OPENBURNING_MODEL_MAPPING["other"]

        ds_kwargs = dict(
            scenarios=scenarios,
            years=self.years,
            historical_years=self.historical_years,
            channels_last=channels_last,
            openburning_specs=openburning_specs,
            mode=mode,
            output_save_dir=output_save_dir,
            reload_climate_set_data=self.reload_climate_set_data,
            seq_to_seq=seq_to_seq,
            seasonality_removal=self.seasonality_removal,
        )
        self.seq_len = seq_len
        self.lat = lat
        self.lon = lon
        # creates on cmip and on input4mip dataset
        #print("creating input4mips")
        self.input4mips_ds = Input4MipsDataset(variables=in_variables, **ds_kwargs)
        #print("creating cmip6")
        # self.cmip6_ds=self.input4mips_ds
        self.cmip6_ds = CMIP6Dataset(
            climate_model=climate_model, num_ensembles=num_ensembles, variables=out_variables, **ds_kwargs
        )

    # NOTE:() changing this so it can deal with with grib files and netcdf files
    # this operates variable wise now.... #TODO: sizes for input4mips / adapt to mulitple vars
    def load_into_mem(
        self, paths: List[List[str]], num_vars: int, channels_last=True, seq_to_seq=True
    ):  # -> np.ndarray():
        """
        Take a file structure of netcdf or grib files and load them into memory.

        Args:
            paths (List[List[str]]): absolute to filepath
            num_vars (int): number of input variables e.g. pr, tas, etc.
            channels_last (bool, optional): reshape data to channels. Defaults to True.
            seq_to_seq (bool, optional): TBC. Defaults to True. #TODO
        """

        array_list = []
        #print("paths:", paths)
        #print("length paths", len(paths))

        # I need to check here that it is doing the right thing
        for vlist in paths:
            #print("length_paths_list", len(vlist))
            # print the last three characters of the first element of vlist
            # NOTE:() assert that they are either .nc or .grib - and print an error!
            if vlist[0][-3:] == ".nc":
                temp_data = xr.open_mfdataset(
                    vlist, concat_dim="time", combine="nested"
                ).compute()  # .compute is not necessary but eh, doesn't hurt
                # ignore the bnds dimension
                temp_data = temp_data.drop_dims("bnds")
                #print("Temp data at the point of reading it in:", temp_data)
            elif vlist[0][-5:] == ".grib":
                # need to install cfgrib, eccodes and likely ecmwflibs to make sure this cfgrib engine works and is available
                temp_data = xr.open_mfdataset(vlist, engine="cfgrib", concat_dim="time", combine="nested").compute()
                #print("Temp data at the point of reading it in:", temp_data)
            # then get rid of this with some assert ^ see above
            else:
                print("File extension not recognized, please use either .nc or .grib")

            temp_data = temp_data.to_array().to_numpy()  # Should be of shape (vars, 1036*num_scenarios, 96, 144)

            #print("Temp data shape:", temp_data.shape)
            # temp_data = temp_data.squeeze() # (1036*num_scanarios, 96, 144)
            array_list.append(temp_data)

        #print("length of the array list:", len(array_list))
        temp_data = np.concatenate(array_list, axis=0)

        #print("Temp data shape after concatenation:", temp_data.shape)

        # this is not very neat, but it calc
        if paths[0][0][-5:] == ".grib":
            years = len(paths[0])
            temp_data = temp_data.reshape(num_vars, years, self.seq_len, -1)
            #print("temp data shape", temp_data.shape)

        else:
            years = len(paths[0])
            temp_data = temp_data.reshape(num_vars, years, self.seq_len, self.lon, self.lat)
            #print("temp data shape", temp_data.shape)

        # create a new array with the first 3 columns, and then tuple(lon, lat)

        if seq_to_seq is False:
            temp_data = temp_data[:, :, -1, :, :]  # only take last time step
            temp_data = np.expand_dims(temp_data, axis=2)
            #print("seq to 1 temp data shape", temp_data.shape)
        if channels_last:
            temp_data = temp_data.transpose((1, 2, 3, 4, 0))
        elif paths[0][0][-5:] == ".grib":
            #print("In elif paths[0][0][-5:] == '.grib'")
            temp_data = temp_data.transpose((1, 2, 0, 3))
        else:
            temp_data = temp_data.transpose((1, 2, 0, 3, 4))
        #print("final temp data shape", temp_data.shape)
        return temp_data

        # (86*num_scenarios!, 12, vars, 96, 144). Desired shape where 86*num_scenaiors can be the batch dimension. Can get items of shape (batch_size, 12, 96, 144) -> #TODO: confirm that one item should be one year of one scenario
        # or maybe without being split into lats and lons...if we are working on the icosahedral? (years, months, no. of vars, no. of unique coords)

    # NOTE:() rewriting this currently to try to use icosahedral code...
    def load_coordinates_into_mem(self, paths: List[List[str]]) -> np.ndarray:
        """
        Load the coordinates into memory.

        Args:
            paths (List[List[str]]): absolute to filepaths to the data

        Returns:
            np.ndarray: coordinates
        """
        print("length paths", len(paths))
        if paths[0][0][-5:] == ".grib":
            # we have no lat and lon in grib files, so we need to fill it up from elsewhere, from the mapping.txt file:
            coordinates = np.loadtxt(
                MAPPINGS_DIR / "vertex_lonlat_mapping.txt"
            )
            coordinates = coordinates[:, 1:]

        else:
            for vlist in [paths[0]]:
                #print("I am in the else of load_coordinates_into_mem")
                #print("length_paths_list", len(vlist))
                temp_data = xr.open_mfdataset(
                    vlist, concat_dim="time", combine="nested"
                ).compute()  # .compute is not necessary but eh, doesn't hurt
                #print("self.in_variables:", self.in_variables)
                # NOTE:() - should this be for all possible variables? Not sure...
                if (
                    "tas" in self.in_variables
                    or "pr" in self.in_variables
                    or "psl" in self.in_variables
                    or "ts" in self.in_variables
                ):
                    array_list_lon = temp_data.lon.to_numpy()
                    # print('array_list_lon shape:', array_list_lon.shape)
                    array_list_lon = array_list_lon[:]
                    array_list_lat = temp_data.lat.to_numpy()
                    array_list_lat = array_list_lat[:]
                else:
                    array_list_lon = temp_data.lon.to_numpy()
                    array_list_lat = temp_data.lat.to_numpy()
            coordinates = np.meshgrid(array_list_lon, array_list_lat)
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

        #print("Inside aggregate_months, and the data before reshaping is:", data.shape)
        reshaped_data = data.reshape(data.shape[0], -1, num_months_aggregated, *data.shape[2:])
        #print("Still inside aggregate months, reshaped_data shape:", reshaped_data.shape)

        # average over the new dimension
        aggregated_data = np.nanmean(reshaped_data, axis=2)
        #print("Shape of the aggregated data?:", aggregated_data.shape)
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

    def get_overlapping_sequences(self, data, idxs, tau):
        """
        Given a dataset, time indices, and lag, generate sequences.

        Return input sequences and next step labels.
        """
        x_list, y_list = [], []
        for idx in idxs:
            x_idx = data[idx - tau : idx]  # input includes tau lagged time steps
            y_idx = data[idx]  # labels are the next time step
            x_list.append(x_idx)
            y_list.append(y_idx)

        return x_list, y_list

    def get_causal_data(
        self,
        tau,
        channels_last,
        num_vars,
        num_scenarios,
        num_ensembles,
        num_years,
        mode,
        num_months_aggregated=1,
        ratio_train=None,
        interval_length=100,
    ):
        """
        Constructs dataset for causal discovery model.

        Splits each scenario into training and validation sets, then generates overlapping sequences.
        """
        #print(f"Getting causal data [mode={mode}] ...")

        # NOTE:() hack to overwrite the number of years
        num_years = self.length
        #print("In get_causal_data, num_years:", num_years)

        data = self.Data

        #print("Here in get_causal_data, self.length:", self.length)

        if channels_last:
            # (n, t, lon, lat, n_vars) -> (n, t, n_vars, lon, lat)
            data = data.transpose((0, 1, 4, 2, 3))

        # n = num_scenarios, t = n_years * 12
        # TODO: breaks if not same number of years in each scenario i.e. historical vs ssp

        try:
            # (years, months, vars, lon, lat) -> (scenrios, years*months, vars, lon, lat)
            # Regular data shape before reshaping: (101, 12, 1, 96, 144)
            # Regular data shape after reshaping: (1, 1212, 1, 96, 144)
            #print("Trying to regrid to lon, lat if we have regular data...")
            # data = data.reshape(num_scenarios, num_years, num_vars, LON, LAT)

            data = data.reshape(num_scenarios, num_years * 12, num_vars, self.lon, self.lat)

        except ValueError:
            print(
                "I saw a ValueError and now I am reshaping the data differently, probably as I have icosahedral data!"
            )



            #print("Data shape before reshaping:", data.shape)
            #print("JUST CHECKING I AM HERE")
            # note that this was returning the wrong shape if we have more than one ensemble member, of course, as it gets stuffed into -1
            # data = data.reshape(num_scenarios, num_years*12, num_vars, -1)
            # 26/08/24
            # Now we don't split up the ensemble members

            data = data.reshape(1, num_years * 12, num_vars, -1)
            #print("Data shape after reshaping:", data.shape)

        if isinstance(num_months_aggregated, (int, np.integer)) and num_months_aggregated > 1:
            data = self.aggregate_months(data, num_months_aggregated)
            # for each scenario in data, generate overlapping sequences
            if mode == "train" or mode == "train+val":
                #print("IN IF")
                x_train_list, y_train_list = [], []
                x_valid_list, y_valid_list = [], []

                for scenario in data:
                    idx_train, idx_valid = self.split_data_by_interval(scenario, tau, ratio_train, interval_length)
                    # np.random.shuffle(idx_train)
                    # np.random.shuffle(idx_valid)

                    x_train, y_train = self.get_overlapping_sequences(scenario, idx_train, tau)
                    x_train_list.extend(x_train)
                    y_train_list.extend(y_train)

                    x_valid, y_valid = self.get_overlapping_sequences(scenario, idx_valid, tau)
                    x_valid_list.extend(x_valid)
                    y_valid_list.extend(y_valid)

                train_x, train_y = np.stack(x_train_list), np.stack(y_train_list)
                if ratio_train == 1:
                    valid_x, valid_y = np.array(x_valid_list), np.array(y_valid_list)
                else:
                    valid_x, valid_y = np.stack(x_valid_list), np.stack(y_valid_list)
                train_y = np.expand_dims(train_y, axis=1)
                valid_y = np.expand_dims(valid_y, axis=1)

                # z-score normalization
                # make train_y go from (2550, 4, 96, 144) to (2550, 1, 4, 96, 144)
                mean_x, std_x = self.get_mean_std(train_x)
                stats_x = {"mean": mean_x, "std": std_x}

                mean_y, std_y = self.get_mean_std(train_y)
                stats_y = {"mean": mean_y, "std": std_y}

                # Was normalizing twice
                # stats_fname, coordinates_fname = self.get_save_name_from_kwargs(mode=mode, file='statistics', kwargs=self.fname_kwargs, causal=True)
                # stats = stats_x, stats_y
                # out_fname = self.write_dataset_statistics(stats_fname, stats)
                # print(f"Saved statistics to {out_fname}")

                # train = self.normalize_data(train_x, stats_x), self.normalize_data(train_y, stats_y)
                # if ratio_train<1:
                #     valid = self.normalize_data(valid_x, stats_x), self.normalize_data(valid_y, stats_y)
                # else:
                #     valid = None
                train = train_x, train_y
                valid = valid_x, valid_y

                # print(train_y.shape)
                # plot_species(train_y[:, :, 0, :, :], self.coordinates, "tas", "../../TEST_REPO", "after_causal")
                return train, valid
            else:
                x_test_list, y_test_list = [], []
                for scenario in data:
                    idx_test = np.arange(tau, scenario.shape[0])
                    x_test, y_test = self.get_overlapping_sequences(scenario, idx_test, tau)
                    x_test_list.extend(x_test)
                    y_test_list.extend(y_test)

                test_x, test_y = np.stack(x_test_list), np.stack(y_test_list)
                test_y = np.expand_dims(test_y, axis=1)

                # z-score normalization
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode="train+val", file="statistics", kwargs=self.fname_kwargs, causal=True
                )
                stats_fname = os.path.join(self.output_save_dir, stats_fname)
                stats = np.load(stats_fname, allow_pickle=True)
                stats_x, stats_y = stats
                test = test_x, test_y
                # test = self.normalize_data(test_x, stats_x), self.normalize_data(test_y, stats_y)

                return test


        else:
            # TODO create this function and use it -> put it inside the data creation...
            # data = self.create_multi_res_data(data, num_months_aggregated)

            # for each scenario in data, generate overlapping sequences
            if mode == "train" or mode == "train+val":
                x_train_list, y_train_list = [], []
                x_valid_list, y_valid_list = [], []
                for scenario in data:
                    idx_train, idx_valid = self.split_data_by_interval(scenario, tau, ratio_train, interval_length)
                    # np.random.shuffle(idx_train)
                    # np.random.shuffle(idx_valid)

                    x_train, y_train = self.get_overlapping_sequences(scenario, idx_train, tau)
                    x_train_list.extend(x_train)
                    y_train_list.extend(y_train)

                    x_valid, y_valid = self.get_overlapping_sequences(scenario, idx_valid, tau)
                    x_valid_list.extend(x_valid)
                    y_valid_list.extend(y_valid)

                train_x, train_y = np.stack(x_train_list), np.stack(y_train_list)
                if ratio_train == 1:
                    valid_x, valid_y = np.array(x_valid_list), np.array(y_valid_list)
                else:
                    valid_x, valid_y = np.stack(x_valid_list), np.stack(y_valid_list)
                train_y = np.expand_dims(train_y, axis=1)
                valid_y = np.expand_dims(valid_y, axis=1)

                # # z-score normalization ALREADY DONE
                # # make train_y go from (2550, 4, 96, 144) to (2550, 1, 4, 96, 144)
                # mean_x, std_x = self.get_mean_std(train_x)
                # stats_x = {'mean': mean_x, 'std': std_x}
                #
                # mean_y, std_y = self.get_mean_std(train_y)
                # stats_y = {'mean': mean_y, 'std': std_y}
                #
                # stats_fname, coordinates_fname = self.get_save_name_from_kwargs(mode=mode, file='statistics',
                #                                                                 kwargs=self.fname_kwargs,
                #                                                                 causal=True)
                # stats = stats_x, stats_y
                # out_fname = self.write_dataset_statistics(stats_fname, stats)
                # print(f"Saved statistics to {out_fname}")
                #
                # train = self.normalize_data(train_x, stats_x), self.normalize_data(train_y, stats_y)
                # if ratio_train < 1:
                #     valid = self.normalize_data(valid_x, stats_x), self.normalize_data(valid_y, stats_y)
                # else:
                #     valid = None

                train = train_x, train_y
                valid = valid_x, valid_y
                # print(train_y.shape)
                # plot_species(train_y[:, 0, 0, :, :], self.coordinates, "tas", "../../TEST_REPO", "after_causal")
                return train, valid
            else:
                x_test_list, y_test_list = [], []
                for scenario in data:
                    idx_test = np.arange(tau, scenario.shape[0])
                    x_test, y_test = self.get_overlapping_sequences(scenario, idx_test, tau)
                    x_test_list.extend(x_test)
                    y_test_list.extend(y_test)

                test_x, test_y = np.stack(x_test_list), np.stack(y_test_list)
                test_y = np.expand_dims(test_y, axis=1)

                # z-score normalization
                # stats_fname, coordinates_fname = self.get_save_name_from_kwargs(mode="train+val", file='statistics',
                #                                                                 kwargs=self.fname_kwargs,
                #                                                                 causal=True)
                # stats_fname = os.path.join(self.output_save_dir, stats_fname)
                # stats = np.load(stats_fname, allow_pickle=True)
                # stats_x, stats_y = stats
                # test = self.normalize_data(test_x, stats_x), self.normalize_data(test_y, stats_y)
                test = test_x, test_y
                return test

    def save_data_into_disk(self, data: np.ndarray, fname: str, output_save_dir: str) -> str:

        np.savez(os.path.join(output_save_dir, fname), data=data)
        return os.path.join(output_save_dir, fname)

    def get_save_name_from_kwargs(self, mode: str, file: str, kwargs: Dict, causal: Optional[bool] = False):
        fname = ""
        coordinates_fname = ""
        #print("KWARGs:", kwargs)

        if file == "statistics":
            # only cmip 6
            if "climate_model" in kwargs:
                fname += f"{kwargs['climate_model']}_"
                coordinates_fname += f"{kwargs['climate_model']}_"
            if "num_ensembles" in kwargs:
                fname += f"{str(kwargs['num_ensembles'])}_"
                coordinates_fname += f"{str(kwargs['num_ensembles'])}_"  # all
            fname += f"{'_'.join(kwargs['variables'])}_"
            coordinates_fname += f"{'_'.join(kwargs['variables'])}_"
            if causal:
                fname += "causal_"
                coordinates_fname += "causal_"
        else:

            for k in kwargs:
                if isinstance(kwargs[k], List):
                    fname += f"{k}_{'_'.join(kwargs[k])}_"
                    coordinates_fname += f"{k}_{'_'.join(kwargs[k])}_"
                else:
                    fname += f"{k}_{kwargs[k]}_"
                    coordinates_fname += f"{k}_{kwargs[k]}_"

        if file == "statistics":
            fname += f"{mode}_{file}.npy"
            coordinates_fname += f"{mode}_coordinates.npy"
        else:
            fname += f"{mode}_{file}.npz"
            coordinates_fname += f"{mode}_coordinates.npy"

        #print(fname)
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

        #print("Normalizing data...")
        data = np.moveaxis(data, 2, 0)  # DATA shape (258, 12, 4, 96, 144) -> (4, 258, 12, 96, 144)
        norm_data = (data - stats["mean"]) / (stats["std"])
        print("I completed the normalisation of the data.")

        norm_data = np.moveaxis(norm_data, 0, 2)  # Switch back to (258, 12, 4, 96, 144)

        # Replace NaNs with 0s
        norm_data = np.nan_to_num(norm_data)

        #print("Really, I completed the normalisation of the data, just about to return.")
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

        #print("Removing seasonality from the data.")

        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        # make a numpy array containing the mean and std for each month:
        remove_season_stats = np.array([mean, std])

        np.save(os.path.join(self.output_save_dir, "remove_season_stats"), remove_season_stats, allow_pickle=True)

        print("Just about to return the data after removing seasonality.")

        return (data - mean[None]) / std[None]

    def write_dataset_statistics(self, fname, stats):
        #            fname = fname.replace('.npz.npy', '.npy')
        np.save(os.path.join(self.output_save_dir, fname), stats, allow_pickle=True)
        return os.path.join(self.output_save_dir, fname)

    def load_dataset_statistics(self, fname, mode, mips):
        if "train_" in fname:
            fname = fname.replace("train", "train+val")
        elif "test" in fname:
            fname = fname.replace("test", "train+val")

        stats_data = np.load(os.path.join(self.output_save_dir, fname), allow_pickle=True).item()

        return stats_data

    def load_dataset_coordinates(self, fname, mode, mips):
        if "train_" in fname:
            fname = fname.replace("train", "train+val")
        elif "test" in fname:
            fname = fname.replace("test", "train+val")

        coordinates_data = np.load(os.path.join(self.output_save_dir, fname), allow_pickle=True)

        return coordinates_data

    def __getitem__(self, index):  # Dict[str, Tensor]):

        # access data in input4mips and cmip6 datasets
        raw_Xs = self.input4mips_ds[index]
        raw_Ys = self.cmip6_ds[index]
        # raw_Ys = self.cmip6_ds[index]
        if not self.load_data_into_mem:
            X = raw_Xs
            Y = raw_Ys
        else:
            # if self.in
            # TO-DO: Need to write Normalizer transform and To-Tensor transform
            # Doing norm and to-tensor per-instance here.
            # X_norm = self.input_transforms(self.X[index])
            # Y_norm = self.output_transforms(self.Y[index])
            X = raw_Xs
            Y = raw_Ys

        return X, Y

    def __str__(self):
        s = f" {self.name} dataset: {self.n_years} years used, with a total size of {len(self)} examples."
        return s

    def __len__(self):
        #print("Input4mips", self.input4mips_ds.length, "CMIP6 data", self.cmip6_ds.length)
        assert self.input4mips_ds.length == self.cmip6_ds.length, "Datasets not of same length"
        return self.input4mips_ds.length


class CMIP6Dataset(ClimateDataset):
    """
    Use first ensemble member for now Option to use multile ensemble member later Give option for which variable to use
    Load 3 scenarios for train/val: Take this as a list Process and save this as .npz in $SLURM_TMPDIR Load these in
    train/val/test Dataloader functions.

    Keep one scenario for testing # Target shape (85 * 12, 1, 144, 96) # ! * num_scenarios!!
    """

    def __init__(  # inherits all the stuff from Base
        self,
        years: Union[int, str],
        historical_years: Union[int, str],
        data_dir: Optional[str] = "Climateset_DATA",
        climate_model: str = "NorESM2-LM",
        num_ensembles: int = 1,  # 1 for first ensemble, -1 for all
        scenarios: List[str] = ["ssp126", "ssp370", "ssp585"],
        variables: List[str] = ["pr"],
        mode: str = "train",
        output_save_dir: str = "",
        reload_climate_set_data: bool = True,
        channels_last: bool = True,
        seq_to_seq: bool = True,
        seasonality_removal: bool = True,
        seq_len: int = 12,
        lat: int = 96,
        lon: int = 144,
        *args,
        **kwargs,
    ):

        self.mode = mode
        self.output_save_dir = output_save_dir
        self.reload_climate_set_data = reload_climate_set_data
        self.root_dir = os.path.join(data_dir, "outputs/CMIP6")
        self.input_nc_files = []
        self.output_nc_files = []
        self.in_variables = variables
        self.seasonality_removal = seasonality_removal
        self.seq_len = seq_len
        self.lon = lon
        self.lat = lat

        fname_kwargs = dict(
            climate_model=climate_model,
            num_ensembles=num_ensembles,
            years=f"{years[0]}-{years[-1]}",
            historical_years=f"{historical_years[0]}-{historical_years[-1]}",
            variables=variables,
            scenarios=scenarios,
            channels_last=channels_last,
            seq_to_seq=seq_to_seq,
        )
        self.fname_kwargs = fname_kwargs

        # TO-DO: This is just getting the list of .nc files for targets. Put this logic in a function and get input list as well.
        # In a function, we can call CausalDataset() instance for train and test separately to load the data

        #print("IN CMIP6!!!")

        if isinstance(climate_model, str):
            self.root_dir = os.path.join(self.root_dir, climate_model)
        else:
            # Logic for multiple climate models, not sure how to load/create dataset yet
            log.warn("Data loader not yet implemented for multiple climate models.")
            raise NotImplementedError


        # I am actually going to make this a list to be compatible with the rest of the code
        if num_ensembles == 1:
            ensemble = os.listdir(self.root_dir)
            # if there is only one element in the ensemble list, we can just take the first element
            if len(ensemble) == 1:
                #print("ensemble:", ensemble)
                #print("This often makes a mistake because it does not know if it wants to be a list or not")
                self.ensemble_dirs = [os.path.join(self.root_dir, ensemble[0])]
            else:  # we are just going to select the first ensemble member here
                self.ensemble_dirs = [
                    os.path.join(self.root_dir, ensemble[0])
                ]  # THIS USED TO BE THE CASE: Taking specific ensemble member (#TODO: only this ensemble member has historical data...)
        else:
            log.warn(
                "Data loader not properly yet implemented for multiple ensemble members, but we are trying something here."
            )
            # here I want to make the dataloader work for all ensemble members:
            # I need to loop through all the ensemble members and load the data
            ensembles = os.listdir(self.root_dir)
            #print("Ensemble members present for this model:", ensembles)
            # Now make a list, which consists of the paths to the ensemble members
            self.ensemble_dirs = [os.path.join(self.root_dir, ensemble) for ensemble in ensembles]

            #print("Ensemble directories:", self.ensemble_dirs)
            #print("What is the type of self.ensemble_dirs:", type(self.ensemble_dirs))

        fname, coordinates_fname = self.get_save_name_from_kwargs(mode=mode, file="target", kwargs=fname_kwargs)

        # here we reload files if they exist
        if os.path.isfile(os.path.join(output_save_dir, fname)) and self.reload_climate_set_data:  # we first need to get the name here to test that...

            self.data_path = os.path.join(output_save_dir, fname)
            #print("path exists, reloading")
            self.raw_data = self._reload_data(self.data_path)

            # Load stats and normalize
            stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                mode=mode, file="statistics", kwargs=fname_kwargs
            )
            stats = self.load_dataset_statistics(
                os.path.join(self.output_save_dir, stats_fname), mode=self.mode, mips="cmip6"
            )
            self.coordinates = self.load_dataset_coordinates(
                os.path.join(self.output_save_dir, coordinates_fname), mode=self.mode, mips="cmip6"
            )
            self.Data = self.normalize_data(self.raw_data, stats)
            if self.seasonality_removal:
                self.Data = self.remove_seasonality(self.Data)

            #print("In CMIP6Dataset, just finished removing the seasonality.")

        else:
            # Add code here for adding files for input nc data
            # Similar to the loop below for output files

            # Got all the files paths at this point, now open and merge

            # List of output files
            files_per_var = []
            for var in variables:

                for exp in scenarios:
                    if exp == "historical":
                        get_years = historical_years
                    else:
                        get_years = years
                    # print("ensemble_dirs")
                    # print(self.ensemble_dirs)



                    all_ensemble_output_nc_files = []

                    #print("What is the type of self.ensemble_dirs:", type(self.ensemble_dirs))

                    # assert that self.ensemble_dirs is a list
                    if isinstance(self.ensemble_dirs, list):
                        print("self.ensemble_dirs is a list")
                    else:
                        #print("self.ensemble_dirs is not a list")
                        #print("self.ensemble_dirs is:", self.ensemble_dirs)
                        raise ValueError("self.ensemble_dirs is not a list")

                    for ensemble_dir in self.ensemble_dirs:
                        print("*****************LOOPING THROUGH ENSEMBLE MEMBERS*****************")
                        print("ensemble member path:", ensemble_dir)

                        # I am now identing this:
                        output_nc_files = []

                        for y in get_years:
                            # for y in self.get_years_list(get_years, give_list=True):
                            # print('y is this:', y)
                            # print('here is exp:', exp)
                            var_dir = os.path.join(ensemble_dir, exp, var, f"{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}")
                            files = glob.glob(f"{var_dir}/*.nc", recursive=True)
                            if len(files) == 0:
                                # print(f"No netcdf files found in {var_dir}, trying to find .grib files")
                                files = glob.glob(f"{var_dir}/*.grib", recursive=True)
                            # print('files here:', files)
                            # loads all years! implement splitting
                            output_nc_files += files

                        #print("Here the final var_dir be:", var_dir)
                        # print('files here after looping through all the years:', output_nc_files)
                        # print(
                        #     "length of output_nc_files. after looping through years for 1 of the ensemble members:",
                        #     len(output_nc_files),
                        # )

                        all_ensemble_output_nc_files += output_nc_files

                    # print("Here the final var_dir be:", var_dir)
                    # print(
                    #     "length of all_ensemble_output_nc_files after looping through all ensemble members:",
                    #     len(all_ensemble_output_nc_files),
                    # )
                    # print('files here after looping through all the ensembles and the years:', all_ensemble_output_nc_files)
                files_per_var.append(all_ensemble_output_nc_files)
            #print("length of files_per_var after looping!:", len(files_per_var))
            # print('files_per_var:', files_per_var)

            # self.raw_data_input = self.load_data_into_mem(self.input_nc_files) #currently don't have input paths etc
            self.raw_data = self.load_into_mem(
                files_per_var, num_vars=len(variables), channels_last=channels_last, seq_to_seq=seq_to_seq
            )
            self.coordinates = self.load_coordinates_into_mem(files_per_var)

            if self.mode == "train" or self.mode == "train+val":
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode=mode, file="statistics", kwargs=fname_kwargs
                )
                #print(stats_fname)
                #print(coordinates_fname)

                if os.path.isfile(stats_fname):
                    print("Stats file already exists! Loading from memory.")
                    # stats = self.load_statistics_data(stats_fname)
                    stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip6")

                    self.norm_data = self.normalize_data(self.raw_data, stats)
                    if self.seasonality_removal:
                        self.norm_data = self.remove_seasonality(self.norm_data)

                else:
                    stat1, stat2 = self.get_dataset_statistics(self.raw_data, self.mode, mips="cmip6")
                    stats = {"mean": stat1, "std": stat2}
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                    if self.seasonality_removal:
                        self.norm_data = self.remove_seasonality(self.norm_data)
                    # plot_species(self.norm_data[:, :, 0, :, :], self.coordinates, variables, "../../TEST_REPO", "before_causal")
                    # print("SPECIES PLOTTED")
                    # #
                    # stats_fname = self.get_save_name_from_kwargs(mode=mode, file='statistics', kwargs=fname_kwargs)
                    save_file_name = self.write_dataset_statistics(stats_fname, stats)
                    print("WROTE STATISTICS", save_file_name)
                    save_file_name = self.write_dataset_statistics(coordinates_fname, self.coordinates)
                    print("WROTE COORDINATES", save_file_name)

                # self.norm_data = self.normalize_data(self.raw_data, stats)

            elif self.mode == "test":
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode="train+val", file="statistics", kwargs=fname_kwargs
                )
                save_file_name = os.path.join(self.output_save_dir, fname)
                stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="cmip6")
                self.norm_data = self.normalize_data(self.raw_data, stats)
                if self.seasonality_removal:
                    self.norm_data = self.remove_seasonality(self.norm_data)

            # self.input_path = self.save_data_into_disk(self.raw_data_input, self.mode, 'input')
            #print("In cmip6, just about to save the data.")
            self.data_path = self.save_data_into_disk(self.raw_data, fname, output_save_dir)
            #print("In cmip6, just saved the data.")

            #print("In cmip6, just about to copy the data to slurm.")
            # self.copy_to_slurm(self.input_path)
            self.copy_to_slurm(self.data_path)
            #print("In cmip6, just copied the data to slurm.")

            self.Data = self.norm_data

        # plot_species(self.Data[:, :, 0, :, :], self.coordinates, variables, "../../TEST_REPO", "before_causal")
        # self.Data = self._reload_data(self.data_path)

        # Now X and Y is ready for getitem
        #print("CMIP6 shape", self.Data.shape)
        self.length = self.Data.shape[0]

    def __getitem__(self, index):
        return self.Data[index]


class Input4MipsDataset(ClimateDataset):
    """Loads all scenarios for a given var / for all vars."""

    def __init__(  # inherits all the stuff from Base
        self,
        years: Union[int, str],
        historical_years: Union[int, str],
        data_dir: Optional[str] = "Climateset_DATA",
        variables: List[str] = ["BC_sum"],
        scenarios: List[str] = ["ssp126", "ssp370", "ssp585"],
        channels_last: bool = False,
        openburning_specs: Tuple[str] = ("no_fires", "no_fires"),
        mode: str = "train",
        output_save_dir: str = "",
        reload_climate_set_data: bool = True,
        seasonality_removal: bool = True,
        seq_len: int = 12,
        lat: int = 96,
        lon: int = 144,
        *args,
        **kwargs,
    ):

        self.channels_last = channels_last

        self.mode = mode
        self.root_dir = os.path.join(data_dir, "inputs/input4mips")
        self.output_save_dir = output_save_dir
        self.reload_climate_set_data = reload_climate_set_data
        self.input_nc_files = []
        self.output_nc_files = []
        self.seasonality_removal = seasonality_removal
        self.in_variables = variables
        self.seq_len = seq_len
        self.lon = lon
        self.lat = lat

        if len(historical_years) == 0:
            historical_years_str = "no_historical"
        elif len(historical_years) == 1:
            historical_years_str = f"{historical_years[0]}"
        else:
            historical_years_str = f"{historical_years[0]}-{historical_years[-1]}"

        fname_kwargs = dict(
            years=f"{years[0]}-{years[-1]}",
            historical_years=historical_years_str,
            variables=variables,
            scenarios=scenarios,
            channels_last=channels_last,
            openburning_specs=openburning_specs,
            seq_to_seq=True,
        )
        self.fname_kwargs = fname_kwargs

        historical_openburning, ssp_openburning = openburning_specs

        # Split the data here using n_years if needed,
        # else do random split logic here
        fname, coordinates_fname = self.get_save_name_from_kwargs(mode=mode, file="input", kwargs=fname_kwargs)

        # Check here if os.path.isfile(data.npz) exists #TODO: check if exists on slurm
        # if it does, use self._reload data(path)

        if os.path.isfile(os.path.join(output_save_dir, fname)) and self.reload_climate_set_data:  # we first need to get the name here to test that...
            self.data_path = os.path.join(output_save_dir, fname)
            #print("path exists, reloading")
            self.raw_data = self._reload_data(self.data_path)

            # Load stats and normalize
            stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                mode=mode, file="statistics", kwargs=fname_kwargs
            )
            stats = self.load_dataset_statistics(
                os.path.join(self.output_save_dir, stats_fname), mode=self.mode, mips="input4mips"
            )
            self.coordinates = self.load_dataset_coordinates(
                os.path.join(self.output_save_dir, coordinates_fname), mode=self.mode, mips="input4mips"
            )
            self.Data = self.normalize_data(self.raw_data, stats)
            if self.seasonality_removal:
                self.Data = self.remove_seasonality(self.Data)

            #print("In Input4mips, just finished removing the seasonality.")

        else:
            files_per_var = []
            for var in variables:
                #print("var", var)
                output_nc_files = []

                for exp in scenarios:  # TODO: implement getting by years! also sub selection for historical years
                    #print("exp", exp)
                    if var in NO_OPENBURNING_VARS and exp == "historical":
                       # print("I am in var in no_openburningvars and historical in input4mips")
                        filter_path_by = ""
                        get_years = historical_years
                    elif var in NO_OPENBURNING_VARS:
                        filter_path_by = ""
                        get_years = years
                    elif exp == "historical":
                        #print("I am in historical in input4mips")
                        filter_path_by = historical_openburning
                        get_years = historical_years
                    else:
                        #print("I am in else in INPUT4MIPS")
                        filter_path_by = ssp_openburning
                        get_years = years

                    for y in get_years:
                        # print('Input4mips y:', y )
                        var_dir = os.path.join(self.root_dir, exp, var, f"{CMIP6_NOM_RES}/{CMIP6_TEMP_RES}/{y}")
                        files = glob.glob(f"{var_dir}/**/*{filter_path_by}*.nc", recursive=True)
                        # print('files in input4mips', files)
                        output_nc_files += files
                files_per_var.append(output_nc_files)

            self.raw_data = self.load_into_mem(
                files_per_var, num_vars=len(variables), channels_last=self.channels_last, seq_to_seq=True
            )  # we always want the full sequence for input4mips
            self.coordinates = self.load_coordinates_into_mem(files_per_var)

            if self.mode == "train" or self.mode == "train+val":
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode=mode, file="statistics", kwargs=fname_kwargs
                )

                if os.path.isfile(stats_fname):
                    print("Stats file already exists! Loading from memory.")
                    stats = self.load_statistics_data(stats_fname)

                    self.norm_data = self.normalize_data(self.raw_data, stats)
                    if self.seasonality_removal:
                        self.norm_data = self.remove_seasonality(self.norm_data)

                else:
                    stat1, stat2 = self.get_dataset_statistics(self.raw_data, self.mode, mips="cmip6")
                    stats = {"mean": stat1, "std": stat2}
                    self.norm_data = self.normalize_data(self.raw_data, stats)
                    if self.seasonality_removal:
                        self.norm_data = self.remove_seasonality(self.norm_data)

                    save_file_name = self.write_dataset_statistics(stats_fname, stats)
                    save_file_name = self.write_dataset_statistics(coordinates_fname, self.coordinates)

                # self.norm_data = self.normalize_data(self.raw_data, stats)

            elif self.mode == "test":
                stats_fname, coordinates_fname = self.get_save_name_from_kwargs(
                    mode="train+val", file="statistics", kwargs=fname_kwargs
                )  # Load train stats cause we don't calculcate norm stats for test.
                stats = self.load_dataset_statistics(stats_fname, mode=self.mode, mips="input4mips")
                self.norm_data = self.normalize_data(self.raw_data, stats)
                if self.seasonality_removal:
                    self.norm_data = self.remove_seasonality(self.norm_data)

            # self.input_path = self.save_data_into_disk(self.raw_data_input, self.mode, 'input')
            #print("In input4mips, just about to save the data.")
            self.data_path = self.save_data_into_disk(self.raw_data, fname, output_save_dir)
            #print("In input4mips, just saved the data.")

            #print("In input4mips, just about to copy the data to slurm.")
            # self.copy_to_slurm(self.input_path)
            self.copy_to_slurm(self.data_path)
            #print("In input4mips, just copied the data to slurm.")

            # Call _reload_data here with self.input_path and self.output_path
            # self.X = self._reload_data(input_path)
            self.Data = self.norm_data
            # self.Data = self._reload_data(self.data_path)
            # Write a normalize transform to calculate mean and std
            # Either normalized whole array here or per instance getitem, that maybe faster

            # Now X and Y is ready for getitem
        #print("Input4mips shape", self.Data.shape)
        self.length = self.Data.shape[0]

    def __getitem__(self, index):
        return self.Data[index]