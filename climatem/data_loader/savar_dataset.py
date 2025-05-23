import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from climatem.synthetic_data.generate_savar_datasets import generate_save_savar_data
from climatem.synthetic_data.graph_evaluation import extract_adjacency_matrix


class SavarDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        output_save_dir: Optional[str] = "Savar_DATA",
        lat: int = 125,
        lon: int = 125,
        tau: int = 5,
        global_normalization: bool = True,
        seasonality_removal: bool = True,
        reload_climate_set_data: Optional[bool] = True,
        time_len: int = 10_000,
        comp_size: int = 10,
        noise_val: float = 0.2,
        n_per_col: int = 2,
        difficulty: str = "easy",
        seasonality: bool = False,
        overlap: bool = False,
        is_forced: bool = False,
        plot_original_data: bool = True,
    ):
        super().__init__()
        self.output_save_dir = Path(output_save_dir)
        self.savar_name = f"modes_{n_per_col**2}_tl_{time_len}_isforced_{is_forced}_difficulty_{difficulty}_noisestrength_{noise_val}_seasonality_{seasonality}_overlap_{overlap}"
        self.savar_path = self.output_save_dir / f"{self.savar_name}.npy"

        self.global_normalization = global_normalization
        self.seasonality_removal = seasonality_removal
        self.reload_climate_set_data = reload_climate_set_data

        # TODO: for now this is ok, we create a square grid. Later we might want to look at icosahedral grid :)
        self.lat = lat
        self.lon = lon
        self.coordinates = np.array(np.meshgrid(np.arange(self.lat), np.arange(self.lon))).reshape((2, -1)).T

        self.time_len = time_len
        self.comp_size = comp_size
        self.noise_val = noise_val
        self.n_per_col = n_per_col
        self.difficulty = difficulty
        self.seasonality = seasonality
        self.overlap = overlap
        self.is_forced = is_forced
        self.plot_original_data = plot_original_data

        if self.reload_climate_set_data:
            self.gt_modes = np.load(self.output_save_dir / f"{self.savar_name}_modes.npy")
            self.gt_noise = np.load(self.output_save_dir / f"{self.savar_name}_noise_modes.npy")
            links_coeffs = np.load(
                self.output_save_dir / f"{self.savar_name}_parameters.npy", allow_pickle=True
            ).item()["links_coeffs"]
            self.gt_adj = np.array(extract_adjacency_matrix(links_coeffs, n_per_col**2, tau))[::-1]
        else:
            self.gt_modes = None
            self.gt_noise = None
            links_coeffs = None
            self.gt_adj = None

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

        print("Inside aggregate_months, and the data before reshaping is:", data.shape)
        reshaped_data = data.reshape(data.shape[0], -1, num_months_aggregated, *data.shape[2:])
        print("Still inside aggregate months, reshaped_data shape:", reshaped_data.shape)

        # average over the new dimension
        aggregated_data = np.nanmean(reshaped_data, axis=2)
        print("Shape of the aggregated data?:", aggregated_data.shape)
        return aggregated_data

    def split_data_by_interval(self, data, tau, ratio_train, interval_length=100):
        """Given a dataset and interval length, divide the data into intervals, then splits each interval into training
        and validation indices based on ratio."""
        # interval_length=10
        print(f"intervallength{interval_length}")
        print(f"datashape{data.shape[0]}")
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

    # change this one

    # This method loads the savar data from the given path and reshapes
    # the loaded data from savar's (lon*lat, years*months) to CDSD's
    # (year, months, lon, lat)
    def load_savar_data(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        print(f"Loaded data shape: {data.shape}")
        time_steps = data.shape[1]
        data_reshaped = data.T.reshape((time_steps, self.lat, self.lon))
        print(f"Loaded data shape after: {data_reshaped.shape}")
        return data_reshaped

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
    ):
        """
        Constructs dataset for causal discovery model.

        Splits each scenario into training and validation sets, then generates overlapping sequences.
        """
        print(f"Getting causal data [mode={mode}] ...")
        # TODO: change + .npy...
        if os.path.exists(self.savar_path) and self.reload_climate_set_data:
            data = self.load_savar_data(self.savar_path)
        else:
            print("CREATE SAVAR DATA")
            data = generate_save_savar_data(
                self.output_save_dir,
                self.savar_name,
                self.time_len,
                self.comp_size,
                self.noise_val,
                self.n_per_col,
                self.difficulty,
                self.seasonality,
                self.overlap,
                self.is_forced,
                self.plot_original_data,
                tau,
            )
            time_steps = data.shape[1]
            data = data.T.reshape((time_steps, self.lat, self.lon))

            self.gt_modes = np.load(self.output_save_dir / f"{self.savar_name}_modes.npy")
            self.gt_noise = np.load(self.output_save_dir / f"{self.savar_name}_noise_modes.npy")
            links_coeffs = np.load(
                self.output_save_dir / f"{self.savar_name}_parameters.npy", allow_pickle=True
            ).item()["links_coeffs"]
            self.gt_adj = np.array(extract_adjacency_matrix(links_coeffs, self.n_per_col**2, tau))

        data = data.astype("float32")
        # TODO: normalize by saveing std/mean from train data and then normalize test by reloading
        # Very important to avoid normalizing differently test and train data
        if self.global_normalization:
            data = (data - data.mean()) / data.std()
        if self.seasonality_removal:
            self.norm_data = self.remove_seasonality(self.norm_data)

        print(f"data is {data.dtype}")

        try:
            # NOTE:(seb) this is what we do when we have the regularly gridded data!
            # (years, months, vars, lon, lat) -> (scenrios, years*months, vars, lon, lat)
            # Regular data shape before reshaping: (101, 12, 1, 96, 144)
            # Regular data shape after reshaping: (1, 1212, 1, 96, 144)
            print("Trying to regrid to lon, lat if we have regular data...")
            # data = data.reshape(num_scenarios, num_years, num_vars, LON, LAT)

            data = data.reshape(1, data.shape[0], 1, self.lon, self.lat)

        except ValueError:
            print(
                "I saw a ValueError and now I am reshaping the data differently, probably as I have icosahedral data!"
            )
            # I need to include the number of years in the reshape here...!
            # How to access it? As the length of the list of paths?
            # NOTE: currently hardcoding 101 year long sequences...need to unhack this...
            # NOTE:(seb) now we hard code that we want to change to num_years*12, like we had before
            # this -1 should probably be changed to reflect the number of coordinates that we have for the icosahedral grid...
            # also the .txt file will not be right for different resolutions!!!!

            print("Data shape before reshaping:", data.shape)
            data = data.reshape(1, data.shape[0], 1, -1)
            print("Data shape after reshaping:", data.shape)

        if isinstance(num_months_aggregated, (int, np.integer)) and num_months_aggregated > 1:
            data = self.aggregate_months(data, num_months_aggregated)
            # for each scenario in data, generate overlapping sequences
            if mode == "train" or mode == "train+val":
                print("IN IF")
                x_train_list, y_train_list = [], []
                x_valid_list, y_valid_list = [], []

                for scenario in data:
                    idx_train, idx_valid = self.split_data_by_interval(scenario, tau, ratio_train, interval_length)
                    # np.random.shuffle(idx_train)
                    # np.random.shuffle(idx_valid)

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
                train_y = np.expand_dims(train_y, axis=1)
                valid_y = np.expand_dims(valid_y, axis=1)

                # z-score normalization
                # make train_y go from (2550, 4, 96, 144) to (2550, 1, 4, 96, 144)
                train = train_x, train_y
                valid = valid_x, valid_y

                # print(train_y.shape)
                # plot_species(train_y[:, :, 0, :, :], self.coordinates, "tas", "../../TEST_REPO", "after_causal")
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

        # NOTE:seb delete commented code

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
                train_y = np.expand_dims(train_y, axis=1)
                valid_y = np.expand_dims(valid_y, axis=1)

                train = train_x, train_y
                valid = valid_x, valid_y
                print(f"train: {train[0].dtype}")
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

        np.savez(os.path.join(output_save_dir, fname), data=data)
        return os.path.join(output_save_dir, fname)

    def get_mean_std(self, data):
        # DATA shape (258, 12, 4, 96, 144) or DATA shape (258, 12, 2, 96, 144)
        # NOTE:(seb) 13th May, 2024: this is the original of the code:
        if data.ndim == 5:
            data = np.moveaxis(
                data, 2, 0
            )  # DATA shape (258, 12, 4, 96, 144) -> (4, 258, 12, 96, 144) easier to calulate statistics
            vars_mean = np.nanmean(data, axis=(1, 2, 3, 4))  # sDATA shape (258, 12, 4, 96, 144)
            vars_std = np.nanstd(data, axis=(1, 2, 3, 4))
            vars_mean = np.expand_dims(vars_mean, (1, 2, 3, 4))  # Shape of mean & std (4, 1, 1, 1, 1)
            vars_std = np.expand_dims(vars_std, (1, 2, 3, 4))

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

    # important?
    # NOTE:(seb) I need to check the axis is correct here?
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

        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        # return data

        # NOTE: SH - do we not do this above?
        # standardise - I hope this is doing by month, to check

        return (data - mean[None]) / std[None]

        # now just divide by std...
        # return data / std[None]

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

    def __getitem__(self, index):  # Dict[str, Tensor]):

        # access data in input4mips and cmip6 datasets
        X = self.input4mips_ds[index]
        Y = self.cmip6_ds[index]

        return X, Y

    def __str__(self):
        s = f" {self.name} dataset: {self.n_years} years used, with a total size of {len(self)} examples."
        return s

    # NOTE(seb): is this a good way to get the length?
    def __len__(self):
        print("Input4mips", self.input4mips_ds.length, "CMIP6 data", self.cmip6_ds.length)
        assert self.input4mips_ds.length == self.cmip6_ds.length, "Datasets not of same length"
        return self.input4mips_ds.length
