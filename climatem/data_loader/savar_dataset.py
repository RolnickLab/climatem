import os
from pathlib import Path
from typing import List, Optional, Sequence

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
        periods: List[float] = [365, 182.5, 60],
        amplitudes: List[float] = [0.06, 0.02, 0.01],
        phases: List[float] = [0.0, 0.7853981634, 1.5707963268],
        yearly_jitter_amp: float = 0.05,
        yearly_jitter_phase: float = 0.10,
        overlap: bool = False,
        is_forced: bool = False,
        f_1: float = 0.1,
        f_2: float = 0.2,
        f_time_1: int = 4000,
        f_time_2: int = 8000,
        ramp_type: str = "linear",
        linearity: str = "linear",
        poly_degrees: List[int] = [2, 3],
        plot_original_data: bool = True,
        use_separate_forcings: bool = False,  # enable dual exogenous forcings
        aerosol_scale: float = 0.02,
        aerosol_spatial_contrast: float = 1.05,
        aerosol_ramp_up_time: int = 2000,
        aerosol_peak_time: int = 5000,
        aerosol_decline_time: int = 8000,
        # Forcing causal structure parameters
        n_co2_latents: int = 1,
        n_aerosol_latents: int = 4,
        co2_effect_strength: float = 0.15,
        aerosol_effect_strength: float = 0.10,
    ):
        super().__init__()
        self.output_save_dir = Path(output_save_dir)
        self.savar_name = (
            f"m_{n_per_col**2}_tl_{time_len}_ifd_{is_forced}_dif_{difficulty}_ns_"
            f"{noise_val}_ses_{seasonality}_ol_{overlap}_f1_{f_1}_f2_{f_2}_ft1_{f_time_1}_ft2_{f_time_2}"
            f"_rmp_{ramp_type}_lin_{linearity}_pds_{poly_degrees}_asp_{aerosol_scale}_asc_{aerosol_spatial_contrast}"
            f"_art_{aerosol_ramp_up_time}_apt_{aerosol_peak_time}_adt_{aerosol_decline_time}"
        )

        # SAVAR data is stored in a subfolder named after the dataset
        self.savar_dataset_dir = self.output_save_dir / self.savar_name
        self.savar_path = self.savar_dataset_dir / "savar.npy"

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
        self.periods = periods
        self.amplitudes = amplitudes
        self.phases = phases
        self.yearly_jitter_amp = yearly_jitter_amp
        self.yearly_jitter_phase = yearly_jitter_phase
        self.overlap = overlap
        self.is_forced = is_forced
        self.f_1 = f_1
        self.f_2 = f_2
        self.f_time_1 = f_time_1
        self.f_time_2 = f_time_2
        self.ramp_type = ramp_type
        self.linearity = linearity
        self.poly_degrees = poly_degrees
        self.plot_original_data = plot_original_data
        self.use_separate_forcings = use_separate_forcings
        self.aerosol_scale = aerosol_scale
        self.aerosol_spatial_contrast = aerosol_spatial_contrast
        self.aerosol_ramp_up_time = aerosol_ramp_up_time
        self.aerosol_peak_time = aerosol_peak_time
        self.aerosol_decline_time = aerosol_decline_time
        # Forcing causal structure parameters
        self.n_co2_latents = n_co2_latents
        self.n_aerosol_latents = n_aerosol_latents
        self.co2_effect_strength = co2_effect_strength
        self.aerosol_effect_strength = aerosol_effect_strength
        self.tau = tau

        if self.reload_climate_set_data:
            self.gt_modes = np.load(self.savar_dataset_dir / "modes.npy")
            self.gt_noise = np.load(self.savar_dataset_dir / "noise_modes.npy")
            params = np.load(self.savar_dataset_dir / "parameters.npy", allow_pickle=True).item()
            links_coeffs = params["links_coeffs"]

            # Use n_total_latents if available (includes forcing latents), otherwise fall back to n_per_col**2
            n_total_latents = params.get("n_total_latents", n_per_col**2)
            self.forcing_indices = params.get("forcing_indices", None)
            self.n_climate_modes = params.get("n_climate_modes", n_per_col**2)

            self.gt_adj = np.array(extract_adjacency_matrix(links_coeffs, n_total_latents, tau))[::-1]

            if self.forcing_indices is not None:
                print(
                    f"Loaded extended causal graph with {n_total_latents} latents "
                    f"(climate: {self.n_climate_modes}, CO2: {len(self.forcing_indices.get('co2', []))}, "
                    f"aerosol: {len(self.forcing_indices.get('aerosol', []))})"
                )

            # Load separate forcing files if requested
            if self.use_separate_forcings and self.is_forced:
                co2_forcing_path = self.savar_dataset_dir / "co2_forcing.npy"
                aerosol_forcing_path = self.savar_dataset_dir / "aerosol_forcing.npy"

                if co2_forcing_path.exists():
                    self.co2_forcing = np.load(co2_forcing_path)
                    print(f"Loaded CO2 forcing from {co2_forcing_path}, shape: {self.co2_forcing.shape}")
                else:
                    print(f"Warning: CO2 forcing file not found: {co2_forcing_path}")
                    self.co2_forcing = None

                if aerosol_forcing_path.exists():
                    self.aerosol_forcing = np.load(aerosol_forcing_path)
                    print(f"Loaded aerosol forcing from {aerosol_forcing_path}, shape: {self.aerosol_forcing.shape}")
                else:
                    print(f"Warning: Aerosol forcing file not found: {aerosol_forcing_path}")
                    self.aerosol_forcing = None

                # Load ground truth forcing latent trajectories for supervision
                co2_latent_path = self.savar_dataset_dir / "co2_latent_trajectory.npy"
                aerosol_latent_path = self.savar_dataset_dir / "aerosol_latent_trajectory.npy"

                if co2_latent_path.exists():
                    self.gt_co2_latent = np.load(co2_latent_path)
                    print(f"Loaded CO2 latent trajectory from {co2_latent_path}, shape: {self.gt_co2_latent.shape}")
                else:
                    print(f"Warning: CO2 latent trajectory not found: {co2_latent_path}")
                    self.gt_co2_latent = None

                if aerosol_latent_path.exists():
                    self.gt_aerosol_latent = np.load(aerosol_latent_path)
                    print(
                        f"Loaded aerosol latent trajectory from {aerosol_latent_path}, shape: {self.gt_aerosol_latent.shape}"
                    )
                else:
                    print(f"Warning: Aerosol latent trajectory not found: {aerosol_latent_path}")
                    self.gt_aerosol_latent = None
            else:
                self.co2_forcing = None
                self.aerosol_forcing = None
                self.gt_co2_latent = None
                self.gt_aerosol_latent = None
        else:
            self.gt_modes = None
            self.gt_noise = None
            links_coeffs = None
            self.gt_adj = None
            self.co2_forcing = None
            self.aerosol_forcing = None

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

    def reshape_forcing_data(self, forcing_data):
        """
        Reshape forcing data from SAVAR format (spatial_res, time) to match observations.

        Args:
            forcing_data: Array of shape (spatial_resolution, time_length)

        Returns:
            Reshaped array of shape (time_length, lat, lon)
        """
        if forcing_data is None:
            return None

        print(f"Reshaping forcing data from shape: {forcing_data.shape}")
        time_steps = forcing_data.shape[1]
        # Transpose and reshape to match observation format
        forcing_reshaped = forcing_data.T.reshape((time_steps, self.lat, self.lon))
        print(f"Reshaped forcing data to: {forcing_reshaped.shape}")
        return forcing_reshaped

    def get_forcing_sequences(self, forcing_data, idxs):
        """
        Extract forcing values at specified timestep indices.

        Args:
            forcing_data: Array of shape (time, lat, lon) - already reshaped forcing data
            idxs: Timestep indices to extract

        Returns:
            Array of shape (len(idxs), lat, lon) containing forcing at each timestep
        """
        if forcing_data is None:
            return None

        forcing_sequences = []
        for idx in idxs:
            forcing_sequences.append(forcing_data[idx])

        return np.stack(forcing_sequences)

    def _load_or_generate_savar_data(self, tau):
        """Load existing SAVAR data or generate new data."""
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
                self.periods,
                self.amplitudes,
                self.phases,
                self.yearly_jitter_amp,
                self.yearly_jitter_phase,
                self.overlap,
                self.is_forced,
                self.f_1,
                self.f_2,
                self.f_time_1,
                self.f_time_2,
                self.ramp_type,
                self.linearity,
                self.poly_degrees,
                self.plot_original_data,
                self.aerosol_scale,
                self.aerosol_spatial_contrast,
                self.aerosol_ramp_up_time,
                self.aerosol_peak_time,
                self.aerosol_decline_time,
                n_co2_latents=self.n_co2_latents,
                n_aerosol_latents=self.n_aerosol_latents,
                co2_effect_strength=self.co2_effect_strength,
                aerosol_effect_strength=self.aerosol_effect_strength,
                tau=self.tau,
            )
            time_steps = data.shape[1]
            data = data.T.reshape((time_steps, self.lat, self.lon))

            self.gt_modes = np.load(self.savar_dataset_dir / "modes.npy")
            self.gt_noise = np.load(self.savar_dataset_dir / "noise_modes.npy")
            params = np.load(self.savar_dataset_dir / "parameters.npy", allow_pickle=True).item()
            links_coeffs = params["links_coeffs"]

            n_total_latents = params.get("n_total_latents", self.n_per_col**2)
            self.forcing_indices = params.get("forcing_indices", None)
            self.n_climate_modes = params.get("n_climate_modes", self.n_per_col**2)

            self.gt_adj = np.array(extract_adjacency_matrix(links_coeffs, n_total_latents, tau))

        return data

    def _load_forcing_files(self):
        """Load separate CO2 and aerosol forcing files if available."""
        if self.use_separate_forcings and self.is_forced:
            co2_forcing_path = self.savar_dataset_dir / "co2_forcing.npy"
            aerosol_forcing_path = self.savar_dataset_dir / "aerosol_forcing.npy"

            if co2_forcing_path.exists():
                self.co2_forcing = np.load(co2_forcing_path)
                print(f"Loaded CO2 forcing from {co2_forcing_path}, shape: {self.co2_forcing.shape}")
            else:
                print(f"Warning: CO2 forcing file not found: {co2_forcing_path}")
                self.co2_forcing = None

            if aerosol_forcing_path.exists():
                self.aerosol_forcing = np.load(aerosol_forcing_path)
                print(f"Loaded aerosol forcing from {aerosol_forcing_path}, shape: {self.aerosol_forcing.shape}")
            else:
                print(f"Warning: Aerosol forcing file not found: {aerosol_forcing_path}")
                self.aerosol_forcing = None

            co2_latent_path = self.savar_dataset_dir / "co2_latent_trajectory.npy"
            aerosol_latent_path = self.savar_dataset_dir / "aerosol_latent_trajectory.npy"

            if co2_latent_path.exists():
                self.gt_co2_latent = np.load(co2_latent_path)
                print(f"Loaded CO2 latent trajectory from {co2_latent_path}, shape: {self.gt_co2_latent.shape}")
            else:
                print(f"Warning: CO2 latent trajectory not found: {co2_latent_path}")
                self.gt_co2_latent = None

            if aerosol_latent_path.exists():
                self.gt_aerosol_latent = np.load(aerosol_latent_path)
                print(
                    f"Loaded aerosol latent trajectory from {aerosol_latent_path}, shape: {self.gt_aerosol_latent.shape}"
                )
            else:
                print(f"Warning: Aerosol latent trajectory not found: {aerosol_latent_path}")
                self.gt_aerosol_latent = None
        else:
            self.co2_forcing = None
            self.aerosol_forcing = None
            self.gt_co2_latent = None
            self.gt_aerosol_latent = None

    def _reshape_data(self, data):
        """Reshape data to proper dimensions."""
        try:
            print("Trying to regrid to lon, lat if we have regular data...")
            data = data.reshape(1, data.shape[0], 1, self.lon, self.lat)
        except ValueError:
            print("Reshaping data for icosahedral grid...")
            print("Data shape before reshaping:", data.shape)
            data = data.reshape(1, data.shape[0], 1, -1)
            print("Data shape after reshaping:", data.shape)
        return data

    def _process_aggregated_data(self, data, tau, future_timesteps, mode, ratio_train, interval_length):
        """Process aggregated monthly data for training/validation."""
        data = self.aggregate_months(data, num_months_aggregated=1)

        if mode == "train" or mode == "train+val":
            train, valid = self._generate_train_valid_sequences(
                data, tau, future_timesteps, ratio_train, interval_length
            )
            return train, valid
        else:
            test = self._generate_test_sequences(data, tau, future_timesteps)
            return test

    def _generate_train_valid_sequences(self, data, tau, future_timesteps, ratio_train, interval_length):
        """Generate training and validation sequences from data."""
        x_train_list, y_train_list, x_valid_list, y_valid_list = [], [], [], []

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

        train_y = np.expand_dims(train_y, axis=1)
        valid_y = np.expand_dims(valid_y, axis=1)

        self._extract_forcing_sequences(data, tau, ratio_train, interval_length)

        return (train_x, train_y), (valid_x, valid_y)

    def _generate_test_sequences(self, data, tau, future_timesteps):
        """Generate test sequences from data."""
        x_test_list, y_test_list = [], []
        for scenario in data:
            idx_test = np.arange(tau, scenario.shape[0])
            x_test, y_test = self.get_overlapping_sequences(scenario, idx_test, tau, future_timesteps)
            x_test_list.extend(x_test)
            y_test_list.extend(y_test)

        test_x, test_y = np.stack(x_test_list), np.stack(y_test_list)
        test_y = np.expand_dims(test_y, axis=1)
        return test_x, test_y

    def _extract_forcing_sequences(self, data, tau, ratio_train, interval_length):
        """Extract forcing sequences for training and validation."""
        if not (self.use_separate_forcings and hasattr(self, "co2_forcing") and self.co2_forcing is not None):
            self.co2_forcing_train = None
            self.aerosol_forcing_train = None
            self.co2_forcing_valid = None
            self.aerosol_forcing_valid = None
            return

        co2_reshaped = self.reshape_forcing_data(self.co2_forcing)
        aerosol_reshaped = self.reshape_forcing_data(self.aerosol_forcing)

        co2_train_list, co2_valid_list = [], []
        aerosol_train_list, aerosol_valid_list = [], []

        for scenario in data:
            idx_train, idx_valid = self.split_data_by_interval(scenario, tau, ratio_train, interval_length)

            co2_train_list.extend([co2_reshaped[idx - tau : idx + 1] for idx in idx_train])
            aerosol_train_list.extend([aerosol_reshaped[idx - tau : idx + 1] for idx in idx_train])

            co2_valid_list.extend([co2_reshaped[idx - tau : idx + 1] for idx in idx_valid])
            aerosol_valid_list.extend([aerosol_reshaped[idx - tau : idx + 1] for idx in idx_valid])

        co2_train_stacked = np.stack(co2_train_list).astype("float32")
        aerosol_train_stacked = np.stack(aerosol_train_list).astype("float32")
        co2_valid_stacked = np.stack(co2_valid_list).astype("float32") if len(co2_valid_list) > 0 else None
        aerosol_valid_stacked = np.stack(aerosol_valid_list).astype("float32") if len(aerosol_valid_list) > 0 else None

        self.co2_forcing_train = co2_train_stacked.mean(axis=(-2, -1), keepdims=True).reshape(
            co2_train_stacked.shape[0], co2_train_stacked.shape[1], 1
        )
        self.aerosol_forcing_train = aerosol_train_stacked.reshape(
            aerosol_train_stacked.shape[0], aerosol_train_stacked.shape[1], -1
        )

        if co2_valid_stacked is not None:
            self.co2_forcing_valid = co2_valid_stacked.mean(axis=(-2, -1), keepdims=True).reshape(
                co2_valid_stacked.shape[0], co2_valid_stacked.shape[1], 1
            )
            self.aerosol_forcing_valid = aerosol_valid_stacked.reshape(
                aerosol_valid_stacked.shape[0], aerosol_valid_stacked.shape[1], -1
            )
        else:
            self.co2_forcing_valid = None
            self.aerosol_forcing_valid = None

        print(
            f"Extracted forcing sequences - CO2 train: {self.co2_forcing_train.shape}, "
            f"aerosol train: {self.aerosol_forcing_train.shape}"
        )
        print("CO2 is global (spatially averaged), aerosols are spatially varying")

    def _generate_non_aggregated_data(self, data, tau, future_timesteps, mode, ratio_train, interval_length):
        """Generate non-aggregated causal data for training/validation or testing."""
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

            train_y = np.expand_dims(train_y, axis=1)
            valid_y = np.expand_dims(valid_y, axis=1)

            self._extract_forcing_sequences(data, tau, ratio_train, interval_length, train_y, valid_y)

            return (train_x, train_y), (valid_x, valid_y)
        else:
            x_test_list, y_test_list = [], []

            for scenario in data:
                idx_test = np.arange(tau, scenario.shape[0])
                x_test, y_test = self.get_overlapping_sequences(scenario, idx_test, tau, future_timesteps)
                x_test_list.extend(x_test)
                y_test_list.extend(y_test)

            test_x, test_y = np.stack(x_test_list), np.stack(y_test_list)
            test_y = np.expand_dims(test_y, axis=1)

            return test_x, test_y

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
        data = self._load_or_generate_savar_data(tau)
        self._load_forcing_files()

        data = data.astype("float32")
        if self.global_normalization:
            data = (data - data.mean()) / data.std()
        if self.seasonality_removal:
            data = self.remove_seasonality(
                data,
                periods=self.periods,
                demean=True,
                normalise=False,
                rolling=True,
                w=10,
            )

        print(f"data is {data.dtype}")
        data = self._reshape_data(data)

        if isinstance(num_months_aggregated, (int, np.integer)) and num_months_aggregated > 1:
            return self._process_aggregated_data(data, tau, future_timesteps, mode, ratio_train, interval_length)
        else:
            return self._generate_non_aggregated_data(data, tau, future_timesteps, mode, ratio_train, interval_length)

    def get_forcing_data(self):
        """
        Get CO2 and aerosol forcing data, properly reshaped to match observations.

        Returns:
            Tuple of (co2_forcing, aerosol_forcing), each of shape (time, lat, lon) or None
        """
        if not self.use_separate_forcings:
            return None, None

        co2_reshaped = self.reshape_forcing_data(self.co2_forcing) if self.co2_forcing is not None else None
        aerosol_reshaped = self.reshape_forcing_data(self.aerosol_forcing) if self.aerosol_forcing is not None else None

        return co2_reshaped, aerosol_reshaped

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

    def remove_seasonality(
        self,
        data: np.ndarray,
        periods: int | Sequence[int] | Sequence[float] = (12, 6, 3),
        demean: bool = True,
        normalise: bool = False,
        rolling: bool = True,  # ← default TRUE because of jitter
        w: int = 10,  # (10 years ≈ 120 steps @ monthly)
    ):
        """
        Remove deterministic periodic seasonality from a [time, …] array.

        Parameters
        ----------
        period      single cycle length **or** list/tuple of lengths
                    (e.g. [12, 6] for annual + semi-annual)
        …
        """

        def _remove_one(x: np.ndarray, p: int) -> np.ndarray:
            """Inner helper that handles a single period length."""
            t = x.shape[0]
            rem = t % p
            if rem:
                x = x[:-rem]
                t -= rem
            folded = x.reshape((t // p, p) + x.shape[1:])
            if rolling:
                k = min(w, folded.shape[0])
                mean = np.nanmean(folded[-k:], axis=0)
                std = np.nanstd(folded[-k:], axis=0)
            else:
                mean = np.nanmean(folded, axis=0)
                std = np.nanstd(folded, axis=0)
            mean_full = np.tile(mean, (t // p, *[1] * (x.ndim - 1)))
            std_full = np.tile(std, (t // p, *[1] * (x.ndim - 1)))
            out = x.copy()
            if demean:
                out -= mean_full
            if normalise:
                out /= np.where(std_full == 0, 1, std_full)
            return out.astype(np.float32)

        # handle one or many cycle lengths
        if isinstance(periods, (list, tuple, np.ndarray)):
            # remove the longest cycle first to avoid leakage
            _periods = sorted([int(round(p)) for p in periods], reverse=True)
        else:  # single scalar
            _periods = [int(round(periods))]

        out = data.astype(np.float32)
        for p in _periods:
            out = _remove_one(out, p)
        return out

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
