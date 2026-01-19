"""
This code is inspired by the SAVAR data generation paper and code "A spatiotemporal stochastic climate model for
benchmarking causal discovery methods for teleconnections", Tibau et al.

2022 The main difference with the provided code is the torch/GPU implementation which considerably speeds up the data
generation process
"""

import itertools as it
import math
from copy import deepcopy
from math import pi
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm.auto import tqdm

from climatem.utils import get_logger

logger = get_logger(__name__)


def dict_to_matrix(links_coeffs, default=0):
    """
    Maps to the coefficient matrix.

    Without time :param links_coeffs: :param default: :return: a matrix coefficient of [j, i, \tau-1] where a link is i
    -> j at \tau
    """
    tau_max = max(abs(lag) for (_, lag), _ in it.chain.from_iterable(links_coeffs.values()))

    n_vars = len(links_coeffs)

    graph = np.ones((n_vars, n_vars, tau_max), dtype=float)
    graph *= default

    for j, values in links_coeffs.items():
        for (i, tau), coeff in values:
            graph[j, i, abs(tau) - 1] = coeff

    return graph


class SAVAR:
    """Main class containing SAVAR model."""

    __slots__ = [
        "links_coeffs",
        "n_vars",
        "n_climate_modes",
        "time_length",
        "transient",
        "spatial_resolution",
        "tau_max",
        "mode_weights",
        "noise_weights",
        "noise_cov",
        "noise_strength",
        "noise_variance",
        "latent_noise_cov",
        "fast_noise_cov",
        "forcing_dict",
        "forcing_indices",
        "forcing_coeffs",
        "season_dict",
        "data_field",
        "noise_data_field",
        "seasonal_data_field",
        "forcing_data_field",
        "co2_forcing_data_field",
        "aerosol_forcing_data_field",
        "co2_latent_trajectory",
        "aerosol_latent_trajectory",
        "linearity",
        "poly_degrees",
        "verbose",
        "model_seed",
        "nnar_model",
        "output_save_dir",
    ]

    def __init__(
        self,
        links_coeffs: dict,
        time_length: int,
        mode_weights: np.ndarray,
        transient: int = 200,
        noise_weights: np.ndarray = None,
        noise_strength: float = 1,
        noise_variance: float = 1,
        noise_cov: np.ndarray = None,
        latent_noise_cov: np.ndarray = None,
        fast_cov: np.ndarray = None,
        forcing_dict: dict = None,
        forcing_indices: dict = None,
        linearity: str = "linear",
        poly_degrees: List[int] = [2],
        season_dict: dict = None,
        data_field: np.ndarray = None,
        noise_data_field: np.ndarray = None,
        seasonal_data_field: np.ndarray = None,
        forcing_data_field: np.ndarray = None,
        co2_forcing_data_field: np.ndarray = None,
        aerosol_forcing_data_field: np.ndarray = None,
        verbose: bool = False,
        model_seed: int = None,
        output_save_dir: str = None,
    ):

        self.links_coeffs = links_coeffs
        self.time_length = time_length
        self.transient = transient
        self.noise_strength = noise_strength
        self.noise_variance = noise_variance  # TODO: NOT USED.
        self.noise_cov = noise_cov

        self.latent_noise_cov = latent_noise_cov  # D_x
        self.fast_noise_cov = fast_cov  # D_y

        self.mode_weights = mode_weights
        self.noise_weights = noise_weights

        self.forcing_dict = forcing_dict
        self.forcing_indices = forcing_indices
        self.season_dict = season_dict
        self.linearity = linearity
        self.poly_degrees = poly_degrees

        self.data_field = data_field

        self.verbose = verbose
        self.model_seed = model_seed
        self.output_save_dir = output_save_dir

        # Computed attributes
        print("Creating attributes")
        # n_climate_modes is the number of climate variables (from mode_weights)
        self.n_climate_modes = mode_weights.shape[0]
        # n_vars is total latents (climate + forcing) if forcing_indices provided
        if forcing_indices is not None:
            self.n_vars = forcing_indices.get("n_total", self.n_climate_modes)
        else:
            self.n_vars = self.n_climate_modes
        self.tau_max = max(abs(lag) for (_, lag), _ in it.chain.from_iterable(self.links_coeffs.values()))
        self.spatial_resolution = deepcopy(self.mode_weights.reshape(self.n_climate_modes, -1).shape[1])

        # Extract forcing → mode coefficients if forcing is used
        self.forcing_coeffs = self._extract_forcing_coefficients() if forcing_indices else None

        # Initialize forcing latent trajectories (populated during forcing generation)
        self.co2_latent_trajectory = None
        self.aerosol_latent_trajectory = None
        print("spatial-resolution done")

        if self.noise_weights is None:
            self.noise_weights = deepcopy(self.mode_weights)
        if self.latent_noise_cov is None:
            self.latent_noise_cov = np.eye(self.n_vars)
        if self.fast_noise_cov is None:
            self.fast_noise_cov = np.zeros((self.spatial_resolution, self.spatial_resolution))
        print("copies done")

        # Empty attributes
        self.noise_data_field = noise_data_field
        self.seasonal_data_field = seasonal_data_field
        self.forcing_data_field = forcing_data_field
        self.co2_forcing_data_field = co2_forcing_data_field
        self.aerosol_forcing_data_field = aerosol_forcing_data_field

        if np.random is not None:
            np.random.seed(model_seed)

    def _extract_forcing_coefficients(self):
        """
        Extract forcing → climate mode causal coefficients from links_coeffs.

        Returns a dictionary with structure:
        {
            'co2_to_modes': {mode_idx: [(forcing_idx, lag, coeff), ...]},
            'aerosol_to_modes': {mode_idx: [(forcing_idx, lag, coeff), ...]},
        }
        """
        if self.forcing_indices is None:
            return None

        co2_indices = set(self.forcing_indices.get("co2", []))
        aerosol_indices = set(self.forcing_indices.get("aerosol", []))

        forcing_coeffs = {
            "co2_to_modes": {m: [] for m in range(self.n_climate_modes)},
            "aerosol_to_modes": {m: [] for m in range(self.n_climate_modes)},
        }

        # Scan links_coeffs for forcing → mode connections
        for target_idx, links in self.links_coeffs.items():
            # Only consider climate modes as targets
            if target_idx >= self.n_climate_modes:
                continue

            for (source_idx, lag), coeff in links:
                if source_idx in co2_indices:
                    forcing_coeffs["co2_to_modes"][target_idx].append((source_idx, lag, coeff))
                elif source_idx in aerosol_indices:
                    forcing_coeffs["aerosol_to_modes"][target_idx].append((source_idx, lag, coeff))

        # Print summary
        n_co2_links = sum(len(v) for v in forcing_coeffs["co2_to_modes"].values())
        n_aerosol_links = sum(len(v) for v in forcing_coeffs["aerosol_to_modes"].values())
        print(f"Extracted forcing coefficients: {n_co2_links} CO2→mode links, {n_aerosol_links} aerosol→mode links")

        return forcing_coeffs

    def generate_data(self, train_nnar=True) -> None:
        """Generates the data of savar :return:"""
        # Prepare the datafield
        if self.data_field is None:
            if self.verbose:
                print("Creating empty data field")
            # Compute the field
            self.data_field = np.zeros((self.spatial_resolution, self.time_length + self.transient))

        # Add noise first
        if self.noise_data_field is None:
            if self.verbose:
                print("Creating noise data field")
            self._add_noise_field()
        else:
            self.data_field += self.noise_data_field

        # Add seasonality
        if self.season_dict is not None:
            if self.verbose:
                print("Adding seasonality forcing")
            self._add_seasonality_forcing()
        else:
            print("No seasonality")

        # Add external forcing
        if self.forcing_dict is not None:
            if self.verbose:
                print("Adding external forcing")
            initial_data = self.data_field.copy()
            # Merge greenhouse gas and aerosol forcings into the simulation baseline.
            self._consume_radiative_forcing()
            # self._add_external_forcing()
            diff = self.data_field - initial_data
            print(f"Max change in data field: {diff.max()}")
            print(f"Mean change in data field: {diff.mean()}")
            print(f"Sample values after forcing applied:\n{diff[:, :5]}")

        # Compute the data
        if self.linearity == "linear":
            if self.verbose:
                print("Creating linear data")
            self._create_linear()
        elif self.linearity == "polynomial":
            if self.verbose:
                print("Creating polynomial data")
            self._create_polynomial()
        else:
            if self.verbose:
                print("Creating nonlinear data")
            if train_nnar:
                print("Training NNAR model before data generation...")
                self.train_nnar(num_epochs=50, learning_rate=0.001, batch_size=32)
            self._create_nonlinear()

    def generate_cov_noise_matrix(self) -> np.ndarray:
        """
        W in NxL data_field L times T.

        :return:
        """

        # Use n_climate_modes (not n_vars) since noise_weights only covers climate modes
        W = deepcopy(self.noise_weights).reshape(self.n_climate_modes, -1)
        print(f"noise_weights copied, {W.shape}")
        W_plus = np.linalg.pinv(W)
        print("noise_weights inverted")
        # Can we speed this up? since they are all np.eye
        cov = self.noise_strength * W_plus @ W_plus.transpose()  # + self.fast_noise_cov
        print("cov created inverted")

        return cov

    def _add_noise_field(self):

        if self.noise_cov is None:
            print("Generate covariance matrix")
            self.noise_cov = self.generate_cov_noise_matrix()
            self.noise_cov += 1e-6 * np.eye(self.noise_cov.shape[0])

        # Generate noise from cov
        print("Generate noise_data_field multivariate random")
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        mean_torch = torch.zeros(self.spatial_resolution, device=dev, dtype=dtype)
        cov = torch.tensor(self.noise_cov, device=dev, dtype=dtype)
        distrib = MultivariateNormal(loc=mean_torch, covariance_matrix=cov)  # . to(device="cuda")
        noise_data_field = distrib.sample(sample_shape=torch.Size([self.time_length + self.transient]))
        self.noise_data_field = noise_data_field.detach().cpu().numpy().transpose()

        # self.noise_data_field = np.random.multivariate_normal(mean=np.zeros(self.spatial_resolution), cov=self.noise_cov,
        #                                                       size=self.time_length + self.transient).transpose()

        self.data_field += self.noise_data_field

    def _add_seasonality_forcing(self):

        periods = self.season_dict["periods"]  # e.g. [12, 6, 3] for year, half-year, quarter-year
        amplitudes = self.season_dict["amplitudes"]  # same length as periods
        phases = self.season_dict.get("phases", [0.0] * len(periods))

        # year-to-year amplitude / phase jitter
        jitter_cfg = self.season_dict.get("yearly_jitter")  # None or dict
        base_P = periods[0]  # assume first is annual (12 months)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        T = self.time_length + self.transient
        ncy = math.ceil(T / base_P)  # # of whole cycles

        L = self.data_field.shape[0]
        T = self.time_length + self.transient
        t = torch.arange(T, device=dev, dtype=dtype)
        seasonal = torch.zeros((L, T), device=dev, dtype=dtype)

        σ_A = jitter_cfg["amplitude"] if jitter_cfg else 0.0
        σ_φ = jitter_cfg["phase"] if jitter_cfg else 0.0

        # allow vector inputs, default to identical values otherwise
        σ_Ak = torch.as_tensor(σ_A).expand(len(periods)).to(dtype=dtype, device=dev)
        σ_φk = torch.as_tensor(σ_φ).expand(len(periods)).to(dtype=dtype, device=dev)

        for k, (A, P, φ) in enumerate(zip(amplitudes, periods, phases)):
            # one jitter draw *per year* for this harmonic
            amp_noise_k = 1 + σ_Ak[k] * torch.randn(ncy, device=dev, dtype=dtype)
            phase_noise_k = σ_φk[k] * torch.randn(ncy, device=dev, dtype=dtype)

            amp_series_k = amp_noise_k.repeat_interleave(base_P)[:T]  # (T,)
            phase_series_k = phase_noise_k.repeat_interleave(base_P)[:T]  # (T,)

            seasonal += amp_series_k * A * torch.sin(2 * math.pi / P * (t + phase_series_k) + φ)

        w = self.season_dict.get("season_weight")
        if w is not None:
            if not torch.is_tensor(w):
                w = torch.as_tensor(w, dtype=dtype, device=dev)
            else:
                w = w.to(device=dev, dtype=dtype)
            if w.ndim > 1:
                w = w.reshape(-1)
            if w.numel() != L:
                raise ValueError(f"season_weight has length {w.numel()} but grid has {L} points")
            seasonal *= w.reshape(L, 1)

        seasonal_np = seasonal.cpu().numpy()
        self.seasonal_data_field = seasonal_np
        self.data_field += seasonal_np

    def _apply_season_forcing_interaction(self, forcing_field, interaction_cfg=None):
        """Modulate a forcing field by the seasonal cycle if requested."""
        if interaction_cfg is None:
            interaction_cfg = (self.forcing_dict or {}).get("season_interaction")

        if not interaction_cfg:
            return forcing_field

        if self.seasonal_data_field is None:
            logger.warning("season_interaction requested but seasonal_data_field is missing; skipping interaction")
            return forcing_field

        # Ensure forcing and seasonal fields share the same device / dtype for math
        np_dtype = None
        if torch.is_tensor(forcing_field):
            dev = forcing_field.device
            dtype = forcing_field.dtype
            forcing_tensor = forcing_field
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32
            forcing_tensor = torch.as_tensor(forcing_field, device=dev, dtype=dtype)
            np_dtype = getattr(forcing_field, "dtype", None)

        # Bring the precomputed seasonal cycle onto the same device so interactions are cheap.
        seasonal_tensor = torch.as_tensor(self.seasonal_data_field, device=dev, dtype=dtype)

        if seasonal_tensor.shape != forcing_tensor.shape:
            # Using mismatched grids would silently broadcast; guard it so users catch config errors.
            raise ValueError(
                f"seasonal_data_field shape {seasonal_tensor.shape} does not match forcing shape {forcing_tensor.shape}"
            )

        # Optional normalisation lets us work with comparable seasonal amplitudes
        eps = float(interaction_cfg.get("eps", 1e-6))
        norm = str(interaction_cfg.get("normalisation", "zscore")).lower()

        if norm == "zscore":
            # Remove the seasonal mean and scale by variance so anomalies are dimensionless.
            mean = seasonal_tensor.mean(dim=1, keepdim=True)
            std = seasonal_tensor.std(dim=1, keepdim=True)
            seasonal_tensor = (seasonal_tensor - mean) / (std + eps)
        elif norm == "minmax":
            # Stretch to [-0.5, 0.5] so the strength parameter is intuitive.
            s_min = seasonal_tensor.amin(dim=1, keepdim=True)
            s_max = seasonal_tensor.amax(dim=1, keepdim=True)
            seasonal_tensor = (seasonal_tensor - s_min) / (s_max - s_min + eps)
            seasonal_tensor = seasonal_tensor - 0.5
        elif norm in ("none", "identity"):
            pass
        else:
            raise ValueError(f"Unsupported season_interaction normalisation '{norm}'")

        # Apply the requested interaction mode to blend seasonality with forcing
        mode = str(interaction_cfg.get("mode", "multiplicative")).lower()
        strength = float(interaction_cfg.get("strength", 1.0))

        if mode == "multiplicative":
            # Seasonal anomalies rescale the forcing field, raising or lowering its amplitude.
            scale = 1.0 + strength * seasonal_tensor
            min_scale = interaction_cfg.get("min_scale")
            max_scale = interaction_cfg.get("max_scale")
            if min_scale is not None:
                scale = torch.clamp(scale, min=float(min_scale))
            if max_scale is not None:
                scale = torch.clamp(scale, max=float(max_scale))
            forcing_tensor = forcing_tensor * scale
        elif mode == "additive":
            # Inject the seasonal fluctuations directly as an additional perturbation.
            forcing_tensor = forcing_tensor + strength * seasonal_tensor
        elif mode == "hybrid":
            # Combine both: a multiplicative scaling plus an additive share (controlled via mix).
            mix = float(interaction_cfg.get("mix", 0.5))
            scale = 1.0 + strength * seasonal_tensor
            forcing_tensor = forcing_tensor * scale + mix * strength * seasonal_tensor
        else:
            raise ValueError(f"Unsupported season_interaction mode '{mode}'")

        # Final affine tweak so users can bias the modulation if desired
        bias = float(interaction_cfg.get("bias", 0.0))
        if bias != 0.0:
            forcing_tensor = forcing_tensor + bias

        if torch.is_tensor(forcing_field):
            return forcing_tensor

        result = forcing_tensor.detach().cpu().numpy()
        if np_dtype is not None:
            # Preserve the caller's dtype to avoid surprising precision changes.
            result = result.astype(np_dtype, copy=False)
        return result

    def _add_external_forcing(self):
        """
        Adds external forcing to the data field using PyTorch tensors for GPU acceleration.

        Allows for both linear and nonlinear ramps.
        """
        if self.forcing_dict is None:
            raise TypeError("Forcing dict is empty")

        w_f = deepcopy(self.forcing_dict.get("w_f"))
        f_1 = float(self.forcing_dict.get("f_1", 0))
        f_2 = float(self.forcing_dict.get("f_2", 0))
        f_time_1 = self.forcing_dict.get("f_time_1", 0)
        f_time_2 = self.forcing_dict.get("f_time_2", self.time_length)
        ramp_type = self.forcing_dict.get("ramp_type", "linear")  # Default to linear

        if w_f is None:
            w_f = deepcopy(self.mode_weights)
            w_f = (w_f != 0).astype(int)  # Convert non-zero elements to 1

        print(self.mode_weights.shape)
        # w_f = w_f / (w_f.max() + 1e-8)  # Normalize to range [0,1]

        # Merge last two dims first => shape (d_z, lat*lon)
        temp = w_f.reshape(w_f.shape[0], w_f.shape[1] * w_f.shape[2])
        # sum over dim=0 => shape (lat*lon,)

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        w_f_sum = torch.tensor(temp.sum(axis=0), dtype=dtype, device=dev)
        f_time_1 += self.transient
        f_time_2 += self.transient
        time_length = self.time_length + self.transient

        # Generate the forcing trend using torch tensors
        if ramp_type == "linear":
            ramp = torch.linspace(f_1, f_2, f_time_2 - f_time_1, dtype=dtype, device=dev)
        elif ramp_type == "quadratic":
            t = torch.linspace(0, 1, f_time_2 - f_time_1, dtype=dtype, device=dev)
            ramp = f_1 + (f_2 - f_1) * t**2
        elif ramp_type == "exponential":
            t = torch.linspace(0, 1, f_time_2 - f_time_1, dtype=torch.float32, device=dev)
            ramp = f_1 + (f_2 - f_1) * (torch.exp(t) - 1) / (torch.exp(torch.tensor(1.0)) - 1)
        elif ramp_type == "sigmoid":
            t = torch.linspace(-6, 6, f_time_2 - f_time_1, dtype=dtype, device=dev)
            ramp = f_1 + (f_2 - f_1) * (1 / (1 + torch.exp(-t)))
        elif ramp_type == "sinusoidal":
            t = torch.linspace(0, pi, f_time_2 - f_time_1, dtype=dtype, device=dev)
            ramp = f_1 + (f_2 - f_1) * (0.5 * (1 - torch.cos(t)))
        else:
            raise ValueError(
                "Unsupported ramp type. Choose from 'linear', 'quadratic', 'exponential', 'sigmoid', or 'sinusoidal'."
            )

        # Generate the forcing trend using torch tensors
        trend = torch.cat(
            [
                torch.full((f_time_1,), f_1, dtype=dtype, device=dev),
                ramp,
                torch.full((time_length - f_time_2,), f_2, dtype=dtype, device=dev),
            ]
        ).reshape(1, time_length)

        if w_f_sum.dim() == 2:
            w_f_sum = w_f_sum.sum(dim=0, keepdim=True)  # Sum across the correct dimension

        # Compute the forcing field on GPU
        forcing_field = (w_f_sum.reshape(1, -1) * trend.T).T
        # Optionally modulate the forcing by the seasonal cycle so ramp strength depends on time of year.
        forcing_field = self._apply_season_forcing_interaction(forcing_field)
        self.forcing_data_field = forcing_field.cpu().numpy()

        print(f"Using {ramp_type} ramp: f_1={f_1}, f_2={f_2}, f_time_1={f_time_1}, f_time_2={f_time_2}")

        print(f"Forcing data field mean: {self.forcing_data_field.mean()}")

        print(f"Before addition - Data field mean: {self.data_field.mean()}")

        self.data_field += self.forcing_data_field

        print(f"After addition - Data field mean: {self.data_field.mean()}")

    def create_co2_forcing(self) -> np.ndarray:
        """
        Create a CO2 forcing field that grows over time with mild spatial variability.

        Uses f_1, f_2, f_time_1, f_time_2, and ramp_type from forcing_dict to control the temporal evolution of CO2
        forcing.

        Returns an array shaped (spatial_resolution, time_length + transient) that can be added to the synthetic field
        or used as an external driver.
        """

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        time_len = self.time_length + self.transient
        if time_len <= 0:
            raise ValueError("Time length including transient must be positive")

        spatial_len = self.spatial_resolution
        if spatial_len <= 0:
            raise ValueError("Spatial resolution must be positive")

        # Get forcing parameters from forcing_dict
        forcing_cfg = self.forcing_dict or {}
        f_1 = float(forcing_cfg.get("f_1", 0.0))
        f_2 = float(forcing_cfg.get("f_2", 0.1))
        f_time_1 = int(forcing_cfg.get("f_time_1", 0))
        f_time_2 = int(forcing_cfg.get("f_time_2", time_len))
        ramp_type = forcing_cfg.get("ramp_type", "linear")

        # Adjust times to include transient period
        f_time_1 += self.transient
        f_time_2 += self.transient

        # Generate the forcing trend using the specified ramp type
        if ramp_type == "linear":
            ramp = torch.linspace(f_1, f_2, f_time_2 - f_time_1, dtype=dtype, device=dev)
        elif ramp_type == "quadratic":
            t = torch.linspace(0, 1, f_time_2 - f_time_1, dtype=dtype, device=dev)
            ramp = f_1 + (f_2 - f_1) * t**2
        elif ramp_type == "exponential":
            t = torch.linspace(0, 1, f_time_2 - f_time_1, dtype=dtype, device=dev)
            ramp = f_1 + (f_2 - f_1) * (torch.exp(t) - 1) / (torch.exp(torch.tensor(1.0)) - 1)
        elif ramp_type == "sigmoid":
            t = torch.linspace(-6, 6, f_time_2 - f_time_1, dtype=dtype, device=dev)
            ramp = f_1 + (f_2 - f_1) * (1 / (1 + torch.exp(-t)))
        elif ramp_type == "sinusoidal":
            from math import pi

            t = torch.linspace(0, pi, f_time_2 - f_time_1, dtype=dtype, device=dev)
            ramp = f_1 + (f_2 - f_1) * (0.5 * (1 - torch.cos(t)))
        else:
            raise ValueError(
                f"Unsupported ramp type '{ramp_type}'. Choose from 'linear', 'quadratic', 'exponential', 'sigmoid', or 'sinusoidal'."
            )

        # Construct the full temporal trend
        co2_trend = torch.cat(
            [
                torch.full((f_time_1,), f_1, dtype=dtype, device=dev),
                ramp,
                torch.full((time_len - f_time_2,), f_2, dtype=dtype, device=dev),
            ]
        )

        logger.info(
            f"CO2 forcing: Using {ramp_type} ramp from f_1={f_1} to f_2={f_2} "
            f"(time {f_time_1} to {f_time_2}, total length {time_len})"
        )

        print(
            f"CO2 forcing: Using {ramp_type} ramp from f_1={f_1} to f_2={f_2} "
            f"(time {f_time_1} to {f_time_2}, total length {time_len})"
        )

        if spatial_len == 1:
            base_pattern = torch.ones(1, device=dev, dtype=dtype)
        else:
            coords = torch.linspace(-1.0, 1.0, spatial_len, device=dev, dtype=dtype)
            num_lobes = max(3, min(8, spatial_len // 64 + 3))
            rand_params = torch.as_tensor(np.random.rand(num_lobes, 3), device=dev, dtype=dtype)
            centers = rand_params[:, 0] * 2.0 - 1.0
            widths = 0.3 + rand_params[:, 1] * 0.7
            amplitudes = 0.3 + rand_params[:, 2] * 0.7

            diff = coords.unsqueeze(0) - centers.unsqueeze(1)
            gaussians = torch.exp(-0.5 * (diff / widths.unsqueeze(1)) ** 2)
            base_pattern = (amplitudes.unsqueeze(1) * gaussians).sum(dim=0)
            base_pattern = base_pattern / (base_pattern.mean() + 1e-6)

        spatial_pattern = base_pattern
        spatial_pattern = spatial_pattern / (spatial_pattern.mean() + 1e-8)

        # Add a deterministic oscillation so the grid is not uniform.
        if spatial_len > 1:
            idx = torch.linspace(0.0, 2.0 * math.pi, spatial_len, device=dev, dtype=dtype)
            spatial_pattern = spatial_pattern * (1.0 + 0.1 * torch.sin(idx))

        spatial_pattern = torch.clamp(spatial_pattern, min=0.05)
        forcing = spatial_pattern.unsqueeze(1) * co2_trend.unsqueeze(0)

        forcing_np = forcing.detach().cpu().numpy()

        return forcing_np

    def create_aerosol_forcing(self) -> np.ndarray:
        """
        Create an aerosol forcing field with distinct temporal dynamics per region.

        Each spatial region (corresponding to an aerosol latent) has a staggered temporal envelope with unique timing
        and frequency modulation. This ensures aerosol latents are distinguishable for causal discovery.

        Uses aerosol_ramp_up_time, aerosol_peak_time, and aerosol_decline_time from forcing_dict as base timing, with
        staggered offsets per region.
        """

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        time_len = self.time_length + self.transient
        if time_len <= 0:
            raise ValueError("Time length including transient must be positive")

        spatial_len = self.spatial_resolution
        if spatial_len <= 0:
            raise ValueError("Spatial resolution must be positive")

        forcing_cfg = self.forcing_dict or {}
        aerosol_scale = float(forcing_cfg.get("aerosol_scale", 0.03))
        aerosol_contrast = float(forcing_cfg.get("aerosol_spatial_contrast", 1.05))
        aerosol_scale = max(aerosol_scale, 0.0)
        aerosol_contrast = max(aerosol_contrast, 0.5)

        # Get aerosol temporal parameters (base timing, independent from CO2)
        base_ramp_up_time = int(forcing_cfg.get("aerosol_ramp_up_time", int(0.2 * self.time_length)))
        base_peak_time = int(forcing_cfg.get("aerosol_peak_time", int(0.6 * self.time_length)))
        base_decline_time = int(forcing_cfg.get("aerosol_decline_time", int(0.85 * self.time_length)))

        # Stagger fraction: how much to spread timing across latents (default 30% of time span)
        timing_stagger = float(forcing_cfg.get("aerosol_timing_stagger", 0.3))

        # Adjust base times to include transient period
        base_ramp_up_time += self.transient
        base_peak_time += self.transient
        base_decline_time += self.transient

        # Time vector
        t = torch.linspace(0.0, 1.0, time_len, device=dev, dtype=dtype)

        # Generate spatial pattern (used to modulate amplitude within each region)
        if spatial_len == 1:
            spatial_pattern = torch.ones(1, device=dev, dtype=dtype)
        else:
            coords = torch.linspace(-1.0, 1.0, spatial_len, device=dev, dtype=dtype)
            num_lobes = max(3, min(8, spatial_len // 64 + 3))
            rand_params = torch.as_tensor(np.random.rand(num_lobes, 3), device=dev, dtype=dtype)
            centers = rand_params[:, 0] * 2.0 - 1.0
            widths = 0.3 + rand_params[:, 1] * 0.7
            amplitudes = 0.3 + rand_params[:, 2] * 0.7

            diff = coords.unsqueeze(0) - centers.unsqueeze(1)
            gaussians = torch.exp(-0.5 * (diff / widths.unsqueeze(1)) ** 2)
            spatial_pattern = (amplitudes.unsqueeze(1) * gaussians).sum(dim=0)
            spatial_pattern = spatial_pattern / (spatial_pattern.mean() + 1e-6)

            idx = torch.linspace(0.0, 2.0 * math.pi, spatial_len, device=dev, dtype=dtype)
            regional_variability = 0.6 + 0.4 * torch.sin(idx * 3.0 + 0.8)
            hemispheric_gradient = 0.8 + 0.2 * torch.cos(idx)
            spatial_pattern = spatial_pattern * regional_variability * hemispheric_gradient

        spatial_pattern = spatial_pattern.pow(aerosol_contrast)
        spatial_pattern = spatial_pattern / (spatial_pattern.abs().mean() + 1e-8)

        # Get number of aerosol latents from forcing_indices
        n_aerosol_latents = len(self.forcing_indices.get("aerosol", [])) if self.forcing_indices else 1
        n_aerosol_latents = max(n_aerosol_latents, 1)  # At least 1 region
        region_size = spatial_len // n_aerosol_latents

        # Create forcing with distinct temporal dynamics per region
        forcing = torch.zeros((spatial_len, time_len), device=dev, dtype=dtype)

        logger.info(
            f"Aerosol forcing: Creating {n_aerosol_latents} distinct temporal patterns "
            f"(base_ramp={base_ramp_up_time}, base_peak={base_peak_time}, base_decline={base_decline_time}, "
            f"timing_stagger={timing_stagger}, scale={aerosol_scale})"
        )
        print(
            f"Aerosol forcing: Creating {n_aerosol_latents} distinct temporal patterns " f"(stagger={timing_stagger})"
        )

        for i in range(n_aerosol_latents):
            # Stagger timing for each latent to create distinct temporal patterns
            offset_fraction = i / n_aerosol_latents

            # Each latent has progressively later timing
            latent_ramp_up = base_ramp_up_time + int(offset_fraction * timing_stagger * time_len)
            latent_peak = base_peak_time + int(offset_fraction * timing_stagger * 0.7 * time_len)
            latent_decline = base_decline_time + int(offset_fraction * timing_stagger * 0.5 * time_len)

            # Clamp to valid range
            latent_peak = min(latent_peak, time_len - 100)
            latent_decline = min(latent_decline, time_len - 10)

            # Normalized time positions for this latent's envelope
            t_ramp = latent_ramp_up / time_len
            t_peak = latent_peak / time_len
            t_decline = latent_decline / time_len

            # Create envelope: sharp rise and decline using sigmoid
            ramp_up = torch.sigmoid((t - t_ramp) * 10.0 / (t_peak - t_ramp + 1e-6))
            ramp_down = torch.sigmoid((t_decline - t) * 10.0 / (t_decline - t_peak + 1e-6))
            envelope = ramp_up * ramp_down

            # Add unique frequency modulation per latent (makes temporal patterns distinguishable)
            base_freq = 3.0 + i * 2.0  # Different base frequency per latent
            freq_mod = 1.0 + 0.2 * torch.sin(2.0 * math.pi * base_freq * t + i * 0.5)

            # Episodic variations unique to each latent
            latent_seasonal = torch.sin(2.0 * math.pi * (6.0 + i) * t)
            latent_bursts = torch.sin(2.0 * math.pi * (18.0 + i * 3) * t + 0.3 * i)

            # Combine into temporal trend for this latent
            latent_trend = -aerosol_scale * envelope * freq_mod * (1.0 + 0.15 * latent_seasonal + 0.05 * latent_bursts)

            # Apply to this region's spatial locations
            start_idx = i * region_size
            end_idx = start_idx + region_size if i < n_aerosol_latents - 1 else spatial_len
            region_pattern = spatial_pattern[start_idx:end_idx]

            # Each spatial location in the region gets this latent's temporal pattern
            forcing[start_idx:end_idx] = region_pattern.unsqueeze(1) * latent_trend.unsqueeze(0)

            print(
                f"  Latent {i}: ramp_up={latent_ramp_up}, peak={latent_peak}, decline={latent_decline}, freq={base_freq}"
            )

        forcing_np = forcing.detach().cpu().numpy()

        return forcing_np

    def _apply_causal_forcing(self, co2_forcing: np.ndarray, aerosol_forcing: np.ndarray) -> np.ndarray:
        """
        Apply forcing through the causal structure defined in links_coeffs.

        Instead of uniform forcing, each climate mode receives forcing contributions
        weighted by the causal coefficients. This creates the ground truth causal
        relationship between forcings and climate modes.

        Args:
            co2_forcing: CO2 forcing field, shape (spatial_resolution, time_length + transient)
            aerosol_forcing: Aerosol forcing field, same shape

        Returns:
            combined_contrib: Total forcing contribution to add to data_field
        """
        time_len = co2_forcing.shape[1]
        combined_contrib = np.zeros_like(co2_forcing)

        # Get mode weights for projecting back to observation space
        mode_weights = self.mode_weights.reshape(self.n_climate_modes, -1)  # (n_modes, spatial)

        # Create forcing latent trajectories
        # CO2: Use spatial mean as the CO2 latent signal (it's meant to be global)
        co2_latent = co2_forcing.mean(axis=0)  # shape: (time,)
        self.co2_latent_trajectory = co2_latent

        # Aerosol: Project to latent space using mode_weights
        # Each aerosol latent corresponds to how aerosol affects each spatial region
        # We create n_aerosol_latents trajectories by dividing space into regions
        n_aerosol_latents = len(self.forcing_indices.get("aerosol", []))
        if n_aerosol_latents > 0:
            # Divide spatial domain into regions for each aerosol latent
            region_size = self.spatial_resolution // n_aerosol_latents
            aerosol_latents = np.zeros((n_aerosol_latents, time_len))
            for i in range(n_aerosol_latents):
                start_idx = i * region_size
                end_idx = start_idx + region_size if i < n_aerosol_latents - 1 else self.spatial_resolution
                aerosol_latents[i] = aerosol_forcing[start_idx:end_idx].mean(axis=0)
            self.aerosol_latent_trajectory = aerosol_latents
        else:
            self.aerosol_latent_trajectory = None

        # Apply CO2 → mode causal contributions
        for mode_idx, links in self.forcing_coeffs["co2_to_modes"].items():
            mode_contrib = np.zeros(time_len)
            for forcing_idx, lag, coeff in links:
                # Apply time lag (lag is negative in links_coeffs convention)
                shift = abs(lag)
                if shift < time_len:
                    # forcing at t-shift affects mode at t
                    mode_contrib[shift:] += coeff * co2_latent[:-shift] if shift > 0 else coeff * co2_latent

            # Project mode contribution to observation space
            mode_weight = mode_weights[mode_idx]  # (spatial,)
            combined_contrib += np.outer(mode_weight, mode_contrib)

        # Apply Aerosol → mode causal contributions
        if self.aerosol_latent_trajectory is not None:
            aerosol_idx_offset = self.n_climate_modes + len(self.forcing_indices.get("co2", []))
            for mode_idx, links in self.forcing_coeffs["aerosol_to_modes"].items():
                mode_contrib = np.zeros(time_len)
                for forcing_idx, lag, coeff in links:
                    # Map forcing_idx to aerosol latent index
                    aerosol_latent_idx = forcing_idx - aerosol_idx_offset
                    if 0 <= aerosol_latent_idx < n_aerosol_latents:
                        shift = abs(lag)
                        if shift < time_len:
                            aerosol_signal = self.aerosol_latent_trajectory[aerosol_latent_idx]
                            mode_contrib[shift:] += (
                                coeff * aerosol_signal[:-shift] if shift > 0 else coeff * aerosol_signal
                            )

                # Project mode contribution to observation space
                mode_weight = mode_weights[mode_idx]
                combined_contrib += np.outer(mode_weight, mode_contrib)

        # Apply seasonal interaction
        combined_contrib = self._apply_season_forcing_interaction(combined_contrib)

        return combined_contrib

    def _consume_radiative_forcing(self) -> None:
        """
        Combine CO2 and aerosol forcing fields and inject them into the data field.

        If causal coefficients are available (forcing_coeffs), the forcing is applied
        through the causal structure: each climate mode receives forcing contributions
        weighted by the causal coefficients from links_coeffs.
        """
        if self.data_field is None:
            raise ValueError("Data field must be initialised before applying forcing")

        co2_forcing = self.create_co2_forcing()
        aerosol_forcing = self.create_aerosol_forcing()

        if co2_forcing.shape != self.data_field.shape:
            raise ValueError("CO2 forcing shape mismatch with data field")
        if aerosol_forcing.shape != self.data_field.shape:
            raise ValueError("Aerosol forcing shape mismatch with data field")

        if self.forcing_data_field is None:
            self.forcing_data_field = np.zeros_like(self.data_field)
        if self.co2_forcing_data_field is None:
            self.co2_forcing_data_field = np.zeros_like(self.data_field)
        if self.aerosol_forcing_data_field is None:
            self.aerosol_forcing_data_field = np.zeros_like(self.data_field)

        # Store separate CO2 and aerosol forcings (for model training)
        np.add(self.co2_forcing_data_field, co2_forcing, out=self.co2_forcing_data_field)
        np.add(self.aerosol_forcing_data_field, aerosol_forcing, out=self.aerosol_forcing_data_field)

        # If we have causal coefficients, apply forcing through the causal structure
        if self.forcing_coeffs is not None and self.forcing_indices is not None:
            combined_contrib = self._apply_causal_forcing(co2_forcing, aerosol_forcing)
            print("Applied forcing through causal structure")
        else:
            # Legacy behavior: uniform addition
            combined_contrib = co2_forcing.copy()
            np.add(combined_contrib, aerosol_forcing, out=combined_contrib)
            combined_contrib = self._apply_season_forcing_interaction(combined_contrib)

        # Accumulate the forcing
        np.add(self.forcing_data_field, combined_contrib, out=self.forcing_data_field)

        # Feed the combined forcing into the simulator state
        np.add(self.data_field, combined_contrib, out=self.data_field)

        instantaneous_mean = combined_contrib.mean(axis=0)
        time_index = np.arange(1, instantaneous_mean.shape[-1] + 1, dtype=float)
        cumulative_mean = np.cumsum(instantaneous_mean, axis=-1) / time_index

        logger.info(
            "Forcing diagnostics: instant mean=%.4f +/- %.4f, cumulative mean=%.4f +/- %.4f",
            float(instantaneous_mean.mean()),
            float(instantaneous_mean.std()),
            float(cumulative_mean.mean()),
            float(cumulative_mean.std()),
        )

        # Save forcing diagnostics plot to the SAVAR dataset directory
        if self.output_save_dir is not None:
            plot_path = Path(self.output_save_dir) / "forcing_diagnostics.png"
            print(f"Saving forcing diagnostics to: {plot_path}")
            self._plot_mean_forcing(instantaneous_mean, cumulative_mean, plot_path)

    def _plot_mean_forcing(self, instantaneous_mean: np.ndarray, cumulative_mean: np.ndarray, output_path: str) -> None:
        """Persist diagnostic plots summarising the applied radiative forcing."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available; skipping forcing diagnostics plot at %s", output_path)
            return

        time_steps = np.arange(instantaneous_mean.shape[-1])
        fig, ax = plt.subplots()
        ax.plot(time_steps, instantaneous_mean, label="Instantaneous mean")
        ax.plot(time_steps, cumulative_mean, label="Cumulative mean")
        ax.set_title("Radiative forcing mean over time")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Mean forcing")
        ax.legend(loc="best")
        fig.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _create_linear(self):
        """Weights N \times L data_field L \times T."""
        weights = deepcopy(self.mode_weights.reshape(self.n_climate_modes, -1))
        # weights_inv = np.linalg.pinv(weights)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_inv = torch.Tensor(np.linalg.pinv(weights)).to(device=dev)
        weights = torch.Tensor(weights).to(device=dev)
        time_len = deepcopy(self.time_length)
        time_len += self.transient
        tau_max = self.tau_max

        # phi = dict_to_matrix(self.links_coeffs)
        phi_full = torch.Tensor(dict_to_matrix(self.links_coeffs)).to(device=dev)
        # Only use climate mode dynamics (forcing is applied separately via _apply_causal_forcing)
        phi = phi_full[: self.n_climate_modes, : self.n_climate_modes, :]
        # data_field = deepcopy(self.data_field)
        data_field = torch.Tensor(self.data_field).to(device=dev)

        print("create_linear")
        for t in tqdm(range(tau_max, time_len)):
            for i in range(tau_max):
                data_field[..., t : t + 1] += weights_inv @ phi[..., i] @ weights @ data_field[..., t - 1 - i : t - i]
                # data_field[..., t:t + 1] += torch.matmul(torch.matmul(torch.matmul(weights_inv, phi[..., i]), weights), data_field[..., t - 1 - i:t - i])

        self.data_field = data_field[..., self.transient :].detach().cpu().numpy()

    def train_nnar(self, num_epochs=50, learning_rate=0.001, batch_size=32):
        """
        Method for training a very simple single-layer neural network with sigmoid activation (one neuron).

        We train it here on pairs (past_values, future_value), but this can be adapted as needed.
        """

        # A trivial net:  data_in -> [Linear] -> [Sigmoid] -> data_out
        self.nnar_model = nn.Sequential(nn.Linear(self.spatial_resolution, self.spatial_resolution), nn.Sigmoid()).to(
            "cuda"
        )

        optimizer = torch.optim.Adam(self.nnar_model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        # Create a training dataset from self.data_field: each sample is (X_t, X_{t+1}),
        # (we might later incorporate more lags)

        # collect input-output pairs:
        X = torch.from_numpy(self.data_field[:, :-1].T).float().to("cuda")
        Y = torch.from_numpy(self.data_field[:, 1:].T).float().to("cuda")
        dataset_size = X.shape[0]

        # Simple mini-batch loop
        for epoch in range(num_epochs):
            perm = torch.randperm(dataset_size, device="cuda")
            batch_losses = []

            for i in range(0, dataset_size, batch_size):
                idx = perm[i : i + batch_size]
                x_batch = X[idx]
                y_batch = Y[idx]

                # forward pass
                pred = self.nnar_model(x_batch)
                loss = loss_fn(pred, y_batch)
                batch_losses.append(loss.item())

                # backward + update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {sum(batch_losses)/len(batch_losses):.6f}")

        print("Training of single-layer NNAR model completed.")

    def _create_nonlinear(self):
        """
        Generates nonlinear data by applying a (trained or simple) nonlinearity at each time step. This method uses the
        same logic as _create_linear to step forward in time and adds the nonlinearity (sigmoid) before adding to
        data_field.

        If train_nnar=True was set, we assume self.nnar_model was trained in generate_data().
        Otherwise, we can do a direct inline "torch.sigmoid(...)" approach.
        Can be increased in complexity if needed
        """

        weights = torch.Tensor(np.linalg.pinv(self.mode_weights.reshape(self.n_climate_modes, -1))).to("cuda")
        phi_full = torch.Tensor(dict_to_matrix(self.links_coeffs)).to("cuda")
        # Only use climate mode dynamics (forcing is applied separately)
        phi = phi_full[: self.n_climate_modes, : self.n_climate_modes, :]
        mode_weights_tensor = torch.Tensor(self.mode_weights.reshape(self.n_climate_modes, -1)).to("cuda")
        data_field = torch.Tensor(self.data_field).to("cuda")

        time_len = self.time_length + self.transient
        tau_max = self.tau_max

        print("create_nonlinear (single-layer net + sigmoid)")

        for t in tqdm(range(tau_max, time_len)):
            # Sum up influences from each lag
            nonlinear_contrib = 0.0
            for i in range(tau_max):
                # get linear combination as in _create_linear
                lincombo = weights @ phi[..., i] @ mode_weights_tensor @ data_field[..., (t - 1 - i) : (t - i)]
                # Apply a sigmoid (or feed it through the small neural net if you want more complexity)
                lincombo_nl = torch.sigmoid(lincombo)
                # accumulate
                nonlinear_contrib += lincombo_nl.squeeze(-1)

            # Add the (nonlinear) effect to the data field at time t
            data_field[:, t] += nonlinear_contrib

        self.data_field = data_field[:, self.transient :].detach().cpu().numpy()

    def _create_polynomial(self):
        """Example polynomial autoregression, e.g. x^2 for poly_degree=2."""
        w_np = np.linalg.pinv(self.mode_weights.reshape(self.n_climate_modes, -1))
        phi_np_full = dict_to_matrix(self.links_coeffs)
        # Only use climate mode dynamics (forcing is applied separately)
        phi_np = phi_np_full[: self.n_climate_modes, : self.n_climate_modes, :]

        # choose GPU if available, else CPU — and use float32 everywhere
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        w_torch = torch.tensor(w_np, device=dev, dtype=dtype)
        phi_torch = torch.tensor(phi_np, device=dev, dtype=dtype)
        mw_torch = torch.tensor(
            self.mode_weights.reshape(self.n_climate_modes, -1),
            device=dev,
            dtype=dtype,
        )
        data_field = torch.tensor(self.data_field, device=dev, dtype=dtype)

        time_len = self.time_length + self.transient
        tau_max = self.tau_max

        print(f"create_polynomial with degrees={self.poly_degrees}")

        for t in tqdm(range(tau_max, time_len)):
            # For each time step, sum over the contributions of all lags
            for i in range(tau_max):
                lincombo = w_torch @ phi_torch[..., i] @ mw_torch @ data_field[..., (t - 1 - i) : (t - i)]

                # For each requested polynomial degree, add its effect
                poly_sum = torch.zeros_like(lincombo)
                for deg in self.poly_degrees:
                    poly_sum += lincombo**deg

                data_field[:, t] += poly_sum.squeeze(-1)

        self.data_field = data_field[:, self.transient :].detach().cpu().numpy()

    def _create_intervened_nextstep(self, input_data, intervened_mode=None, intervention_value=None, intervened_t=None):
        """
        Not tested yet!!!

        input_data are the tau timesteps that get intervened on
        at mode intervened_mode, with value +intervention_value, at timestep intervened_t

        input_data is here of shape `self.spatial_resolution * self.time_length`.
        This is to keep the savar structure similar to the one of `self.data_field`
        """

        weights = deepcopy(self.mode_weights.reshape(self.n_climate_modes, -1))
        # weights_inv = np.linalg.pinv(weights)
        weights_inv = torch.Tensor(np.linalg.pinv(weights)).to(device="cuda")
        weights = torch.Tensor(weights).to(device="cuda")
        tau = input_data.shape[1]

        # phi = dict_to_matrix(self.links_coeffs)
        phi_full = torch.Tensor(dict_to_matrix(self.links_coeffs)).to(device="cuda")
        # Only use climate mode dynamics (forcing is applied separately via _apply_causal_forcing)
        phi = phi_full[: self.n_climate_modes, : self.n_climate_modes, :]
        # data_field = deepcopy(self.data_field)
        next_step = torch.zeros(self.spatial_resolution).to(device="cuda")

        # perform intervention
        input_data[
            intervened_mode * self.spatial_resolution : (intervened_mode + 1) * self.spatial_resolution, intervened_t
        ] += intervention_value

        for i in range(tau):
            next_step += weights_inv @ phi[..., i] @ weights @ input_data[..., tau - 1 - i : tau - i]

        return next_step
