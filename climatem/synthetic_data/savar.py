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
        "season_dict",
        "data_field",
        "noise_data_field",
        "seasonal_data_field",
        "forcing_data_field",
        "linearity",
        "poly_degrees",
        "verbose",
        "model_seed",
        "nnar_model",
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
        linearity: str = "linear",
        poly_degrees: List[int] = [2],
        season_dict: dict = None,
        data_field: np.ndarray = None,
        noise_data_field: np.ndarray = None,
        seasonal_data_field: np.ndarray = None,
        forcing_data_field: np.ndarray = None,
        verbose: bool = False,
        model_seed: int = None,
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
        self.season_dict = season_dict
        self.linearity = linearity
        self.poly_degrees = poly_degrees

        self.data_field = data_field

        self.verbose = verbose
        self.model_seed = model_seed

        # Computed attributes
        print("Creating attributes")
        self.n_vars = len(links_coeffs)
        self.tau_max = max(abs(lag) for (_, lag), _ in it.chain.from_iterable(self.links_coeffs.values()))
        self.spatial_resolution = deepcopy(self.mode_weights.reshape(self.n_vars, -1).shape[1])
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

        if np.random is not None:
            np.random.seed(model_seed)

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

        W = deepcopy(self.noise_weights).reshape(self.n_vars, -1)
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

        # # Convert tensors to numpy for plotting if necessary
        # if isinstance(w_f_sum, torch.Tensor):
        #     w_f_sum = w_f_sum.cpu().numpy()
        # if isinstance(forcing_field, torch.Tensor):
        #     forcing_field = forcing_field.cpu().numpy()
        # if isinstance(data_field_before, torch.Tensor):
        #     data_field_before = data_field_before.cpu().numpy()
        # if isinstance(data_field_after, torch.Tensor):
        #     data_field_after = data_field_after.cpu().numpy()

        # # Compute mean values over spatial dimensions
        # mean_forcing = forcing_field.mean(axis=0)
        # mean_data_before = data_field_before.mean(axis=0)
        # mean_data_after = data_field_after.mean(axis=0)

        # # Plot 1: Mean Forcing over Time
        # plt.figure(figsize=(10, 4))
        # plt.plot(range(time_length), mean_forcing, label="Mean Forcing", color="blue")
        # plt.axvline(x=f_time_1, linestyle="--", color="gray", label="Start Forcing")
        # plt.axvline(x=f_time_2, linestyle="--", color="gray", label="End Forcing")
        # plt.xlabel("Time Steps")
        # plt.ylabel("Forcing Intensity")
        # plt.title("Evolution of External Forcing Over Time")
        # plt.legend()
        # plt.grid()
        # plt.savefig(f"mean_forcing_over_time_{f_1}_{f_2}_{ramp_type}.png")  # Save to a file
        # plt.close()

        # # Plot 2: Mean Data Before and After Forcing
        # plt.figure(figsize=(10, 4))
        # plt.plot(range(time_length), mean_data_before, label="Data Before Forcing", color="red", linestyle="dashed")
        # plt.plot(range(time_length), mean_data_after, label="Data After Forcing", color="green")
        # plt.axvline(x=f_time_1, linestyle="--", color="gray", label="Start Forcing")
        # plt.axvline(x=f_time_2, linestyle="--", color="gray", label="End Forcing")
        # plt.xlabel("Time Steps")
        # plt.ylabel("Mean Data Value")
        # plt.title("Effect of Forcing on Data Field")
        # plt.legend()
        # plt.grid()
        # plt.savefig(f"mean_data_before_after_forcing_{f_1}_{f_2}_{ramp_type}.png")  # Save to a file
        # plt.close()

    def create_co2_forcing(self) -> np.ndarray:
        """
        Create a CO2 forcing field that grows over time with mild spatial variability.

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

        # Smoothly accelerating increase to mimic CO2 growth over time.
        t = torch.linspace(0.0, 1.0, time_len, device=dev, dtype=dtype)
        co2_trend = t.pow(1.5)

        # Base spatial pattern derived from mode weights (falls back to ones).
        spatial_pattern = torch.ones(spatial_len, device=dev, dtype=dtype)
        if self.mode_weights is not None:
            mw = torch.as_tensor(self.mode_weights, device=dev, dtype=dtype)
            spatial_pattern = mw.reshape(self.n_vars, -1).abs().mean(dim=0)
            spatial_pattern = spatial_pattern / (spatial_pattern.mean() + 1e-8)

        # Add a deterministic oscillation so the grid is not uniform.
        if spatial_len > 1:
            idx = torch.linspace(0.0, 2.0 * math.pi, spatial_len, device=dev, dtype=dtype)
            spatial_pattern = spatial_pattern * (1.0 + 0.1 * torch.sin(idx))

        spatial_pattern = torch.clamp(spatial_pattern, min=0.05)
        forcing = spatial_pattern.unsqueeze(1) * co2_trend.unsqueeze(0)

        forcing_np = forcing.detach().cpu().numpy()

        self._maybe_plot_co2_forcing(spatial_pattern, forcing_np, spatial_len)

        return forcing_np

    def _maybe_plot_co2_forcing(self, spatial_pattern: torch.Tensor, forcing_np: np.ndarray, spatial_len: int) -> None:
        diagnostics_cfg = (self.forcing_dict or {}).get("diagnostics", {})
        if not diagnostics_cfg.get("co2_plots", True):
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib import animation
        except ImportError:
            logger.warning("matplotlib not available; skipping CO2 forcing visualisations")
            return

        output_dir = Path("/hkfs/work/workspace_haic/scratch/qa4548-climate_ws/SAVAR_DATA_TEST/co2_forcing_diagnostics")
        output_dir.mkdir(parents=True, exist_ok=True)

        spatial_pattern_np = spatial_pattern.detach().cpu().numpy()

        fig, ax = plt.subplots()
        ax.plot(np.arange(spatial_len), spatial_pattern_np, color="tab:red")
        ax.set_title("CO2 forcing spatial pattern")
        ax.set_xlabel("Grid point")
        ax.set_ylabel("Relative intensity")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "co2_forcing_spatial_pattern.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots()
        im = ax.imshow(forcing_np, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title("CO2 forcing over space and time")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Grid point")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / "co2_forcing_heatmap.png", dpi=150)
        plt.close(fig)

        time_axis = np.arange(forcing_np.shape[1])
        mean_intensity = forcing_np.mean(axis=0)
        cumulative_intensity = np.cumsum(mean_intensity)

        fig, ax1 = plt.subplots()
        (line_immediate,) = ax1.plot(time_axis, mean_intensity, color="tab:blue")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Average intensity")
        ax1.grid(alpha=0.2)

        ax2 = ax1.twinx()
        (line_cumulative,) = ax2.plot(time_axis, cumulative_intensity, color="tab:orange")
        ax2.set_ylabel("Cumulative intensity", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax1.set_title("CO2 forcing timeline")
        ax1.legend((line_immediate, line_cumulative), ("Immediate CO2", "Cumulative CO2"), loc="upper left")
        fig.tight_layout()
        fig.savefig(output_dir / "co2_forcing_timeline.png", dpi=150)
        plt.close(fig)

        grid_shape = None
        if self.mode_weights is not None:
            mw_shape = tuple(self.mode_weights.shape)
            if len(mw_shape) >= 3:
                grid_shape = mw_shape[-2:]
            elif len(mw_shape) == 2:
                grid_shape = (1, mw_shape[-1])
        if grid_shape and np.prod(grid_shape) != spatial_len:
            grid_shape = None

        max_frames_cfg = diagnostics_cfg.get("co2_animation_max_frames", 120)
        try:
            max_frames = int(max_frames_cfg)
        except (TypeError, ValueError):
            max_frames = 120
        max_frames = max(1, max_frames)

        frame_stride = max(1, forcing_np.shape[1] // max_frames)
        frame_indices = np.arange(0, forcing_np.shape[1], frame_stride, dtype=int)
        if frame_indices.size == 0 or frame_indices[-1] != forcing_np.shape[1] - 1:
            frame_indices = np.append(frame_indices, forcing_np.shape[1] - 1)

        if grid_shape:
            grid_series = forcing_np.reshape(*grid_shape, forcing_np.shape[1])
            vmin = float(grid_series.min())
            vmax = float(grid_series.max())
            if vmin == vmax:
                vmax = vmin + 1e-6
            fig, ax = plt.subplots()
            im = ax.imshow(grid_series[..., frame_indices[0]], vmin=vmin, vmax=vmax, animated=True)
            ax.set_title("CO2 forcing progression")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            def update(idx):
                frame = frame_indices[idx]
                im.set_array(grid_series[..., frame])
                ax.set_title(f"CO2 forcing progression (t={frame})")
                return (im,)

        else:
            y_min = float(forcing_np.min())
            y_max = float(forcing_np.max())
            if y_min == y_max:
                y_max = y_min + 1e-6
            fig, ax = plt.subplots()
            ax.set_title("CO2 forcing progression")
            ax.set_xlabel("Grid point")
            ax.set_ylabel("Forcing")
            (line,) = ax.plot(np.arange(spatial_len), forcing_np[:, frame_indices[0]])
            ax.set_ylim(y_min, y_max)

            def update(idx):
                frame = frame_indices[idx]
                line.set_ydata(forcing_np[:, frame])
                ax.set_title(f"CO2 forcing progression (t={frame})")
                return (line,)

        anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=80, blit=True)

        fps_cfg = diagnostics_cfg.get("co2_animation_fps", 10)
        try:
            fps = int(fps_cfg)
        except (TypeError, ValueError):
            fps = 10
        fps = max(1, fps)

        writer = animation.PillowWriter(fps=fps)
        anim.save(output_dir / "co2_forcing.gif", writer=writer)
        plt.close(fig)

    def _maybe_plot_aerosol_forcing(
        self, spatial_pattern: torch.Tensor, forcing_np: np.ndarray, spatial_len: int
    ) -> None:
        diagnostics_cfg = (self.forcing_dict or {}).get("diagnostics", {})
        if not diagnostics_cfg.get("aerosol_plots", True):
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib import animation
        except ImportError:
            logger.warning("matplotlib not available; skipping aerosol forcing visualisations")
            return

        output_dir = Path(
            "/hkfs/work/workspace_haic/scratch/qa4548-climate_ws/SAVAR_DATA_TEST/aerosol_forcing_diagnostics"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        spatial_pattern_np = spatial_pattern.detach().cpu().numpy()

        fig, ax = plt.subplots()
        ax.plot(np.arange(spatial_len), spatial_pattern_np, color="tab:blue")
        ax.set_title("Aerosol forcing spatial pattern")
        ax.set_xlabel("Grid point")
        ax.set_ylabel("Relative intensity")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "aerosol_forcing_spatial_pattern.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots()
        im = ax.imshow(forcing_np, aspect="auto", origin="lower", interpolation="nearest")
        ax.set_title("Aerosol forcing over space and time")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Grid point")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / "aerosol_forcing_heatmap.png", dpi=150)
        plt.close(fig)

        time_axis = np.arange(forcing_np.shape[1])
        mean_intensity = forcing_np.mean(axis=0)
        cumulative_intensity = np.cumsum(mean_intensity)

        fig, ax1 = plt.subplots()
        (line_immediate,) = ax1.plot(time_axis, mean_intensity, color="tab:purple")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Average intensity")
        ax1.grid(alpha=0.2)

        ax2 = ax1.twinx()
        (line_cumulative,) = ax2.plot(time_axis, cumulative_intensity, color="tab:orange")
        ax2.set_ylabel("Cumulative intensity", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax1.set_title("Aerosol forcing timeline")
        ax1.legend((line_immediate, line_cumulative), ("Immediate aerosol", "Cumulative aerosol"), loc="upper left")
        fig.tight_layout()
        fig.savefig(output_dir / "aerosol_forcing_timeline.png", dpi=150)
        plt.close(fig)

        grid_shape = None
        if self.mode_weights is not None:
            mw_shape = tuple(self.mode_weights.shape)
            if len(mw_shape) >= 3:
                grid_shape = mw_shape[-2:]
            elif len(mw_shape) == 2:
                grid_shape = (1, mw_shape[-1])
        if grid_shape and np.prod(grid_shape) != spatial_len:
            grid_shape = None

        max_frames_cfg = diagnostics_cfg.get("aerosol_animation_max_frames", 120)
        try:
            max_frames = int(max_frames_cfg)
        except (TypeError, ValueError):
            max_frames = 120
        max_frames = max(1, max_frames)

        frame_stride = max(1, forcing_np.shape[1] // max_frames)
        frame_indices = np.arange(0, forcing_np.shape[1], frame_stride, dtype=int)
        if frame_indices.size == 0 or frame_indices[-1] != forcing_np.shape[1] - 1:
            frame_indices = np.append(frame_indices, forcing_np.shape[1] - 1)

        if grid_shape:
            grid_series = forcing_np.reshape(*grid_shape, forcing_np.shape[1])
            vmin = float(grid_series.min())
            vmax = float(grid_series.max())
            if vmin == vmax:
                vmax = vmin + 1e-6
            fig, ax = plt.subplots()
            im = ax.imshow(grid_series[..., frame_indices[0]], vmin=vmin, vmax=vmax, animated=True)
            ax.set_title("Aerosol forcing progression")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")

            def update(idx):
                frame = frame_indices[idx]
                im.set_array(grid_series[..., frame])
                ax.set_title(f"Aerosol forcing progression (t={frame})")
                return (im,)

        else:
            y_min = float(forcing_np.min())
            y_max = float(forcing_np.max())
            if y_min == y_max:
                y_max = y_min + 1e-6
            fig, ax = plt.subplots()
            ax.set_title("Aerosol forcing progression")
            ax.set_xlabel("Grid point")
            ax.set_ylabel("Forcing")
            (line,) = ax.plot(np.arange(spatial_len), forcing_np[:, frame_indices[0]])
            ax.set_ylim(y_min, y_max)

            def update(idx):
                frame = frame_indices[idx]
                line.set_ydata(forcing_np[:, frame])
                ax.set_title(f"Aerosol forcing progression (t={frame})")
                return (line,)

        anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=80, blit=True)

        fps_cfg = diagnostics_cfg.get("aerosol_animation_fps", 10)
        try:
            fps = int(fps_cfg)
        except (TypeError, ValueError):
            fps = 10
        fps = max(1, fps)

        writer = animation.PillowWriter(fps=fps)
        anim.save(output_dir / "aerosol_forcing.gif", writer=writer)
        plt.close(fig)

    def create_aerosol_forcing(self) -> np.ndarray:
        """
        Create an aerosol forcing field with short-lived, regional cooling hotspots.

        The pattern ramps up quickly, peaks mid-period, and tapers off as regulations take effect, while also exhibiting
        high-frequency variability consistent with episodic aerosol emissions.
        """

        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        time_len = self.time_length + self.transient
        if time_len <= 0:
            raise ValueError("Time length including transient must be positive")

        spatial_len = self.spatial_resolution
        if spatial_len <= 0:
            raise ValueError("Spatial resolution must be positive")

        # Aerosols respond quickly and decay fast: sharp rise, plateau, and decline.
        t = torch.linspace(0.0, 1.0, time_len, device=dev, dtype=dtype)
        ramp_up = torch.sigmoid((t - 0.2) * 10.0)
        ramp_down = torch.sigmoid((0.85 - t) * 10.0)
        envelope = ramp_up * ramp_down

        # Episodic spikes to mimic industrial cycles and volcanic-like bursts.
        seasonal_cycle = torch.sin(2.0 * math.pi * 6.0 * t)
        short_bursts = torch.sin(2.0 * math.pi * 18.0 * t + 0.3)
        aerosol_trend = -0.45 * envelope * (1.0 + 0.15 * seasonal_cycle + 0.05 * short_bursts)

        # Spatial hotspots emphasize regions with stronger aerosol influence.
        spatial_pattern = torch.ones(spatial_len, device=dev, dtype=dtype)
        if self.mode_weights is not None:
            mw = torch.as_tensor(self.mode_weights, device=dev, dtype=dtype)
            spatial_pattern = mw.reshape(self.n_vars, -1).abs().mean(dim=0)
            spatial_pattern = spatial_pattern / (spatial_pattern.mean() + 1e-8)

        if spatial_len > 1:
            idx = torch.linspace(0.0, 2.0 * math.pi, spatial_len, device=dev, dtype=dtype)
            regional_variability = 0.6 + 0.4 * torch.sin(idx * 3.0 + 0.8)
            hemispheric_gradient = 0.8 + 0.2 * torch.cos(idx)
            spatial_pattern = spatial_pattern * regional_variability * hemispheric_gradient

        spatial_pattern = spatial_pattern.pow(1.2)
        spatial_pattern = spatial_pattern / (spatial_pattern.abs().mean() + 1e-8)

        forcing = spatial_pattern.unsqueeze(1) * aerosol_trend.unsqueeze(0)

        forcing_np = forcing.detach().cpu().numpy()

        self._maybe_plot_aerosol_forcing(spatial_pattern, forcing_np, spatial_len)

        return forcing_np

    def _consume_radiative_forcing(self) -> None:
        """Combine CO2 and aerosol forcing fields and inject them into the data field."""
        if self.data_field is None:
            raise ValueError("Data field must be initialised before applying forcing")

        co2_forcing = self.create_co2_forcing()
        aerosol_forcing = self.create_aerosol_forcing()

        if co2_forcing.shape != self.data_field.shape:
            raise ValueError("CO2 forcing shape mismatch with data field")
        if aerosol_forcing.shape != self.data_field.shape:
            raise ValueError("Aerosol forcing shape mismatch with data field")

        if self.forcing_data_field is None:
            # Keep an accumulator so diagnostics can inspect the total applied forcing.
            self.forcing_data_field = np.zeros_like(self.data_field)

        # Merge the long-lived CO2 warming and short-lived aerosol cooling into a single field.
        # Start from greenhouse-gas warming and add aerosol cooling to get a net radiative signal.
        combined_contrib = co2_forcing.copy()
        np.add(combined_contrib, aerosol_forcing, out=combined_contrib)
        # Re-weight the combined radiative forcing by the seasonal cycle if configured
        combined_contrib = self._apply_season_forcing_interaction(combined_contrib)

        # Accumulate the forcing so any pre-existing external drivers stay accounted for.
        np.add(self.forcing_data_field, combined_contrib, out=self.forcing_data_field)

        # Feed the combined forcing into the simulator state so the autoregressive solver consumes it step by step.
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

        plot_path = "/hkfs/work/workspace_haic/scratch/qa4548-climate_ws/SAVAR_DATA/forcing_diagnostics.png"
        print(plot_path)
        if plot_path:
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
        weights = deepcopy(self.mode_weights.reshape(self.n_vars, -1))
        # weights_inv = np.linalg.pinv(weights)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_inv = torch.Tensor(np.linalg.pinv(weights)).to(device=dev)
        weights = torch.Tensor(weights).to(device=dev)
        time_len = deepcopy(self.time_length)
        time_len += self.transient
        tau_max = self.tau_max

        # phi = dict_to_matrix(self.links_coeffs)
        phi = torch.Tensor(dict_to_matrix(self.links_coeffs)).to(device=dev)
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

        weights = torch.Tensor(np.linalg.pinv(self.mode_weights.reshape(self.n_vars, -1))).to("cuda")
        phi = torch.Tensor(dict_to_matrix(self.links_coeffs)).to("cuda")
        mode_weights_tensor = torch.Tensor(self.mode_weights.reshape(self.n_vars, -1)).to("cuda")
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
        w_np = np.linalg.pinv(self.mode_weights.reshape(self.n_vars, -1))
        phi_np = dict_to_matrix(self.links_coeffs)

        # choose GPU if available, else CPU — and use float32 everywhere
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32
        w_torch = torch.tensor(w_np, device=dev, dtype=dtype)
        phi_torch = torch.tensor(phi_np, device=dev, dtype=dtype)
        mw_torch = torch.tensor(
            self.mode_weights.reshape(self.n_vars, -1),
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
