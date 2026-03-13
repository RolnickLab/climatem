"""
This code is inspired by the SAVAR data generation paper and code "A spatiotemporal stochastic climate model for
benchmarking causal discovery methods for teleconnections", Tibau et al.

2022 The main difference with the provided code is the torch/GPU implementation which considerably speeds up the data
generation process
"""

import itertools as it
import math
from copy import deepcopy
from math import sqrt
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


def normalize_transition_matrix(phi: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize the transition matrix P so that for each target mode, the sum of all incoming link coefficients equals 1.

    This follows the SAVAR SNR formulation constraint: sum of links in P = 1.

    Args:
        phi: Transition matrix of shape (n_modes, n_modes, tau_max)
             where phi[j, i, tau] is the coefficient from mode i to mode j at lag tau+1
        eps: Small constant to avoid division by zero

    Returns:
        Normalized phi where sum over (i, tau) of |phi[j, i, tau]| = 1 for each j
    """
    phi_normalized = phi.copy()
    n_modes = phi.shape[0]

    for j in range(n_modes):
        # Sum of absolute values of all incoming links to mode j
        total = np.abs(phi[j, :, :]).sum()
        if total > eps:
            phi_normalized[j, :, :] = phi[j, :, :] / total

    return phi_normalized


def normalize_transition_with_forcing(
    phi_climate: np.ndarray, phi_forcing: np.ndarray, n_climate_modes: int, eps: float = 1e-8
) -> tuple:
    """
    Normalize transition matrix including forcing coefficients.

    For each climate mode j, the sum of all incoming coefficients equals 1:
    sum(|phi_climate[j,:,:]|) + sum(|phi_forcing[j,:,:]|) = 1

    This ensures the SAVAR SNR constraint is maintained when forcing
    is included in the dynamics.

    Args:
        phi_climate: Climate-to-climate coefficients (n_climate, n_climate, tau_max)
        phi_forcing: Forcing-to-climate coefficients (n_climate, n_forcing, tau_max)
        n_climate_modes: Number of climate modes
        eps: Small constant to avoid division by zero

    Returns:
        Tuple of (normalized phi_climate, normalized phi_forcing)
    """
    phi_climate_norm = phi_climate.copy()
    phi_forcing_norm = phi_forcing.copy()

    for j in range(n_climate_modes):
        # Total incoming coefficient sum (climate + forcing)
        climate_sum = np.abs(phi_climate[j, :, :]).sum()
        forcing_sum = np.abs(phi_forcing[j, :, :]).sum()
        total = climate_sum + forcing_sum

        if total > eps:
            phi_climate_norm[j, :, :] = phi_climate[j, :, :] / total
            phi_forcing_norm[j, :, :] = phi_forcing[j, :, :] / total

        logger.debug(
            f"Mode {j}: climate links = {climate_sum:.4f}, " f"forcing links = {forcing_sum:.4f}, total = {total:.4f}"
        )

    return phi_climate_norm, phi_forcing_norm


class SAVAR:
    """
    SAVAR synthetic climate data generator.

    Generates spatiotemporal synthetic climate data with a known causal structure,
    used to benchmark causal discovery methods. Based on Tibau et al. (2022), extended
    with GPU acceleration, external forcing (CO2 + aerosol), and background state.

    Data generation pipeline:
        1. Seasonality: harmonic components with optional yearly jitter
        2. Forcing trajectories: CO2 (monotonic ramp) and aerosol (regional, staggered)
           latent time series are generated as exogenous inputs
        3. Dynamics: autoregressive evolution in latent (mode) space:
               z_j(t) = sum_{k,tau} P_climate[j,k,tau] * z_k(t-tau)
                       + sum_{f,tau} P_forcing[j,f,tau] * forcing_f(t-tau) + N_j(t)
               S_t = W^{-1} * z(t)    (project back to observation space)
           where W = mode_weights (mixing matrix), P_climate/P_forcing = transition
           matrices, and N_t is observation-space noise with covariance s * W^T * W.
        4. Background: optional slow, spatially-smooth AR(1) drift added post-dynamics

    Key attributes:
        links_coeffs (dict): Causal graph as {target_idx: [((source_idx, lag), coeff), ...]}.
        n_climate_modes (int): Number of climate latent variables (from mode_weights).
        n_vars (int): Total latent count (climate + forcing modes).
        mode_weights (ndarray): Spatial patterns (mixing matrix W), shape (n_modes, D_x, D_y)
            where D_x * D_y = spatial_resolution.
        data_field (ndarray): Generated spatiotemporal data, shape (spatial_resolution, time_length).
        noise_data_field (ndarray): Pre-generated noise, shape (spatial_resolution, time_length + transient).
        seasonal_data_field (ndarray): Seasonality component (same shape as data_field before trimming).
        co2_latent_trajectory (ndarray): CO2 latent time series, shape (time,).
        aerosol_latent_trajectory (ndarray): Aerosol latent time series, shape (n_aerosol, time).
        aerosol_spatial_templates (ndarray): Orthogonal spatial templates for aerosol extraction,
            shape (n_aerosol, spatial_resolution).
        forcing_coeffs (dict): Extracted forcing-to-climate causal coefficients.
        forcing_amplification (float): Scaling factor for forcing contributions.
    """

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
        "noise_ar1",
        "noise_ar1_rho",
        "latent_noise_cov",
        "fast_noise_cov",
        "forcing_dict",
        "forcing_indices",
        "forcing_coeffs",
        "forcing_amplification",
        "season_dict",
        "data_field",
        "noise_data_field",
        "seasonal_data_field",
        "forcing_data_field",
        "co2_forcing_data_field",
        "aerosol_forcing_data_field",
        "co2_latent_trajectory",
        "aerosol_latent_trajectory",
        "aerosol_spatial_templates",
        "background_data_field",
        "linearity",
        "poly_degrees",
        "verbose",
        "model_seed",
        "nnar_model",
        "output_save_dir",
        # Background state parameters
        "enable_background",
        "background_strength",
        "background_strength_mode",
        "background_smoothness",
        "background_timescale_rho",
        "background_n_modes",
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
        noise_ar1: bool = True,
        noise_ar1_rho: float = 0.95,
        noise_cov: np.ndarray = None,
        latent_noise_cov: np.ndarray = None,
        fast_cov: np.ndarray = None,
        forcing_dict: dict = None,
        forcing_indices: dict = None,
        forcing_amplification: float = 5.0,
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
        # Background state parameters
        enable_background: bool = False,
        background_strength: float = 0.3,
        background_strength_mode: str = "relative",
        background_smoothness: float = 0.15,
        background_timescale_rho: float = 0.995,
        background_n_modes: int = 3,
    ):

        self.links_coeffs = links_coeffs
        self.time_length = time_length
        self.transient = transient
        self.noise_strength = noise_strength
        self.noise_variance = noise_variance  # TODO: NOT USED.
        self.noise_ar1 = noise_ar1
        self.noise_ar1_rho = noise_ar1_rho
        self.noise_cov = noise_cov

        self.latent_noise_cov = latent_noise_cov  # D_x
        self.fast_noise_cov = fast_cov  # D_y

        self.mode_weights = mode_weights
        self.noise_weights = noise_weights

        self.forcing_dict = forcing_dict
        self.forcing_indices = forcing_indices
        self.forcing_amplification = forcing_amplification
        self.season_dict = season_dict
        self.linearity = linearity
        self.poly_degrees = poly_degrees

        self.data_field = data_field

        self.verbose = verbose
        self.model_seed = model_seed
        self.output_save_dir = output_save_dir

        # Background state parameters
        self.enable_background = enable_background
        self.background_strength = background_strength
        self.background_strength_mode = background_strength_mode
        self.background_smoothness = background_smoothness
        self.background_timescale_rho = background_timescale_rho
        self.background_n_modes = background_n_modes

        # Computed attributes
        logger.debug("Creating attributes")
        # n_climate_modes is the number of climate variables (from mode_weights)
        self.n_climate_modes = mode_weights.shape[0]
        # n_vars is total latents (climate + forcing) if forcing_indices provided
        if forcing_indices is not None:
            self.n_vars = forcing_indices.get("n_total", self.n_climate_modes)
        else:
            self.n_vars = self.n_climate_modes
        self.tau_max = max(abs(lag) for (_, lag), _ in it.chain.from_iterable(self.links_coeffs.values()))
        self.spatial_resolution = deepcopy(self.mode_weights.reshape(self.n_climate_modes, -1).shape[1])
        print("spatial-resolution done")

        # Extract forcing → mode coefficients if forcing is used
        self.forcing_coeffs = self._extract_forcing_coefficients() if forcing_indices else None

        # Initialize forcing latent trajectories (populated during forcing generation)
        self.co2_latent_trajectory = None
        self.aerosol_latent_trajectory = None
        self.aerosol_spatial_templates = None  # Orthogonal spatial templates for aerosol extraction
        logger.debug("spatial-resolution done")

        if self.noise_weights is None:
            self.noise_weights = deepcopy(self.mode_weights)
        if self.latent_noise_cov is None:
            self.latent_noise_cov = np.eye(self.n_vars)
        if self.fast_noise_cov is None:
            self.fast_noise_cov = np.zeros((self.spatial_resolution, self.spatial_resolution))
        logger.debug("copies done")

        # Empty attributes
        self.noise_data_field = noise_data_field
        self.seasonal_data_field = seasonal_data_field
        self.forcing_data_field = forcing_data_field
        self.co2_forcing_data_field = co2_forcing_data_field
        self.aerosol_forcing_data_field = aerosol_forcing_data_field
        self.background_data_field = None

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
        logger.info(
            f"Extracted forcing coefficients: {n_co2_links} CO2→mode links, {n_aerosol_links} aerosol→mode links"
        )

        return forcing_coeffs

    def generate_data(self, train_nnar=True, include_noise=True) -> None:
        """
        Generates the data of savar.

        Args:
            train_nnar: Whether to train NNAR model for nonlinear data
            include_noise: Whether to include noise in the generated data
        """
        # Prepare the datafield
        if self.data_field is None:
            logger.debug("Creating empty data field")
            self.data_field = np.zeros((self.spatial_resolution, self.time_length + self.transient))

        self._apply_noise(include_noise)
        self._apply_seasonality()
        self._apply_forcing()
        self._apply_dynamics(train_nnar)
        self._apply_background()

    def _apply_noise(self, include_noise):
        """Add noise to data_field BEFORE dynamics (baseline behavior)."""
        if include_noise:
            if self.noise_data_field is None:
                logger.debug("Creating noise data field and adding to data_field")
                self._add_noise_field()
            else:
                logger.debug("Reusing existing noise_data_field, adding to data_field")
                self.data_field += self.noise_data_field
        else:
            logger.debug("Skipping noise (generating deterministic data)")

    def _apply_seasonality(self):
        """Add seasonality to data_field if configured."""
        if self.season_dict is None:
            logger.info("No seasonality")
            return
        logger.debug("Adding seasonality forcing")
        if self.seasonal_data_field is not None:
            logger.debug("Reusing existing seasonal_data_field")
            self.data_field += self.seasonal_data_field
        else:
            self._add_seasonality_forcing()

    def _apply_forcing(self):
        """Generate forcing latent trajectories if configured."""
        if self.forcing_dict is None:
            return
        logger.debug("Generating forcing latent trajectories (applied during dynamics)")
        if self.co2_latent_trajectory is None:
            self._generate_forcing_trajectories()
        else:
            logger.debug("Reusing existing forcing latent trajectories")

    def _apply_dynamics(self, train_nnar):
        """Compute data using the configured linearity model."""
        # Compute the data
        if self.linearity == "linear":
            logger.debug("Creating linear data")
            self._create_linear()
        elif self.linearity == "polynomial":
            logger.debug("Creating polynomial data")
            self._create_polynomial()
        else:
            logger.debug("Creating nonlinear data")
            if train_nnar:
                logger.info("Training NNAR model before data generation...")
                self.train_nnar(num_epochs=50, learning_rate=0.001, batch_size=32)
            self._create_nonlinear()

    def _apply_background(self):
        """Add background state AFTER mode dynamics to avoid entanglement with causal graph."""
        if not self.enable_background:
            return
        logger.debug("Adding low-frequency background state")
        if self.background_data_field is not None:
            logger.debug("Reusing existing background_data_field")
            self.data_field += self.background_data_field
        else:
            self._add_background_field(
                background_strength=self.background_strength,
                background_strength_mode=self.background_strength_mode,
                spatial_smoothness=self.background_smoothness,
                time_rho=self.background_timescale_rho,
                n_bg_modes=self.background_n_modes,
            )

    def generate_cov_noise_matrix(self) -> np.ndarray:
        """
        Generate the noise covariance matrix: Cov = s · W^+ · (W^+)^T.

        This gives noise that is anti-correlated with the mode spatial structure,
        ensuring the model can distinguish signal from noise. W^+ is the
        pseudoinverse of the noise_weights mixing matrix.

        Returns:
            cov: Covariance matrix of shape (spatial_resolution, spatial_resolution).
        """
        W = deepcopy(self.noise_weights).reshape(self.n_climate_modes, -1)
        logger.debug(f"noise_weights copied, {W.shape}")
        W_plus = np.linalg.pinv(W)
        logger.debug("noise_weights inverted")
        cov = self.noise_strength * W_plus @ W_plus.transpose()
        logger.debug("cov created")

        return cov

    def _add_noise_field(self):
        """
        Generate noise in observation space with covariance s · W^+ · (W^+)^T.

        Uses the pseudoinverse of the mixing matrix so that the noise covariance
        is *anti-correlated* with the mode spatial patterns. This is essential for
        the model to distinguish signal (mode-aligned) from noise.

        NOTE: This method stores the noise in self.noise_data_field but does NOT
        add it to self.data_field. The noise is applied during dynamics
        (_create_linear, etc.) following:  S_t = W^{-1}·P·W·S_{t-τ} + N_t
        """
        logger.info("Generate noise_data_field with covariance s·W^+·(W^+)^T")

        if self.noise_cov is None:
            if self.verbose:
                print("Generate covariance matrix")
            self.noise_cov = self.generate_cov_noise_matrix()
            self.noise_cov += 1e-6 * np.eye(self.noise_cov.shape[0])

        # Generate noise from cov
        if self.verbose:
            print("Generate noise_data_field multivariate random")
        mean_torch = torch.Tensor(np.zeros(self.spatial_resolution)).to(device="cuda")
        cov = torch.Tensor(self.noise_cov).to(device="cuda")
        distrib = MultivariateNormal(loc=mean_torch, covariance_matrix=cov)  # . to(device="cuda")
        noise_data_field = distrib.sample(sample_shape=torch.Size([self.time_length + self.transient]))
        self.noise_data_field = noise_data_field.detach().cpu().numpy().transpose()

        logger.info(
            f"Generated noise with shape {self.noise_data_field.shape}, "
            f"std={self.noise_data_field.std():.4f}, "
            f"noise_strength (s)={self.noise_strength:.4f}"
        )
        # Add noise to data_field BEFORE dynamics (baseline behavior).
        # The AR dynamics will then evolve the noise-inclusive field,
        # effectively dampening the noise through the transition coefficients.
        self.data_field += self.noise_data_field

    def _add_seasonality_forcing(self):
        """
        Add deterministic seasonal cycle to the data field.

        Constructs a sum of sinusoidal harmonics (e.g. annual, semi-annual) with
        optional year-to-year jitter in amplitude and phase. The result is stored
        in ``self.seasonal_data_field`` and added to ``self.data_field``.

        Configuration comes from ``self.season_dict`` with keys:
            periods (list[float]): harmonic periods in timesteps, e.g. [365, 182.5, 60].
            amplitudes (list[float]): amplitude for each harmonic.
            phases (list[float]): phase offset (radians) for each harmonic.
            yearly_jitter (dict|None): {"amplitude": float, "phase": float} controlling
                inter-annual variability of each harmonic.
            season_weight (ndarray|None): per-gridpoint multiplier (latitude-dependent
                seasonality strength), shape (spatial_resolution,).
        Adds external forcing to the data field using PyTorch tensors for GPU acceleration.

        Allows for both linear and nonlinear ramps.
        """
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

        # Generate the forcing trend using torch tensors
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
            t = torch.linspace(0, math.pi, f_time_2 - f_time_1, dtype=dtype, device=dev)
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
                torch.full((time_len - f_time_2,), f_2, dtype=dtype, device=dev),
            ]
        ).reshape(1, time_len)

        logger.info(
            f"CO2 forcing: Using {ramp_type} ramp from f_1={f_1} to f_2={f_2} "
            f"(time {f_time_1} to {f_time_2}, total length {time_len})"
        )

        logger.debug(
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
        if self.verbose:
            print(f"Using {ramp_type} ramp: f_1={f_1}, f_2={f_2}, f_time_1={f_time_1}, f_time_2={f_time_2}")
            print(f"Forcing data field mean: {self.forcing_data_field.mean()}")
            print(f"Before addition - Data field mean: {self.data_field.mean()}")

        # data_field_before = self.data_field.copy()

        spatial_pattern = torch.clamp(spatial_pattern, min=0.05)
        forcing = spatial_pattern.unsqueeze(1) * trend

        forcing_np = forcing.detach().cpu().numpy()

        # data_field_after = self.data_field
        if self.verbose:
            print(f"After addition - Data field mean: {self.data_field.mean()}")

        return forcing_np

    def _resolve_aerosol_warming_index(self, warming_index, n_aerosol_latents: int):
        """Resolve and validate the optional warming latent index."""
        if warming_index is None and n_aerosol_latents >= 4:
            warming_index = 3
        try:
            return int(warming_index) if warming_index is not None else None
        except (TypeError, ValueError):
            return None

    def _create_aerosol_spatial_templates(
        self, n_aerosol_latents: int, spatial_len: int, aerosol_contrast: float, dev, dtype
    ) -> torch.Tensor:
        """Create and normalize aerosol spatial templates for latent projection."""
        if spatial_len == 1:
            return torch.ones(n_aerosol_latents, 1, device=dev, dtype=dtype)

        grid_size = int(math.sqrt(spatial_len))
        is_2d_grid = (grid_size * grid_size == spatial_len) and grid_size > 1

        if is_2d_grid:
            logger.info(f"Creating 2D aerosol templates for {grid_size}x{grid_size} grid")
            spatial_templates = self._create_aerosol_templates_2d(n_aerosol_latents, grid_size, dev, dtype)
        else:
            logger.info(f"Creating 1D aerosol templates for {spatial_len} points")
            spatial_templates = self._create_aerosol_templates_1d(n_aerosol_latents, spatial_len, dev, dtype)

        self._normalize_aerosol_templates(spatial_templates, aerosol_contrast)
        self._log_aerosol_template_orthogonality(spatial_templates)
        return spatial_templates

    def _create_aerosol_templates_2d(self, n_aerosol_latents: int, grid_size: int, dev, dtype) -> torch.Tensor:
        """Build 2D region-localized aerosol templates."""
        spatial_templates = torch.zeros(n_aerosol_latents, grid_size * grid_size, device=dev, dtype=dtype)

        lat_coords = torch.linspace(-1.0, 1.0, grid_size, device=dev, dtype=dtype)
        lon_coords = torch.linspace(-1.0, 1.0, grid_size, device=dev, dtype=dtype)
        lat_grid, lon_grid = torch.meshgrid(lat_coords, lon_coords, indexing="ij")

        def gaussian_2d(lat_center, lon_center, lat_sigma, lon_sigma, angle=0.0):
            x = lon_grid - lon_center
            y = lat_grid - lat_center
            if angle != 0.0:
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                x_rot = x * cos_a + y * sin_a
                y_rot = -x * sin_a + y * cos_a
            else:
                x_rot = x
                y_rot = y
            return torch.exp(-0.5 * ((x_rot / lon_sigma) ** 2 + (y_rot / lat_sigma) ** 2))

        def smooth_window(x, start, end, sharpness=8.0):
            return torch.sigmoid((x - start) * sharpness) * torch.sigmoid((end - x) * sharpness)

        north_mask = smooth_window(lat_grid, -1.0, -0.05, sharpness=8.0)
        north_mask_strict = smooth_window(lat_grid, -1.0, -0.30, sharpness=8.0)

        if n_aerosol_latents >= 1:
            na = (
                1.0 * gaussian_2d(-0.80, -0.80, 0.30, 0.38, angle=-0.25)
                + 0.8 * gaussian_2d(-0.85, -0.95, 0.30, 0.28, angle=-0.05)
                + 0.7 * gaussian_2d(-0.70, -0.35, 0.26, 0.32, angle=0.10)
            )
            na *= north_mask_strict * smooth_window(lon_grid, -1.0, 0.00, sharpness=6.5)
            spatial_templates[0] = -na.reshape(-1) * 1.65

        if n_aerosol_latents >= 2:
            eu = (
                1.0 * gaussian_2d(-0.80, 0.40, 0.24, 0.30, angle=0.20)
                + 0.8 * gaussian_2d(-0.75, 0.50, 0.22, 0.26, angle=0.30)
                + 0.6 * gaussian_2d(-0.70, 0.30, 0.20, 0.24, angle=-0.05)
            )
            eu *= north_mask_strict * smooth_window(lon_grid, -0.30, 0.55, sharpness=7.5)
            spatial_templates[1] = -eu.reshape(-1) * 1.55

        if n_aerosol_latents >= 3:
            ea = (
                1.0 * gaussian_2d(0.45, 0.65, 0.16, 0.22, angle=-0.20)
                + 0.6 * gaussian_2d(0.35, 0.80, 0.12, 0.18, angle=-0.10)
                + 0.5 * gaussian_2d(0.50, 0.45, 0.14, 0.18, angle=0.10)
            )
            ea *= north_mask * smooth_window(lon_grid, 0.35, 0.98, sharpness=8.0)
            spatial_templates[2] = -ea.reshape(-1) * 1.10

        if n_aerosol_latents >= 4:
            ind = (
                1.0 * gaussian_2d(0.25, 0.25, 0.10, 0.12, angle=0.30)
                + 0.6 * gaussian_2d(0.15, 0.35, 0.08, 0.10, angle=-0.10)
                + 0.4 * gaussian_2d(0.35, 0.15, 0.10, 0.12, angle=0.00)
            )
            ind *= smooth_window(lat_grid, -0.55, -0.05, sharpness=8.0) * smooth_window(
                lon_grid, 0.20, 0.65, sharpness=8.0
            )
            spatial_templates[3] = -ind.reshape(-1) * 0.95

        for i in range(4, n_aerosol_latents):
            wave_num = i - 2
            pattern_2d = torch.abs(torch.sin(wave_num * math.pi * lat_grid)) * torch.abs(
                torch.cos((wave_num + 1) * math.pi * lon_grid)
            )
            spatial_templates[i] = -pattern_2d.reshape(-1) * (1.0 + 0.2 * i)

        return spatial_templates

    def _create_aerosol_templates_1d(self, n_aerosol_latents: int, spatial_len: int, dev, dtype) -> torch.Tensor:
        """Build fallback 1D aerosol templates for non-square grids."""
        spatial_templates = torch.zeros(n_aerosol_latents, spatial_len, device=dev, dtype=dtype)
        coords = torch.linspace(-1.0, 1.0, spatial_len, device=dev, dtype=dtype)

        if n_aerosol_latents >= 1:
            northern_mask = (coords < 0.0).float()
            spatial_templates[0] = -torch.sigmoid((-coords - 0.3) * 8.0) * northern_mask * 1.5

        if n_aerosol_latents >= 2:
            southern_mask = (coords > 0.0).float()
            spatial_templates[1] = -torch.sigmoid((coords - 0.3) * 8.0) * southern_mask * 1.0

        if n_aerosol_latents >= 3:
            spatial_templates[2] = -torch.exp(-((coords / 0.3) ** 2)) * 0.8

        if n_aerosol_latents >= 4:
            mid_lat_mask = ((coords > -0.5) & (coords < 0.5)).float()
            spatial_templates[3] = -torch.abs(torch.sin(2 * math.pi * coords)) * mid_lat_mask * 1.2

        for i in range(4, n_aerosol_latents):
            wave_num = i - 2
            spatial_templates[i] = -torch.abs(torch.sin(wave_num * math.pi * coords + i * 0.3)) * (1.0 + 0.2 * i)

        return spatial_templates

    def _normalize_aerosol_templates(self, spatial_templates: torch.Tensor, aerosol_contrast: float) -> None:
        """Apply contrast and L2 normalization in-place."""
        n_aerosol_latents = spatial_templates.shape[0]
        for i in range(n_aerosol_latents):
            if aerosol_contrast != 1.0:
                spatial_templates[i] = -torch.abs(spatial_templates[i]).pow(aerosol_contrast)

            norm = torch.sqrt((spatial_templates[i] ** 2).sum())
            if norm > 1e-8:
                spatial_templates[i] = spatial_templates[i] / norm
            else:
                logger.warning(f"Aerosol template {i} has near-zero norm, skipping normalization")

    def _log_aerosol_template_orthogonality(self, spatial_templates: torch.Tensor) -> None:
        """Log pairwise inner products for diagnostics."""
        n_aerosol_latents = spatial_templates.shape[0]
        if n_aerosol_latents <= 1:
            return

        logger.info("Aerosol spatial template orthogonality:")
        for i in range(n_aerosol_latents):
            for j in range(i + 1, n_aerosol_latents):
                inner_prod = (spatial_templates[i] * spatial_templates[j]).sum().item()
                logger.info(f"  Template {i} · Template {j} = {inner_prod:.4f}")

    def _accumulate_aerosol_latent_forcing(
        self,
        forcing: torch.Tensor,
        spatial_templates: torch.Tensor,
        latent_signs: List[float],
        t: torch.Tensor,
        time_len: int,
        base_ramp_up_time: int,
        base_peak_time: int,
        base_decline_time: int,
        timing_stagger: float,
        aerosol_scale: float,
    ) -> None:
        """Accumulate latent-specific temporal patterns into the forcing field."""
        n_aerosol_latents = spatial_templates.shape[0]
        for i in range(n_aerosol_latents):
            offset_fraction = i / n_aerosol_latents
            latent_ramp_up = base_ramp_up_time + int(offset_fraction * timing_stagger * time_len)
            latent_peak = base_peak_time + int(offset_fraction * timing_stagger * 0.7 * time_len)
            latent_decline = base_decline_time + int(offset_fraction * timing_stagger * 0.5 * time_len)

            latent_peak = min(latent_peak, time_len - 100)
            latent_decline = min(latent_decline, time_len - 10)

            t_ramp = latent_ramp_up / time_len
            t_peak = latent_peak / time_len
            t_decline = latent_decline / time_len
            ramp_up = torch.sigmoid((t - t_ramp) * 10.0 / (t_peak - t_ramp + 1e-6))
            ramp_down = torch.sigmoid((t_decline - t) * 10.0 / (t_decline - t_peak + 1e-6))
            envelope = ramp_up * ramp_down

            base_freq = 3.0 + i * 2.0
            freq_mod = 1.0 + 0.2 * torch.sin(2.0 * math.pi * base_freq * t + i * 0.5)
            latent_seasonal = torch.sin(2.0 * math.pi * (6.0 + i) * t)
            latent_bursts = torch.sin(2.0 * math.pi * (18.0 + i * 3) * t + 0.3 * i)

            trend_sign = latent_signs[i] if i < len(latent_signs) else -1.0
            latent_trend = (
                trend_sign * aerosol_scale * envelope * freq_mod * (1.0 + 0.15 * latent_seasonal + 0.05 * latent_bursts)
            )
            forcing += spatial_templates[i].unsqueeze(1) * latent_trend.unsqueeze(0)

            sign_label = "warming" if trend_sign > 0 else "cooling"
            logger.debug(
                f"  Latent {i}: ramp_up={latent_ramp_up}, peak={latent_peak}, decline={latent_decline}, "
                f"freq={base_freq}, {sign_label}"
            )

    def create_aerosol_forcing(self) -> np.ndarray:
        """
        Create an aerosol forcing field with distinct temporal dynamics per region.

        Each spatial region (corresponding to an aerosol latent) has a staggered temporal signal with unique timing and
        frequency modulation. This ensures aerosol latents are distinguishable for causal discovery.

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
        aerosol_scale = max(float(forcing_cfg.get("aerosol_scale", 0.03)), 0.0)
        aerosol_contrast = max(float(forcing_cfg.get("aerosol_spatial_contrast", 1.05)), 0.5)

        base_ramp_up_time = int(forcing_cfg.get("aerosol_ramp_up_time", int(0.2 * self.time_length))) + self.transient
        base_peak_time = int(forcing_cfg.get("aerosol_peak_time", int(0.6 * self.time_length))) + self.transient
        base_decline_time = int(forcing_cfg.get("aerosol_decline_time", int(0.85 * self.time_length))) + self.transient
        timing_stagger = float(forcing_cfg.get("aerosol_timing_stagger", 0.3))

        t = torch.linspace(0.0, 1.0, time_len, device=dev, dtype=dtype)
        n_aerosol_latents = len(self.forcing_indices.get("aerosol", [])) if self.forcing_indices else 1
        n_aerosol_latents = max(n_aerosol_latents, 1)

        warming_index = self._resolve_aerosol_warming_index(
            forcing_cfg.get("aerosol_warming_index", None),
            n_aerosol_latents,
        )
        latent_signs = [-1.0] * n_aerosol_latents
        if warming_index is not None and 0 <= warming_index < n_aerosol_latents:
            latent_signs[warming_index] = 1.0

        spatial_templates = self._create_aerosol_spatial_templates(
            n_aerosol_latents,
            spatial_len,
            aerosol_contrast,
            dev,
            dtype,
        )
        forcing = torch.zeros((spatial_len, time_len), device=dev, dtype=dtype)

        logger.info(
            f"Aerosol forcing: Creating {n_aerosol_latents} orthogonal spatial templates with distinct temporal patterns "
            f"(base_ramp={base_ramp_up_time}, base_peak={base_peak_time}, base_decline={base_decline_time}, "
            f"timing_stagger={timing_stagger}, scale={aerosol_scale}, spatial_contrast={aerosol_contrast})"
        )
        if warming_index is not None and 0 <= warming_index < n_aerosol_latents:
            logger.info(f"Aerosol forcing: warming latent index={warming_index}")

        self._accumulate_aerosol_latent_forcing(
            forcing=forcing,
            spatial_templates=spatial_templates,
            latent_signs=latent_signs,
            t=t,
            time_len=time_len,
            base_ramp_up_time=base_ramp_up_time,
            base_peak_time=base_peak_time,
            base_decline_time=base_decline_time,
            timing_stagger=timing_stagger,
            aerosol_scale=aerosol_scale,
        )

        forcing_np = forcing.detach().cpu().numpy()
        self.aerosol_spatial_templates = spatial_templates.detach().cpu().numpy()
        logger.info(f"[AEROSOL FORCING] Stored {n_aerosol_latents} spatial templates for latent extraction")
        return forcing_np

    def _generate_forcing_trajectories(self) -> None:
        """
        Generate forcing latent trajectories BEFORE dynamics.

        This creates the CO2 and aerosol forcing fields and extracts their
        latent representations (time series) that will be used as exogenous
        inputs during the dynamics computation.

        Must be called BEFORE _create_linear(), _create_nonlinear(), or _create_polynomial().

        The forcing latent trajectories are stored in:
        - self.co2_latent_trajectory: shape (time,) - scalar CO2 latent over time
        - self.aerosol_latent_trajectory: shape (n_aerosol_latents, time) - aerosol latents over time
        """
        if self.forcing_dict is None or self.forcing_indices is None:
            logger.info("No forcing configured, skipping trajectory generation")
            return

        time_len = self.time_length + self.transient

        # Initialize forcing data fields if needed
        if self.forcing_data_field is None:
            self.forcing_data_field = np.zeros((self.spatial_resolution, time_len))
        if self.co2_forcing_data_field is None:
            self.co2_forcing_data_field = np.zeros((self.spatial_resolution, time_len))
        if self.aerosol_forcing_data_field is None:
            self.aerosol_forcing_data_field = np.zeros((self.spatial_resolution, time_len))

        # Generate CO2 forcing field and extract latent trajectory
        co2_forcing = self.create_co2_forcing()
        self.co2_forcing_data_field = co2_forcing
        self.co2_latent_trajectory = co2_forcing.mean(axis=0)  # shape: (time,)

        logger.info(
            f"[FORCING TRAJ] Generated CO2 latent trajectory: shape {self.co2_latent_trajectory.shape}, "
            f"range [{self.co2_latent_trajectory.min():.4f}, {self.co2_latent_trajectory.max():.4f}]"
        )

        # Generate aerosol forcing field and extract latent trajectories via template projection
        aerosol_forcing = self.create_aerosol_forcing()
        self.aerosol_forcing_data_field = aerosol_forcing

        n_aerosol_latents = len(self.forcing_indices.get("aerosol", []))

        if n_aerosol_latents > 0 and self.aerosol_spatial_templates is not None:
            aerosol_latents = np.zeros((n_aerosol_latents, time_len))
            for i in range(n_aerosol_latents):
                template = self.aerosol_spatial_templates[i]  # (spatial_resolution,)
                # Inner product of template with forcing field at each timestep
                aerosol_latents[i] = template @ aerosol_forcing  # (time,)
            self.aerosol_latent_trajectory = aerosol_latents

            logger.info(
                f"[FORCING TRAJ] Generated aerosol latent trajectories: shape {self.aerosol_latent_trajectory.shape}"
            )
            for i in range(n_aerosol_latents):
                logger.debug(
                    f"  Aerosol latent {i}: range [{aerosol_latents[i].min():.4f}, {aerosol_latents[i].max():.4f}]"
                )
        else:
            self.aerosol_latent_trajectory = None
            if n_aerosol_latents > 0:
                logger.info(
                    f"[FORCING TRAJ] Warning: {n_aerosol_latents} aerosol latents configured but no spatial templates"
                )

        # Store combined forcing field for diagnostics (but DON'T add to data_field)
        self.forcing_data_field = co2_forcing + aerosol_forcing

    def _create_linear(self):
        """
        Create linear SAVAR dynamics with forcing in the dynamics equation:

        z_j(t) = sum_{k,τ} P_climate[j,k,τ] · z_k(t-τ) + sum_{f,τ} P_forcing[j,f,τ] · forcing_f(t-τ) + N_j(t)
        S_t = W^{-1} · z(t)

        where:
        - W is the mixing matrix (mode_weights)
        - P_climate is the climate-to-climate transition matrix
        - P_forcing is the forcing-to-climate transition matrix
        - forcing_f are exogenous forcing latent trajectories (CO2, aerosol)
        - N_t ~ N(0, s·W·W^T) is the noise at each timestep
        - The sum of all links (climate + forcing) equals 1 for each target mode
        """
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        weights = deepcopy(self.mode_weights.reshape(self.n_climate_modes, -1))
        weights_inv = torch.tensor(np.linalg.pinv(weights), device=dev, dtype=dtype)
        weights_torch = torch.tensor(weights, device=dev, dtype=dtype)

        time_len = self.time_length + self.transient
        tau_max = self.tau_max

        # Get full transition matrix
        phi_full_np = dict_to_matrix(self.links_coeffs)

        # Extract climate-to-climate submatrix
        phi_climate_np = phi_full_np[: self.n_climate_modes, : self.n_climate_modes, :]

        # Check if we have forcing
        has_forcing = self.forcing_indices is not None and self.co2_latent_trajectory is not None

        if has_forcing:
            # Extract forcing-to-climate submatrix: phi_full[climate_targets, forcing_sources, :]
            phi_forcing_np = phi_full_np[: self.n_climate_modes, self.n_climate_modes :, :]

            # Normalize both climate and forcing coefficients together
            phi_climate_np, phi_forcing_np = normalize_transition_with_forcing(
                phi_climate_np, phi_forcing_np, self.n_climate_modes
            )
            phi_forcing = torch.tensor(phi_forcing_np, device=dev, dtype=dtype)

            # Prepare forcing latent trajectories
            co2_latent = torch.tensor(self.co2_latent_trajectory, device=dev, dtype=dtype)

            # Concatenate all forcing latents: (n_forcing, time)
            if self.aerosol_latent_trajectory is not None:
                aerosol_latent = torch.tensor(self.aerosol_latent_trajectory, device=dev, dtype=dtype)
                forcing_latents = torch.cat(
                    [co2_latent.unsqueeze(0), aerosol_latent], dim=0  # (1, time)  # (n_aerosol, time)
                )
            else:
                forcing_latents = co2_latent.unsqueeze(0)  # (1, time)

            logger.debug(f"[LINEAR] Forcing latents shape: {forcing_latents.shape}")
            logger.debug(f"[LINEAR] Forcing coefficient matrix shape: {phi_forcing_np.shape}")
        else:
            # No forcing - use raw coefficients (already bounded < 1 by create_links_coeffs)
            pass

        phi_climate = torch.tensor(phi_climate_np, device=dev, dtype=dtype)

        # Initialize data_field from existing (includes noise added by _add_noise_field)
        data_field = torch.tensor(self.data_field, device=dev, dtype=dtype)

        if has_forcing:
            logger.info("create_linear (with forcing in dynamics: S_t = W^{-1}·(P_c·z + P_f·f))")
        else:
            logger.info("create_linear (no forcing: S_t = W^{-1}·P·W·S)")

        # data_field = deepcopy(self.data_field)
        data_field = torch.Tensor(self.data_field).to(device="cuda")
        if self.verbose:
            print("create_linear")
        for t in tqdm(range(tau_max, time_len)):
            # Climate-to-climate AR contribution: W^{-1} · P_climate · W · S_{t-τ:t-1}
            ar_contribution = torch.zeros(self.spatial_resolution, 1, device=dev, dtype=dtype)
            for i in range(tau_max):
                ar_contribution += (
                    weights_inv @ phi_climate[..., i] @ weights_torch @ data_field[..., t - 1 - i : t - i]
                )

            # Forcing-to-climate contribution (if forcing exists)
            if has_forcing:
                # Forcing contribution in latent space: P_forcing @ forcing_latents
                forcing_contrib_latent = torch.zeros(self.n_climate_modes, 1, device=dev, dtype=dtype)
                for i in range(tau_max):
                    lag_idx = t - 1 - i
                    if lag_idx >= 0:
                        # phi_forcing: (n_climate, n_forcing, tau_max)
                        # forcing_latents: (n_forcing, time)
                        forcing_at_lag = forcing_latents[:, lag_idx].unsqueeze(1)  # (n_forcing, 1)
                        forcing_contrib_latent += phi_forcing[..., i] @ forcing_at_lag

                # Project forcing contribution to observation space: W^{-1} @ forcing_contrib
                forcing_contrib_obs = weights_inv @ forcing_contrib_latent
                ar_contribution += forcing_contrib_obs

            # S_t = AR_contribution + forcing_contribution
            # Noise is already in data_field (added before dynamics by _add_noise_field)
            data_field[..., t : t + 1] += ar_contribution

        self.data_field = data_field[..., self.transient :].detach().cpu().numpy()

    def _create_intervened_nextstep(self, input_data, intervened_mode=None, intervention_value=None, intervened_t=None):
        """
        Not tested yet!!! see causal_graph_comparison for a proper function.

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

        phi = torch.Tensor(dict_to_matrix(self.links_coeffs)).to(device="cuda")
        next_step = torch.zeros(self.spatial_resolution).to(device="cuda")

        change_indices = []

        quadrant_row = intervened_mode // int(sqrt(self.n_vars))
        quadrant_col = intervened_mode % int(sqrt(self.n_vars))

        start_row = quadrant_row * self.spatial_resolution
        start_col = quadrant_col * self.spatial_resolution

        for i in range(self.spatial_resolution):
            for j in range(self.spatial_resolution):
                change_idx = (start_row + i) * int(sqrt(self.n_vars) * self.spatial_resolution) + (start_col + j)
                change_indices.append(change_idx)

        # perform intervention
        input_data[change_indices, intervened_t] += intervention_value

        for i in range(tau):
            next_step += weights_inv @ phi[..., i] @ weights @ input_data[:, -i]

        return next_step

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
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {sum(batch_losses)/len(batch_losses):.6f}")

        if self.verbose:
            print("Training of single-layer NNAR model completed.")

    def _create_nonlinear(self):
        """
        Generates nonlinear data by applying a (trained or simple) nonlinearity at each time step. This method uses the
        same logic as _create_linear to step forward in time and adds the nonlinearity (sigmoid) before adding to
        data_field.

        z_j(t) = tanh(sum_{k,τ} P_climate[j,k,τ] · z_k(t-τ)) + sum_{f,τ} P_forcing[j,f,τ] · forcing_f(t-τ) + N_j(t)
        S_t = W^{-1} · z(t)

        Note: The nonlinearity (tanh) is applied only to climate contributions.
        Forcing contributions remain linear as they are exogenous inputs.
        """
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        weights_np = np.linalg.pinv(self.mode_weights.reshape(self.n_climate_modes, -1))
        weights_inv = torch.tensor(weights_np, device=dev, dtype=dtype)
        mode_weights_tensor = torch.tensor(self.mode_weights.reshape(self.n_climate_modes, -1), device=dev, dtype=dtype)

        time_len = self.time_length + self.transient
        tau_max = self.tau_max

        # Get full transition matrix
        phi_full_np = dict_to_matrix(self.links_coeffs)

        # Extract climate-to-climate submatrix
        phi_climate_np = phi_full_np[: self.n_climate_modes, : self.n_climate_modes, :]

        # Check if we have forcing
        has_forcing = self.forcing_indices is not None and self.co2_latent_trajectory is not None

        if has_forcing:
            # Extract forcing-to-climate submatrix
            phi_forcing_np = phi_full_np[: self.n_climate_modes, self.n_climate_modes :, :]

            # Normalize both climate and forcing coefficients together
            phi_climate_np, phi_forcing_np = normalize_transition_with_forcing(
                phi_climate_np, phi_forcing_np, self.n_climate_modes
            )
            phi_forcing = torch.tensor(phi_forcing_np, device=dev, dtype=dtype)

            # Prepare forcing latent trajectories
            co2_latent = torch.tensor(self.co2_latent_trajectory, device=dev, dtype=dtype)

            if self.aerosol_latent_trajectory is not None:
                aerosol_latent = torch.tensor(self.aerosol_latent_trajectory, device=dev, dtype=dtype)
                forcing_latents = torch.cat([co2_latent.unsqueeze(0), aerosol_latent], dim=0)
            else:
                forcing_latents = co2_latent.unsqueeze(0)

            logger.debug(f"[NONLINEAR] Forcing latents shape: {forcing_latents.shape}")
        else:
            # No forcing - use raw coefficients (already bounded < 1 by create_links_coeffs)
            pass

        phi_climate = torch.tensor(phi_climate_np, device=dev, dtype=dtype)
        data_field = torch.tensor(self.data_field, device=dev, dtype=dtype)

        if has_forcing:
            logger.info("create_nonlinear (with forcing: tanh(climate) + linear(forcing))")
        else:
            logger.info("create_nonlinear (no forcing: tanh(climate) + noise)")

        for t in tqdm(range(tau_max, time_len)):
            # Climate contribution with nonlinearity
            nonlinear_contrib = torch.zeros(self.spatial_resolution, device=dev, dtype=dtype)
            for i in range(tau_max):
                lincombo = (
                    weights_inv @ phi_climate[..., i] @ mode_weights_tensor @ data_field[..., (t - 1 - i) : (t - i)]
                )
                lincombo_nl = torch.sigmoid(lincombo)
                nonlinear_contrib += lincombo_nl.squeeze(-1)

            # Forcing contribution (linear, exogenous)
            if has_forcing:
                forcing_contrib_latent = torch.zeros(self.n_climate_modes, 1, device=dev, dtype=dtype)
                for i in range(tau_max):
                    lag_idx = t - 1 - i
                    if lag_idx >= 0:
                        forcing_at_lag = forcing_latents[:, lag_idx].unsqueeze(1)
                        forcing_contrib_latent += phi_forcing[..., i] @ forcing_at_lag

                forcing_contrib_obs = weights_inv @ forcing_contrib_latent
                nonlinear_contrib += forcing_contrib_obs.squeeze(-1)

            # S_t = nonlinear_contribution + forcing_contribution
            # Noise is already in data_field (added before dynamics by _add_noise_field)
            data_field[:, t] += nonlinear_contrib

        self.data_field = data_field[:, self.transient :].detach().cpu().numpy()

    def _create_polynomial(self):
        """
        Example polynomial autoregression, e.g. x^2 for poly_degree=2. w_np =
        np.linalg.pinv(self.mode_weights.reshape(self.n_climate_modes, -1)) phi_np = dict_to_matrix(self.links_coeffs)

        z_j(t) = sum_deg (sum_{k,τ} P_climate[j,k,τ] · z_k(t-τ))^deg + sum_{f,τ} P_forcing[j,f,τ] · forcing_f(t-τ) + N_j(t)
        S_t = W^{-1} · z(t)

        Note: The polynomial nonlinearity is applied only to climate contributions.
        Forcing contributions remain linear as they are exogenous inputs.
        """
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float32

        w_np = np.linalg.pinv(self.mode_weights.reshape(self.n_climate_modes, -1))
        w_torch = torch.tensor(w_np, device=dev, dtype=dtype)
        mw_torch = torch.tensor(
            self.mode_weights.reshape(self.n_climate_modes, -1),
            device=dev,
            dtype=dtype,
        )

        time_len = self.time_length + self.transient
        tau_max = self.tau_max

        # Get full transition matrix
        phi_full_np = dict_to_matrix(self.links_coeffs)

        # Extract climate-to-climate submatrix
        phi_climate_np = phi_full_np[: self.n_climate_modes, : self.n_climate_modes, :]

        # Check if we have forcing
        has_forcing = self.forcing_indices is not None and self.co2_latent_trajectory is not None

        if has_forcing:
            # Extract forcing-to-climate submatrix
            phi_forcing_np = phi_full_np[: self.n_climate_modes, self.n_climate_modes :, :]

            # Normalize both climate and forcing coefficients together
            phi_climate_np, phi_forcing_np = normalize_transition_with_forcing(
                phi_climate_np, phi_forcing_np, self.n_climate_modes
            )
            phi_forcing = torch.tensor(phi_forcing_np, device=dev, dtype=dtype)

            # Prepare forcing latent trajectories
            co2_latent = torch.tensor(self.co2_latent_trajectory, device=dev, dtype=dtype)

            if self.aerosol_latent_trajectory is not None:
                aerosol_latent = torch.tensor(self.aerosol_latent_trajectory, device=dev, dtype=dtype)
                forcing_latents = torch.cat([co2_latent.unsqueeze(0), aerosol_latent], dim=0)
            else:
                forcing_latents = co2_latent.unsqueeze(0)

            logger.debug(f"[POLYNOMIAL] Forcing latents shape: {forcing_latents.shape}")
        else:
            # No forcing - use raw coefficients (already bounded < 1 by create_links_coeffs)
            pass

        phi_climate = torch.tensor(phi_climate_np, device=dev, dtype=dtype)
        data_field = torch.tensor(self.data_field, device=dev, dtype=dtype)

        if has_forcing:
            logger.info(
                f"create_polynomial (with forcing: poly(climate) + linear(forcing) + noise) degrees={self.poly_degrees}"
            )
        else:
            logger.info(f"create_polynomial (no forcing: poly(climate) + noise) degrees={self.poly_degrees}")

        for t in tqdm(range(tau_max, time_len)):
            # Climate contribution with polynomial nonlinearity
            poly_contrib = torch.zeros(self.spatial_resolution, device=dev, dtype=dtype)
            for i in range(tau_max):
                lincombo = w_torch @ phi_climate[..., i] @ mw_torch @ data_field[..., (t - 1 - i) : (t - i)]

                # For each requested polynomial degree, add its effect
                for deg in self.poly_degrees:
                    poly_contrib += lincombo**deg

            # Forcing contribution (linear, exogenous)
            if has_forcing:
                forcing_contrib_latent = torch.zeros(self.n_climate_modes, 1, device=dev, dtype=dtype)
                for i in range(tau_max):
                    lag_idx = t - 1 - i
                    if lag_idx >= 0:
                        forcing_at_lag = forcing_latents[:, lag_idx].unsqueeze(1)
                        forcing_contrib_latent += phi_forcing[..., i] @ forcing_at_lag

                forcing_contrib_obs = w_torch @ forcing_contrib_latent
                poly_contrib += forcing_contrib_obs.squeeze(-1)

            # S_t = polynomial_contribution + forcing_contribution
            # Noise is already in data_field (added before dynamics by _add_noise_field)
            data_field[:, t] += poly_contrib

        self.data_field = data_field[:, self.transient :].detach().cpu().numpy()
