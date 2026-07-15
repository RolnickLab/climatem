"""Visualization suite for SAVAR synthetic data: plots modes, causal graphs, time series, forcing trajectories, and evaluation metrics."""

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal

from climatem.plotting.plot_model_output import Plotter
from climatem.synthetic_data.utils import permute_matrices
from climatem.utils import get_logger

# Optional tigramite import for transfer entropy
try:
    from tigramite import data_processing as pp
    from tigramite import plotting as tp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    TIGRAMITE_AVAILABLE = True
except ImportError:
    TIGRAMITE_AVAILABLE = False

logger = get_logger(__name__)


class SavarPlotter(Plotter):
    """
    Specialized plotter for SAVAR synthetic data experiments.

    Inherits from the base Plotter class and adds SAVAR-specific visualization methods including feature map plotting,
    adjacency matrix alignment, and forcing diagnostics.
    """

    def __init__(self):
        """
        Initialize SavarPlotter, inheriting MCC/assignment tracking from Plotter.

        No additional state is stored; SAVAR-specific context is loaded lazily via prepare_savar_context.
        """
        super().__init__()
        logger.info("Initialized SavarPlotter for synthetic data visualization")

    def prepare_savar_context(self, learner):
        """Load the SAVAR ground-truth artifacts needed for plotting."""
        if not learner.plot_params.savar:
            return None

        savar_folder = learner.data_params.data_dir
        savar_params = learner.savar_params
        savar_fname = (
            f"m_{savar_params.n_per_col**2}_tl_{savar_params.time_len}_ifd_{savar_params.is_forced}_dif_{savar_params.difficulty}_ns_"
            f"{savar_params.noise_val}_ses_{savar_params.seasonality}_ol_{savar_params.overlap}_f1_{savar_params.f_1}_f2_{savar_params.f_2}"
            f"_ft1_{savar_params.f_time_1}_ft2_{savar_params.f_time_2}_rmp_{savar_params.ramp_type}_lin_{savar_params.linearity}"
            f"_pds_{savar_params.poly_degrees}_asp_{savar_params.aerosol_scale}_asc_{savar_params.aerosol_spatial_contrast}"
            f"_art_{savar_params.aerosol_ramp_up_time}_apt_{savar_params.aerosol_peak_time}_adt_{savar_params.aerosol_decline_time}"
        )

        savar_dataset_dir = Path(savar_folder) / savar_fname

        # --- Load core SAVAR data ---
        modes_gt = np.load(savar_dataset_dir / "modes.npy")
        savar_data = np.load(savar_dataset_dir / "savar.npy")  # (spatial, time)

        # --- Initialize GT holders ---
        learner.co2_gt_spatial = None
        learner.aerosol_gt_spatial = None
        learner.aerosol_gt_templates = None  # List of separate spatial templates (one per aerosol latent)
        learner.forcing_indices = None

        co2_forcing = None
        aerosol_forcing = None

        # --- Load forcing ground truth ---
        if savar_params.is_forced:
            co2_path = savar_dataset_dir / "co2_forcing.npy"
            aerosol_path = savar_dataset_dir / "aerosol_forcing.npy"
            aerosol_templates_path = savar_dataset_dir / "aerosol_spatial_templates.npy"

            if co2_path.exists():
                co2_forcing = np.load(co2_path)
                learner.co2_gt_spatial = co2_forcing.mean(axis=1).reshape(learner.lat, learner.lon)

            if aerosol_path.exists():
                aerosol_forcing = np.load(aerosol_path)
                learner.aerosol_gt_spatial = aerosol_forcing.mean(axis=1).reshape(learner.lat, learner.lon)

            # Load separate aerosol spatial templates (one per aerosol latent)
            if aerosol_templates_path.exists():
                templates = np.load(aerosol_templates_path)  # Shape: (n_aerosol, spatial_resolution)
                n_aerosol = savar_params.n_aerosol_latents
                # Only load as many templates as there are aerosol latents
                n_to_load = min(n_aerosol, templates.shape[0])
                learner.aerosol_gt_templates = [
                    templates[i].reshape(learner.lat, learner.lon) for i in range(n_to_load)
                ]
                logger.info(
                    f"Loaded {len(learner.aerosol_gt_templates)} aerosol spatial templates "
                    f"(n_aerosol_latents={n_aerosol}, file has {templates.shape[0]} templates)"
                )

            if hasattr(learner, "datamodule"):
                if hasattr(learner.datamodule, "forcing_indices") and learner.datamodule.forcing_indices is not None:
                    learner.forcing_indices = learner.datamodule.forcing_indices
                elif hasattr(learner.datamodule, "savar") and hasattr(learner.datamodule.savar, "forcing_indices"):
                    learner.forcing_indices = learner.datamodule.savar.forcing_indices

        # --- Diagnostic plots (run once, first time context is prepared) ---
        if (
            not getattr(learner, "_savar_gt_plots_done", False)
            and hasattr(learner, "datamodule")
            and hasattr(learner.datamodule, "savar")
            and learner.datamodule.savar is not None
        ):
            savar = learner.datamodule.savar

            # Signal-noise range plots
            # Get deterministic data from SAVAR object (generated without noise)
            deterministic_component = getattr(savar, "deterministic_data_field", None)
            logger.info(f"SAVAR data shape: {savar_data.shape}")
            logger.info(f"Deterministic component: {deterministic_component is not None}")

            if deterministic_component is not None:
                logger.info(f"Deterministic component shape: {deterministic_component.shape}")
                logger.info(
                    f"Deterministic min/max: {deterministic_component.min():.4f} / {deterministic_component.max():.4f}"
                )

                # Compute noise as (full_data - deterministic_data)
                if deterministic_component.shape == savar_data.shape:
                    noise_component = savar_data - deterministic_component
                    logger.info(
                        f"✓ Computed noise component from (data - deterministic), shape: {noise_component.shape}"
                    )
                    logger.info(f"Noise min/max: {noise_component.min():.4f} / {noise_component.max():.4f}")
                else:
                    logger.warning(
                        f"Shape mismatch: savar_data {savar_data.shape} vs deterministic {deterministic_component.shape}. "
                        "Cannot compute noise component."
                    )
                    noise_component = None
            else:
                logger.warning(
                    "deterministic_data_field not found in SAVAR object - cannot compute noise decomposition"
                )
                noise_component = None

            # Get linearity information
            linearity = getattr(savar, "linearity", "linear")
            poly_degrees = getattr(savar, "poly_degrees", None)

            self.plot_savar_signal_noise_decomposition(
                savar_data=savar_data,
                deterministic_component=deterministic_component,
                noise_component=noise_component,
                path=savar_dataset_dir,
                linearity=linearity,
                poly_degrees=poly_degrees,
            )

            # Forcing diagnostics
            if savar_params.is_forced:
                # Comprehensive forcing diagnostics
                self.plot_forcing_diagnostics(
                    savar_data=savar_data,
                    co2_forcing=co2_forcing,
                    aerosol_forcing=aerosol_forcing,
                    gt_co2_latent=getattr(savar, "co2_latent_trajectory", None),
                    gt_aerosol_latent=getattr(savar, "aerosol_latent_trajectory", None),
                    lat=learner.lat,
                    lon=learner.lon,
                    path=savar_dataset_dir,
                )

                # Simple forcing diagnostic plots
                if co2_forcing is not None:
                    self.plot_forcing_diagnostic(co2_forcing, "CO2", savar_dataset_dir, learner.lat, learner.lon)
                if aerosol_forcing is not None:
                    self.plot_forcing_diagnostic(
                        aerosol_forcing, "Aerosol", savar_dataset_dir, learner.lat, learner.lon
                    )

                if hasattr(savar, "aerosol_latent_trajectory") and savar.aerosol_latent_trajectory is not None:
                    self.plot_aerosol_latent_trajectories(
                        savar.aerosol_latent_trajectory,
                        savar_dataset_dir,
                    )

            learner._savar_gt_plots_done = True

        return modes_gt

    def plot_forcing_diagnostic(self, forcing_data, forcing_name, path, lat, lon):
        """
        Create comprehensive forcing diagnostic plots.

        Args:
            forcing_data: Forcing field of shape (spatial_resolution, time)
            forcing_name: Name of forcing (e.g., "CO2", "Aerosol")
            path: Save path (SAVAR data directory)
            lat: Latitude dimension
            lon: Longitude dimension
        """
        if forcing_data is None:
            return

        forcing_np = forcing_data  # Shape: (spatial_resolution, time)
        spatial_len = forcing_np.shape[0]
        time_len = forcing_np.shape[1]

        # Use consistent color for each forcing type
        color = "tab:red" if forcing_name == "CO2" else "tab:blue"

        logger.info(f"Creating comprehensive {forcing_name} forcing diagnostics")

        # 1. Spatial pattern plot (average over time)
        spatial_pattern_avg = forcing_np.mean(axis=1)  # Average over time

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.arange(spatial_len), spatial_pattern_avg, color=color, linewidth=2)
        ax.set_title(f"{forcing_name} Forcing: Spatial Pattern (Time-Averaged)")
        ax.set_xlabel("Grid point")
        ax.set_ylabel("Relative intensity")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        filename = f"{forcing_name.lower()}_spatial_pattern.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

        # 2. Heatmap over space and time
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(forcing_np, aspect="auto", interpolation="nearest", cmap="RdBu_r")
        ax.set_title(f"{forcing_name} Forcing Over Space and Time")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Grid point")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Forcing magnitude")
        fig.tight_layout()
        filename = f"{forcing_name.lower()}_heatmap.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

        # 3. Timeline with immediate and cumulative forcing
        time_axis = np.arange(time_len)
        mean_intensity = forcing_np.mean(axis=0)  # Spatial average at each timestep
        cumulative_intensity = np.cumsum(mean_intensity)

        fig, ax1 = plt.subplots(figsize=(12, 6))
        (line_immediate,) = ax1.plot(
            time_axis, mean_intensity, color=color, linewidth=2, label=f"Immediate {forcing_name}"
        )
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Average intensity", color=color)
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(alpha=0.3)

        ax2 = ax1.twinx()
        (line_cumulative,) = ax2.plot(
            time_axis, cumulative_intensity, color="tab:orange", linewidth=2, label=f"Cumulative {forcing_name}"
        )
        ax2.set_ylabel("Cumulative intensity", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        ax1.set_title(f"{forcing_name} Forcing Timeline")
        ax1.legend(handles=[line_immediate, line_cumulative], loc="upper left")
        fig.tight_layout()
        filename = f"{forcing_name.lower()}_timeline.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

        # 4. Spatial pattern at peak forcing (2D grid view)
        forcing_reshaped = forcing_np.T.reshape((time_len, lat, lon))  # (time, lat, lon)

        # Find peak time
        peak_idx = np.argmax(np.abs(mean_intensity))

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(forcing_reshaped[peak_idx], cmap="RdBu_r", aspect="auto", origin="upper")
        ax.set_title(f"{forcing_name} Forcing: Spatial Pattern at Peak (t={peak_idx})")
        ax.set_xlabel("Longitude index")
        ax.set_ylabel("Latitude index")
        plt.colorbar(im, ax=ax, label=f"{forcing_name} magnitude")
        fig.tight_layout()
        filename = f"{forcing_name.lower()}_peak_spatial.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

        logger.info(f"Completed comprehensive {forcing_name} forcing diagnostics")

    def plot_aerosol_latent_trajectories(self, aerosol_latent_traj, path, n_aerosol_latents=None):
        """
        Plot individual aerosol latent trajectories to verify they have distinct temporal patterns.

        Args:
            aerosol_latent_traj: Aerosol latent trajectories of shape (n_latents, time)
            path: Save path
            n_aerosol_latents: Number of aerosol latents (if None, inferred from data shape)
        """
        if aerosol_latent_traj is None:
            return

        n_latents = aerosol_latent_traj.shape[0]
        if n_aerosol_latents is None:
            n_aerosol_latents = n_latents
        time_len = aerosol_latent_traj.shape[1]
        time_axis = np.arange(time_len)

        # Plot 1: All latent trajectories on same axes
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, n_latents))
        for i in range(n_latents):
            ax.plot(
                time_axis,
                aerosol_latent_traj[i],
                color=colors[i],
                linewidth=1.5,
                label=f"Aerosol Latent {i}",
                alpha=0.8,
            )
        ax.set_title("Aerosol Latent Trajectories (Should Show Distinct Temporal Patterns)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Latent value")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(path / "aerosol_latent_trajectories.png", dpi=150)
        plt.close(fig)
        logger.info("Saved aerosol_latent_trajectories.png")

        # Plot 2: Separate subplots for each latent
        fig, axes = plt.subplots(n_latents, 1, figsize=(14, 3 * n_latents), sharex=True)
        if n_latents == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(time_axis, aerosol_latent_traj[i], color=colors[i], linewidth=1.5)
            ax.set_ylabel(f"Latent {i}")
            ax.grid(alpha=0.3)
            # Add peak marker
            peak_idx = np.argmin(aerosol_latent_traj[i])  # Aerosol is negative
            ax.axvline(x=peak_idx, color="red", linestyle="--", alpha=0.5, label=f"Peak at t={peak_idx}")
            ax.legend(loc="upper right")
        axes[-1].set_xlabel("Time step")
        fig.suptitle("Individual Aerosol Latent Trajectories with Peak Markers", fontsize=12)
        fig.tight_layout()
        fig.savefig(path / "aerosol_latent_trajectories_separate.png", dpi=150)
        plt.close(fig)
        logger.info("Saved aerosol_latent_trajectories_separate.png")

        # Plot 3: Correlation matrix
        corr_matrix = np.corrcoef(aerosol_latent_traj)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_title("Aerosol Latent Correlation Matrix\n(Target: Off-diagonal < 0.5)")
        ax.set_xticks(np.arange(n_latents))
        ax.set_yticks(np.arange(n_latents))
        ax.set_xticklabels([f"L{i}" for i in range(n_latents)])
        ax.set_yticklabels([f"L{i}" for i in range(n_latents)])
        # Add correlation values as text
        for i in range(n_latents):
            for j in range(n_latents):
                text_color = "white" if abs(corr_matrix[i, j]) > 0.5 else "black"
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=10)
        fig.colorbar(im, ax=ax, label="Correlation")
        fig.tight_layout()
        fig.savefig(path / "aerosol_latent_correlation_matrix.png", dpi=150)
        plt.close(fig)
        logger.info("Saved aerosol_latent_correlation_matrix.png")

        # Print correlation summary
        off_diag_corrs = []
        for i in range(n_latents):
            for j in range(i + 1, n_latents):
                off_diag_corrs.append(abs(corr_matrix[i, j]))
        avg_corr = np.mean(off_diag_corrs) if off_diag_corrs else 0
        max_corr = np.max(off_diag_corrs) if off_diag_corrs else 0
        logger.info(f"Aerosol latent correlations: avg={avg_corr:.4f}, max={max_corr:.4f}")
        logger.info(f"Aerosol latent correlations: avg={avg_corr:.4f}, max={max_corr:.4f} (target: < 0.5)")

    def plot_compare_predictions_savar(
        self,
        x_past: np.ndarray,
        y_true: np.ndarray,
        y_recons: np.ndarray,
        y_hat: np.ndarray,
        sample: int,
        lat: int,
        lon: int,
        path,
        filename_prefix=None,
        iteration: int = 0,
        valid: str = False,
        plot_through_time: bool = True,
    ):
        """Plot SAVAR predictions alongside reconstruction and ground truth on a latitude/longitude grid."""

        def _reshape(arr: np.ndarray, var_idx: int) -> np.ndarray:
            arr = np.asarray(arr)
            if arr.ndim == 2:
                if sample >= arr.shape[0]:
                    raise ValueError("Sample index out of bounds for provided array.")
                if var_idx != 0:
                    raise ValueError("Variable index must be 0 for 2D inputs shaped (n_samples, lat*lon).")
                flat = arr[sample]
            elif arr.ndim >= 3:
                if sample >= arr.shape[0] or var_idx >= arr.shape[1]:
                    raise ValueError("Sample or variable index out of bounds for provided array.")
                flat = arr[sample, var_idx]
            else:
                raise ValueError("Expected arrays with at least 2 dimensions.")
            flat = np.nan_to_num(flat)
            return flat.reshape(lat, lon)

        y_true = np.asarray(y_true)
        y_recons = np.asarray(y_recons)
        y_hat = np.asarray(y_hat)

        if y_true.ndim >= 4:
            y_true_current = y_true[:, 0]
            y_true_next = y_true[:, 1] if y_true.shape[1] > 1 else None
        else:
            y_true_current = y_true
            y_true_next = None

        n_vars = y_true_current.shape[1] if y_true_current.ndim > 2 else 1
        num_cols = 5 if y_true_next is not None else 4
        fig_width = 8 * num_cols
        fig_height = 16 if n_vars > 1 else 8

        fig, axs = plt.subplots(n_vars, num_cols, figsize=(fig_width, fig_height), layout="constrained")
        ax_rows = axs if n_vars > 1 else [axs]

        # Panel descriptions:
        # x_past: Last timestep from history (t-1)
        # y_true_current: Ground truth target (t)
        # y_recons: Reconstruction of y through encoder-decoder (t)
        # y_hat: Model prediction of y from history (t)
        panels = [
            ("Ground truth (t-1)\n[Last history step]", x_past),
            ("Ground truth (t)\n[Target]", y_true_current),
            ("Reconstruction (t)\n[Encode-Decode]", y_recons),
            ("Prediction (t)\n[Model output]", y_hat),
        ]
        if y_true_next is not None:
            panels.append(("Ground truth (t+1)\n[Future step]", y_true_next))

        for var_idx, ax_row in enumerate(ax_rows):
            grids = [_reshape(arr, var_idx) for _, arr in panels]
            vmin = min(grid.min() for grid in grids)
            vmax = max(grid.max() for grid in grids)
            im = None
            for ax, (title, _), grid in zip(ax_row, panels, grids):
                im = ax.imshow(grid, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("Longitude index")
                ax.set_ylabel("Latitude index")
            if im is not None:
                fig.colorbar(im, ax=ax_row[-1], orientation="vertical", shrink=1.0, label=f"Variable {var_idx}")

        if not valid:
            if plot_through_time:
                fname = f"compare_predictions_savar_{iteration}_sample_{sample}_train.png"
            else:
                fname = "compare_predictions_savar_train.png"
        else:
            if plot_through_time:
                fname = f"compare_predictions_savar_{iteration}_sample_{sample}_valid.png"
            else:
                fname = "compare_predictions_savar_valid.png"

        # Create descriptive overall title
        if y_true_next is not None:
            title = "SAVAR Prediction Comparison: History (t-1) | Target (t) | Reconstruction (t) | Prediction (t) | Future (t+1)"
        else:
            title = (
                "SAVAR Prediction Comparison: History (t-1) | Ground Truth (t) | Reconstruction (t) | Prediction (t)"
            )
        plt.suptitle(title, fontsize=24)
        if filename_prefix:
            fname = f"{filename_prefix}_{fname}"
        plt.savefig(path / fname, format="png")
        plt.close()

    def plot_savar_feature_maps(
        self,
        learner,
        w_adj,
        coordinates: np.ndarray,
        iteration: int,
        plot_through_time: bool,
        path,
    ):
        """
        Plot learned latent feature maps for SAVAR data.

        Creates separate visualizations for climate latents and forcing latents.
        """
        grid_shape = (learner.lat, learner.lon)
        # grid_shape_co2 = (1, 1)
        logger.info("Creating SAVAR feature maps visualization")
        w_adj = w_adj[0]  # Now w_adj_mean should be (lat*lon, num_latents)
        d_z = w_adj.shape[1]
        logger.info(f"Model dimension: d_z = {d_z}")

        # Get model reference (needed for decoder weight visualization)
        model = learner.model.module if hasattr(learner.model, "module") else learner.model

        # Get forcing configuration from SAVAR params if available, otherwise from model
        logger.info(
            f"Checking for savar_params: hasattr={hasattr(learner, 'savar_params')}, is not None={getattr(learner, 'savar_params', None) is not None}"
        )
        if hasattr(learner, "savar_params") and learner.savar_params is not None:
            # Use ground truth configuration from SAVAR params
            n_co2 = learner.savar_params.n_co2_latents
            n_aerosol = learner.savar_params.n_aerosol_latents
            # Get ground truth number of climate modes (n_per_col^2)
            n_climate = learner.savar_params.n_per_col**2  # + 1
            logger.info(
                f"Using SAVAR params (ground truth): n_co2={n_co2}, n_aerosol={n_aerosol}, n_climate={n_climate}"
            )

            # Sanity check: warn if model dimension doesn't match ground truth
            expected_d_z = n_climate + n_co2 + n_aerosol
            if d_z != expected_d_z:
                logger.warning(
                    f"Model dimension mismatch! Model has d_z={d_z} latents, "
                    f"but ground truth has {expected_d_z} ({n_climate} climate + {n_co2} CO2 + {n_aerosol} aerosol). "
                    f"Will plot only the first {n_climate} as climate latents."
                )
        else:
            # Fall back to model configuration
            use_forced_latents = getattr(model, "use_forced_latents", False)
            n_co2 = getattr(model, "n_forced_latents_co2", 0) if use_forced_latents else 0
            n_aerosol = getattr(model, "n_forced_latents_aerosol", 0) if use_forced_latents else 0
            n_climate = d_z - n_co2 - n_aerosol
            logger.info(f"Using model config: n_co2={n_co2}, n_aerosol={n_aerosol}, n_climate={n_climate}")

        # Split latent indices
        climate_indices = list(range(n_climate))
        co2_indices = list(range(n_co2))
        aerosol_indices = list(range(n_aerosol))

        logger.info(
            f"Climate latents: {climate_indices}, CO2 latents: {co2_indices}, Aerosol latents: {aerosol_indices}"
        )

        # ==== Figure 1: Climate Latents vs Ground Truth ====
        if len(climate_indices) > 0:
            self._plot_climate_feature_maps(
                learner, w_adj, grid_shape, climate_indices, iteration, plot_through_time, path
            )

        # ==== Figure 2: CO2 Forcing - Ground Truth vs Forcing Decoder Reconstruction ====
        if len(co2_indices) > 0 and learner.co2_gt_spatial is not None:
            self._plot_co2_feature_maps(learner, model, grid_shape, iteration, plot_through_time, path)

        # ==== Figure 3: Aerosol Forcing - Ground Truth vs Forcing Decoder Reconstruction ====
        has_templates = learner.aerosol_gt_templates is not None and len(learner.aerosol_gt_templates) > 0
        if len(aerosol_indices) > 0 and not (has_templates or learner.aerosol_gt_spatial is not None):
            logger.warning(
                f"Aerosol latent indices {aerosol_indices} found but no GT spatial data "
                "(aerosol_forcing.npy / aerosol_spatial_templates.npy missing). "
                "Regenerate SAVAR data to include aerosol GT files."
            )
        if len(aerosol_indices) > 0 and (has_templates or learner.aerosol_gt_spatial is not None):
            self._plot_aerosol_feature_maps(
                learner, model, grid_shape, aerosol_indices, has_templates, iteration, plot_through_time, path
            )

    def _plot_climate_feature_maps(
        self, learner, w_adj, grid_shape, climate_indices, iteration, plot_through_time, path
    ):
        """Plot climate latent feature maps vs ground truth."""
        n_climate_plots = len(climate_indices) + 1  # +1 for ground truth
        combined_map_n_rows = int(np.sqrt(n_climate_plots)) + 1
        combined_map_n_columns = int(np.ceil(n_climate_plots / combined_map_n_rows))

        fig, axs = plt.subplots(
            nrows=combined_map_n_rows,
            ncols=combined_map_n_columns,
            figsize=(combined_map_n_columns * 3, combined_map_n_rows * 3),
        )
        if combined_map_n_rows == 1:
            axs = axs.reshape(1, -1)

        # Plot ground truth climate modes
        ax = axs.flat[0]
        gt_modes_sum = (
            learner.datamodule.savar_gt_modes.sum(axis=0)
            if learner.datamodule.savar_gt_modes.ndim == 3
            else learner.datamodule.savar_gt_modes
        )
        vmin = np.min(learner.datamodule.savar_gt_noise + gt_modes_sum)
        vmax = np.max(learner.datamodule.savar_gt_noise + gt_modes_sum)
        im = ax.imshow(learner.datamodule.savar_gt_noise + gt_modes_sum, vmin=vmin, vmax=vmax, cmap="viridis")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title("Ground-Truth\nClimate Modes", fontsize="large")
        ax.tick_params(axis="both", labelsize="large")

        # Plot climate latent features
        for plot_idx, latent_idx in enumerate(climate_indices):
            ax = axs.flat[plot_idx + 1]
            data = w_adj[:, latent_idx].reshape(grid_shape)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(data, cmap="viridis")  # , vmin=vmin, vmax=vmax)
            plt.colorbar(im, cax=cax)
            ax.set_title(f"Climate Latent {latent_idx}", fontsize="large")
            ax.tick_params(axis="both", labelsize="large")

        for ax in axs.flat[n_climate_plots:]:
            fig.delaxes(ax)

        fig.tight_layout()
        fname = (
            f"spatial_aggregation_climate_{iteration}.png" if plot_through_time else "spatial_aggregation_climate.png"
        )
        plt.savefig(path / fname)
        plt.close()
        logger.info(f"Saved climate latent feature maps to {fname}")

    def _plot_co2_feature_maps(self, learner, model, grid_shape, iteration, plot_through_time, path):
        """Plot CO2 forcing ground truth vs learned decoder reconstruction."""
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))

        # Left: Ground truth
        im0 = axs[0].imshow(learner.co2_gt_spatial, cmap="RdBu_r")
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im0, cax=cax0)
        axs[0].set_title(f"Ground-Truth CO2 Forcing {learner.co2_gt_spatial.mean():.3f}", fontsize="large")

        # Right: Learned decoder weights (w_co2)
        try:
            autoencoder = model.autoencoder if hasattr(model, "autoencoder") else None
            if autoencoder is not None and hasattr(autoencoder, "get_w_co2"):
                w_co2 = autoencoder.get_w_co2().detach()
                if w_co2 is not None:
                    w_co2_np = w_co2.cpu().numpy()
                    spatial_pattern = w_co2_np[:, 0].reshape(grid_shape)
                    im1 = axs[1].imshow(spatial_pattern, cmap="RdBu_r")
                    divider1 = make_axes_locatable(axs[1])
                    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                    plt.colorbar(im1, cax=cax1)
                    axs[1].set_title(f"Learned CO2 Decoder Weights {spatial_pattern.mean():.3f}", fontsize="large")
                else:
                    axs[1].text(0.5, 0.5, "w_co2 not available", ha="center", va="center", transform=axs[1].transAxes)
                    axs[1].set_title("CO2 Decoder Weights", fontsize="large")
            else:
                axs[1].text(0.5, 0.5, "Old model\n(no w_co2)", ha="center", va="center", transform=axs[1].transAxes)
                axs[1].set_title("CO2 Decoder Weights", fontsize="large")
        except Exception as e:
            axs[1].text(
                0.5,
                0.5,
                f"Error, mean is {w_co2_np[:, 0].mean():.3f}",
                ha="center",
                va="center",
                transform=axs[1].transAxes,
            )
            logger.warning(f"Could not visualize CO2 decoder weights: {e}")

        for ax in axs:
            ax.tick_params(axis="both", labelsize="large")
        fig.tight_layout()
        fname = f"spatial_aggregation_co2_{iteration}.png" if plot_through_time else "spatial_aggregation_co2.png"
        plt.savefig(path / fname)
        plt.close()
        logger.info(f"Saved CO2 forcing comparison to {fname}")

    def _plot_aerosol_feature_maps(
        self, learner, model, grid_shape, aerosol_indices, has_templates, iteration, plot_through_time, path
    ):
        """Plot aerosol forcing ground truth vs learned decoder reconstruction."""
        n_aerosol = len(aerosol_indices)
        fig, axs = plt.subplots(2, n_aerosol, figsize=(4 * n_aerosol, 8))
        if n_aerosol == 1:
            axs = axs.reshape(2, 1)

        # Top row: Ground truth aerosol patterns
        for i in range(n_aerosol):
            if has_templates and i < len(learner.aerosol_gt_templates):
                im = axs[0, i].imshow(learner.aerosol_gt_templates[i], cmap="RdBu_r")
                divider = make_axes_locatable(axs[0, i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                axs[0, i].set_title(f"GT Aerosol Template {i}", fontsize="large")
            elif learner.aerosol_gt_spatial is not None and i == 0:
                im = axs[0, i].imshow(learner.aerosol_gt_spatial, cmap="RdBu_r")
                divider = make_axes_locatable(axs[0, i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)
                axs[0, i].set_title("GT Aerosol (combined)", fontsize="large")
            else:
                axs[0, i].text(
                    0.5, 0.5, "No template\n(regenerate data)", ha="center", va="center", transform=axs[0, i].transAxes
                )
                axs[0, i].set_title(f"GT Aerosol {i}", fontsize="large")

        # Bottom row: Forcing decoder weights
        try:
            autoencoder = model.autoencoder if hasattr(model, "autoencoder") else None
            if autoencoder is not None and hasattr(autoencoder, "get_w_aerosol"):
                w_aerosol = autoencoder.get_w_aerosol().detach()
                if w_aerosol is not None:
                    w_aerosol_np = w_aerosol.cpu().numpy()
                    for i in range(n_aerosol):
                        spatial_pattern = w_aerosol_np[:, i].reshape(grid_shape)
                        im = axs[1, i].imshow(spatial_pattern, cmap="RdBu_r")
                        divider = make_axes_locatable(axs[1, i])
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        plt.colorbar(im, cax=cax)
                        axs[1, i].set_title(f"Learned Decoder Weights {i}", fontsize="large")
                else:
                    for i in range(n_aerosol):
                        axs[1, i].text(
                            0.5, 0.5, "w_aerosol not available", ha="center", va="center", transform=axs[1, i].transAxes
                        )
            else:
                for i in range(n_aerosol):
                    axs[1, i].text(
                        0.5, 0.5, "Old model\n(no w_aerosol)", ha="center", va="center", transform=axs[1, i].transAxes
                    )
        except Exception as e:
            logger.warning(f"Could not visualize aerosol decoder weights: {e}")
            for i in range(n_aerosol):
                axs[1, i].text(0.5, 0.5, "Error", ha="center", va="center", transform=axs[1, i].transAxes)

        for ax in axs.flat:
            ax.tick_params(axis="both", labelsize="large")
        fig.tight_layout()
        fname = (
            f"spatial_aggregation_aerosol_{iteration}.png" if plot_through_time else "spatial_aggregation_aerosol.png"
        )
        plt.savefig(path / fname)
        plt.close()
        logger.info(f"Saved aerosol forcing comparison to {fname}")

    def plot_decoder_connectivity_heatmap(
        self,
        learner,
        w_adj,
        iteration: int,
        plot_through_time: bool,
        path,
    ):
        """
        Plot decoder connectivity heatmap showing spatial × latents connections.

        NOTE: With the architectural fix, only CLIMATE latents are used in observation decoding.
        Forcing latents (CO2, aerosol) are excluded from the observation decoder and only
        contribute through forcing decoders and the causal transition model. When forced
        latents are present, they are shown here for reference.
        """
        logger.info("Creating decoder connectivity heatmap")
        w_adj = w_adj[0]  # Shape: (spatial_resolution, num_latents)
        d_z = w_adj.shape[1]

        # Detect forcing configuration from model
        model = learner.model.module if hasattr(learner.model, "module") else learner.model
        use_forced_latents = getattr(model, "use_forced_latents", False)
        n_co2 = getattr(model, "n_forced_latents_co2", 0) if use_forced_latents else 0
        n_aerosol = getattr(model, "n_forced_latents_aerosol", 0) if use_forced_latents else 0
        n_climate = d_z - n_co2 - n_aerosol

        include_forcings = use_forced_latents and (n_co2 + n_aerosol) > 0

        if include_forcings:
            w_adj_plot = w_adj
            latent_labels = (
                [f"Climate {i}" for i in range(n_climate)]
                + [f"CO2 {i}" for i in range(n_co2)]
                + [f"Aerosol {i}" for i in range(n_aerosol)]
            )
            latent_types = ["Climate"] * n_climate + ["CO2"] * n_co2 + ["Aerosol"] * n_aerosol
        else:
            # Only show climate latent columns (actually used in observation decoding)
            w_adj_plot = w_adj[:, :n_climate]
            latent_labels = [f"Climate {i}" for i in range(n_climate)]
            latent_types = ["Climate"] * n_climate

        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # Left plot: Decoder connectivity heatmap (spatial × latents)
        im1 = ax1.imshow(np.abs(w_adj_plot), aspect="auto", cmap="viridis", interpolation="nearest")
        ax1.set_xlabel("Latent Index", fontsize=12)
        ax1.set_ylabel("Spatial Location", fontsize=12)
        if include_forcings:
            ax1.set_title("Observation Decoder Connectivity\n(Forcing latents shown for reference)", fontsize=14)
        else:
            ax1.set_title("Observation Decoder Connectivity\n(Climate Latents Only)", fontsize=14)
        ax1.set_xticks(range(len(latent_labels)))
        ax1.set_xticklabels(latent_labels, rotation=45, ha="right", fontsize=10)
        plt.colorbar(im1, ax=ax1, label="Weight Magnitude")

        # Right plot: Latent-wise L2 norms (bar chart)
        latent_norms = np.linalg.norm(w_adj_plot, axis=0)  # L2 norm for each latent
        if include_forcings:
            type_colors = {"Climate": "tab:green", "CO2": "tab:red", "Aerosol": "tab:blue"}
        else:
            type_colors = {"Climate": "tab:blue"}
        bar_colors = [type_colors[label] for label in latent_types]

        ax2.bar(range(len(latent_norms)), latent_norms, color=bar_colors, alpha=0.7)
        ax2.set_xlabel("Latent Index", fontsize=12)
        ax2.set_ylabel("Decoder Weight L2 Norm", fontsize=12)
        if include_forcings:
            ax2.set_title("Latent Usage\n(Forcing latents shown for reference)", fontsize=14)
        else:
            ax2.set_title("Climate Latent Usage\n(Forcing latents excluded from obs decoder)", fontsize=14)
        ax2.set_xticks(range(len(latent_labels)))
        ax2.set_xticklabels(latent_labels, rotation=45, ha="right", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)
        if include_forcings:
            legend_patches = [
                Patch(facecolor=type_colors[name], label=name)
                for name in ["Climate", "CO2", "Aerosol"]
                if name in latent_types
            ]
            ax2.legend(handles=legend_patches, loc="upper right", fontsize=9)

        # Log latent norms
        logger.info(f"Latent norms: {latent_norms}")
        if use_forced_latents and include_forcings:
            logger.info(f"Included forcing latents in plot for reference ({n_co2} CO2, {n_aerosol} aerosol)")
        elif use_forced_latents:
            logger.info(f"Note: Forcing latent columns ({n_co2} CO2, {n_aerosol} aerosol) excluded from obs decoder")

        fig.tight_layout()

        if plot_through_time:
            fname = f"decoder_connectivity_{iteration}.png"
        else:
            fname = "decoder_connectivity.png"

        plt.savefig(path / fname, dpi=150)
        plt.close()
        logger.info(f"Saved decoder connectivity heatmap to {fname}")

    def plot_adjacency_matrix_savar(
        self,
        learner,
        mat1: np.ndarray,
        mat2: np.ndarray,
        modes_gt,
        modes_inferred,
        path,
        name_suffix: str,
        no_gt: bool = False,
        iteration: int = 0,
        plot_through_time: bool = True,
    ):
        """
        Plot adjacency matrices for SAVAR runs after aligning inferred modes with the ground truth.

        Uses spatial proximity of mode centroids to find the optimal permutation before plotting. Fully self-contained —
        does not delegate to Plotter.plot_adjacency_matrix.
        """

        mask = np.array(mat1, copy=True)
        tau, d_x, _ = mask.shape
        n_climate = modes_gt.shape[0]
        n_forcings = d_x - n_climate
        names = [f"C{n}" for n in range(n_climate)] + [f"F{n}" for n in range(n_forcings)]

        ncols = min(tau, 3)
        nrows = math.ceil(tau / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.2 * ncols, 3.2 * nrows),
            squeeze=False,
        )

        for t in range(tau):
            ax = axes[t // ncols, t % ncols]
            # matrix is [target, source] -> imshow
            # matrix is [source, target] -> imshow transpose
            im = ax.imshow(mask[t], cmap="viridis", vmin=0, vmax=1)

            ax.set_title(f"t-{tau-t-1}")
            ax.set_xlabel("source")
            ax.set_ylabel("target")

            ax.set_xticks(range(d_x))
            ax.set_yticks(range(d_x))

            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_yticklabels(names)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        # hide unused axes
        for t in range(tau, nrows * ncols):
            axes[t // ncols, t % ncols].axis("off")

        if plot_through_time:
            fname = f"adjacency_{name_suffix}_{iteration}_raw.png"
        else:
            fname = f"adjacency_{name_suffix}_raw.png"

        plt.savefig(path / fname, format="png")
        plt.close()

        effective_no_gt = no_gt or mat2 is None
        lat = getattr(learner, "lat", None)
        lon = getattr(learner, "lon", None)

        mat1_to_plot = np.array(mat1, copy=True)[: mat2.shape[0]]
        tau = mat1_to_plot.shape[0]
        # Permute learned adjacency to align with ground-truth modes.
        # Only permute climate-mode submatrix; forcing latent rows/columns stay in place.
        if (
            not effective_no_gt
            and lat is not None
            and lon is not None
            and modes_gt is not None
            and modes_inferred is not None
        ):
            # modes_inferred may be (d, d_x, d_z) from get_w_decoder(); squeeze to (d_x, d_z)
            mi = np.squeeze(modes_inferred)

            n_climate = modes_gt.shape[0]
            d_total = mat1_to_plot.shape[1]
            if n_climate < d_total:
                # Slice latent axis (last) to climate-only columns, then permute climate submatrix
                climate_sub = mat1_to_plot[:, :n_climate, :n_climate].copy()
                mi_climate = mi[..., :n_climate]
                climate_sub = permute_matrices(lat, lon, mi_climate, modes_gt, climate_sub, tau)
                mat1_to_plot[:, :n_climate, :n_climate] = climate_sub
            else:
                mat1_to_plot = permute_matrices(lat, lon, mi, modes_gt, mat1_to_plot, tau)

        # Prepare ground truth adjacency matrix
        forcing_indices = getattr(learner, "forcing_indices", None)
        mat2_aligned = None
        if not effective_no_gt:
            mat2_aligned = np.array(mat2, copy=True)
            if mat2_aligned.ndim == 2:
                mat2_aligned = mat2_aligned[None, ...]
            if mat2_aligned.shape[0] != tau:
                if mat2_aligned.shape[0] == 1:
                    mat2_aligned = np.repeat(mat2_aligned, tau, axis=0)
                else:
                    mat2_aligned = mat2_aligned[:tau]
            target_d = mat1_to_plot.shape[1]
            if mat2_aligned.shape[1] != target_d or mat2_aligned.shape[2] != target_d:
                resized = np.zeros((tau, target_d, target_d), dtype=mat2_aligned.dtype)
                min_d = min(target_d, mat2_aligned.shape[1], mat2_aligned.shape[2])
                resized[:, :min_d, :min_d] = mat2_aligned[:, :min_d, :min_d]
                mat2_aligned = resized

        # Create figure and plot
        fig = plt.figure(constrained_layout=True)
        fig.suptitle("Adjacency matrices: learned vs ground-truth")

        if tau == 1:
            self._plot_adjacency_single_time(fig, mat1_to_plot, mat2_aligned, effective_no_gt, forcing_indices)
        else:
            self._plot_adjacency_through_time(fig, mat1_to_plot, mat2_aligned, effective_no_gt, tau, forcing_indices)

        if plot_through_time:
            fname = f"adjacency_{name_suffix}_{iteration}.png"
        else:
            fname = f"adjacency_{name_suffix}.png"

        plt.savefig(path / fname, format="png")
        plt.close()

    def plot_original_savar(self, data, lat, lon, path):
        """
        Create an animated GIF of the original SAVAR data over time.

        Args:
            data: SAVAR data of shape (n_modes, spatial_resolution, time) or similar
            lat: Latitude dimension
            lon: Longitude dimension
            path: Save path for the GIF
        """
        logger.info(f"Creating SAVAR original data animation - data shape: {data.shape}")

        # Get the dimensions
        time_steps = data.shape[1]
        data_reshaped = data.T.reshape((time_steps, lat, lon))

        # Calculate the average over the time axis
        avg_data = np.mean(data_reshaped, axis=0)

        # Determine the global min and max from the averaged data for consistent color scaling
        vmin = np.min(avg_data)
        vmax = np.max(avg_data)

        fig, ax = plt.subplots(figsize=(lon / 10, lat / 10))
        cax = ax.imshow(data_reshaped[0], aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)

        def animate(i):
            cax.set_data(data_reshaped[i])
            ax.set_title(f"SAVAR Original Data - Time step: {i+1}/{time_steps}")
            return (cax,)

        # Create an animation (first 100 timesteps to keep file size reasonable)
        n_frames = min(100, time_steps)
        ani = animation.FuncAnimation(fig, animate, frames=n_frames, blit=False)

        # Save the animation as a GIF
        ani.save(path, writer="pillow", fps=10)
        plt.close()

        logger.info(f"Saved SAVAR original data animation to {path}")

    def compute_snr_metrics(self, signal_data: np.ndarray, noise_data: np.ndarray) -> dict:
        """
        Compute signal-to-noise ratio metrics.

        Args:
            signal_data: Signal component (spatial, time)
            noise_data: Noise component (spatial, time)

        Returns:
            Dictionary with SNR metrics
        """
        # Compute standard deviations
        signal_std = np.std(signal_data)
        noise_std = np.std(noise_data)

        # SNR by standard deviation
        snr_std = signal_std / noise_std if noise_std > 0 else np.inf

        # SNR in dB
        snr_db = 10 * np.log10(snr_std) if snr_std > 0 else -np.inf

        # Amplitude-based SNR (max range)
        signal_range = signal_data.max() - signal_data.min()
        noise_range = noise_data.max() - noise_data.min()
        snr_amplitude = signal_range / noise_range if noise_range > 0 else np.inf

        # Root mean square
        signal_rms = np.sqrt(np.mean(signal_data**2))
        noise_rms = np.sqrt(np.mean(noise_data**2))
        snr_rms = signal_rms / noise_rms if noise_rms > 0 else np.inf

        return {
            "signal_std": signal_std,
            "noise_std": noise_std,
            "snr_std": snr_std,
            "snr_db": snr_db,
            "signal_range": signal_range,
            "noise_range": noise_range,
            "snr_amplitude": snr_amplitude,
            "signal_rms": signal_rms,
            "noise_rms": noise_rms,
            "snr_rms": snr_rms,
        }

    def plot_savar_data_noise_ranges(
        self,
        savar_data: np.ndarray,
        noise_data: Optional[np.ndarray],
        path: Path,
        title: str = "SAVAR data",
    ) -> None:
        """
        Plot data field range and noise field range over time, similar to signal-to-noise diagnostics.

        Creates a filled area plot showing the min-max range of data values and noise values
        across spatial dimensions at each time step.

        Args:
            savar_data: Climate data of shape (spatial, time)
            noise_data: Noise data of shape (spatial, time) or None
            path: Save directory path
            title: Title prefix for the plot
        """
        logger.info(f"Plotting data/noise field ranges for {title}")
        logger.info(f"  savar_data shape: {savar_data.shape}")
        logger.info(f"  noise_data: {noise_data is not None}")
        if noise_data is not None:
            logger.info(f"  noise_data shape: {noise_data.shape}")

        spatial_dim, time_len = savar_data.shape

        # Compute min/max range across spatial dimension at each time step
        data_min = savar_data.min(axis=0)
        data_max = savar_data.max(axis=0)
        time_axis = np.arange(time_len)

        fig, ax = plt.subplots(figsize=(14, 5))

        # Plot data field range
        ax.fill_between(time_axis, data_min, data_max, color="steelblue", alpha=0.7, label="data field range")

        # Plot noise field range if available and compute SNR
        snr_text = ""
        if noise_data is not None:
            logger.info("  → Plotting noise field range")
            noise_min = noise_data.min(axis=0)
            noise_max = noise_data.max(axis=0)
            ax.fill_between(time_axis, noise_min, noise_max, color="sandybrown", alpha=0.7, label="noise field range")

            # Compute SNR metrics
            snr_metrics = self.compute_snr_metrics(savar_data, noise_data)
            logger.info(f"  → SNR (std): {snr_metrics['snr_std']:.3f} ({snr_metrics['snr_db']:.2f} dB)")
            logger.info(f"  → Signal std: {snr_metrics['signal_std']:.4f}, Noise std: {snr_metrics['noise_std']:.4f}")
            logger.info(f"  → SNR (amplitude): {snr_metrics['snr_amplitude']:.3f}")

            # Create SNR annotation text
            snr_text = (
                f"SNR: {snr_metrics['snr_std']:.2f} ({snr_metrics['snr_db']:.1f} dB)\n"
                f"Signal std: {snr_metrics['signal_std']:.4f}  |  Noise std: {snr_metrics['noise_std']:.4f}"
            )
        else:
            logger.warning("  → Noise data is None, skipping noise plot")

        ax.set_xlabel("Time step", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(alpha=0.3)

        # Add SNR text box if available
        if snr_text:
            ax.text(
                0.02,
                0.02,
                snr_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="bottom",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

        fig.tight_layout()
        filename = f"{title.lower().replace(' ', '_')}_field_ranges.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

    def plot_savar_signal_noise_decomposition(
        self,
        savar_data: np.ndarray,
        deterministic_component: Optional[np.ndarray],
        noise_component: Optional[np.ndarray],
        path: Path,
        linearity: str = "linear",
        poly_degrees: Optional[List[int]] = None,
    ) -> None:
        """
        Create separate plots for linear/nonlinear/polynomial components showing data vs noise ranges.

        This mimics the style of the reference plots showing signal-to-noise characteristics
        for different data types.

        Args:
            savar_data: Full SAVAR climate data (spatial, time)
            deterministic_component: Deterministic signal component (spatial, time) or None
            noise_component: Noise component (spatial, time) or None
            path: Save directory path
            linearity: Type of data ("linear", "nonlinear", "polynomial")
            poly_degrees: List of polynomial degrees if linearity is "polynomial"
        """
        logger.info(f"Creating SAVAR signal-noise decomposition plots (linearity={linearity})")

        # Determine title prefix based on linearity type
        if linearity == "linear":
            data_type = "Linear data"
        elif linearity == "nonlinear":
            data_type = "Nonlinear data"
        elif linearity == "polynomial":
            if poly_degrees:
                deg_str = ",".join(map(str, poly_degrees))
                data_type = f"Polynomial data (degrees {deg_str})"
            else:
                data_type = "Polynomial data"
        else:
            data_type = f"{linearity.capitalize()} data"

        # Plot 1: Full data range
        self.plot_savar_data_noise_ranges(
            savar_data=savar_data,
            noise_data=noise_component,
            path=path,
            title=data_type,
        )

        # Plot 2: If we have deterministic component, plot it separately
        if deterministic_component is not None:
            self.plot_savar_data_noise_ranges(
                savar_data=deterministic_component,
                noise_data=noise_component,
                path=path,
                title=f"{data_type} - Deterministic Component",
            )

            # Plot 3: Residual (data - deterministic) vs noise
            residual = savar_data - deterministic_component
            self.plot_savar_data_noise_ranges(
                savar_data=residual,
                noise_data=noise_component,
                path=path,
                title=f"{data_type} - Residual",
            )

        logger.info(f"Completed SAVAR signal-noise decomposition plots for {data_type}")

    # =========================================================================
    # FORCING DIAGNOSTIC PLOTS - Global Correlation Analysis
    # =========================================================================

    # needs to be edited, gotta ask julien again how he wants it
    def plot_correlation_heatmap_over_time(
        self,
        forcing: np.ndarray,
        climate: np.ndarray,
        forcing_name: str,
        window_size: int,
        path: Path,
    ) -> None:
        """
        Plot sliding window correlation between forcing and climate over time.

        Args:
            forcing: Forcing data of shape (spatial, time) or (time,)
            climate: Climate data of shape (spatial, time)
            forcing_name: Name of forcing
            window_size: Size of sliding window
            path: Save path
        """
        logger.info(f"Computing sliding window correlation for {forcing_name}")

        # Ensure forcing and climate have matching time dimensions
        time_forcing = forcing.shape[-1]
        time_climate = climate.shape[-1]
        if time_forcing != time_climate:
            min_time = min(time_forcing, time_climate)
            logger.warning(
                f"Time dimension mismatch: forcing has {time_forcing} timesteps, "
                f"climate has {time_climate} timesteps. Trimming both to {min_time}."
            )
            forcing = forcing[..., :min_time]
            climate = climate[..., :min_time]

        # Aggregate spatially
        forcing_ts = forcing.mean(axis=0) if forcing.ndim == 2 else forcing
        climate_ts = climate.mean(axis=0) if climate.ndim == 2 else climate

        time_len = len(forcing_ts)
        n_windows = time_len - window_size + 1

        if n_windows < 10:
            logger.warning(f"Too few windows ({n_windows}), skipping correlation heatmap")
            return

        # Compute correlation in sliding windows
        correlations = []
        window_centers = []

        for i in range(n_windows):
            window_forcing = forcing_ts[i : i + window_size]
            window_climate = climate_ts[i : i + window_size]
            corr = np.corrcoef(window_forcing, window_climate)[0, 1]
            correlations.append(corr)
            window_centers.append(i + window_size // 2)

        correlations = np.array(correlations)
        window_centers = np.array(window_centers)

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        color = "tab:red" if forcing_name == "CO2" else "tab:blue"

        # Top: forcing and climate time series
        ax1_twin = ax1.twinx()
        ax1.plot(np.arange(time_len), forcing_ts, color=color, linewidth=1, label=f"{forcing_name}", alpha=0.8)
        ax1_twin.plot(np.arange(time_len), climate_ts, color="green", linewidth=1, label="Climate", alpha=0.8)
        ax1.set_ylabel(f"{forcing_name} (normalized)", color=color)
        ax1_twin.set_ylabel("Climate (normalized)", color="green")
        ax1.legend(loc="upper left")
        ax1_twin.legend(loc="upper right")
        ax1.set_title(f"{forcing_name} and Climate Time Series", fontsize=12)

        # Bottom: sliding window correlation
        ax2.fill_between(window_centers, 0, correlations, where=(correlations > 0), color="green", alpha=0.5)
        ax2.fill_between(window_centers, 0, correlations, where=(correlations < 0), color="red", alpha=0.5)
        ax2.plot(window_centers, correlations, color="black", linewidth=1)
        ax2.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
        ax2.set_xlabel("Time step", fontsize=12)
        ax2.set_ylabel(f"Correlation (window={window_size})", fontsize=12)
        ax2.set_title(f"Sliding Window Correlation: {forcing_name} vs Climate", fontsize=12)
        ax2.set_ylim(-1, 1)
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        filename = f"{forcing_name.lower()}_correlation_over_time.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

    # =========================================================================
    # FORCING DIAGNOSTIC PLOTS - Spatial Correlation Analysis
    # =========================================================================

    def plot_pointwise_correlation_map(
        self,
        forcing: np.ndarray,
        climate: np.ndarray,
        forcing_name: str,
        lat: int,
        lon: int,
        path: Path,
        lag: int = 0,
    ) -> None:
        """
        Plot correlation between forcing and climate at each grid point.

        Args:
            forcing: Forcing data (spatial, time) or (time,) - will broadcast if 1D
            climate: Climate data (spatial, time)
            forcing_name: Name of forcing
            lat, lon: Grid dimensions
            path: Save path
            lag: Time lag to apply (positive = forcing leads)
        """
        logger.info(f"Computing pointwise correlation map for {forcing_name} (lag={lag})")

        # Ensure forcing and climate have matching time dimensions
        time_forcing = forcing.shape[-1]
        time_climate = climate.shape[-1]
        if time_forcing != time_climate:
            min_time = min(time_forcing, time_climate)
            logger.warning(
                f"Time dimension mismatch: forcing has {time_forcing} timesteps, "
                f"climate has {time_climate} timesteps. Trimming both to {min_time}."
            )
            forcing = forcing[..., :min_time]
            climate = climate[..., :min_time]

        spatial_size = lat * lon

        # Handle forcing shape
        if forcing.ndim == 1:
            # Broadcast 1D forcing to all spatial points
            forcing_broadcast = np.tile(forcing, (spatial_size, 1))
        else:
            forcing_broadcast = forcing

        # Apply lag
        if lag > 0:
            forcing_aligned = forcing_broadcast[:, lag:]
            climate_aligned = climate[:, :-lag]
        elif lag < 0:
            forcing_aligned = forcing_broadcast[:, :lag]
            climate_aligned = climate[:, -lag:]
        else:
            forcing_aligned = forcing_broadcast
            climate_aligned = climate

        # Compute correlation at each spatial point
        correlations = np.zeros(spatial_size)
        for i in range(spatial_size):
            f = forcing_aligned[i] if forcing_aligned.ndim == 2 else forcing_aligned
            c = climate_aligned[i]
            if np.std(f) > 1e-10 and np.std(c) > 1e-10:
                correlations[i] = np.corrcoef(f, c)[0, 1]
            else:
                correlations[i] = 0

        # Reshape to grid
        corr_map = correlations.reshape(lat, lon)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_map, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xlabel("Longitude index", fontsize=12)
        ax.set_ylabel("Latitude index", fontsize=12)
        lag_str = f" (lag={lag})" if lag != 0 else ""
        ax.set_title(f"Pointwise Correlation: {forcing_name} vs Climate{lag_str}", fontsize=14)
        fig.colorbar(im, ax=ax, label="Correlation")

        # Add statistics
        mean_corr = np.nanmean(corr_map)
        max_corr = np.nanmax(np.abs(corr_map))
        ax.text(
            0.02,
            0.98,
            f"Mean: {mean_corr:.3f}\nMax |r|: {max_corr:.3f}",
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        fig.tight_layout()
        filename = f"{forcing_name.lower()}_pointwise_correlation_lag{lag}.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename} (mean r={mean_corr:.3f})")

    # =========================================================================
    # FORCING DIAGNOSTIC PLOTS - Joint Animations
    # =========================================================================

    def plot_joint_forcing_climate_animation(
        self,
        co2_forcing: Optional[np.ndarray],
        aerosol_forcing: Optional[np.ndarray],
        climate: np.ndarray,
        lat: int,
        lon: int,
        path: Path,
        max_frames: int = 200,
    ) -> None:
        """
        Create side-by-side animation of forcing fields and climate field.

        Args:
            co2_forcing: CO2 forcing (spatial, time) or None
            aerosol_forcing: Aerosol forcing (spatial, time) or None
            climate: Climate data (spatial, time)
            lat, lon: Grid dimensions
            path: Save path
            max_frames: Maximum number of frames
        """
        logger.info("Creating joint forcing-climate animation")

        # Ensure forcing and climate have matching time dimensions
        time_climate = climate.shape[-1]
        if co2_forcing is not None:
            time_co2 = co2_forcing.shape[-1]
            if time_co2 != time_climate:
                min_time = min(time_co2, time_climate)
                logger.warning(
                    f"Time dimension mismatch: CO2 forcing has {time_co2} timesteps, "
                    f"climate has {time_climate} timesteps. Trimming both to {min_time}."
                )
                co2_forcing = co2_forcing[..., :min_time]
                climate = climate[..., :min_time]
                time_climate = min_time

        if aerosol_forcing is not None:
            time_aerosol = aerosol_forcing.shape[-1]
            if time_aerosol != time_climate:
                min_time = min(time_aerosol, time_climate)
                logger.warning(
                    f"Time dimension mismatch: aerosol forcing has {time_aerosol} timesteps, "
                    f"climate has {time_climate} timesteps. Trimming both to {min_time}."
                )
                aerosol_forcing = aerosol_forcing[..., :min_time]
                climate = climate[..., :min_time]

        time_len = climate.shape[1]
        frame_stride = max(1, time_len // max_frames)
        frame_indices = np.arange(0, time_len, frame_stride, dtype=int)

        # Reshape data
        climate_reshaped = climate.T.reshape(time_len, lat, lon)

        # Determine number of columns
        n_cols = 1  # Climate is always shown
        forcings_to_plot = []
        if co2_forcing is not None:
            co2_reshaped = co2_forcing.T.reshape(time_len, lat, lon)
            forcings_to_plot.append(("CO2", co2_reshaped, "Reds"))
            n_cols += 1
        if aerosol_forcing is not None:
            aerosol_reshaped = aerosol_forcing.T.reshape(time_len, lat, lon)
            forcings_to_plot.append(("Aerosol", aerosol_reshaped, "Blues"))
            n_cols += 1

        # Create figure
        fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]

        # Initialize images
        images = []
        for idx, (name, data, cmap) in enumerate(forcings_to_plot):
            vmin, vmax = data.min(), data.max()
            if vmin == vmax:
                vmax = vmin + 1e-6
            im = axes[idx].imshow(data[0], cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", animated=True)
            axes[idx].set_title(f"{name} (t=0)")
            axes[idx].set_xlabel("Lon")
            axes[idx].set_ylabel("Lat")
            fig.colorbar(im, ax=axes[idx], shrink=0.8)
            images.append((im, name, data))

        # Climate
        climate_vmin, climate_vmax = climate_reshaped.min(), climate_reshaped.max()
        climate_im = axes[-1].imshow(
            climate_reshaped[0], cmap="RdBu_r", vmin=climate_vmin, vmax=climate_vmax, aspect="auto", animated=True
        )
        axes[-1].set_title("Climate (t=0)")
        axes[-1].set_xlabel("Lon")
        fig.colorbar(climate_im, ax=axes[-1], shrink=0.8)

        def update(frame_idx):
            t = frame_indices[frame_idx]
            for im, name, data in images:
                im.set_array(data[t])
            climate_im.set_array(climate_reshaped[t])
            for idx, (name, _, _) in enumerate(forcings_to_plot):
                axes[idx].set_title(f"{name} (t={t})")
            axes[-1].set_title(f"Climate (t={t})")
            return [im for im, _, _ in images] + [climate_im]

        anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=100, blit=True)
        writer = animation.PillowWriter(fps=10)
        filename = "joint_forcing_climate.gif"
        anim.save(path / filename, writer=writer)
        plt.close(fig)
        logger.info(f"Saved {filename}")

    # =========================================================================
    # FORCING DIAGNOSTIC PLOTS - Variance Attribution
    # =========================================================================

    # rescale this vertically - variance difference too big?
    def plot_variance_explained_by_forcing(
        self,
        co2_forcing: Optional[np.ndarray],
        aerosol_forcing: Optional[np.ndarray],
        climate: np.ndarray,
        lat: int,
        lon: int,
        path: Path,
        window_size: int = 100,
    ) -> None:
        """
        Plot R² of forcing → climate regression over time (sliding window).

        Args:
            co2_forcing: CO2 forcing (spatial, time) or None
            aerosol_forcing: Aerosol forcing (spatial, time) or None
            climate: Climate data (spatial, time)
            lat, lon: Grid dimensions
            path: Save path
            window_size: Sliding window size
        """
        logger.info("Computing variance explained by forcings over time")

        # Ensure forcing and climate have matching time dimensions
        time_climate = climate.shape[-1]
        if co2_forcing is not None:
            time_co2 = co2_forcing.shape[-1]
            if time_co2 != time_climate:
                min_time = min(time_co2, time_climate)
                logger.warning(
                    f"Time dimension mismatch: CO2 forcing has {time_co2} timesteps, "
                    f"climate has {time_climate} timesteps. Trimming both to {min_time}."
                )
                co2_forcing = co2_forcing[..., :min_time]
                climate = climate[..., :min_time]
                time_climate = min_time

        if aerosol_forcing is not None:
            time_aerosol = aerosol_forcing.shape[-1]
            if time_aerosol != time_climate:
                min_time = min(time_aerosol, time_climate)
                logger.warning(
                    f"Time dimension mismatch: aerosol forcing has {time_aerosol} timesteps, "
                    f"climate has {time_climate} timesteps. Trimming both to {min_time}."
                )
                aerosol_forcing = aerosol_forcing[..., :min_time]
                climate = climate[..., :min_time]

        # Spatially average
        climate_ts = climate.mean(axis=0)
        time_len = len(climate_ts)
        n_windows = time_len - window_size + 1

        if n_windows < 10:
            logger.warning("Too few windows for variance explained plot")
            return

        window_centers = np.arange(window_size // 2, time_len - window_size // 2 + 1)

        results = {}

        # Process each forcing
        forcings = []
        if co2_forcing is not None:
            forcings.append(("CO2", co2_forcing.mean(axis=0) if co2_forcing.ndim == 2 else co2_forcing, "tab:red"))
        if aerosol_forcing is not None:
            forcings.append(
                ("Aerosol", aerosol_forcing.mean(axis=0) if aerosol_forcing.ndim == 2 else aerosol_forcing, "tab:blue")
            )

        for name, forcing_ts, color in forcings:
            r2_values = []
            for i in range(n_windows):
                window_forcing = forcing_ts[i : i + window_size]
                window_climate = climate_ts[i : i + window_size]

                # Simple linear regression R²
                corr = np.corrcoef(window_forcing, window_climate)[0, 1]
                r2 = corr**2
                r2_values.append(r2)

            results[name] = (np.array(r2_values), color)

        # Combined model (if both forcings available)
        if len(forcings) == 2:
            r2_combined = []
            f1_ts = forcings[0][1]
            f2_ts = forcings[1][1]

            for i in range(n_windows):
                w_f1 = f1_ts[i : i + window_size]
                w_f2 = f2_ts[i : i + window_size]
                w_climate = climate_ts[i : i + window_size]

                # Multiple regression: climate = a*f1 + b*f2 + c
                X = np.column_stack([w_f1, w_f2, np.ones(window_size)])
                try:
                    coeffs, residuals, rank, s = np.linalg.lstsq(X, w_climate, rcond=None)
                    y_pred = X @ coeffs
                    ss_res = np.sum((w_climate - y_pred) ** 2)
                    ss_tot = np.sum((w_climate - w_climate.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                except Exception:
                    r2 = 0
                r2_combined.append(r2)

            results["Combined"] = (np.array(r2_combined), "green")

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))

        for name, (r2_vals, color) in results.items():
            linestyle = "-" if name != "Combined" else "--"
            ax.plot(window_centers, r2_vals, color=color, linewidth=2, label=f"{name} R²", linestyle=linestyle)

        ax.set_xlabel("Time step (window center)", fontsize=12)
        ax.set_ylabel("Variance Explained (R²)", fontsize=12)
        ax.set_title(f"Variance Explained by Forcings Over Time (window={window_size})", fontsize=14)
        ax.set_ylim(0, 1)
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        filename = "variance_explained_over_time.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

    def plot_forcing_attribution_summary(
        self,
        co2_forcing: Optional[np.ndarray],
        aerosol_forcing: Optional[np.ndarray],
        climate: np.ndarray,
        path: Path,
    ) -> None:
        """
        Create summary bar chart of variance explained by CO2, aerosol, and internal variability.

        Args:
            co2_forcing: CO2 forcing (spatial, time) or None
            aerosol_forcing: Aerosol forcing (spatial, time) or None
            climate: Climate data (spatial, time)
            path: Save path
        """
        logger.info("Creating forcing attribution summary")

        # Ensure forcing and climate have matching time dimensions
        time_climate = climate.shape[-1]
        if co2_forcing is not None:
            time_co2 = co2_forcing.shape[-1]
            if time_co2 != time_climate:
                min_time = min(time_co2, time_climate)
                logger.warning(
                    f"Time dimension mismatch: CO2 forcing has {time_co2} timesteps, "
                    f"climate has {time_climate} timesteps. Trimming both to {min_time}."
                )
                co2_forcing = co2_forcing[..., :min_time]
                climate = climate[..., :min_time]
                time_climate = min_time

        if aerosol_forcing is not None:
            time_aerosol = aerosol_forcing.shape[-1]
            if time_aerosol != time_climate:
                min_time = min(time_aerosol, time_climate)
                logger.warning(
                    f"Time dimension mismatch: aerosol forcing has {time_aerosol} timesteps, "
                    f"climate has {time_climate} timesteps. Trimming both to {min_time}."
                )
                aerosol_forcing = aerosol_forcing[..., :min_time]
                climate = climate[..., :min_time]

        # Spatially average
        climate_ts = climate.mean(axis=0)
        _ = np.var(climate_ts)

        # Build feature matrix
        features = []
        feature_names = []
        feature_colors = []

        if co2_forcing is not None:
            co2_ts = co2_forcing.mean(axis=0) if co2_forcing.ndim == 2 else co2_forcing
            features.append(co2_ts)
            feature_names.append("CO2")
            feature_colors.append("tab:red")

        if aerosol_forcing is not None:
            aerosol_ts = aerosol_forcing.mean(axis=0) if aerosol_forcing.ndim == 2 else aerosol_forcing
            features.append(aerosol_ts)
            feature_names.append("Aerosol")
            feature_colors.append("tab:blue")

        if len(features) == 0:
            logger.warning("No forcing data for attribution summary")
            return

        # Full model R²
        X = np.column_stack(features + [np.ones(len(climate_ts))])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, climate_ts, rcond=None)
            y_pred = X @ coeffs
            ss_res = np.sum((climate_ts - y_pred) ** 2)
            ss_tot = np.sum((climate_ts - climate_ts.mean()) ** 2)
            r2_full = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        except Exception:
            r2_full = 0

        # Individual R² (marginal contribution)
        r2_individual = []
        for i, f in enumerate(features):
            corr = np.corrcoef(f, climate_ts)[0, 1]
            r2_individual.append(corr**2)

        # Internal variability = 1 - R²_full
        internal_var = 1 - r2_full

        # Create pie chart and bar chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Pie chart
        pie_values = r2_individual + [internal_var]
        pie_labels = feature_names + ["Internal"]
        pie_colors = feature_colors + ["gray"]

        ax1.pie(
            pie_values,
            labels=pie_labels,
            colors=pie_colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=[0.05] * len(pie_values),
        )
        ax1.set_title("Variance Attribution\n(Marginal R²)", fontsize=12)

        # Bar chart
        bar_positions = np.arange(len(feature_names) + 2)
        bar_heights = r2_individual + [r2_full, internal_var]
        bar_labels = feature_names + ["Combined", "Internal"]
        bar_colors = feature_colors + ["green", "gray"]

        ax2.bar(bar_positions, bar_heights, color=bar_colors, alpha=0.7, edgecolor="black")
        ax2.set_xticks(bar_positions)
        ax2.set_xticklabels(bar_labels, rotation=15)
        ax2.set_ylabel("Fraction of Variance", fontsize=11)
        ax2.set_title("Variance Explained by Each Source", fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(axis="y", alpha=0.3)

        # Add values on bars
        for i, h in enumerate(bar_heights):
            ax2.text(i, h + 0.02, f"{h:.2f}", ha="center", fontsize=10)

        fig.suptitle("Forcing Attribution Summary", fontsize=14)
        fig.tight_layout()
        filename = "forcing_attribution_summary.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename} (R²_full={r2_full:.3f})")

    # =========================================================================
    # FORCING DIAGNOSTIC PLOTS - Spectral Analysis
    # =========================================================================

    def plot_phase_relationship(
        self,
        forcing: np.ndarray,
        climate: np.ndarray,
        forcing_name: str,
        path: Path,
        fs: float = 1.0,
        n_dominant_freqs: int = 5,
    ) -> None:
        """
        Plot detailed phase lag analysis between forcing and climate oscillations.

        Extracts dominant frequencies and shows phase lags with interpretation (leading/lagging).

        Args:
            forcing: Forcing data (spatial, time) or (time,)
            climate: Climate data (spatial, time)
            forcing_name: Name of forcing
            path: Save path
            fs: Sampling frequency (default 1.0 = 1 sample per timestep)
            n_dominant_freqs: Number of dominant frequencies to highlight
        """
        logger.info(f"Analyzing phase relationship for {forcing_name}")

        # Ensure forcing and climate have matching time dimensions
        time_forcing = forcing.shape[-1]
        time_climate = climate.shape[-1]
        if time_forcing != time_climate:
            min_time = min(time_forcing, time_climate)
            logger.warning(
                f"Time dimension mismatch: forcing has {time_forcing} timesteps, "
                f"climate has {time_climate} timesteps. Trimming both to {min_time}."
            )
            forcing = forcing[..., :min_time]
            climate = climate[..., :min_time]

        # Get time series
        forcing_ts = forcing.mean(axis=0) if forcing.ndim == 2 else forcing
        climate_ts = climate.mean(axis=0)

        # Compute cross-spectrum
        nperseg = min(len(forcing_ts) // 4, 256)
        f, Pxy = signal.csd(forcing_ts, climate_ts, fs=fs, nperseg=nperseg)
        phase = np.angle(Pxy, deg=True)

        # Also compute coherence to identify significant frequencies
        f_coh, Cxy = signal.coherence(forcing_ts, climate_ts, fs=fs, nperseg=nperseg)

        # Find dominant frequencies (high coherence)
        coherence_threshold = 0.3
        significant_mask = Cxy > coherence_threshold
        if not significant_mask.any():
            logger.warning(f"No significant coherence found for {forcing_name}, using top {n_dominant_freqs} peaks")
            dominant_indices = np.argsort(Cxy)[-n_dominant_freqs:]
        else:
            coherent_indices = np.where(significant_mask)[0]
            # Among coherent frequencies, pick top ones by coherence
            coherent_indices_sorted = coherent_indices[np.argsort(Cxy[coherent_indices])[-n_dominant_freqs:]]
            dominant_indices = coherent_indices_sorted

        # Create comprehensive figure
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

        color = "tab:red" if forcing_name == "CO2" else "tab:blue"

        # Panel 1: Full phase spectrum
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(f, phase, color=color, linewidth=1.5, alpha=0.7)
        ax1.scatter(
            f[dominant_indices],
            phase[dominant_indices],
            s=100,
            c="orange",
            edgecolors="black",
            zorder=10,
            label="Dominant freqs",
        )
        ax1.axhline(y=0, color="gray", linestyle="--", linewidth=1)
        ax1.axhline(y=90, color="green", linestyle=":", linewidth=1, alpha=0.5, label="90° (quadrature)")
        ax1.axhline(y=-90, color="green", linestyle=":", linewidth=1, alpha=0.5)
        ax1.set_xlabel("Frequency", fontsize=11)
        ax1.set_ylabel("Phase (degrees)", fontsize=11)
        ax1.set_title(f"Phase Spectrum: {forcing_name} → Climate", fontsize=13, fontweight="bold")
        ax1.set_ylim(-180, 180)
        ax1.legend(loc="upper right")
        ax1.grid(alpha=0.3)

        # Panel 2: Coherence (for reference)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(f_coh, Cxy, color=color, linewidth=1.5)
        ax2.scatter(f[dominant_indices], Cxy[dominant_indices], s=100, c="orange", edgecolors="black", zorder=10)
        ax2.axhline(
            y=coherence_threshold, color="gray", linestyle="--", alpha=0.7, label=f"Threshold ({coherence_threshold})"
        )
        ax2.set_xlabel("Frequency", fontsize=11)
        ax2.set_ylabel("Coherence", fontsize=11)
        ax2.set_title("Spectral Coherence", fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")
        ax2.grid(alpha=0.3)

        # Panel 3: Phase at dominant frequencies (bar chart)
        ax3 = fig.add_subplot(gs[1, 1])
        dominant_freqs = f[dominant_indices]
        dominant_phases = phase[dominant_indices]
        _ = Cxy[dominant_indices]

        colors_bars = [
            "green" if -45 < p < 45 else "orange" if abs(abs(p) - 90) < 45 else "red" for p in dominant_phases
        ]
        ax3.bar(range(len(dominant_indices)), dominant_phases, color=colors_bars, alpha=0.7, edgecolor="black")
        ax3.axhline(y=0, color="black", linestyle="-", linewidth=1)
        ax3.set_xticks(range(len(dominant_indices)))
        ax3.set_xticklabels([f"{freq:.3f}" for freq in dominant_freqs], rotation=45, ha="right")
        ax3.set_xlabel("Frequency", fontsize=11)
        ax3.set_ylabel("Phase (degrees)", fontsize=11)
        ax3.set_title("Phase at Dominant Frequencies", fontsize=12)
        ax3.set_ylim(-180, 180)
        ax3.grid(axis="y", alpha=0.3)

        # Panel 4: Interpretation table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis("off")

        # Create interpretation text
        table_data = []
        table_data.append(["Freq", "Period", "Phase", "Coherence", "Interpretation"])
        table_data.append(["-" * 8, "-" * 8, "-" * 8, "-" * 10, "-" * 40])

        for i, idx in enumerate(dominant_indices):
            freq = f[idx]
            period = 1 / freq if freq > 0 else np.inf
            ph = phase[idx]
            coh = Cxy[idx]

            # Interpretation
            if -45 < ph < 45:
                interp = f"In-phase: {forcing_name} and climate move together"
            elif 45 <= ph < 135:
                interp = f"{forcing_name} leads climate by ~1/4 period"
            elif ph >= 135 or ph <= -135:
                interp = f"Anti-phase: {forcing_name} and climate opposite"
            else:  # -135 < ph <= -45
                interp = f"{forcing_name} lags climate by ~1/4 period"

            table_data.append(
                [
                    f"{freq:.4f}",
                    f"{period:.1f}" if period != np.inf else "∞",
                    f"{ph:.1f}°",
                    f"{coh:.3f}",
                    interp,
                ]
            )

        # Render table
        table = ax4.table(
            cellText=table_data,
            cellLoc="left",
            loc="center",
            colWidths=[0.12, 0.12, 0.12, 0.14, 0.5],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)

        # Style header row
        for i in range(5):
            cell = table[(0, i)]
            cell.set_facecolor("#4CAF50")
            cell.set_text_props(weight="bold", color="white")

        fig.suptitle(f"Phase Relationship Analysis: {forcing_name} ↔ Climate", fontsize=15, fontweight="bold")
        filename = f"{forcing_name.lower()}_phase_relationship.png"
        fig.savefig(path / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {filename}")

    # =========================================================================
    # FORCING DIAGNOSTIC PLOTS - Tigramite Transfer Entropy
    # =========================================================================

    def plot_transfer_entropy_matrix(
        self,
        co2_latent: Optional[np.ndarray],
        aerosol_latents: Optional[np.ndarray],
        climate_modes: np.ndarray,
        path: Path,
        tau_max: int = 5,
        significance_level: float = 0.05,
    ) -> None:
        """
        Compute and visualize transfer entropy from forcings to climate modes using Tigramite.

        Args:
            co2_latent: CO2 latent trajectory (time,) or None
            aerosol_latents: Aerosol latent trajectories (n_aerosol, time) or None
            climate_modes: Climate mode trajectories (n_modes, time)
            path: Save path
            tau_max: Maximum time lag for PCMCI
            significance_level: Significance level for link detection
        """
        if not TIGRAMITE_AVAILABLE:
            logger.warning("Tigramite not available, skipping transfer entropy analysis")
            return

        logger.info("Computing transfer entropy using Tigramite PCMCI")

        # Build data array (n_vars, time)
        var_list = []
        var_names = []

        # Add forcing latents
        if co2_latent is not None:
            if co2_latent.ndim == 1:
                var_list.append(co2_latent)
                var_names.append("CO2")
            else:
                for i in range(co2_latent.shape[0]):
                    var_list.append(co2_latent[i])
                    var_names.append(f"CO2_{i}")

        if aerosol_latents is not None:
            if aerosol_latents.ndim == 1:
                var_list.append(aerosol_latents)
                var_names.append("Aero")
            else:
                for i in range(aerosol_latents.shape[0]):
                    var_list.append(aerosol_latents[i])
                    var_names.append(f"A{i}")

        # Add climate modes
        if climate_modes.ndim == 1:
            var_list.append(climate_modes)
            var_names.append("M0")
        else:
            for i in range(climate_modes.shape[0]):
                var_list.append(climate_modes[i])
                var_names.append(f"M{i}")

        if len(var_list) < 2:
            logger.warning("Need at least 2 variables for transfer entropy")
            return

        # Stack into (time, n_vars) array for tigramite
        data_array = np.column_stack(var_list)

        # Create tigramite dataframe
        dataframe = pp.DataFrame(data_array, var_names=var_names)

        # Run PCMCI with partial correlation test
        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

        try:
            results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=significance_level)
        except Exception as e:
            logger.error(f"PCMCI failed: {e}")
            return

        # Extract link matrix
        q_matrix = pcmci.get_corrected_pvalues(p_matrix=results["p_matrix"], fdr_method="fdr_bh")
        link_matrix = np.where(q_matrix < significance_level, results["val_matrix"], 0)

        # Create custom visualization
        n_vars = len(var_names)
        n_forcing = (1 if co2_latent is not None else 0) + (
            aerosol_latents.shape[0]
            if aerosol_latents is not None and aerosol_latents.ndim > 1
            else (1 if aerosol_latents is not None else 0)
        )

        # Summary: forcing → climate links (sum over lags)
        forcing_to_climate = np.zeros((n_forcing, n_vars - n_forcing))
        for i in range(n_forcing):
            for j in range(n_forcing, n_vars):
                # Sum absolute link strengths over all lags
                forcing_to_climate[i, j - n_forcing] = np.sum(np.abs(link_matrix[i, j, :]))

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))

        forcing_names = var_names[:n_forcing]
        climate_names = var_names[n_forcing:]

        im = ax.imshow(forcing_to_climate, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(climate_names)))
        ax.set_yticks(range(len(forcing_names)))
        ax.set_xticklabels(climate_names, rotation=45, ha="right")
        ax.set_yticklabels(forcing_names)
        ax.set_xlabel("Climate Modes", fontsize=12)
        ax.set_ylabel("Forcing Latents", fontsize=12)
        ax.set_title(f"Causal Influence: Forcings → Climate (PCMCI, τ_max={tau_max})", fontsize=14)

        # Add values
        for i in range(len(forcing_names)):
            for j in range(len(climate_names)):
                val = forcing_to_climate[i, j]
                color = "white" if val > forcing_to_climate.max() / 2 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

        fig.colorbar(im, ax=ax, label="Summed |link strength|")
        fig.tight_layout()
        filename = "transfer_entropy_matrix.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

        # Also save Tigramite's built-in graph plot
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            tp.plot_graph(
                val_matrix=results["val_matrix"],
                graph=results["graph"],
                var_names=var_names,
                link_colorbar_label="Cross-MCI",
                node_colorbar_label="Auto-MCI",
                fig_ax=(fig, ax),
            )
            filename_graph = "tigramite_causal_graph.png"
            fig.savefig(path / filename_graph, dpi=150)
            plt.close(fig)
            logger.info(f"Saved {filename_graph}")
        except Exception as e:
            logger.warning(f"Could not save Tigramite graph plot: {e}")

    def plot_mutual_information_matrix(
        self,
        co2_latent: Optional[np.ndarray],
        aerosol_latents: Optional[np.ndarray],
        climate_modes: np.ndarray,
        path: Path,
        n_bins: int = 20,
    ) -> None:
        """
        Compute and visualize mutual information between forcing latents and climate modes.

        Args:
            co2_latent: CO2 latent trajectory (time,) or None
            aerosol_latents: Aerosol latent trajectories (n_aerosol, time) or None
            climate_modes: Climate mode trajectories (n_modes, time)
            path: Save path
            n_bins: Number of bins for histogram-based MI estimation
        """
        logger.info("Computing mutual information matrix")

        # Build variable lists
        forcing_list = []
        forcing_names = []

        if co2_latent is not None:
            if co2_latent.ndim == 1:
                forcing_list.append(co2_latent)
                forcing_names.append("CO2")
            else:
                for i in range(co2_latent.shape[0]):
                    forcing_list.append(co2_latent[i])
                    forcing_names.append(f"CO2_{i}")

        if aerosol_latents is not None:
            if aerosol_latents.ndim == 1:
                forcing_list.append(aerosol_latents)
                forcing_names.append("Aero")
            else:
                for i in range(aerosol_latents.shape[0]):
                    forcing_list.append(aerosol_latents[i])
                    forcing_names.append(f"A{i}")

        climate_list = []
        climate_names = []
        if climate_modes.ndim == 1:
            climate_list.append(climate_modes)
            climate_names.append("M0")
        else:
            for i in range(climate_modes.shape[0]):
                climate_list.append(climate_modes[i])
                climate_names.append(f"M{i}")

        if len(forcing_list) == 0 or len(climate_list) == 0:
            logger.warning("Need forcing and climate variables for MI computation")
            return

        # Compute MI matrix (forcing × climate)
        mi_matrix = np.zeros((len(forcing_list), len(climate_list)))

        for i, f_var in enumerate(forcing_list):
            for j, c_var in enumerate(climate_list):
                # Mutual information using histogram method
                # MI(X,Y) = H(X) + H(Y) - H(X,Y)
                # Where H is entropy
                c_xy, _, _ = np.histogram2d(f_var, c_var, bins=n_bins)
                c_x = np.histogram(f_var, bins=n_bins)[0]
                c_y = np.histogram(c_var, bins=n_bins)[0]

                # Normalize to probabilities
                p_xy = c_xy / np.sum(c_xy)
                p_x = c_x / np.sum(c_x)
                p_y = c_y / np.sum(c_y)

                # Compute entropies (ignore zero probabilities)
                h_x = -np.sum(p_x[p_x > 0] * np.log2(p_x[p_x > 0]))
                h_y = -np.sum(p_y[p_y > 0] * np.log2(p_y[p_y > 0]))
                h_xy = -np.sum(p_xy[p_xy > 0] * np.log2(p_xy[p_xy > 0]))

                mi_matrix[i, j] = h_x + h_y - h_xy

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mi_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(climate_names)))
        ax.set_yticks(range(len(forcing_names)))
        ax.set_xticklabels(climate_names, rotation=45, ha="right")
        ax.set_yticklabels(forcing_names)
        ax.set_xlabel("Climate Modes", fontsize=12)
        ax.set_ylabel("Forcing Latents", fontsize=12)
        ax.set_title("Mutual Information: Forcings ↔ Climate", fontsize=14)

        # Add values
        for i in range(len(forcing_names)):
            for j in range(len(climate_names)):
                val = mi_matrix[i, j]
                color = "white" if val > mi_matrix.max() / 2 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

        fig.colorbar(im, ax=ax, label="MI (bits)")
        fig.tight_layout()
        filename = "mutual_information_matrix.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename} (mean MI={np.mean(mi_matrix):.3f} bits)")

    def plot_conditional_correlation_network(
        self,
        co2_latent: Optional[np.ndarray],
        aerosol_latents: Optional[np.ndarray],
        climate_modes: np.ndarray,
        path: Path,
        threshold: float = 0.2,
    ) -> None:
        """
        Plot network graph showing partial correlations to distinguish direct vs indirect effects.

        Args:
            co2_latent: CO2 latent trajectory (time,) or None
            aerosol_latents: Aerosol latent trajectories (n_aerosol, time) or None
            climate_modes: Climate mode trajectories (n_modes, time)
            path: Save path
            threshold: Minimum absolute partial correlation to display
        """
        logger.info("Computing conditional correlation network")

        # Build data array
        var_list = []
        var_names = []
        var_types = []  # 'forcing' or 'climate'

        if co2_latent is not None:
            if co2_latent.ndim == 1:
                var_list.append(co2_latent)
                var_names.append("CO2")
                var_types.append("forcing")
            else:
                for i in range(co2_latent.shape[0]):
                    var_list.append(co2_latent[i])
                    var_names.append(f"CO2_{i}")
                    var_types.append("forcing")

        if aerosol_latents is not None:
            if aerosol_latents.ndim == 1:
                var_list.append(aerosol_latents)
                var_names.append("Aero")
                var_types.append("forcing")
            else:
                for i in range(aerosol_latents.shape[0]):
                    var_list.append(aerosol_latents[i])
                    var_names.append(f"A{i}")
                    var_types.append("forcing")

        if climate_modes.ndim == 1:
            var_list.append(climate_modes)
            var_names.append("M0")
            var_types.append("climate")
        else:
            for i in range(climate_modes.shape[0]):
                var_list.append(climate_modes[i])
                var_names.append(f"M{i}")
                var_types.append("climate")

        if len(var_list) < 2:
            logger.warning("Need at least 2 variables for correlation network")
            return

        # Stack into array (time, n_vars)
        data_array = np.column_stack(var_list)
        n_vars = len(var_list)

        # Compute correlation matrix
        corr_matrix = np.corrcoef(data_array.T)

        # Compute partial correlations (simple inverse covariance method)
        # Partial corr: -cov_inv[i,j] / sqrt(cov_inv[i,i] * cov_inv[j,j])
        try:
            precision = np.linalg.inv(corr_matrix + 1e-6 * np.eye(n_vars))
            partial_corr = np.zeros((n_vars, n_vars))
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j:
                        partial_corr[i, j] = -precision[i, j] / np.sqrt(precision[i, i] * precision[j, j])
        except np.linalg.LinAlgError:
            logger.warning("Could not compute partial correlations, using regular correlations")
            partial_corr = corr_matrix

        # Create network visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Plot 1: Heatmap of partial correlations
        im1 = ax1.imshow(partial_corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax1.set_xticks(range(n_vars))
        ax1.set_yticks(range(n_vars))
        ax1.set_xticklabels(var_names, rotation=45, ha="right")
        ax1.set_yticklabels(var_names)
        ax1.set_title("Partial Correlation Matrix", fontsize=14)
        fig.colorbar(im1, ax=ax1, label="Partial Correlation")

        # Plot 2: Network graph (simple circular layout)
        ax2.set_xlim(-1.5, 1.5)
        ax2.set_ylim(-1.5, 1.5)
        ax2.set_aspect("equal")
        ax2.axis("off")
        ax2.set_title(f"Correlation Network (|r| > {threshold})", fontsize=14)

        # Node positions (circular layout)
        angles = np.linspace(0, 2 * np.pi, n_vars, endpoint=False)
        positions = np.column_stack([np.cos(angles), np.sin(angles)])

        # Draw edges (links above threshold)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                pc = partial_corr[i, j]
                if abs(pc) > threshold:
                    x_vals = [positions[i, 0], positions[j, 0]]
                    y_vals = [positions[i, 1], positions[j, 1]]
                    color = "red" if pc > 0 else "blue"
                    width = abs(pc) * 3
                    ax2.plot(x_vals, y_vals, color=color, linewidth=width, alpha=0.6)

        # Draw nodes
        for i, (name, typ) in enumerate(zip(var_names, var_types)):
            color = "orange" if typ == "forcing" else "lightblue"
            ax2.scatter(positions[i, 0], positions[i, 1], s=500, c=color, edgecolors="black", linewidths=2, zorder=10)
            ax2.text(
                positions[i, 0] * 1.2,
                positions[i, 1] * 1.2,
                name,
                fontsize=11,
                ha="center",
                va="center",
                fontweight="bold",
            )

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="Forcing"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="lightblue", markersize=10, label="Climate"),
            Line2D([0], [0], color="red", linewidth=2, label="Positive corr"),
            Line2D([0], [0], color="blue", linewidth=2, label="Negative corr"),
        ]
        ax2.legend(handles=legend_elements, loc="upper right", fontsize=10)

        fig.tight_layout()
        filename = "conditional_correlation_network.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

    def plot_causal_graph_tigramite(
        self,
        data_array: np.ndarray,
        var_names: List[str],
        path: Path,
        tau_max: int = 5,
        pc_alpha: float = 0.05,
    ) -> None:
        """
        Generate Tigramite-style causal graph visualization with lag information.

        Args:
            data_array: Data (time, n_vars) or (n_vars, time) - will be transposed if needed
            var_names: Variable names ['CO2', 'A0', 'A1', ..., 'M0', 'M1', ...]
            path: Save path
            tau_max: Maximum time lag for PCMCI
            pc_alpha: Significance level for link detection
        """
        if not TIGRAMITE_AVAILABLE:
            logger.warning("Tigramite not available, skipping causal graph")
            return

        logger.info("Generating Tigramite causal graph")

        # Ensure data is (time, n_vars)
        if data_array.shape[0] < data_array.shape[1]:
            data_array = data_array.T

        # Create tigramite dataframe
        dataframe = pp.DataFrame(data_array, var_names=var_names)

        # Run PCMCI
        parcorr = ParCorr(significance="analytic")
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

        try:
            results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
        except Exception as e:
            logger.error(f"PCMCI failed: {e}")
            return

        # Create Tigramite's graph visualization
        fig, ax = plt.subplots(figsize=(14, 10))

        try:
            tp.plot_graph(
                val_matrix=results["val_matrix"],
                graph=results["graph"],
                var_names=var_names,
                link_colorbar_label="MCI value",
                node_colorbar_label="Auto-MCI",
                fig_ax=(fig, ax),
            )

            ax.set_title(
                f"Tigramite Causal Graph (τ_max={tau_max}, α={pc_alpha})", fontsize=14, fontweight="bold", pad=20
            )

            filename = "tigramite_causal_graph_full.png"
            fig.savefig(path / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved {filename}")

        except Exception as e:
            logger.error(f"Could not create Tigramite graph: {e}")
            plt.close(fig)

    def plot_information_flow_arrows(
        self,
        transfer_entropy_matrix: np.ndarray,
        source_names: List[str],
        target_names: List[str],
        lat: int,
        lon: int,
        path: Path,
        source_positions: Optional[np.ndarray] = None,
        target_positions: Optional[np.ndarray] = None,
    ) -> None:
        """
        Overlay information flow arrows on spatial grid.

        Arrow thickness represents transfer entropy strength from sources to targets.

        Args:
            transfer_entropy_matrix: Transfer entropy (n_sources, n_targets)
            source_names: Source variable names (e.g., ['CO2', 'A0', ...])
            target_names: Target variable names (e.g., ['M0', 'M1', ...])
            lat, lon: Grid dimensions
            path: Save path
            source_positions: Optional source positions on grid (n_sources, 2) as [lat_idx, lon_idx]
            target_positions: Optional target positions on grid (n_targets, 2) as [lat_idx, lon_idx]
        """
        logger.info("Creating information flow arrow visualization")

        n_sources = len(source_names)
        n_targets = len(target_names)

        # If positions not provided, create default positions
        if source_positions is None:
            # Place sources on left side
            source_positions = np.zeros((n_sources, 2))
            for i in range(n_sources):
                source_positions[i] = [lat // 2, lon // 4]  # Same position for all (will offset in plot)

        if target_positions is None:
            # Place targets on right side, distributed vertically
            target_positions = np.zeros((n_targets, 2))
            for i in range(n_targets):
                target_positions[i] = [int(lat * (i + 1) / (n_targets + 1)), 3 * lon // 4]

        # Create grid background
        fig, ax = plt.subplots(figsize=(12, 8))

        # Draw a simple grid
        ax.set_xlim(0, lon)
        ax.set_ylim(0, lat)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        ax.set_xlabel("Longitude index", fontsize=12)
        ax.set_ylabel("Latitude index", fontsize=12)
        ax.set_title("Information Flow: Forcings → Climate Modes", fontsize=14, fontweight="bold")

        # Normalize transfer entropy for arrow widths
        te_max = np.max(transfer_entropy_matrix)
        te_min = np.min(transfer_entropy_matrix)

        if te_max > te_min:
            te_normalized = (transfer_entropy_matrix - te_min) / (te_max - te_min)
        else:
            te_normalized = np.zeros_like(transfer_entropy_matrix)

        # Draw arrows
        for i in range(n_sources):
            for j in range(n_targets):
                te_val = transfer_entropy_matrix[i, j]
                if te_val > 0.01:  # Only draw significant flows
                    # Source and target positions
                    src_pos = source_positions[i]
                    tgt_pos = target_positions[j]

                    # Arrow properties
                    arrow_width = te_normalized[i, j] * 5  # Scale width
                    color_intensity = te_normalized[i, j]
                    arrow_color = plt.cm.YlOrRd(color_intensity)

                    # Draw arrow
                    ax.annotate(
                        "",
                        xy=(tgt_pos[1], tgt_pos[0]),
                        xytext=(src_pos[1], src_pos[0]),
                        arrowprops=dict(
                            arrowstyle="->", lw=arrow_width, color=arrow_color, alpha=0.7, shrinkA=10, shrinkB=10
                        ),
                    )

        # Draw source nodes
        for i, name in enumerate(source_names):
            pos = source_positions[i]
            ax.scatter(pos[1], pos[0], s=300, c="orange", edgecolors="black", linewidths=2, zorder=10)
            ax.text(pos[1], pos[0] + lat * 0.05, name, fontsize=11, ha="center", fontweight="bold")

        # Draw target nodes
        for j, name in enumerate(target_names):
            pos = target_positions[j]
            ax.scatter(pos[1], pos[0], s=300, c="lightblue", edgecolors="black", linewidths=2, zorder=10)
            ax.text(pos[1], pos[0] + lat * 0.05, name, fontsize=11, ha="center", fontweight="bold")

        # Add colorbar for transfer entropy
        sm = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd, norm=plt.Normalize(vmin=te_min, vmax=te_max))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Transfer Entropy", shrink=0.7)

        # Legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="Forcing"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="lightblue", markersize=10, label="Climate Mode"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

        fig.tight_layout()
        filename = "information_flow_arrows.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

    # =========================================================================
    # FORCING DIAGNOSTIC PLOTS - Ground Truth Comparison
    # =========================================================================

    def plot_gt_vs_learned_forcing_effect(
        self,
        gt_adj: np.ndarray,
        learned_adj: np.ndarray,
        forcing_indices: Dict[str, List[int]],
        path: Path,
        iteration: int,
    ) -> None:
        """
        Compare known ground truth causal coefficients with learned adjacency.

        Args:
            gt_adj: Ground truth adjacency matrix (tau, n_latents, n_latents)
            learned_adj: Learned adjacency matrix (tau, n_latents, n_latents)
            forcing_indices: Dict with 'co2' and 'aerosol' latent indices
            path: Save path
            iteration: Current iteration
        """
        logger.info("Comparing GT vs learned forcing effects")

        if forcing_indices is None:
            logger.warning("No forcing indices available for GT comparison")
            return

        # Get forcing and climate indices
        co2_idx = forcing_indices.get("co2", [])
        aerosol_idx = forcing_indices.get("aerosol", [])
        n_total = forcing_indices.get("n_total", learned_adj.shape[1])
        n_climate = n_total - len(co2_idx) - len(aerosol_idx)
        climate_idx = list(range(n_climate))

        all_forcing_idx = co2_idx + aerosol_idx

        # Extract forcing → climate submatrices (sum over tau)
        gt_forcing_to_climate = np.zeros((len(all_forcing_idx), n_climate))
        learned_forcing_to_climate = np.zeros((len(all_forcing_idx), n_climate))

        for fi, forcing_i in enumerate(all_forcing_idx):
            for ci, climate_i in enumerate(climate_idx):
                # Sum over all lags
                gt_forcing_to_climate[fi, ci] = np.sum(np.abs(gt_adj[:, forcing_i, climate_i]))
                learned_forcing_to_climate[fi, ci] = np.sum(np.abs(learned_adj[:, forcing_i, climate_i]))

        # Create comparison figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # GT
        im0 = axes[0].imshow(gt_forcing_to_climate, cmap="YlOrRd", aspect="auto")
        axes[0].set_title("Ground Truth\nForcing → Climate", fontsize=12)
        axes[0].set_xlabel("Climate Mode")
        axes[0].set_ylabel("Forcing Latent")
        fig.colorbar(im0, ax=axes[0], shrink=0.8)

        # Learned
        im1 = axes[1].imshow(learned_forcing_to_climate, cmap="YlOrRd", aspect="auto")
        axes[1].set_title("Learned\nForcing → Climate", fontsize=12)
        axes[1].set_xlabel("Climate Mode")
        fig.colorbar(im1, ax=axes[1], shrink=0.8)

        # Difference
        diff = learned_forcing_to_climate - gt_forcing_to_climate
        vabs = max(np.abs(diff).max(), 0.01)
        im2 = axes[2].imshow(diff, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")
        axes[2].set_title("Difference\n(Learned - GT)", fontsize=12)
        axes[2].set_xlabel("Climate Mode")
        fig.colorbar(im2, ax=axes[2], shrink=0.8)

        # Add y-axis labels
        forcing_labels = [f"CO2_{i}" for i in range(len(co2_idx))] + [f"A{i}" for i in range(len(aerosol_idx))]
        for ax in axes:
            ax.set_yticks(range(len(forcing_labels)))
            ax.set_yticklabels(forcing_labels)
            ax.set_xticks(range(n_climate))
            ax.set_xticklabels([f"M{i}" for i in range(n_climate)])

        fig.suptitle(f"GT vs Learned Forcing Effects (iter={iteration})", fontsize=14)
        fig.tight_layout()
        filename = f"gt_vs_learned_forcing_{iteration}.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

    def plot_forcing_latent_reconstruction_error(
        self,
        gt_co2_latent: Optional[np.ndarray],
        gt_aerosol_latent: Optional[np.ndarray],
        learned_co2_latent: Optional[np.ndarray],
        learned_aerosol_latent: Optional[np.ndarray],
        path: Path,
        iteration: int,
    ) -> None:
        """
        Track how well learned forcing latents match ground truth trajectories.

        Args:
            gt_co2_latent: Ground truth CO2 latent (time,) or (n, time)
            gt_aerosol_latent: Ground truth aerosol latents (n_aerosol, time)
            learned_co2_latent: Learned CO2 latent (time,) or (n, time)
            learned_aerosol_latent: Learned aerosol latents (n_aerosol, time)
            path: Save path
            iteration: Current iteration
        """
        logger.info("Computing forcing latent reconstruction error")

        n_plots = sum([gt_co2_latent is not None, gt_aerosol_latent is not None])
        if n_plots == 0:
            logger.warning("No ground truth latents available for comparison")
            return

        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        plot_idx = 0

        # CO2 comparison
        if gt_co2_latent is not None and learned_co2_latent is not None:
            ax = axes[plot_idx]
            gt = gt_co2_latent.flatten() if gt_co2_latent.ndim > 1 else gt_co2_latent
            learned = learned_co2_latent.flatten() if learned_co2_latent.ndim > 1 else learned_co2_latent

            # Align lengths
            min_len = min(len(gt), len(learned))
            gt = gt[:min_len]
            learned = learned[:min_len]

            # Normalize for comparison (correlation doesn't care about scale)
            gt_norm = (gt - gt.mean()) / (gt.std() + 1e-8)
            learned_norm = (learned - learned.mean()) / (learned.std() + 1e-8)

            time_axis = np.arange(min_len)
            ax.plot(time_axis, gt_norm, label="GT", color="black", linewidth=1.5)
            ax.plot(time_axis, learned_norm, label="Learned", color="tab:red", linewidth=1.5, alpha=0.8)

            corr = np.corrcoef(gt_norm, learned_norm)[0, 1]
            mse = np.mean((gt_norm - learned_norm) ** 2)

            ax.set_title(f"CO2 Latent (r={corr:.3f}, MSE={mse:.3f})", fontsize=12)
            ax.set_xlabel("Time")
            ax.set_ylabel("Normalized value")
            ax.legend()
            ax.grid(alpha=0.3)
            plot_idx += 1

        # Aerosol comparison
        if gt_aerosol_latent is not None and learned_aerosol_latent is not None:
            ax = axes[plot_idx]

            # Handle different shapes
            gt_aero = gt_aerosol_latent
            learned_aero = learned_aerosol_latent

            if gt_aero.ndim == 1:
                gt_aero = gt_aero.reshape(1, -1)
            if learned_aero.ndim == 1:
                learned_aero = learned_aero.reshape(1, -1)

            n_gt = gt_aero.shape[0]
            n_learned = learned_aero.shape[0]
            n_plot = min(n_gt, n_learned)
            min_len = min(gt_aero.shape[1], learned_aero.shape[1])

            colors_gt = plt.cm.Blues(np.linspace(0.4, 0.9, n_plot))
            colors_learned = plt.cm.Reds(np.linspace(0.4, 0.9, n_plot))

            time_axis = np.arange(min_len)
            corrs = []

            for i in range(n_plot):
                gt_i = gt_aero[i, :min_len]
                learned_i = learned_aero[i, :min_len]

                gt_norm = (gt_i - gt_i.mean()) / (gt_i.std() + 1e-8)
                learned_norm = (learned_i - learned_i.mean()) / (learned_i.std() + 1e-8)

                ax.plot(time_axis, gt_norm, color=colors_gt[i], linewidth=1, label=f"GT A{i}" if i == 0 else None)
                ax.plot(
                    time_axis,
                    learned_norm,
                    color=colors_learned[i],
                    linewidth=1,
                    linestyle="--",
                    label=f"Learned A{i}" if i == 0 else None,
                )

                corrs.append(np.corrcoef(gt_norm, learned_norm)[0, 1])

            mean_corr = np.mean(corrs)
            ax.set_title(f"Aerosol Latents (mean r={mean_corr:.3f})", fontsize=12)
            ax.set_xlabel("Time")
            ax.set_ylabel("Normalized value")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)

        fig.suptitle(f"Forcing Latent Reconstruction (iter={iteration})", fontsize=14)
        fig.tight_layout()
        filename = f"forcing_latent_reconstruction_{iteration}.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

    # =========================================================================
    # INTEGRATION METHOD - Call all forcing diagnostics
    # =========================================================================

    def plot_forcing_diagnostics(
        self,
        savar_data: np.ndarray,
        co2_forcing: Optional[np.ndarray],
        aerosol_forcing: Optional[np.ndarray],
        gt_co2_latent: Optional[np.ndarray],
        gt_aerosol_latent: Optional[np.ndarray],
        lat: int,
        lon: int,
        path: Path,
        max_lag: int = 20,
        window_size: int = 200,
    ) -> None:
        """
        Generate all forcing diagnostic plots.

        This is the main entry point for comprehensive forcing analysis.
        Called during data generation and optionally during training.

        Args:
            savar_data: Climate data (spatial, time)
            co2_forcing: CO2 forcing (spatial, time) or None
            aerosol_forcing: Aerosol forcing (spatial, time) or None
            gt_co2_latent: Ground truth CO2 latent trajectory
            gt_aerosol_latent: Ground truth aerosol latent trajectories
            lat, lon: Grid dimensions
            path: Save path
            max_lag: Maximum lag for cross-correlation
            window_size: Window size for sliding correlations
        """
        logger.info("=" * 60)
        logger.info("GENERATING COMPREHENSIVE FORCING DIAGNOSTICS")
        logger.info("=" * 60)

        # 1. Global Correlation Analysis
        logger.info("--- Global Correlation Analysis ---")
        if co2_forcing is not None:
            self.plot_correlation_heatmap_over_time(co2_forcing, savar_data, "CO2", window_size, path)
        if aerosol_forcing is not None:
            self.plot_correlation_heatmap_over_time(aerosol_forcing, savar_data, "Aerosol", window_size, path)

        # 2. Spatial Correlation Analysis
        logger.info("--- Spatial Correlation Analysis ---")
        if co2_forcing is not None:
            self.plot_pointwise_correlation_map(co2_forcing, savar_data, "CO2", lat, lon, path, lag=0)
            self.plot_pointwise_correlation_map(co2_forcing, savar_data, "CO2", lat, lon, path, lag=1)
        if aerosol_forcing is not None:
            self.plot_pointwise_correlation_map(aerosol_forcing, savar_data, "Aerosol", lat, lon, path, lag=0)

        # 3. Joint Animations
        logger.info("--- Joint Animations ---")
        self.plot_joint_forcing_climate_animation(co2_forcing, aerosol_forcing, savar_data, lat, lon, path)

        # 4. Variance Attribution
        logger.info("--- Variance Attribution ---")
        self.plot_variance_explained_by_forcing(co2_forcing, aerosol_forcing, savar_data, lat, lon, path, window_size)
        self.plot_forcing_attribution_summary(co2_forcing, aerosol_forcing, savar_data, path)

        # 5. Causal/Information-Theoretic Analysis
        if gt_co2_latent is not None or gt_aerosol_latent is not None:
            logger.info("--- Causal & Information-Theoretic Analysis ---")
            # For SAVAR, we use the latent trajectories, not the full spatial data
            # We need to extract climate mode trajectories - for now use spatial mean as proxy
            climate_proxy = savar_data.mean(axis=0).reshape(1, -1)  # (1, time)

            # Mutual information
            self.plot_mutual_information_matrix(gt_co2_latent, gt_aerosol_latent, climate_proxy, path)

            # Conditional correlation network
            self.plot_conditional_correlation_network(gt_co2_latent, gt_aerosol_latent, climate_proxy, path)

            # Transfer entropy (if Tigramite available)
            if TIGRAMITE_AVAILABLE:
                self.plot_transfer_entropy_matrix(gt_co2_latent, gt_aerosol_latent, climate_proxy, path)

                # Tigramite causal graph
                var_list = []
                var_names = []
                if gt_co2_latent is not None:
                    var_list.append(gt_co2_latent if gt_co2_latent.ndim == 1 else gt_co2_latent[0])
                    var_names.append("CO2")
                if gt_aerosol_latent is not None:
                    if gt_aerosol_latent.ndim == 1:
                        var_list.append(gt_aerosol_latent)
                        var_names.append("Aero")
                    else:
                        for i in range(gt_aerosol_latent.shape[0]):
                            var_list.append(gt_aerosol_latent[i])
                            var_names.append(f"A{i}")
                var_list.append(climate_proxy[0])
                var_names.append("M0")

                if len(var_list) >= 2:
                    data_array = np.column_stack(var_list)
                    self.plot_causal_graph_tigramite(data_array, var_names, path, tau_max=5)

        logger.info("=" * 60)
        logger.info("FORCING DIAGNOSTICS COMPLETE")
        logger.info("=" * 60)

    def plot_training_forcing_diagnostics(
        self,
        learner,
        iteration: int,
        path: Path,
    ) -> None:
        """
        Subset of forcing diagnostics suitable for training checkpoints.

        Avoids expensive computations like full transfer entropy.
        Called at plot_freq intervals during training.

        Args:
            learner: TrainingLatent instance
            iteration: Current iteration
            path: Save path
        """
        logger.info(f"Generating training forcing diagnostics (iter={iteration})")

        # Get forcing data from datamodule
        datamodule = getattr(learner, "datamodule", None)
        if datamodule is None:
            return

        co2_forcing = getattr(datamodule, "co2_forcing", None)
        aerosol_forcing = getattr(datamodule, "aerosol_forcing", None)

        if co2_forcing is None and aerosol_forcing is None:
            return

        # Get climate data
        savar_data = getattr(datamodule, "savar_data", None)
        if savar_data is None:
            return

        # GT vs learned comparison (if available)
        gt_adj = getattr(datamodule, "savar_gt_adj", None)
        forcing_indices = getattr(datamodule, "forcing_indices", None)

        if gt_adj is not None and forcing_indices is not None:
            learned_adj = learner.model.get_adj().cpu().detach().numpy()
            self.plot_gt_vs_learned_forcing_effect(gt_adj, learned_adj, forcing_indices, path, iteration)

        # Forcing latent reconstruction (if model has forcing encoders)
        # This would require extracting latent trajectories from the model
        # For now, skip - can be added when model provides this interface

        # Less frequent: variance explained (every 5 * plot_freq)
        plot_freq = learner.plot_params.plot_freq
        if iteration % (5 * plot_freq) == 0:
            lat = learner.lat
            lon = learner.lon
            self.plot_variance_explained_by_forcing(
                co2_forcing, aerosol_forcing, savar_data, lat, lon, path, window_size=200
            )

    def plot_sparsity(self, learner, save=False):
        """
        Override parent method to handle SAVAR-specific plotting completely.

        This avoids conflicts with parent's SAVAR handling by implementing the full plotting logic for SAVAR
        experiments.
        """
        # Save coordinates
        np.save(learner.plots_path / "coordinates.npy", learner.coordinates)

        if save:
            self.save(learner)

        # 1. Plot learning curves (same for all experiments)
        if learner.latent:
            self.plot_learning_curves(
                train_loss=learner.train_loss_list,
                train_recons=learner.train_recons_list,
                train_kl=learner.train_kl_list,
                valid_loss=learner.valid_loss_list,
                valid_recons=learner.valid_recons_list,
                valid_kl=learner.valid_kl_list,
                best_metrics=learner.best_metrics,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
                path=learner.plots_path,
            )

            # Plot penalties and losses
            losses = [
                {"name": "tr ortho", "data": learner.train_ortho_cons_list, "s": ":"},
                {"name": "mu ortho", "data": learner.mu_ortho_list, "s": ":"},
                {"name": "tr ortho_spatial", "data": learner.train_ortho_spatial_forcing_cons_list, "s": ":"},
                {"name": "mu ortho_spatial", "data": learner.mu_ortho_spatial_forcing_list, "s": ":"},
                {"name": "tr sparsity", "data": learner.train_sparsity_cons_list, "s": ":"},
                {"name": "tr var adj", "data": learner.train_transition_var_list, "s": ":"},
                {"name": "mu sparsity", "data": learner.mu_sparsity_list, "s": ":"},
            ]
            self.plot_learning_curves2(
                losses=losses,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
                path=learner.plots_path,
                fname="penalties",
                yaxis_log=True,
            )

            losses = [
                {"name": "tr loss", "data": learner.train_loss_list, "s": "-."},
                {"name": "tr recons", "data": learner.train_recons_list, "s": "-"},
                {"name": "val recons", "data": learner.valid_recons_list, "s": "-"},
                {"name": "KL", "data": learner.train_kl_list, "s": "-"},
                {"name": "val loss", "data": learner.valid_loss_list, "s": "-."},
                {"name": "tr ELBO", "data": learner.train_elbo_list, "s": "-."},
                {"name": "val ELBO", "data": learner.valid_elbo_list, "s": "-."},
            ]
            self.plot_learning_curves2(
                losses=losses,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
                path=learner.plots_path,
                fname="losses",
            )

            logvar = [
                {"name": "logvar encoder", "data": learner.logvar_encoder_tt, "s": "-"},
                {"name": "logvar decoder", "data": learner.logvar_decoder_tt, "s": "-"},
                {"name": "logvar transition", "data": learner.logvar_transition_tt, "s": "-"},
            ]
            self.plot_learning_curves2(
                losses=logvar,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
                path=learner.plots_path,
                fname="logvar",
            )

        # 2. SAVAR-specific: prepare context and plot original data
        logger.info("Preparing SAVAR-specific visualizations")
        modes_gt = self.prepare_savar_context(learner)

        # 3. Get adjacency matrices
        adj = learner.model.get_adj().cpu().detach().numpy()
        adj_w = learner.model.autoencoder.get_w_decoder().cpu().detach().numpy()
        adj_w2 = learner.model.autoencoder.get_w_encoder().cpu().detach().numpy()

        # 4. Plot SAVAR adjacency matrix with spatial alignment
        self.plot_adjacency_matrix_savar(
            learner=learner,
            mat1=adj,
            mat2=learner.datamodule.savar_gt_adj,
            modes_gt=modes_gt,
            modes_inferred=adj_w,
            path=learner.plots_path,
            name_suffix="transition",
            no_gt=False,
            iteration=learner.iteration,
            plot_through_time=learner.plot_params.plot_through_time,
        )

        # 5. Plot weight matrices
        if learner.latent:
            # Plot decoder and encoder weight matrices
            self.plot_adjacency_matrix_w(adj_w, None, learner.plots_path, "w", no_gt=True)
            adj_w2 = np.swapaxes(adj_w2, 1, 2)
            self.plot_adjacency_matrix_w(adj_w2, None, learner.plots_path, "encoder_w", no_gt=True)

            # Plot SAVAR feature maps (spatial patterns of learned latents)
            self.plot_savar_feature_maps(
                learner,
                adj_w,
                coordinates=learner.coordinates,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
                path=learner.plots_path,
            )

            # Plot decoder connectivity heatmap (NEW)
            self.plot_decoder_connectivity_heatmap(
                learner,
                adj_w,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
                path=learner.plots_path,
            )

        logger.info("Completed SAVAR-specific plotting")
