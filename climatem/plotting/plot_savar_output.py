"""SAVAR-specific plotting functions for synthetic data experiments."""

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from climatem.plotting.plot_model_output import Plotter
from climatem.utils import get_logger

logger = get_logger(__name__)


class SavarPlotter(Plotter):
    """
    Specialized plotter for SAVAR synthetic data experiments.

    Inherits from the base Plotter class and adds SAVAR-specific visualization methods including feature map plotting,
    adjacency matrix alignment, and forcing diagnostics.
    """

    def __init__(self):
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
        # SAVAR data is now in a subfolder
        savar_dataset_dir = Path(savar_folder) / savar_fname
        modes_gt = np.load(savar_dataset_dir / "modes.npy")

        if learner.iteration <= 10000:
            savar_data = np.load(savar_dataset_dir / "savar.npy")
            savar_anim_path = savar_dataset_dir / "original_savar_data.gif"
            self.plot_original_savar(savar_data, learner.lat, learner.lon, savar_anim_path)
            print(f"SAVAR ground truth modes shape: {modes_gt.shape}")
            print(f"SAVAR data shape: {savar_data.shape}")

        # Load forcing ground truth spatial patterns if forcing is enabled
        learner.co2_gt_spatial = None
        learner.aerosol_gt_spatial = None
        learner.forcing_indices = None

        if savar_params.is_forced:
            co2_forcing_path = savar_dataset_dir / "co2_forcing.npy"
            aerosol_forcing_path = savar_dataset_dir / "aerosol_forcing.npy"

            if co2_forcing_path.exists():
                co2_forcing = np.load(co2_forcing_path)  # Shape: (spatial_resolution, time)
                # Compute spatial pattern (time average)
                learner.co2_gt_spatial = co2_forcing.mean(axis=1).reshape(learner.lat, learner.lon)
                logger.info(f"Loaded CO2 forcing ground truth, spatial shape: {learner.co2_gt_spatial.shape}")

            if aerosol_forcing_path.exists():
                aerosol_forcing = np.load(aerosol_forcing_path)  # Shape: (spatial_resolution, time)
                # Compute spatial pattern (time average)
                learner.aerosol_gt_spatial = aerosol_forcing.mean(axis=1).reshape(learner.lat, learner.lon)
                logger.info(f"Loaded aerosol forcing ground truth, spatial shape: {learner.aerosol_gt_spatial.shape}")

            # Load forcing indices from datamodule if available
            if hasattr(learner, "datamodule") and hasattr(learner.datamodule, "forcing_indices"):
                learner.forcing_indices = learner.datamodule.forcing_indices
                logger.info(f"Forcing indices: {learner.forcing_indices}")

        # Plot forcing diagnostics if available from SAVAR instance
        if hasattr(learner, "datamodule") and hasattr(learner.datamodule, "savar"):
            savar = learner.datamodule.savar

            # Plot CO2 forcing if it was generated
            # Save forcing plots to SAVAR data subfolder, not results directory
            if savar is not None and hasattr(savar, "co2_forcing") and savar.co2_forcing is not None:
                savar_data_path = savar.savar_dataset_dir
                self.plot_forcing_diagnostic(savar.co2_forcing, "CO2", savar_data_path, learner.lat, learner.lon)
            # Plot aerosol forcing if it was generated
            if savar is not None and hasattr(savar, "aerosol_forcing") and savar.aerosol_forcing is not None:
                savar_data_path = savar.savar_dataset_dir
                self.plot_forcing_diagnostic(
                    savar.aerosol_forcing, "Aerosol", savar_data_path, learner.lat, learner.lon
                )

            # Plot aerosol latent trajectories to verify they have distinct patterns
            if (
                savar is not None
                and hasattr(savar, "aerosol_latent_trajectory")
                and savar.aerosol_latent_trajectory is not None
            ):
                savar_data_path = savar.savar_dataset_dir
                self.plot_aerosol_latent_trajectories(savar.aerosol_latent_trajectory, savar_data_path)

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
        im = ax.imshow(forcing_np, aspect="auto", origin="lower", interpolation="nearest", cmap="RdBu_r")
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
        im = ax.imshow(forcing_reshaped[peak_idx], cmap="RdBu_r", aspect="auto", origin="lower")
        ax.set_title(f"{forcing_name} Forcing: Spatial Pattern at Peak (t={peak_idx})")
        ax.set_xlabel("Longitude index")
        ax.set_ylabel("Latitude index")
        plt.colorbar(im, ax=ax, label=f"{forcing_name} magnitude")
        fig.tight_layout()
        filename = f"{forcing_name.lower()}_peak_spatial.png"
        fig.savefig(path / filename, dpi=150)
        plt.close(fig)
        logger.info(f"Saved {filename}")

        # 5. Animated GIF showing forcing progression
        max_frames = 120
        frame_stride = max(1, time_len // max_frames)
        frame_indices = np.arange(0, time_len, frame_stride, dtype=int)
        if frame_indices.size == 0 or frame_indices[-1] != time_len - 1:
            frame_indices = np.append(frame_indices, time_len - 1)

        # Create 2D spatial animation
        vmin = float(forcing_reshaped.min())
        vmax = float(forcing_reshaped.max())
        if vmin == vmax:
            vmax = vmin + 1e-6

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(
            forcing_reshaped[frame_indices[0]],
            vmin=vmin,
            vmax=vmax,
            cmap="RdBu_r",
            animated=True,
            origin="lower",
            aspect="auto",
        )
        ax.set_title(f"{forcing_name} Forcing Progression (t=0)")
        ax.set_xlabel("Longitude index")
        ax.set_ylabel("Latitude index")
        plt.colorbar(im, ax=ax, label=f"{forcing_name} magnitude")

        def update(idx):
            frame = frame_indices[idx]
            im.set_array(forcing_reshaped[frame])
            ax.set_title(f"{forcing_name} Forcing Progression (t={frame})")
            return (im,)

        anim = animation.FuncAnimation(fig, update, frames=len(frame_indices), interval=80, blit=True)

        fps = 10
        writer = animation.PillowWriter(fps=fps)
        filename = f"{forcing_name.lower()}_progression.gif"
        anim.save(path / filename, writer=writer)
        plt.close(fig)
        logger.info(f"Saved animated {filename}")

        logger.info(f"Completed comprehensive {forcing_name} forcing diagnostics")

    def plot_aerosol_latent_trajectories(self, aerosol_latent_traj, path, n_aerosol_latents=4):
        """
        Plot individual aerosol latent trajectories to verify they have distinct temporal patterns.

        Args:
            aerosol_latent_traj: Aerosol latent trajectories of shape (n_latents, time)
            path: Save path
            n_aerosol_latents: Number of aerosol latents
        """
        if aerosol_latent_traj is None:
            return

        n_latents = aerosol_latent_traj.shape[0]
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
        print(f"Aerosol latent correlations: avg={avg_corr:.4f}, max={max_corr:.4f} (target: < 0.5)")

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
                im = ax.imshow(grid, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower", aspect="auto")
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
        logger.info("Creating SAVAR feature maps visualization")
        w_adj = w_adj[0]  # Now w_adj_mean should be (lat*lon, num_latents)
        d_z = w_adj.shape[1]

        # Detect forcing configuration from model
        model = learner.model.module if hasattr(learner.model, "module") else learner.model
        use_forced_latents = getattr(model, "use_forced_latents", False)
        n_co2 = getattr(model, "n_forced_latents_co2", 0) if use_forced_latents else 0
        n_aerosol = getattr(model, "n_forced_latents_aerosol", 0) if use_forced_latents else 0
        n_climate = d_z - n_co2 - n_aerosol

        # Split latent indices
        climate_indices = list(range(n_climate))
        co2_indices = list(range(n_climate, n_climate + n_co2))
        aerosol_indices = list(range(n_climate + n_co2, d_z))

        logger.info(
            f"Climate latents: {climate_indices}, CO2 latents: {co2_indices}, Aerosol latents: {aerosol_indices}"
        )

        # ==== Figure 1: Climate Latents vs Ground Truth ====
        if len(climate_indices) > 0:
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
            im = ax.imshow(learner.datamodule.savar_gt_noise + learner.datamodule.savar_gt_modes, cmap="viridis")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title("Ground-Truth\nClimate Modes", fontsize="large")
            ax.tick_params(axis="both", labelsize="large")

            # Plot climate latent features
            for plot_idx, latent_idx in enumerate(climate_indices):
                ax = axs.flat[plot_idx + 1]
                feature_data = w_adj[:, latent_idx]
                data = feature_data.reshape(grid_shape)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                im = ax.imshow(data, cmap="viridis")
                plt.colorbar(im, cax=cax)
                ax.set_title(f"Climate Latent {latent_idx}", fontsize="large")
                ax.tick_params(axis="both", labelsize="large")

            # Remove unused subplots
            for ax in axs.flat[n_climate_plots:]:
                fig.delaxes(ax)

            fig.tight_layout()

            if plot_through_time:
                fname = f"spatial_aggregation_climate_{iteration}.png"
            else:
                fname = "spatial_aggregation_climate.png"

            plt.savefig(path / fname)
            plt.close()
            logger.info(f"Saved climate latent feature maps to {fname}")

        # ==== Figure 2: CO2 Forcing Latent ====
        if len(co2_indices) > 0 and learner.co2_gt_spatial is not None:
            n_co2_plots = len(co2_indices) + 1  # +1 for ground truth
            fig, axs = plt.subplots(
                nrows=1,
                ncols=n_co2_plots,
                figsize=(n_co2_plots * 4, 4),
            )
            if n_co2_plots == 1:
                axs = [axs]

            # Plot ground truth CO2 forcing spatial pattern
            ax = axs[0]
            im = ax.imshow(learner.co2_gt_spatial, cmap="RdBu_r")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title("Ground-Truth\nCO2 Forcing", fontsize="large")
            ax.tick_params(axis="both", labelsize="large")

            # Plot CO2 latent features
            for plot_idx, latent_idx in enumerate(co2_indices):
                ax = axs[plot_idx + 1]
                feature_data = w_adj[:, latent_idx]
                data = feature_data.reshape(grid_shape)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                im = ax.imshow(data, cmap="RdBu_r")
                plt.colorbar(im, cax=cax)
                ax.set_title(f"CO2 Latent {latent_idx}", fontsize="large")
                ax.tick_params(axis="both", labelsize="large")

            fig.tight_layout()

            if plot_through_time:
                fname = f"spatial_aggregation_co2_{iteration}.png"
            else:
                fname = "spatial_aggregation_co2.png"

            plt.savefig(path / fname)
            plt.close()
            logger.info(f"Saved CO2 latent feature maps to {fname}")

        # ==== Figure 3: Aerosol Forcing Latents ====
        if len(aerosol_indices) > 0 and learner.aerosol_gt_spatial is not None:
            n_aerosol_plots = len(aerosol_indices) + 1  # +1 for ground truth
            combined_map_n_rows = int(np.sqrt(n_aerosol_plots)) + 1
            combined_map_n_columns = int(np.ceil(n_aerosol_plots / combined_map_n_rows))

            fig, axs = plt.subplots(
                nrows=combined_map_n_rows,
                ncols=combined_map_n_columns,
                figsize=(combined_map_n_columns * 3, combined_map_n_rows * 3),
            )
            if combined_map_n_rows == 1:
                axs = axs.reshape(1, -1)

            # Plot ground truth aerosol forcing spatial pattern
            ax = axs.flat[0]
            im = ax.imshow(learner.aerosol_gt_spatial, cmap="RdBu_r")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            ax.set_title("Ground-Truth\nAerosol Forcing", fontsize="large")
            ax.tick_params(axis="both", labelsize="large")

            # Plot aerosol latent features
            for plot_idx, latent_idx in enumerate(aerosol_indices):
                ax = axs.flat[plot_idx + 1]
                feature_data = w_adj[:, latent_idx]
                data = feature_data.reshape(grid_shape)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                im = ax.imshow(data, cmap="RdBu_r")
                plt.colorbar(im, cax=cax)
                ax.set_title(f"Aerosol Latent {latent_idx}", fontsize="large")
                ax.tick_params(axis="both", labelsize="large")

            # Remove unused subplots
            for ax in axs.flat[n_aerosol_plots:]:
                fig.delaxes(ax)

            fig.tight_layout()

            if plot_through_time:
                fname = f"spatial_aggregation_aerosol_{iteration}.png"
            else:
                fname = "spatial_aggregation_aerosol.png"

            plt.savefig(path / fname)
            plt.close()
            logger.info(f"Saved aerosol latent feature maps to {fname}")

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

        Reveals which latents have strong/weak decoder connections and spatial structure.
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

        # Create latent labels
        latent_labels = []
        for i in range(n_climate):
            latent_labels.append(f"Climate {i}")
        for i in range(n_co2):
            latent_labels.append(f"CO2 {i}")
        for i in range(n_aerosol):
            latent_labels.append(f"Aerosol {i}")

        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Left plot: Full connectivity heatmap (spatial × latents)
        im1 = ax1.imshow(np.abs(w_adj), aspect="auto", cmap="viridis", interpolation="nearest")
        ax1.set_xlabel("Latent Index", fontsize=12)
        ax1.set_ylabel("Spatial Location", fontsize=12)
        ax1.set_title("Decoder Connectivity (Absolute Values)", fontsize=14)
        ax1.set_xticks(range(d_z))
        ax1.set_xticklabels(latent_labels, rotation=45, ha="right", fontsize=10)

        # Add vertical lines to separate latent types
        if n_climate > 0:
            ax1.axvline(n_climate - 0.5, color="red", linestyle="--", linewidth=2, label="Climate|CO2")
        if n_co2 > 0:
            ax1.axvline(n_climate + n_co2 - 0.5, color="orange", linestyle="--", linewidth=2, label="CO2|Aerosol")

        plt.colorbar(im1, ax=ax1, label="Weight Magnitude")

        # Right plot: Latent-wise L2 norms (bar chart)
        latent_norms = np.linalg.norm(w_adj, axis=0)  # L2 norm for each latent
        colors = ["blue"] * n_climate + ["red"] * n_co2 + ["orange"] * n_aerosol

        ax2.bar(range(d_z), latent_norms, color=colors, alpha=0.7)
        ax2.set_xlabel("Latent Index", fontsize=12)
        ax2.set_ylabel("Decoder Weight L2 Norm", fontsize=12)
        ax2.set_title("Latent Usage (Decoder Norms)", fontsize=14)
        ax2.set_xticks(range(d_z))
        ax2.set_xticklabels(latent_labels, rotation=45, ha="right", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="blue", alpha=0.7, label="Climate Latents"),
            Patch(facecolor="red", alpha=0.7, label="CO2 Latents"),
            Patch(facecolor="orange", alpha=0.7, label="Aerosol Latents"),
        ]
        ax2.legend(handles=legend_elements, loc="upper right")

        # Log latent norms
        logger.info(f"Climate latent norms: {latent_norms[:n_climate]}")
        logger.info(f"CO2 latent norms: {latent_norms[n_climate:n_climate+n_co2]}")
        logger.info(f"Aerosol latent norms: {latent_norms[n_climate+n_co2:]}")

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

        Uses spatial proximity of mode centroids to find the optimal permutation before plotting.
        """
        effective_no_gt = no_gt or mat2 is None
        lat = getattr(learner, "lat", None)
        lon = getattr(learner, "lon", None)

        mat1_to_plot = np.array(mat1, copy=True)

        if (
            not effective_no_gt
            and lat is not None
            and lon is not None
            and modes_gt is not None
            and modes_inferred is not None
        ):

            def _flatten_modes(arr):
                if arr is None:
                    return None
                arr = np.asarray(arr)
                try:
                    if arr.ndim == 3 and arr.shape[1:] == (lat, lon):
                        return arr.reshape(arr.shape[0], -1)
                    if arr.ndim == 2:
                        if arr.shape[1] == lat * lon:
                            return arr
                        if arr.shape[0] == lat * lon:
                            return arr.T
                    if arr.ndim == 1 and arr.size == lat * lon:
                        return arr.reshape(1, -1)
                    if arr.ndim >= 2:
                        return arr.reshape(arr.shape[0], -1)
                except Exception:
                    return None
                return None

            gt_flat = _flatten_modes(modes_gt)
            inferred_flat = _flatten_modes(modes_inferred)

            if gt_flat is not None and inferred_flat is not None:
                expected_size = lat * lon
                if gt_flat.shape[1] != expected_size or inferred_flat.shape[1] != expected_size:
                    gt_flat = None
                    inferred_flat = None

            if gt_flat is not None and inferred_flat is not None:
                n_modes = min(mat1_to_plot.shape[1], gt_flat.shape[0], inferred_flat.shape[0])
                if n_modes > 0:
                    # Find centroids (peak locations) for each mode
                    gt_idx = gt_flat[:n_modes].argmax(axis=1)
                    inf_idx = inferred_flat[:n_modes].argmax(axis=1)
                    if np.max(gt_idx) < expected_size and np.max(inf_idx) < expected_size:
                        # Convert flat indices to 2D coordinates
                        coords_gt = np.stack(np.unravel_index(gt_idx, (lat, lon)), axis=-1)
                        coords_inf = np.stack(np.unravel_index(inf_idx, (lat, lon)), axis=-1)
                        # Compute pairwise distances
                        distance = ((coords_gt[:, None, :] - coords_inf[None, :, :]) ** 2).sum(axis=2)
                        # Find optimal permutation (greedy matching)
                        permutation = distance.argmin(axis=1)
                        # Apply permutation to adjacency matrix
                        mat1_to_plot = np.take(mat1_to_plot, permutation, axis=1)
                        mat1_to_plot = np.take(mat1_to_plot, permutation, axis=2)
                        logger.info(f"Applied spatial alignment permutation: {permutation}")

        # Call parent class method to do the actual plotting
        self.plot_adjacency_matrix(
            learner=learner,
            mat1=mat1_to_plot,
            mat2=mat2,
            modes_gt=None,
            modes_inferred=None,
            path=path,
            name_suffix=name_suffix,
            savar=False,  # Set to False since we've already done SAVAR-specific processing
            no_gt=effective_no_gt,
            iteration=iteration,
            plot_through_time=plot_through_time,
        )

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
        adj = learner.model.module.get_adj().cpu().detach().numpy()
        adj_w = learner.model.module.autoencoder.get_w_decoder().cpu().detach().numpy()
        adj_w2 = learner.model.module.autoencoder.get_w_encoder().cpu().detach().numpy()

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
