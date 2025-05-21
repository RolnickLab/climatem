import torch
import subprocess
import numpy as np
import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from shapely.geometry import Point

import random
import argparse
from pathlib import Path

from climatem.plotting.plot_model_output import Plotter


class PlotterERA5:
    def __init__(self, coordinates, land_mask, plotter=None):
        self.coordinates = coordinates
        self.land_mask = land_mask
        self.plotter = plotter or Plotter()

    def get_spectral_loss_gridded(self, y_true_grid, take_log=True):
        """
        Compute 2D spatial power spectrum over lat-lon for gridded ERA5 data.

        Args:
            y_true_grid: Tensor of shape [B, 1, lat, lon]
            take_log: Whether to apply log scaling

        Returns:
            fft_true: Tensor of shape [freq_lat, freq_lon] (mean over batch)
        """
        assert y_true_grid.ndim == 4  # [B, 1, lat, lon]

        # Take real FFT over last two spatial dimensions: (lat, lon)
        fft2d = torch.fft.rfft2(y_true_grid, dim=(-2, -1))  # shape [B, 1, f_lat, f_lon]

        # Compute power spectrum
        power = torch.abs(fft2d) ** 2  # shape [B, 1, f_lat, f_lon]

        # Mean across batch and variable dims → [f_lat, f_lon]
        fft_true = power.mean(dim=(0, 1))

        if take_log:
            eps = 1e-8
            fft_true = torch.log(fft_true + eps)

        return fft_true  # shape [freq_lat, freq_lon]
    
    def get_spectral_loss(self, y_true, y_recons, y_pred, take_log=True):
        """
        Calculate the spectral loss between the true values and the predicted values. We need to calculate the spectra
        of thhe true values and the predicted values, and then determine an appropriate metric to compare them.

        There are a lot of design choices here that may not make a lot of sense.
        Averaging across batches? Square of the difference? Absolute value of the difference?

        Separating out the contributions of the different variables? All unclear.

        I might actually want to log this, so that the loss is not just dominated by the very low frequency, high power components.

        I should be setting some kind of limit at which I do this here - I am still not sure if it is an upper or lower bound that is the right threshold on the power spectrum.

        I am going to add this to both the reconstruction and the prediction.

        Args:
            y: torch.Tensor, the true values
            y_pred: torch.Tensor, the predicted values
        """

        # assert that y_true has 3 dimensions
        print("y_true.shape", y_true.shape, "y_recons.shape", y_recons.shape, "y_pred.shape", y_pred.shape)
        assert y_true.dim() == 3
        assert y_recons.dim() == 3
        assert y_pred.dim() == 3

        if y_true.size(-1) == 96 * 144:

            y_true = torch.reshape(y_true, (y_true.size(0), y_true.size(1), 96, 144))
            y_recons = torch.reshape(y_recons, (y_recons.size(0), y_recons.size(1), 96, 144))
            y_pred = torch.reshape(y_pred, (y_pred.size(0), y_pred.size(1), 96, 144))

            # calculate the spectra of the true values
            # note we calculate the spectra across space, and then take the mean across the batch
            fft_true = torch.mean(torch.abs(torch.fft.rfft(y_true[:, :, :], dim=3)), dim=0)
            # calculate the spectra of the reconstructed values
            fft_recons = torch.mean(torch.abs(torch.fft.rfft(y_recons[:, :, :], dim=3)), dim=0)
            # calculate the spectra of the predicted values
            fft_pred = torch.mean(torch.abs(torch.fft.rfft(y_pred[:, :, :], dim=3)), dim=0)

        elif y_true.size(-1) == 6250:

            y_true = y_true
            y_recons = y_recons
            y_pred = y_pred

            # calculate the spectra of the true values
            # note we calculate the spectra across space, and then take the mean across the batch
            fft_true = torch.mean(torch.abs(torch.fft.rfft(y_true[:, :, :], dim=2)), dim=0)
            # calculate the spectra of the reconstructed values
            fft_recons = torch.mean(torch.abs(torch.fft.rfft(y_recons[:, :, :], dim=2)), dim=0)
            # calculate the spectra of the predicted values
            fft_pred = torch.mean(torch.abs(torch.fft.rfft(y_pred[:, :, :], dim=2)), dim=0)
        else:
            raise ValueError("The size of the input is a surprise, and should be addressed here.")

        if take_log:
            fft_true = torch.log(fft_true)
            fft_recons = torch.log(fft_recons)
            fft_pred = torch.log(fft_pred)

        # Calculate the power spectrum
        spectral_loss_recons = torch.abs(fft_recons - fft_true)
        spectral_loss_pred = torch.abs(fft_pred - fft_true)

        spectral_loss = spectral_loss_recons + spectral_loss_pred

        spectral_loss = torch.mean(spectral_loss[..., :])
        print('what is the shape of the spectral loss?', spectral_loss)

        return fft_true, fft_recons, fft_pred, spectral_loss

    def get_temporal_spectral_loss(self, x, y_true, y_recons, y_pred):
        """
        Calculate the temporal power spectra for each grid cell and compare reconstructed and predicted values to ground truth.

        Args:
            x: Tensor of shape [B, T, 1, D]
            y_true, y_recons, y_pred: Tensor of shape [B, 1, D]
        Returns:
            fft_true, fft_recons, fft_pred (all [T//2+1, 1, D]), and scalar loss
        """

        y_true = y_true.unsqueeze(1)   # [B, 1, 1, D]
        y_recons = y_recons.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)

        obs = torch.cat((x, y_true), dim=1)     # [B, T+1, 1, D]
        recons = torch.cat((x, y_recons), dim=1)
        pred = torch.cat((x, y_pred), dim=1)

        # FFT over time (dim=1)
        fft_true = torch.abs(torch.fft.rfft(obs, dim=1))        # [B, T//2+1, 1, D]
        fft_recons = torch.abs(torch.fft.rfft(recons, dim=1))
        fft_pred = torch.abs(torch.fft.rfft(pred, dim=1))

        # Average over batch
        fft_true_mean = fft_true.mean(dim=0)        # [T//2+1, 1, D]
        fft_recons_mean = fft_recons.mean(dim=0)
        fft_pred_mean = fft_pred.mean(dim=0)

        # Spectral loss
        spectral_loss = torch.abs(fft_recons_mean - fft_true_mean) + torch.abs(fft_pred_mean - fft_true_mean)
        loss = torch.mean(spectral_loss)

        return fft_true_mean, fft_recons_mean, fft_pred_mean, loss
    
    @staticmethod
    def get_temporal_spectral_loss_gridded(arr, axis=0):
        fft = np.fft.fft(arr, axis=axis)
        power = np.abs(fft)**2
        return power.mean(axis=tuple(i for i in range(power.ndim) if i != axis))[:arr.shape[axis] // 2]

    def plot_icosahedral_spatial_spectrum(self, fft_true, fft_recons, fft_pred, save_path=None):
        freq_bins = fft_true.shape[-1]
        
        plt.figure(figsize=(12, 6))
        plt.plot(range(freq_bins), fft_true.squeeze(), label='True', alpha=0.8)
        plt.plot(range(freq_bins), fft_recons.squeeze(), label='Reconstruction', alpha=0.8)
        plt.plot(range(freq_bins), fft_pred.squeeze(), label='Prediction', alpha=0.8)
        plt.xlabel('Frequency Bin (Icosahedral)')
        plt.ylabel('Power Spectrum (log scale)')
        plt.title('Icosahedral Spatial Spectrum')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(Path(save_path) / "icosahedral_spatial_spectrum.png")
            plt.close()
        else:
            plt.show()
            
    def plot_gridded_spatial_spectrum(self, gridded_true, gridded_recons=None, gridded_pred=None, save_path=None):
        """
        Plot the gridded spatial frequency spectrum from 2D FFT (averaged over latitude).

        Args:
            gridded_true: Tensor or array of shape [freq_lat, freq_lon]
            gridded_recons, gridded_pred: Optional matching arrays
            save_path: Output directory
        """
        # If input is 2D (from rfft2), average over latitude (dim=0)
        if gridded_true.ndim == 2:
            gridded_true = gridded_true.mean(dim=0) if isinstance(gridded_true, torch.Tensor) else gridded_true.mean(axis=0)
        if gridded_recons is not None and gridded_recons.ndim == 2:
            gridded_recons = gridded_recons.mean(dim=0) if isinstance(gridded_recons, torch.Tensor) else gridded_recons.mean(axis=0)
        if gridded_pred is not None and gridded_pred.ndim == 2:
            gridded_pred = gridded_pred.mean(dim=0) if isinstance(gridded_pred, torch.Tensor) else gridded_pred.mean(axis=0)

        freq_bins = gridded_true.shape[-1]
        plt.figure(figsize=(12, 6))
        plt.plot(range(freq_bins), gridded_true.squeeze(), label='Gridded True', linestyle='--')
        if gridded_recons is not None:
            plt.plot(range(freq_bins), gridded_recons.squeeze(), label='Gridded Recons', linestyle='-.')
        if gridded_pred is not None:
            plt.plot(range(freq_bins), gridded_pred.squeeze(), label='Gridded Pred', linestyle=':')

        plt.xlabel('Spatial Frequency Bin (Longitude)')
        plt.ylabel('Mean Power Spectrum (log scale)')
        plt.title('Gridded Spatial Frequency Spectrum (Averaged over Latitude)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(Path(save_path) / "gridded_spatial_spectrum.png")
            plt.close()
        else:
            plt.show()

    def plot_temporal_spectrum_icosahedral(self, fft_true, fft_recons, fft_pred, save_path=None):
        freq_bins = fft_true.shape[0]
        plt.figure(figsize=(12, 6))
        plt.plot(range(freq_bins), fft_true.mean(dim=1).squeeze().numpy(), label="True", alpha=0.8)
        plt.plot(range(freq_bins), fft_recons.mean(dim=1).squeeze().numpy(), label="Reconstruction", alpha=0.8)
        plt.plot(range(freq_bins), fft_pred.mean(dim=1).squeeze().numpy(), label="Prediction", alpha=0.8)
        plt.xlabel("Temporal Frequency Bin")
        plt.ylabel("Power Spectrum (log scale)")
        plt.title("Temporal Spectrum (Icosahedral)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(Path(save_path) / "temporal_spectrum_icosahedral.png")
            plt.close()
        else:
            plt.show()

    def plot_temporal_spectrum_gridded(self, gridded_power, save_path=None):
        plt.figure(figsize=(10, 4))
        plt.semilogy(avg_temporal)
        plt.title("Temporal Power Spectrum (mean over space)")
        plt.xlabel("Frequency Bin")
        plt.ylabel("Power")
        plt.grid(True)
        plt.tight_layout()
        if save_path:
            plt.savefig(Path(save_path) / "temporal_spectrum_gridded.png")
            plt.close()
        else:
            plt.show()

    def plot_avg_temp_histograms(self, y_true, y_recons, y_pred, sampled_days, num_samples=100):
        latitudes = self.coordinates[:, 1]
        masks = {
            "Over Ocean": self.land_mask == 0,
            "Over Land": self.land_mask == 1,
            "In Tropics": np.abs(latitudes) <= 23.5,
            "Outside Tropics": np.abs(latitudes) > 23.5
        }
        for label, mask in masks.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(y_recons[sampled_days][:, mask].mean(axis=1), bins=100, alpha=0.6, label="Reconstruction")
            ax.hist(y_pred[sampled_days][:, mask].mean(axis=1), bins=100, alpha=0.6, label="Prediction")
            ax.hist(y_true[sampled_days][:, mask].mean(axis=1), bins=100, alpha=0.8, label="Observed", edgecolor="black")
            ax.set_title(f"Avg. Daily Temp • {label}")
            ax.set_xlabel("Normalized Temperature")
            ax.set_ylabel("Frequency")
            ax.legend()
            plt.tight_layout()
            plt.show()

    def plot_avg_temp_cdfs(self, y_true, y_recons, y_pred, sampled_days):
        latitudes = self.coordinates[:, 1]
        masks = {
            "Over Ocean": self.land_mask == 0,
            "Over Land": self.land_mask == 1,
            "In Tropics": np.abs(latitudes) <= 23.5,
            "Outside Tropics": np.abs(latitudes) > 23.5
        }
        for label, mask in masks.items():
            fig, ax = plt.subplots(figsize=(4, 6))
            for data, lbl in zip([y_true, y_recons, y_pred], ["Observed", "Reconstruction", "Prediction"]):
                vals = np.sort(data[sampled_days][:, mask].mean(axis=1))
                cdf = np.linspace(0, 1, len(vals))
                ax.plot(vals, cdf, label=lbl, linewidth=2)
            ax.set_title(f"CDF of Avg. Daily Temp • {label}")
            ax.set_xlabel("Normalized Temperature")
            ax.set_ylabel("Cumulative Probability")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            plt.show()

    def generate_prediction_video(self, x_past, y_true, y_recons, y_hat, out_dir, sample_index=0, fps=5):
        out_dir = Path(out_dir)
        frame_dir = out_dir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Save individual frames
        for t in range(y_true.shape[0]):
            frame_path = frame_dir / f"frame_{t:04d}.png"
            self.plotter.plot_compare_predictions_icosahedral(
                x_past, y_true, y_recons, y_hat, sample=t, coordinates=self.coordinates,
                path=frame_path, iteration=t, valid="video"
            )

        # Use ffmpeg to stitch them into a video
        video_path = out_dir / "validation_prediction_vs_truth.mp4"
        try:
            subprocess.run([
                "ffmpeg",
                "-y",  # overwrite if exists
                "-framerate", str(fps),
                "-i", str(frame_dir / "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(video_path)
            ], check=True)
            print(f"Video saved to {video_path}")
        except subprocess.CalledProcessError as e:
            print("Failed to create video with ffmpeg:", e)

    def plot_top_extreme_events(self, y_true, y_recons, y_hat, x_past, output_dir, top_k=10, adj_w_fixed=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        arr = y_true.squeeze(1).cpu().numpy()
        flat_indices = np.argpartition(arr.flatten(), -top_k)[-top_k:]
        top_values = arr.flatten()[flat_indices]
        day_indices, coord_indices = np.unravel_index(flat_indices, arr.shape)

        top_extremes = sorted(zip(day_indices, coord_indices, top_values), key=lambda x: x[2], reverse=True)

        for i, (day, coord, val) in enumerate(top_extremes):
            fig = self.plotter.plot_compare_predictions_icosahedral(
                x_past, y_true, y_recons, y_hat, sample=day, coordinates=self.coordinates,
                path=output_dir, iteration=day, valid="extremes"
            )

            ax = fig.axes[-1]
            lat, lon = self.coordinates[coord]
            ax.plot(lon, lat, 'o', color='lime', markersize=8, transform=ccrs.PlateCarree())
            ax.text(lon + 2, lat, f"{val:.2f}", transform=ccrs.PlateCarree(), color='lime', fontsize=10, bbox=dict(facecolor='black', alpha=0.5, boxstyle='round'))
            fig.savefig(output_dir / f"highlight_extreme_{day}.png")
            plt.close(fig)

            if adj_w_fixed is not None:
                self.plotter.plot_regions_map(adj_w_fixed, self.coordinates, iteration=day, plot_through_time=True, path=output_dir, annotate=True, one_plot=True)

def create_land_mask_from_file(coord_path):
    coordinates = np.load(coord_path)  # expecting shape (N, 2) with (lat, lon)
    land_shp = shapereader.natural_earth(resolution='110m', category='physical', name='land')
    land_geom = list(shapereader.Reader(land_shp).geometries())

    land_mask = np.zeros(len(coordinates), dtype=np.uint8)
    for i, (lat, lon) in enumerate(coordinates):
        point = Point(lon, lat)
        if any(poly.contains(point) for poly in land_geom):
            land_mask[i] = 1

    return land_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, required=True)
    parser.add_argument("--variable", type=str, default="t2m")
    parser.add_argument("--years", type=str, required=True)
    parser.add_argument("--coord_path", type=str, required=True)
    parser.add_argument("--land_mask_path", type=str, required=True)
    parser.add_argument("--y_true_path", type=str, required=True)
    parser.add_argument("--y_recons_path", type=str, required=True)
    parser.add_argument("--y_pred_path", type=str, required=True)
    parser.add_argument("--x_past_path", type=str, required=False)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    coords = np.load(args.coord_path)
    land_mask = create_land_mask_from_file(args.coord_path)

    y_true = torch.from_numpy(
        np.concatenate([np.load(f) for f in sorted(glob.glob(args.y_true_path))], axis=0)
    ).float()

    y_recons = torch.from_numpy(
        np.concatenate([np.load(f) for f in sorted(glob.glob(args.y_recons_path))], axis=0)
    ).float()

    y_pred = torch.from_numpy(
        np.concatenate([np.load(f) for f in sorted(glob.glob(args.y_pred_path))], axis=0)
    ).float()

    if args.x_past_path:
        x_past = torch.from_numpy(
            np.concatenate([np.load(f) for f in sorted(glob.glob(args.x_past_path))], axis=0)
        ).float()
    else:
        x_past = torch.zeros_like(y_true)

    print("x_past, y_true, y_recons, y_pred", x_past.shape, y_true.shape, y_recons.shape, y_pred.shape)

    plotter = PlotterERA5(coordinates=coords, land_mask=land_mask)

    # Efficient gridded sampling
    npz_dir = Path(args.npz_dir)
    if "-" in args.years:
        year_range = list(range(int(args.years.split("-")[0]), int(args.years.split("-")[1]) + 1))
    else:
        year_range = [int(args.years)]

    all_samples = []
    all_temporal = []
    target_sample_count = 128
    samples_per_year = target_sample_count // len(year_range) + 1

    for year in year_range:
        npz_path = npz_dir / f"{args.variable}_{year}.npz"
        if not npz_path.exists():
            continue

        # data = np.load(npz_path, mmap_mode='r')["data"]
        # if data.shape[0] != 365:
        #     continue
        # temporal_power = plotter.get_temporal_spectral_loss_gridded(data)

        # all_temporal.append(temporal_power)

        # indices = np.random.choice(data.shape[0], min(samples_per_year, data.shape[0]), replace=False)
        # sampled = torch.from_numpy(data[indices]).unsqueeze(1).float()  # [N, 1, 721, 1440]
        # all_samples.append(sampled)

    # if all_samples:
    #     y_true_grid = torch.cat(all_samples, dim=0)[:128]  # ensure exactly 128 samples
    #     print("Sampled y_true_grid:", y_true_grid.shape)
    # else:
    #     raise RuntimeError("No valid ERA5 .npz data found for sampling.")

    # Spectral analysis
    fft_true, fft_recons, fft_pred, _ = plotter.get_spectral_loss(y_true, y_recons, y_pred, take_log=True)
    # fft_true_grid = plotter.get_spectral_loss_gridded(y_true_grid, take_log=True)
    fft_temporal_true, fft_temporal_recons, fft_temporal_pred, _ = plotter.get_temporal_spectral_loss(
        x_past, y_true, y_recons, y_pred
    )

    avg_temporal = np.mean(np.stack(all_temporal), axis=0)

    # Plot
    plotter.plot_icosahedral_spatial_spectrum(
        fft_true.numpy(), fft_recons.numpy(), fft_pred.numpy(),
        save_path=Path(args.out_dir)
    )

    # gridded_true = fft_true_grid.squeeze()
    # plotter.plot_gridded_spatial_spectrum(gridded_true.numpy(), save_path=Path(args.out_dir))
    plotter.plot_temporal_spectrum_icosahedral(fft_temporal_true, fft_temporal_recons, fft_temporal_pred, save_path=Path(args.out_dir))
    plotter.plot_temporal_spectrum_gridded(avg_temporal, save_path=Path(args.out_dir))

    print("Saved combined spatial and temporal spectrum plots to", args.out_dir)