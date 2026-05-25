# This is a script to analyze what z2 / pz2_mu encodes.


import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
import numpy as np
import torch
import healpy as hp
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from climatem.model.tsdcd_latent import LatentTSDCD
from climatem.config import *
from climatem.utils import parse_args, update_config_withparse

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")

class PerturbLatentTSDCD(LatentTSDCD):

    def __init__(
        self,
        **kwargs
    ):

        super().__init__(**kwargs)


    def predict_z1z2(self, x, y):

        """
        This is the prediction function for the model.

        We want to take past time steps and predict the next time step, not to reconstruct the past time steps.
        """
        b = x.size(0)

        # NOTE: we are not using y here. We encode using both x and y,
        # but then we discard the latents from the y encoding.

        z, _, _ = self.encode(x, y)
        z2, _, _ = self.encode_global(z)

        mask = self.mask(b)

        if self.instantaneous:
            pz2_mu, _ = self.transition_global(z2[:, :-1].clone())
            pz_mu, pz_std = self.transition(z.clone(), pz2_mu.clone(), mask)
        else:
            pz2_mu, pz2_std = self.transition_global(z2[:, :-1].clone())  # (b,1,d_z2)
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), pz2_mu.clone(), mask)

        # decode
        px_mu, _ = self.decode(pz_mu)

        return z, z2
    def predict_pz2mu_x(self, x, y):

        """
        This is the prediction function for the model.

        We want to take past time steps and predict the next time step, not to reconstruct the past time steps.
        """
        b = x.size(0)

        # NOTE: we are not using y here. We encode using both x and y,
        # but then we discard the latents from the y encoding.

        z, _, _ = self.encode(x, y)
        z2, _, _ = self.encode_global(z)

        mask = self.mask(b)

        if self.instantaneous:
            pz2_mu, _ = self.transition_global(z2[:, :-1].clone())
            pz_mu, pz_std = self.transition(z.clone(), pz2_mu.clone(), mask)
        else:
            pz2_mu, pz2_std = self.transition_global(z2[:, :-1].clone())  # (b,1,d_z2)
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), pz2_mu.clone(), mask)

        # decode
        px_mu, _ = self.decode(pz_mu)

        return pz2_mu, px_mu
    def predict_counterfactual_on_pz2mu(self, x, y, counterfactual_z_global_value):

        b = x.size(0)

        z, _, _ = self.encode(x, y)
        z2, _, _ = self.encode_global(z)

        mask = self.mask(b)

        if self.instantaneous:
            pz2_mu, _ = self.transition_global(z2[:, :-1].clone())
            pz_mu, pz_std = self.transition(z.clone(), pz2_mu, mask)
        else:
            pz2_mu, _ = self.transition_global(z2[:, :-1].clone())  # (b,1,d_z2)
            # Perturb on pz2_mu (AFTER transition):
            pz2_mu[:, 0, 0] = counterfactual_z_global_value
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), pz2_mu, mask)

        # decode
        px_mu, _ = self.decode(pz_mu)

        return px_mu, pz2_mu
    
    def predict_counterfactual_on_z2(self, x, y, counterfactual_z_global_value):
        
        b = x.size(0)

        z, _, _ = self.encode(x, y)
        z2, _, _ = self.encode_global(z)

        # Perturb on z2 BEFORE transition:
        z2[:, -2, 0, 0] = counterfactual_z_global_value

        # print("This is e.g. the new value of the latents after intervention.", z[0, -2, 0, 0])

        mask = self.mask(b)

        if self.instantaneous:
            pz2_mu, _ = self.transition_global(z2[:, :-1].clone())
            pz_mu, pz_std = self.transition(z.clone(), pz2_mu, mask)
        else:
            pz2_mu, _ = self.transition_global(z2[:, :-1].clone())  # (b,1,d_z2)
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), pz2_mu, mask)

        # decode
        px_mu, _ = self.decode(pz_mu)

        return px_mu, pz2_mu
    
    def predict_sample_bayesianfiltering_perturbed(self, x, y, iteration ,num_samples, with_zs_logprob: bool = False):
        """
        Perturb on pz2_mu
        """
        b = x.size(0)

        with torch.no_grad():
            # sample Zs (based on X)
            z, q_mu_y, q_std_y = self.encode(x, y)
            z2, q_mu_z2, q_std_z2 = self.encode_global(z)

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)

            if self.instantaneous:
                pz2_mu, pz2_std = self.transition_global(z2[:, :-1].clone())
                pz_mu, pz_std = self.transition(z.clone(), pz2_mu, mask)
            else:
                pz2_mu, pz2_std = self.transition_global(z2[:, :-1].clone())  # (b,1,d_z2)
                # if  iteration>0:
                #     pz2_mu = pz2_mu - 2*pz2_std
                # if iteration == 0: 
                #     pertrubed_value = pz2_mu[:,0,0].clone()
                #     self.init = pz2_mu[:,0,0].clone()
                # else:
                #     alpha = iteration / 600
                #     pertrubed_value = self.init + alpha * (-20)
                # else:
                #     # linear decay
                #     pertrubed_value = -20.0
                if iteration == 0:
                    pertrubed_value = pz2_mu[:, 0, 0].clone()
                    self.init = pz2_mu[:, 0, 0].clone()
                else:
                    if iteration <= 150:
                        # phase 1: init -> -15
                        alpha = iteration / 150.0
                        pertrubed_value = self.init + alpha * (-15 - self.init)

                    elif iteration <= 450:
                        # phase 2: -15 -> +15
                        alpha = (iteration - 150) / 300.0
                        pertrubed_value = (-15) + alpha * (15 - (-15))

                    else:
                        # phase 3: +15 -> 0
                        alpha = (iteration - 450) / 150.0
                        pertrubed_value = (15) + alpha * (0 - 15)

                pz2_mu[:, 0, 0] = pertrubed_value
                pz_mu, pz_std = self.transition(z[:, :-1].clone(), pz2_mu.clone(), mask)
            dim = pz_mu.ndim
            new_shape = [num_samples]
            for k in range(dim):
                new_shape.append(1)
            z_samples = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).sample()

            if with_zs_logprob:
                z_samples_logprob = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).log_prob(
                    z_samples
                )
            samples_from_zs, some_decoded_samples_std = self.decode(
                z_samples.reshape(z_samples.size(0) * z_samples.size(1), z_samples.size(2), z_samples.size(3))
            )
            samples_from_zs = samples_from_zs.reshape(z_samples.size(0), z_samples.size(1), z_samples.size(2), self.d_x)
            # decode
            px_mu, px_std = self.decode(pz_mu.unsqueeze(1))
            px_mu = px_mu.squeeze(1)
            px_std = px_std.squeeze(1)

            dim = px_mu.ndim
            new_shape = [num_samples]
            for k in range(dim):
                new_shape.append(1)
            # here we decode from pz_mu, and then sample from the distribution over xs.
            # note this will simply give us chequerboards.
            samples_from_xs = torch.zeros(num_samples, b, self.d, self.d_x)
            samples_from_xs = self.distr_decoder(px_mu.repeat(new_shape), px_std.repeat(new_shape)).sample()

        if with_zs_logprob:
            return samples_from_xs, samples_from_zs, y, z_samples_logprob, pz2_mu
        return samples_from_xs, samples_from_zs, y

    def predict_sample_bayesianfiltering_perturbed_x(self, x, y, iteration ,num_samples, with_zs_logprob: bool = False, perturbed_area=None):
        """
        Perturb on x
        """

        b = x.size(0)
        if iteration == 0:
            perturbed_value = x[..., perturbed_area].clone()
            self.init = x[..., perturbed_area].clone()
        else:
            if iteration <= 150:
                # phase 1: init -> -15
                alpha = iteration / 150.0
                perturbed_value = self.init + alpha * (-15 - self.init)
                self.init_next = perturbed_value

            elif iteration <= 450:
                # phase 2: -15 -> +15
                alpha = (iteration - 150) / 300.0
                perturbed_value =  self.init_next + alpha * (15 - (-15))
                self.init_next2 = perturbed_value

            else:
                # phase 3: +15 -> 0
                alpha = (iteration - 450) / 150.0
                perturbed_value = self.init_next2 + alpha * (0 - 15)
    
        x[..., perturbed_area] = perturbed_value

        with torch.no_grad():
            # sample Zs (based on X)
            z, q_mu_y, q_std_y = self.encode(x, y)
            z2, q_mu_z2, q_std_z2 = self.encode_global(z)

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)

            if self.instantaneous:
                pz2_mu, pz2_std = self.transition_global(z2[:, :-1].clone())
                pz_mu, pz_std = self.transition(z.clone(), pz2_mu, mask)
            else:
                pz2_mu, pz2_std = self.transition_global(z2[:, :-1].clone())  # (b,1,d_z2)
                pz_mu, pz_std = self.transition(z[:, :-1].clone(), pz2_mu.clone(), mask)
            dim = pz_mu.ndim
            new_shape = [num_samples]
            for k in range(dim):
                new_shape.append(1)
            z_samples = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).sample()

            if with_zs_logprob:
                z_samples_logprob = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).log_prob(
                    z_samples
                )
            samples_from_zs, some_decoded_samples_std = self.decode(
                z_samples.reshape(z_samples.size(0) * z_samples.size(1), z_samples.size(2), z_samples.size(3))
            )
            samples_from_zs = samples_from_zs.reshape(z_samples.size(0), z_samples.size(1), z_samples.size(2), self.d_x)
            # decode
            px_mu, px_std = self.decode(pz_mu.unsqueeze(1))
            px_mu = px_mu.squeeze(1)
            px_std = px_std.squeeze(1)

            dim = px_mu.ndim
            new_shape = [num_samples]
            for k in range(dim):
                new_shape.append(1)
            # here we decode from pz_mu, and then sample from the distribution over xs.
            # note this will simply give us chequerboards.
            samples_from_xs = torch.zeros(num_samples, b, self.d, self.d_x)
            samples_from_xs = self.distr_decoder(px_mu.repeat(new_shape), px_std.repeat(new_shape)).sample()

        if with_zs_logprob:
            return samples_from_xs, samples_from_zs, y, z_samples_logprob, pz2_mu, perturbed_value
        return samples_from_xs, samples_from_zs, y
def get_all_temp_and_gt(datamodule, accelerator, model, device):
           # Start again at the beginning of the dataloader.
    train_dataloader = iter(datamodule.train_dataloader(accelerator))

    # iterate through the data and append all the y values together
    pred_all = []
    gt_all = []

    for i in range(len(train_dataloader)):
        x, y = next(train_dataloader)

        x = torch.nan_to_num(x).to(device)
        y = torch.nan_to_num(y).to(device)
        y = y[:, 0]        
       
        with torch.no_grad():
           pred, _, _, _, _, _ = model.predict(x, y) 

        pred_all.append(pred)
        gt_all.append(y)
       

    pred_all = torch.cat(pred_all, dim=0)
    pred_all = torch.nan_to_num(pred_all) 

    gt_all = torch.cat(gt_all, dim=0)
    gt_all = torch.nan_to_num(gt_all)

    
    # make sure we reset the dataloader
    train_dataloader = iter(datamodule.train_dataloader(accelerator))


    return pred_all, gt_all     

def get_all_temp_and_z2_after_perturb(datamodule, accelerator, model, device, counterfactual_z_global_value):
        # Start again at the beginning of the dataloader.
    train_dataloader = iter(datamodule.train_dataloader(accelerator))

    # iterate through the data and append all the y values together
    pred_all = []
    z_global_all = []
    for i in range(len(train_dataloader)):
        x, y = next(train_dataloader)

        x = torch.nan_to_num(x).to(device)
        y = torch.nan_to_num(y).to(device)
        y = y[:, 0]        
       
        with torch.no_grad():
            # Which variable you want to perturb at? pz2_mu or z2?
           pred, pz2_mu = model.predict_counterfactual_on_pz2mu(x, y, counterfactual_z_global_value=counterfactual_z_global_value) # z_local shape [b, tau+1, 1, d_z], # z_global shape [b, tau+1, 1, 1]

        pred_all.append(pred)
        z_global_all.append(pz2_mu)

    z_global_all = torch.cat(z_global_all, dim=0)
    z_global_all = torch.nan_to_num(z_global_all) #(torch.Size([4608, 6, 1, 1]))

    pred_all = torch.cat(pred_all, dim=0)
    pred_all = torch.nan_to_num(pred_all) 
 
    # make sure we reset the dataloader
    train_dataloader = iter(datamodule.train_dataloader(accelerator))


    return pred_all, z_global_all
   
def get_all_z1_and_z2(datamodule, accelerator, model, device):
        # Start again at the beginning of the dataloader.
    train_dataloader = iter(datamodule.train_dataloader(accelerator))

    # iterate through the data and append all the y values together
    z_local_all = []
    z_global_all = []

    for i in range(len(train_dataloader)):
        x, y = next(train_dataloader)

        x = torch.nan_to_num(x).to(device)
        y = torch.nan_to_num(y).to(device)
        y = y[:, 0]        
       
        with torch.no_grad():
           z_local, z_global = model.predict_z1z2(x, y) # z_local shape [b, tau+1, 1, d_z], # z_global shape [b, tau+1, 1, 1]

        z_global_all.append(z_global)
        z_local_all.append(z_local)

    z_global_all = torch.cat(z_global_all, dim=0)
    z_global_all = torch.nan_to_num(z_global_all) #(torch.Size([4608, 6, 1, 1]))

    z_local_all = torch.cat(z_local_all, dim=0)
    z_local_all = torch.nan_to_num(z_local_all) #z_local_all torch.Size([4608, 6, 1, 60])

    # make sure we reset the dataloader
    train_dataloader = iter(datamodule.train_dataloader(accelerator))


    return z_local_all, z_global_all

def get_all_pz2mu_and_temp(datamodule, accelerator, model, device):
        # Start again at the beginning of the dataloader.
    train_dataloader = iter(datamodule.train_dataloader(accelerator))

    # iterate through the data and append all the y values together
    pz2_mu_all = []
    y_pred_all = []

    for i in range(len(train_dataloader)):
        x, y = next(train_dataloader)

        x = torch.nan_to_num(x).to(device)
        y = torch.nan_to_num(y).to(device)
        y = y[:, 0]        
       
        with torch.no_grad():
           pz2_mu, y_pred = model.predict_pz2mu_x(x, y) # z_local shape [b, tau+1, 1, d_z], # z_global shape [b, tau+1, 1, 1]

        pz2_mu_all.append(pz2_mu)
        y_pred_all.append(y_pred)

    pz2_mu_all = torch.cat(pz2_mu_all, dim=0)
    pz2_mu_all = torch.nan_to_num(pz2_mu_all) #(torch.Size([4608, 1, 1]))

    y_pred_all = torch.cat(y_pred_all, dim=0)
    y_pred_all = torch.nan_to_num(y_pred_all) #torch.Size([4608, 1, 3072])
 
    # make sure we reset the dataloader
    train_dataloader = iter(datamodule.train_dataloader(accelerator))

    return pz2_mu_all, y_pred_all

def plot_bar_map(statistics, statistics_name:str, save_path):

    # ============================================================
    # BAR PLOT OF CORRELATIONS / R^2
    # ============================================================

    latent_ids = np.arange(statistics.shape[0])

    plt.figure(figsize=(12, 5))
    plt.bar(latent_ids, statistics, width=0.4, label=statistics_name)

    plt.xlabel("z1 latent dimension")
    plt.ylabel(f"{statistics_name} with z2")
    plt.title(f"{statistics_name} between z2 and each z1 latent")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path/f"{statistics_name}_train_all.png")
    plt.close()

def get_centroid(xs, ys):
    """
    http://www.geomidpoint.com/example.html
    http://gis.stackexchange.com/questions/6025/find-the-centroid-of-a-cluster-of-points
    """
    sum_x, sum_y, sum_z = 0, 0, 0
    n = float(xs.shape[0])

    if n > 0:
        for x, y in zip(xs, ys):
            lat = np.radians(y)
            lon = np.radians(x)
            ## convert lat lon to cartesian coordinates
            sum_x += np.cos(lat) * np.cos(lon)
            sum_y += np.cos(lat) * np.sin(lon)
            sum_z += np.sin(lat)
        avg_x = sum_x / n
        avg_y = sum_y / n
        avg_z = sum_z / n
        center_lon = np.arctan2(avg_y, avg_x)
        hyp = np.sqrt(avg_x * avg_x + avg_y * avg_y)
        center_lat = np.arctan2(avg_z, hyp)
        final_x, final_y = np.degrees(center_lon), np.degrees(center_lat)
        return final_x, final_y
    else:
        return 0.0, 0.0
from matplotlib.patches import Rectangle

def add_region_box(ax, lon_min, lon_max, lat_min, lat_max, label, color="black"):
    rect = Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        linewidth=2,
        edgecolor=color,
        facecolor="none",
        transform=ccrs.PlateCarree(),
        zorder=10,
    )

    ax.add_patch(rect)

    ax.text(
        lon_min,
        lat_max + 3,
        label,
        color=color,
        fontsize=12,
        fontweight="bold",
        transform=ccrs.PlateCarree(),
        zorder=11,
    )
def plot_regions_map(
        w_adj,
        statistics,
        statistics_name,
        coordinates: np.ndarray,
        save_path
    ):
        """Here we extend the plot_regions_map function to plot multiple variables."""

        # find the argmax per row
        idx = np.argmax(w_adj, axis=-1) 
        # len(idx) = 3072, each element is a number, corresponding to cluster 0-59, from soft (probabilities) to discrete region label 

        # here we want the number of latents
        d_z = w_adj.shape[-1]

        # plot the regions

        cmap = plt.cm.coolwarm
        # 95 percentile values can better visualize the difference 
        # norm = plt.Normalize(vmin=np.min(statistics), vmax=np.max(statistics))
        v = np.max(abs(statistics))
        norm = plt.Normalize(vmin=-v, vmax=v)

        # First, I will assert that I have two columns.
        assert coordinates.shape[1] == 2

        # Then, swap the columns if the first column at the moment is the longitude column.
        coords = coordinates.copy()
        if np.max(coords[:, 0]) > 91:
            coords = np.flip(coords, axis=1)

        fig, ax = plt.subplots(
                1, 1, subplot_kw={"projection": ccrs.Robinson()}, layout="constrained", figsize=(32, 8)
            )
     
        ax.set_global()
        ax.coastlines()
        # Add some map features for context
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.gridlines(draw_labels=False)
        # ---------------------------
        # ENSO region (Niño3.4)
        # ---------------------------
        add_region_box(
            ax,
            lon_min=-170,
            lon_max=-120,
            lat_min=-5,
            lat_max=5,
            label="ENSO",
            color="red",
        )

        # ---------------------------
        # IOD west pole
        # ---------------------------
        add_region_box(
            ax,
            lon_min=50,
            lon_max=70,
            lat_min=-10,
            lat_max=10,
            label="IOD-West",
            color="green",
        )

        # IOD east pole
        add_region_box(
            ax,
            lon_min=90,
            lon_max=110,
            lat_min=-10,
            lat_max=0,
            label="IOD-East",
            color="green",
        )

        # ---------------------------
        # AMO region
        # ---------------------------
        add_region_box(
            ax,
            lon_min=-80,
            lon_max=0,
            lat_min=0,
            lat_max=70,
            label="AMO",
            color="purple",
        )


        # Vectorized scatter plot with color array
        for k in range(d_z):
            region = coords[idx == k]
            
            corr_value = statistics[k]

            color = cmap(norm(corr_value))

            c = np.repeat(
                np.array([color]),
                region.shape[0],
                axis=0
            )

            ax.scatter(
                x=region[:, 1],
                y=region[:, 0],
                c=c,
                alpha=1.0,
                s=35,
                transform=ccrs.PlateCarree(),

            )

            # add number for each region (that are completely in one of the four quadrants)
            if region.shape[0] >0:
                x, y = get_centroid(region[:, 1], region[:, 0])
                ax.text(x, y, str(k), transform=ccrs.PlateCarree())
            print(f"found region {k} with {region.shape[0]} points")
        
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=norm
        )

        sm.set_array([])

        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            shrink=0.4,
            pad=0.05,
        )

        cbar.set_label(f"{statistics_name}", fontsize=14)
        fname = f"spatial_aggregation_{statistics_name}.png"

        plt.savefig(save_path / fname, format="png")
        print(f"fig {fname} saved!")
        plt.close()

def plot_regions_map_cluster_k(
        w_adj,
        statistics,
        statistics_name,
        coordinates: np.ndarray,
        save_path,
        chosen_idx,
        color
    ):
        """Here we extend the plot_regions_map function to plot multiple variables."""

        # find the argmax per row
        idx = np.argmax(w_adj, axis=-1)
        print(idx)
        print("len(idx)",len(idx))

        # plot the regions

        # First, I will assert that I have two columns.
        
        assert coordinates.shape[1] == 2
        coords = coordinates.copy()
        # Then, swap the columns if the first column at the moment is the longitude column.
        if np.max(coords[:, 0]) > 91:
            coords = np.flip(coords, axis=1)

        fig, ax = plt.subplots(
                1, 1, subplot_kw={"projection": ccrs.Robinson()}, layout="constrained", figsize=(32, 8)
            )
     
        ax.set_global()
        ax.coastlines()
        # Add some map features for context
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.gridlines(draw_labels=False)


        # Vectorized scatter plot with color array
        region = coords[idx == chosen_idx]
        
        corr_value = statistics[chosen_idx]

        ax.scatter(
            x=region[:, 1],
            y=region[:, 0],
            c=color,
            alpha=1.0,
            s=35,
            transform=ccrs.PlateCarree(),

        )

        # add number for each region (that are completely in one of the four quadrants)

        x, y = get_centroid(region[:, 1], region[:, 0])
        ax.text(
        x, y,
        f"{chosen_idx}\n{corr_value:.2f}",
        fontsize=14,
        ha='center',
        transform=ccrs.PlateCarree(),
        bbox=dict(facecolor='white', alpha=0.5, edgecolor='none')
        )

        fname = f"spatial_aggregation_z1_z2_{statistics_name}_region_{chosen_idx}.png"

        plt.savefig(save_path / fname, format="png")
        print("fig saved!")
        plt.close()

def plot_map(statistics,
        statistics_name,
        coordinates: np.ndarray,
        save_path,
        vmin,
        vmax):

    # 1. Define the projection (Mollweide is standard for HEALPix)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Robinson())
    coords = coordinates.copy()
    if np.max(coords[:, 0]) > 91:
        coords = np.flip(coords, axis=1)
    ax.set_global()
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=":")
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor="black")
    ax.gridlines(draw_labels=False)

    sc = ax.scatter(
        coords[:, 1],
        coords[:, 0],
        c=statistics,
        cmap='RdBu_r',
        s=25,
        alpha=1.0,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree()
    )
    plt.colorbar(sc, label=f'{statistics_name}', ax=ax, shrink=0.6, pad=0.05)
    plt.title(f"{statistics_name}")
    fname = f"{statistics_name}.png"
    plt.savefig(save_path / fname, format="png")
    print("fig saved!")
    plt.close()

def plot_perturbation_comparison(
    pred_dict,
    y,
    preds_wo_perturb,
    coordinates,
    save_path,
    num_samples=3,
    seed=42,

):
    """
    pred_dict: e.g.,
        {
            "z2=2.0": pred_all_2,
            "z2=3.0": pred_all_3,
        }

    preds_wo_perturb, y:
        prediction without perturbation, and ground truth [N, 1, 3072]
    """

    np.random.seed(seed)

    keys = list(pred_dict.keys())
    num_conditions = len(keys)
    

    N = y.shape[0]

    # ------------------------------------------------
    # sample random indices
    # ------------------------------------------------

    sample_ids = np.random.choice(N, num_samples, replace=False)

    # ------------------------------------------------
    # ensure lon/lat order consistency
    # ------------------------------------------------
    coords = coordinates.copy()
    if np.max(coords[:, 0]) > 91:
        coords = np.flip(coords, axis=1)

    # ------------------------------------------------
    # figure
    # ------------------------------------------------
    fig, axes = plt.subplots(
        num_samples,
        num_conditions + 2, # additional two cols for gt and pred wo perturbation
        figsize=(5 * (num_conditions + 2), 3.5 * num_samples),
        subplot_kw={"projection": ccrs.Robinson()},
        # constrained_layout=True
    )

    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    # ------------------------------------------------
    # loop samples
    # ------------------------------------------------
    for i, idx in enumerate(sample_ids):

        # -----------------------------
        # GT column (first)
        # -----------------------------
        gt_map = y[idx, 0, :]
        pred_wo_perturb_map = preds_wo_perturb[idx, 0, :]

        ax = axes[i, 0]

        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.gridlines(draw_labels=False)

        sc = ax.scatter(
            coords[:, 1],
            coords[:, 0],
            c=gt_map,
            cmap='RdBu_r',
            s=25,
            alpha=1.0,
            vmin=-3.5,
            vmax=3.5,
            transform=ccrs.PlateCarree()
        )

        ax.set_title(f"GT | sample {idx}")
        plt.colorbar(sc, ax=ax, shrink=0.4, pad=0.05)
        ax = axes[i, 1]

        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.LAND, edgecolor="black")
        ax.gridlines(draw_labels=False)

        sc = ax.scatter(
            coords[:, 1],
            coords[:, 0],
            c=pred_wo_perturb_map,
            cmap='RdBu_r',
            s=25,
            alpha=1.0,
            vmin=-3.5,
            vmax=3.5,
            transform=ccrs.PlateCarree()
        )

        ax.set_title(f"Pred | sample {idx}")
        # -----------------------------
        # predictions
        # -----------------------------
        for j, key in enumerate(keys):

            pred = pred_dict[key][idx, 0, :]
            ax = axes[i, j+2]

            ax.set_global()
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, edgecolor="black")
            ax.gridlines(draw_labels=False)

            sc = ax.scatter(
                coords[:, 1],
                coords[:, 0],
                c=pred,
                cmap='RdBu_r',
                s=25,
                alpha=1.0,
                vmin=-3.5,
                vmax=3.5,
                transform=ccrs.PlateCarree()
            )

            ax.set_title(f"{key} | sample {idx}")
    fname = "compare_pred_gt_sample_perturb_pz2_mu"
    plt.savefig(save_path/fname,bbox_inches="tight", pad_inches=0.05)
    print(f"fig saved to {fname} !")
def plot_scatter_map(z1,z2,index, pearson_corrs, linear_r2, save_path):
    # ============================================================
    # SCATTER PLOTS
    # ============================================================

    # Number of latent dimensions to visualize

    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(6,4))

    z1_i = z1[:, index]

    ax.scatter(
        z2,
        z1_i,
        s=5,
        alpha=0.3
    )

    # Linear fit
    coeffs = np.polyfit(z2, z1_i, deg=1)

    x_line = np.linspace(z2.min(), z2.max(), 200)
    y_line = coeffs[0] * x_line + coeffs[1]

    ax.plot(x_line, y_line)

    ax.set_title(
        f"z1[{index}] vs z2\n"
        f"Pearson={pearson_corrs[index]:.3f}, "
        f"R²={linear_r2[index]:.3f}"
    )

    ax.set_xlabel("z2")
    ax.set_ylabel(f"z1[{index}]")

    plt.tight_layout()
    fname=f"scatter_z1[{index}]_z2.png"
    plt.savefig(save_path / fname)
    print(f"{fname} saved")

def main(
    experiment_params, 
    data_params, 
    # gt_params, 
    train_params, 
    model_params, 
    optim_params, 
    plot_params, 
    savar_params,
    rollout_params,
    exp_id,
    iter_id,
):
    """
    :param hp: object containing hyperparameter values
    """

    # Control as much randomness as possible
    torch.manual_seed(experiment_params.random_seed)
    np.random.seed(experiment_params.random_seed)
    
    device = torch.device("cuda" if (torch.cuda.is_available() and experiment_params.gpu) else "cpu")

    if experiment_params.gpu and torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")

    if data_params.data_format == "hdf5":
        print("IS HDF5")
        return
    else:
        datamodule = CausalClimateDataModule(
            tau=experiment_params.tau,
            num_months_aggregated=data_params.num_months_aggregated,
            train_val_interval_length=data_params.train_val_interval_length,
            in_var_ids=data_params.in_var_ids,
            out_var_ids=data_params.out_var_ids,
            train_years=data_params.train_years,
            train_historical_years=data_params.train_historical_years,
            test_years=data_params.test_years,  # do we want to implement keeping only certain years for testing?
            val_split=1 - train_params.ratio_train,  # fraction of testing to split for valdation
            seq_to_seq=data_params.seq_to_seq,  # if true maps from T->T else from T->1
            channels_last=data_params.channels_last,  # wheather variables come last our after sequence lenght
            train_scenarios=data_params.train_scenarios,
            test_scenarios=data_params.test_scenarios,
            train_models=data_params.train_models,
            # test_models = data_params.test_models,
            batch_size=data_params.batch_size,
            eval_batch_size=data_params.eval_batch_size,
            num_workers=experiment_params.num_workers,
            pin_memory=experiment_params.pin_memory,
            load_train_into_mem=data_params.load_train_into_mem,
            load_test_into_mem=data_params.load_test_into_mem,
            verbose=experiment_params.verbose,
            seed=experiment_params.random_seed,
            seq_len=data_params.seq_len,
            data_dir=data_params.climateset_data,
            output_save_dir=data_params.data_dir,
            num_ensembles=data_params.num_ensembles,  # 1 for first ensemble, -1 for all
            lon=experiment_params.lon,
            lat=experiment_params.lon,
            num_levels=data_params.num_levels,
            global_normalization=data_params.global_normalization,
            seasonality_removal=data_params.seasonality_removal,
            reload_climate_set_data=data_params.reload_climate_set_data,
            icosahedral_coordinates_path=data_params.icosahedral_coordinates_path,
            # Below SAVAR data arguments
            time_len=savar_params.time_len,
            comp_size=savar_params.comp_size,
            noise_val=savar_params.noise_val,
            n_per_col=savar_params.n_per_col,
            difficulty=savar_params.difficulty,
            seasonality=savar_params.seasonality,
            overlap=savar_params.overlap,
            is_forced=savar_params.is_forced,
            plot_original_data=savar_params.plot_original_data,
        )
        datamodule.setup()

    d = len(data_params.in_var_ids)
    print(f"Using {d} variables")

    if model_params.instantaneous:
        print("Using instantaneous connections")
        num_input = d * (experiment_params.tau + 1) 
    else:
        num_input = d * experiment_params.tau 

    # set the model
    model = PerturbLatentTSDCD(
        num_layers=model_params.num_layers,
        num_hidden=model_params.num_hidden,
        num_input=num_input,
        num_output=2,  # This should be parameterized somewhere?
        num_layers_mixing=model_params.num_layers_mixing,
        num_hidden_mixing=model_params.num_hidden_mixing,
        position_embedding_dim=model_params.position_embedding_dim,
        transition_param_sharing=model_params.transition_param_sharing,
        position_embedding_transition=model_params.position_embedding_transition,
        coeff_kl=optim_params.coeff_kl,
        d=d,
        distr_z0="gaussian",
        distr_encoder="gaussian",
        distr_transition="gaussian",
        distr_decoder="gaussian",
        d_x=experiment_params.d_x,
        d_z=experiment_params.d_z,
        d_z_global=experiment_params.d_z_global,
        tau=experiment_params.tau,
        instantaneous=model_params.instantaneous,
        nonlinear_dynamics=model_params.nonlinear_dynamics,
        nonlinear_mixing=model_params.nonlinear_mixing,
        tied_w=model_params.tied_w,
        fixed=model_params.fixed,
        fixed_output_fraction=model_params.fixed_output_fraction,
    )
    
    # read paths 
    coordinates = np.load(data_params.icosahedral_coordinates_path)

    exp_path = Path(experiment_params.exp_path)
    if not os.path.exists(exp_path): 
        raise ValueError(f"Results path {exp_path} doesn't exist. Model should be saved in this folder")
 

    name = exp_id

    exp_path = exp_path / name
    print("exp_path experiment_params.exp_path:", exp_path)
    if not os.path.exists(exp_path): 
        raise ValueError(f"Results path {exp_path} does not exist. Are you using the same parameters?")

    # create path to exp and save hyperparameters
    save_path = exp_path / "inference"
    os.makedirs(save_path, exist_ok=True)

    # seed = 1
    save_path = save_path / f"inf_iter{iter_id}_spherical{str(optim_params.take_spherical_harmonics)}"
    os.makedirs(save_path, exist_ok=True)

    model_path = exp_path #/ "training_results"

    model_file = "model.pth"
    print(f"The model being teseted is under: {model_path / model_file}")
    state_dict_vae_final = torch.load(
        model_path / model_file, 
        map_location=device
    )
    model.load_state_dict({k.replace("module.", ""): v for k, v in state_dict_vae_final.items()})

    # Move the model to the GPU
    model = model.to(device)
    print("Where is the model?", next(model.parameters()).device)
    model.eval()
    
    # ----------- Scatter plot between pz2_mu and true global mean temperature ----------------- #
    pz2_mu, y_pred = get_all_pz2mu_and_temp(datamodule=datamodule, accelerator=accelerator, model=model, device=device)
    
    y_pred_np = y_pred.squeeze(1).detach().cpu().numpy()   # [B, d_x]
    y_pred_np_global_mean = y_pred_np.mean(-1)
    pz2_mu_np = pz2_mu.squeeze(1).squeeze(1).detach().cpu().numpy()  # [B,]


    from sklearn.feature_selection import mutual_info_regression

    x = pz2_mu_np.flatten()
    y = y_pred_np_global_mean.flatten()

    # sklearn expects shape (N, features)
    mi = mutual_info_regression(
        x.reshape(-1, 1),
        y,
        random_state=42
    )[0]

    pearson_r, _ = pearsonr(x, y)
    spearman_r, _ = spearmanr(x, y)

    plt.figure(figsize=(8, 6))

    hb = plt.hexbin(
        x,
        y,
        gridsize=60,
        cmap="viridis",
        mincnt=1
    )

    plt.colorbar(label="Counts")

    plt.xlabel("pz2_mu")
    plt.ylabel("true_global_mean")

    plt.title(
        f"Relationship between variables\n"
        f"MI = {mi:.3f} | "
        f"Pearson r = {pearson_r:.3f} | "
        f"Spearman ρ = {spearman_r:.3f}"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "mi_hexbin.png")
    
    x = pz2_mu_np.flatten()
    y = y_pred_np_global_mean.flatten()
    x =( x -np.mean(x)) / np.std(x)
    y = (y -np.mean(y)) / np.std(y)

    # sklearn expects shape (N, features)
    mi = mutual_info_regression(
        x.reshape(-1, 1),
        y,
        random_state=42
    )[0]

    pearson_r, _ = pearsonr(x, y)
    spearman_r, _ = spearmanr(x, y)

    plt.figure(figsize=(8, 6))

    hb = plt.hexbin(
        x,
        y,
        gridsize=60,
        cmap="viridis",
        mincnt=1
    )

    plt.colorbar(label="Counts")

    plt.xlabel("pz2_mu")
    plt.ylabel("true_global_mean")

    plt.title(
        f"Relationship between variables\n"
        f"MI = {mi:.3f} | "
        f"Pearson r = {pearson_r:.3f} | "
        f"Spearman ρ = {spearman_r:.3f}"
    )

    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path / "mi_hexbin_normalized.png")
    """
    # -----------Analysis 1:  perturb on pz2_mu,  check the predicted temperature maps agaisnt changing pz2_mu ------------- #
    counterfactural_z2_values = [-5.0, -2.5, 0.0, 2.5, 5.0]

    pred_dicts = {}
    z_dicts = {}

    for z2 in counterfactural_z2_values:

        pred_all, z_global_all = get_all_temp_and_z2_after_perturb(
            datamodule=datamodule,
            accelerator=accelerator,
            model=model,
            device=device,
            counterfactual_z_global_value=z2
        ) #
        # print("pred_all",pred_all.shape) # torch.Size([4608, 1, 1])
        # print("z_global_all",z_global_all.shape, z_global_all)# torch.Size([4608, 1, 1])
        # print("y_all",y_all.shape) #y_all torch.Size([4608, 1, 3072])

        pred_all, z_global_all = pred_all.detach().cpu().numpy(), z_global_all.detach().cpu().numpy()
        pred_dicts[z2] = pred_all
        z_dicts[z2] = z_global_all

    pred_all_wo_perturb, y_all= get_all_temp_and_gt(datamodule=datamodule,
        accelerator=accelerator,
        model=model,
        device=device)
    y_all, pred_all_wo_perturb = y_all.detach().cpu().numpy(), pred_all_wo_perturb.detach().cpu().numpy()

    pred_dict = {
        f"z2={v}": pred_dicts[v]
        for v in counterfactural_z2_values
    }
    
    plot_perturbation_comparison(
        pred_dict=pred_dict,
        y=y_all,
        preds_wo_perturb=pred_all_wo_perturb,
        coordinates=coordinates,
        num_samples=3,
        save_path=save_path
    )
    """

    """
    #----------- Analysis 2: Check how z1 and z2 relates, including correlation map, and bar plots ------------------ #
    
    z1, z2 = get_all_z1_and_z2(datamodule=datamodule, accelerator=accelerator, model=model, device=device)
    
    z1_np = z1.squeeze(2).detach().cpu().numpy()  # [B, T, 1, d_z], T= tau+1
    z2_np = z2.squeeze(2).squeeze(2).detach().cpu().numpy() # [B, T, 1, 1]

    B, T, d_z = z1_np.shape

    # ============================================================
    # CORRELATION ANALYSIS
    # ============================================================

    pearson_corrs = np.zeros(d_z)
    linear_slopes = np.zeros(d_z)
    linear_r2 = np.zeros(d_z)
    spearmanrs = np.zeros(d_z)

    # Put the tau dim into batch 
    z1_np_flat = z1_np.reshape(B * T, d_z)
    z2_np_flat = z2_np.reshape(B * T)

    for i in range(d_z):

        z1_i = z1_np_flat[:, i]
        # --------------------------------
        # Pearson correlation
        # --------------------------------
        pearson_corr, _ = pearsonr(z2_np_flat, z1_i)
        spearman_corr, _ = spearmanr(z2_np_flat, z1_i)

        # --------------------------------
        # Linear regression
        # z1_i = a * z2 + b
        # --------------------------------
        reg = LinearRegression()

        reg.fit(
            z2_np_flat.reshape(-1, 1),
            z1_i
        )

        slope = reg.coef_[0]

        r2 = reg.score(
            z2_np_flat.reshape(-1, 1),
            z1_i
        )

        # --------------------------------
        # Store
        # --------------------------------

        pearson_corrs[i] = pearson_corr
        linear_slopes[i] = slope
        linear_r2[i] = r2
        spearmanrs[i] = spearman_corr

    # ---------- Bar plot of the correlation and r2 between z1 and z2 -------------- #
    plot_bar_map(statistics=pearson_corrs, statistics_name="pearson_corrs_z1_z2",save_path=save_path)
    # plot_bar_map(statistics=linear_r2, statistics_name="linear_r2__z1_z2",save_path=save_path)
    plot_bar_map(statistics=spearmanrs, statistics_name="spearmanrs_z1_z2",save_path=save_path)

    adj_w_encoder = model.autoencoder.get_w_encoder().cpu().detach().numpy()[0]#(1, d_z, d_x), then take the first variable
    adj_w_decoder = model.autoencoder.get_w_decoder().cpu().detach().numpy()[0]#(1, d_x, d_z)

    # ---------- Correlation between z1 and z2, mapped to the grids #
    # plot_regions_map(w_adj=adj_w_encoder.T, coordinates=coordinates, statistics=pearson_corrs,statistics_name="pearson_corrs_encoder_z1_z2",save_path=save_path)
    plot_regions_map(w_adj=adj_w_decoder, coordinates=coordinates, statistics=pearson_corrs,statistics_name="pearson_corrs_decoder_z1_z2",save_path=save_path)
    
    # ---------- Spearman correlation coefficient between z1 and z2, mapped to the grids #
    plot_regions_map(w_adj=adj_w_decoder, coordinates=coordinates, statistics=spearmanrs,statistics_name="spearmanrs_decoder_z1_z2",save_path=save_path)
    
    # ---------- R^2 between z1 and z2, mapped to the grids #
    # plot_regions_map(w_adj=adj_w_encoder.T, coordinates=coordinates, statistics=linear_r2,statistics_name="linear_r2_encoder_perturbed",save_path=save_path)
    # plot_regions_map(w_adj=adj_w_decoder, coordinates=coordinates, statistics=linear_r2,statistics_name="linear_r2_decoder_perturbed",save_path=save_path)
    # ---------- A closer look at the correlation between z1 and z2 for a single z1 -------- #
    # chosen_idxs_positive = [7, 12, 33, 34, 36, 55, 58] # large positive corr
    # chosen_idxs_negative = [1, 16, 20, 22, 50] # large negative corr
    # # for id chosen_idxs_positive:
    # #     # plot top positive maps
    # #     plot_regions_map_cluster_k(w_adj=adj_w_encoder.T, coordinates=coordinates, statistics=pearson_corrs,statistics_name="pearson_corrs_encoder_z1_z2",save_path=save_path,chosen_idx=id, color="red")
    # #     plot_regions_map_cluster_k(w_adj=adj_w_decoder, coordinates=coordinates, statistics=pearson_corrs,statistics_name="pearson_corrs_decoder_z1_z2",save_path=save_path,chosen_idx=id, color="red")
    # # for id chosen_idxs_positive:
    # #     # plot top negative maps
    # #     plot_regions_map_cluster_k(w_adj=adj_w_encoder.T, coordinates=coordinates, statistics=pearson_corrs,statistics_name="pearson_corrs_encoder_z1_z2",save_path=save_path,chosen_idx=id, color="blue")
    # #     plot_regions_map_cluster_k(w_adj=adj_w_decoder, coordinates=coordinates, statistics=pearson_corrs,statistics_name="pearson_corrs_decoder_z1_z2",save_path=save_path,chosen_idx=id, color="blue")
    
    # for id in chosen_idxs_positive + chosen_idxs_negative:
    #     plot_scatter_map(z1=z1_np_flat, z2=z2_np_flat, index=id, save_path=save_path, pearson_corrs=pearson_corrs, linear_r2=linear_r2)
      

    """
    """

    #----------- Analysis 3: Check how pz2mu and temp relates, including correlation map (z2[:,-1] returns almost the same corr with temp as pz2_mu)------------------ #   
    # _, y_true= get_all_temp_and_gt(datamodule=datamodule,
    #     accelerator=accelerator,
    #     model=model,
    #     device=device)  
    # print("y_true",y_true.shape)
    # y_pred_np= y_true.squeeze(1).detach().cpu().numpy()
    # pz2_mu_np = y_pred_np.mean(-1)
    pz2_mu, y_pred = get_all_pz2mu_and_temp(datamodule=datamodule, accelerator=accelerator, model=model, device=device)
    
    y_pred_np = y_pred.squeeze(1).detach().cpu().numpy()   # [B, d_x]
    pz2_mu_np = pz2_mu.squeeze(1).squeeze(1).detach().cpu().numpy()  # [B,]

    B, d_x = y_pred_np.shape

    # ============================================================
    # CORRELATION ANALYSIS
    # ============================================================
    spearmanr_pz2_mu_and_pred = np.zeros(d_x)
    pearson_corrs_pz2_mu_and_pred = np.zeros(d_x)
    linear_slopes_pz2_mu_and_pred = np.zeros(d_x)
    linear_r2_pz2_mu_and_pred = np.zeros(d_x)

    for i in range(d_x):

        y_pred_i = y_pred_np[:, i]
        # --------------------------------
        # Pearson correlation
        # --------------------------------
        pearson_corr, _ = pearsonr(pz2_mu_np, y_pred_i)
        spearman_corr, _ = spearmanr(pz2_mu_np, y_pred_i)

        # --------------------------------
        # Linear regression
        # z1_i = a * z2 + b
        # --------------------------------
        reg = LinearRegression()

        reg.fit(
            pz2_mu_np.reshape(-1, 1),
            y_pred_i
        )

        slope = reg.coef_[0]

        r2 = reg.score(
            pz2_mu_np.reshape(-1, 1),
            y_pred_i
        )


        # --------------------------------
        # Store
        # --------------------------------

        pearson_corrs_pz2_mu_and_pred[ i] = pearson_corr
        linear_slopes_pz2_mu_and_pred[ i] = slope
        linear_r2_pz2_mu_and_pred[ i] = r2
        spearmanr_pz2_mu_and_pred[i] = spearman_corr

    # ---------- Correlation between pz2_mu and temperature, mapped to the grids #
    v = np.max(abs(pearson_corrs_pz2_mu_and_pred))
    plot_map(coordinates=coordinates, statistics=pearson_corrs_pz2_mu_and_pred,statistics_name="pearson_pz2mu_pred",save_path=save_path, vmin=-v, vmax=v)
    plot_map(coordinates=coordinates, statistics=spearmanr_pz2_mu_and_pred,statistics_name="spearmanr_pz2mu_pred",save_path=save_path, vmin=-v, vmax=v)
    """

if __name__ == "__main__":

    args = parse_args()
    
    cwd = Path.cwd()
    root_path = cwd.parent
    # config_path = root_path / f"configs"
    exp_id = args.exp_id
    iter_id = args.iter_id
    folder = exp_id.split("/")[0] 
    exp_id = exp_id.split("/")[-1]
    config_path = Path("/home/mila/s/shanz/scratch/results") / folder / exp_id
    json_path = config_path / args.config_path
    
    with open(json_path, "r") as f:
        params = json.load(f)
    params = update_config_withparse(params, args)

    # get user's scratch directory:
    scratch_path = os.getenv("SCRATCH")
    params["data_params"]["data_dir"] = params["data_params"]["data_dir"].replace("$SCRATCH", scratch_path)
    print ("new data path:", params["data_params"]["data_dir"])

    params["exp_params"]["exp_path"] = params["exp_params"]["exp_path"].replace("$SCRATCH", scratch_path)
    print ("new exp path:", params["exp_params"]["exp_path"])

    # get directory of project via current file (aka .../climatem/scripts/main_picabu.py)
    params["data_params"]["icosahedral_coordinates_path"] = params["data_params"]["icosahedral_coordinates_path"].replace("$CLIMATEMDIR", root_path.absolute().as_posix())
    print ("new icosahedron path:", params["data_params"]["icosahedral_coordinates_path"])
    
    # For rollout, most cases we already have the climate dataset during training
    params["data_params"]["reload_climate_set_data"] = True
    print ("new reload_climate_set_data:", params["data_params"]["reload_climate_set_data"])

    experiment_params = expParams(**params["exp_params"])
    data_params = dataParams(**params["data_params"])
    # gt_params = gtParams(**params["gt_params"])
    train_params = trainParams(**params["train_params"])
    model_params = modelParams(**params["model_params"])
    optim_params = optimParams(**params["optim_params"])
    plot_params = plotParams(**params["plot_params"])
    savar_params = savarParams(**params["savar_params"])
    rollout_params = rolloutParams(**params["rollout_params"])

    #Overwrite arguments if using savar
    if "savar" in data_params.in_var_ids:
        experiment_params.lat = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.lon = int(savar_params.comp_size * savar_params.n_per_col)
        experiment_params.d_x = int(experiment_params.lat * experiment_params.lon)
        plot_params.savar = True
    else:
        plot_params.savar = False

    main(experiment_params, data_params, train_params, model_params, optim_params, plot_params, savar_params, rollout_params, exp_id, iter_id)

