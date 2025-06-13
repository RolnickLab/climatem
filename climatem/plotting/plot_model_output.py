# Collection of plotting functions for plotting the results of experiments.

import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from climatem.model.metrics import mcc_latent


def moving_average(a: np.ndarray, n: int = 10):
    """
    Returns: the moving average of the array 'a' with a timewindow of 'n'
    """
    # from https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    if len(a) == 0:
        return 0
    else:
        if torch.is_tensor(a):
            # a = torch.stack(a, dim=0)
            ret = torch.cumsum(a, dim=0).cpu().detach().numpy()
        elif torch.is_tensor(a[0]):
            a = torch.stack(a, dim=0)
            ret = torch.cumsum(a, dim=0).cpu().detach().numpy()
        else:
            ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1 :] / n


class Plotter:
    def __init__(self):
        self.mcc = []
        self.assignments = []

    def save(self, learner):
        """
        Save all the different metrics.

        Can then reload them to plot them.
        """

        if learner.latent:
            # save matrix W of the decoder and encoder
            print("Saving the decoder, encoder and graphs.")
            w_decoder = learner.model.autoencoder.get_w_decoder().cpu().detach().numpy()
            np.save(learner.plots_path / "w_decoder.npy", w_decoder)
            w_encoder = learner.model.autoencoder.get_w_encoder().cpu().detach().numpy()
            np.save(learner.plots_path / "w_encoder.npy", w_encoder)

            # save the graphs G
            adj = learner.model.get_adj().cpu().detach().numpy()
            np.save(learner.plots_path / "graphs.npy", adj)

    def load(self, exp_path, data_loader):
        # load matrix W of the decoder and encoder
        self.w = np.load(exp_path / "w_decoder.npy")
        self.w_encoder = np.load(exp_path / "w_encoder.npy")

        # load adj_tt and adj_w_tt, adjacencies through time
        self.adj_tt = np.load(exp_path / "adj_tt")
        self.adj_w_tt = np.load(exp_path / "adj_w_tt")

        # load log-variance of encoder and decoder
        self.logvar_encoder_tt = np.load(exp_path / "logvar_encoder_tt")
        self.logvar_decoder_tt = np.load(exp_path / "logvar_decoder_tt")
        self.logvar_transition_tt = np.load(exp_path / "logvar_transition_tt")

        # load losses and penalties
        self.penalties = {}
        penalties = [
            {"name": "sparsity", "data": "train_sparsity_reg"},
            {"name": "tr ortho", "data": "train_ortho_cons"},
            {"name": "mu ortho", "data": "mu_ortho"},
        ]
        for p in penalties:
            self.penalties[p["data"]] = np.load(exp_path / p["name"])

        losses = [
            {"name": "tr ELBO", "data": "train_loss"},
            {"name": "Recons", "data": "train_recons"},
            {"name": "KL", "data": "train_kl"},
            {"name": "val ELBO", "data": "valid_loss"},
        ]
        for loss in losses:
            self.losses[loss["data"]] = np.load(exp_path / loss["name"])

        # load GT W and graph
        self.gt_w = data_loader.gt_w
        self.gt_graph = data_loader.gt_dag

    def plot(self, learner, save=False):
        """
        Main plotting function.

        Plot the learning curves and if the ground-truth is known the adjacency and adjacency through time.
        """

        # NOTE:(seb) I am going to save the coordinates here, but this should be moved.
        np.save(learner.plots_path / "coordinates.npy", learner.coordinates)

        if save:
            self.save(learner)

        # plot learning curves
        if learner.latent:

            # NOTE:(seb) adding here capacity to plot the new sparsity constraint!
            losses = [
                {"name": "sparsity", "data": learner.train_sparsity_reg_list, "s": "-"},
                {"name": "tr ortho", "data": learner.train_ortho_cons_list, "s": ":"},
                {"name": "mu ortho", "data": learner.mu_ortho_list, "s": ":"},
                # {"name": "tr sparsity", "data": learner.train_sparsity_cons_list, "s": ":"},
                # {"name": "mu sparsity", "data": learner.mu_sparsity_list, "s": ":"},
                # {"name": "gamma ortho", "data": learner.gamma_ortho_list, "s": ":"},
                # {"name": "gamma sparsity", "data": learner.gamma_sparsity_list, "s": ":"},
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
        else:
            self.plot_learning_curves(
                train_loss=learner.train_loss_list,
                valid_loss=learner.valid_loss_list,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
                path=learner.plots_path,
            )

        # plot the adjacency matrix (learned vs ground-truth)
        adj = learner.model.get_adj().cpu().detach().numpy()
        if not learner.no_gt:
            if learner.latent:
                # for latent models, find the right permutation of the latent
                adj_w = learner.model.autoencoder.get_w_decoder().cpu().detach().numpy()
                adj_w2 = learner.model.autoencoder.get_w_encoder().cpu().detach().numpy()
                # variables using MCC
                if learner.debug_gt_z:
                    gt_dag = learner.gt_dag
                    gt_w = learner.gt_w
                    self.mcc.append(1.0)
                    self.assignments.append(np.arange(learner.gt_dag.shape[1]))
                else:
                    score, cc_program_perm, assignments, z, z_hat, x = mcc_latent(learner.model, learner.data)
                    permutation = np.zeros((learner.gt_dag.shape[1], learner.gt_dag.shape[1]))
                    permutation[np.arange(learner.gt_dag.shape[1]), assignments[1]] = 1
                    self.mcc.append(score.item())
                    self.assignments.append(assignments[1])

                    gt_dag = permutation.T @ learner.gt_dag @ permutation
                    gt_w = learner.gt_w
                    # TODO: put back
                    # adj_w = adj_w[:, :, assignments[1]]
                    # adj_w2 = adj_w2[:, assignments[1], :]
                    adj_w2 = np.swapaxes(adj_w2, 1, 2)
                self.save_mcc_and_assignement(learner.plots_path)

                # draw learned mixing fct vs GT
                if learner.model_params.nonlinear_mixing:
                    self.plot_learned_mixing(z, z_hat, adj_w, gt_w, x, learner.plots_path)

            else:
                gt_dag = learner.gt_dag

            self.plot_adjacency_through_time(
                learner.adj_tt, gt_dag, learner.iteration, learner.plots_path, "transition"
            )
        else:
            gt_dag = None
            gt_w = None

            # for latent models, find the right permutation of the latent
            adj_w = learner.model.autoencoder.get_w_decoder().cpu().detach().numpy()
            adj_w2 = learner.model.autoencoder.get_w_encoder().cpu().detach().numpy()

        # this is where this was before, but I have now added the argument names for myself
        if learner.plot_params.savar:
            self.plot_adjacency_matrix(
                mat1=adj,
                # Below savar dag
                mat2=learner.datamodule.savar_gt_adj,
                path=learner.plots_path,
                name_suffix="transition",
                no_gt=False,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
            )
        else:
            self.plot_adjacency_matrix(
                mat1=adj,
                mat2=gt_dag,
                path=learner.plots_path,
                name_suffix="transition",
                no_gt=learner.no_gt,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
            )

        # plot the weights W for latent models (between the latent Z and the X)
        # hoping that these don't fail due to defaults
        if learner.latent:
            # plot the decoder matrix W
            self.plot_adjacency_matrix_w(adj_w, gt_w, learner.plots_path, "w", learner.no_gt)
            # plot the encoder matrix W_2
            # gt_w2 = np.swapaxes(gt_w, 1, 2)
            gt_w2 = gt_w
            self.plot_adjacency_matrix_w(adj_w2, gt_w2, learner.plots_path, "encoder_w", learner.no_gt)
            if not learner.no_gt:
                self.plot_adjacency_through_time_w(
                    learner.adj_w_tt, learner.gt_w, learner.iteration, learner.plots_path, "w"
                )
            elif learner.plot_params.savar:
                self.plot_savar_feature_maps(
                    learner,
                    adj_w,
                    coordinates=learner.coordinates,
                    iteration=learner.iteration,
                    plot_through_time=learner.plot_params.plot_through_time,
                    path=learner.plots_path,
                )
            else:
                self.plot_regions_map(
                    adj_w,
                    learner.coordinates,
                    learner.iteration,
                    learner.plot_params.plot_through_time,
                    path=learner.plots_path,
                    idx_region=None,
                    annotate=True,
                    one_plot=True,
                )

                self.plot_regions_map(
                    adj_w,
                    learner.coordinates,
                    learner.iteration,
                    learner.plot_params.plot_through_time,
                    path=learner.plots_path,
                    idx_region=None,
                    annotate=True,
                )

    def plot_sparsity(self, learner, input_var_shapes=None, input_var_offsets=None, save=False):
        """
        Main plotting function.

        Plot the learning curves and if the ground-truth is known the adjacency and adjacency through time.
        """

        np.save(learner.plots_path / "coordinates.npy", learner.coordinates)

        if save:
            self.save(learner)

        # plot learning curves
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
            # NOTE:(seb) adding here capacity to plot the new sparsity constraint!
            losses = [  # {"name": "sparsity", "data": learner.train_sparsity_reg_list, "s": "-"},
                {"name": "tr ortho", "data": learner.train_ortho_cons_list, "s": ":"},
                {"name": "mu ortho", "data": learner.mu_ortho_list, "s": ":"},
                {"name": "tr sparsity", "data": learner.train_sparsity_cons_list, "s": ":"},
                {"name": "tr var adj", "data": learner.train_transition_var_list, "s": ":"},
                {"name": "mu sparsity", "data": learner.mu_sparsity_list, "s": ":"},
                # {"name": "gamma ortho", "data": learner.gamma_ortho_list, "s": ":"},
                # {"name": "gamma sparsity", "data": learner.gamma_sparsity_list, "s": ":"},
            ]
            # {"name": "tr acyclic", "data": learner.train_acyclic_cons_list, "s": "-"},
            # {"name": "tr connect", "data": learner.train_connect_reg_list, "s": "-"},
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
        else:
            self.plot_learning_curves(
                train_loss=learner.train_loss_list,
                valid_loss=learner.valid_loss_list,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
                path=learner.plots_path,
            )

        # TODO: plot the prediction vs gt
        # plot_compare_prediction(x, x_hat)

        # plot the adjacency matrix (learned vs ground-truth)
        # Here if SAVAR, learner should have GT and gt_dag should be the SAVAR GT
        adj = learner.model.get_adj().cpu().detach().numpy()
        if not learner.no_gt:
            if learner.latent:
                # for latent models, find the right permutation of the latent
                adj_w = learner.model.autoencoder.get_w_decoder().cpu().detach().numpy()
                adj_w2 = learner.model.autoencoder.get_w_encoder().cpu().detach().numpy()
                # variables using MCC
                if learner.debug_gt_z:
                    gt_dag = learner.gt_dag
                    gt_w = learner.gt_w
                    self.mcc.append(1.0)
                    self.assignments.append(np.arange(learner.gt_dag.shape[1]))
                else:
                    score, cc_program_perm, assignments, z, z_hat, x = mcc_latent(learner.model, learner.data)
                    permutation = np.zeros((learner.gt_dag.shape[1], learner.gt_dag.shape[1]))
                    permutation[np.arange(learner.gt_dag.shape[1]), assignments[1]] = 1
                    self.mcc.append(score.item())
                    self.assignments.append(assignments[1])

                    gt_dag = permutation.T @ learner.gt_dag @ permutation
                    gt_w = learner.gt_w
                    # TODO: put back
                    # adj_w = adj_w[:, :, assignments[1]]
                    # adj_w2 = adj_w2[:, assignments[1], :]
                    adj_w2 = np.swapaxes(adj_w2, 1, 2)
                self.save_mcc_and_assignement(learner.plots_path)

                # draw learned mixing fct vs GT
                if learner.model_params.nonlinear_mixing:
                    self.plot_learned_mixing(z, z_hat, adj_w, gt_w, x, learner.plots_path)

            else:
                gt_dag = learner.gt_dag

            self.plot_adjacency_through_time(
                learner.adj_tt, gt_dag, learner.iteration, learner.plots_path, "transition"
            )
        else:
            gt_dag = None
            gt_w = None

            # for latent models, find the right permutation of the latent
            adj_w = learner.model.autoencoder.get_w_decoder().cpu().detach().numpy()
            adj_w2 = learner.model.autoencoder.get_w_encoder().cpu().detach().numpy()

        # this is where this was before, but I have now added the argument names for myself
        if learner.plot_params.savar:
            self.plot_adjacency_matrix(
                mat1=adj,
                # Below savar dag
                mat2=learner.datamodule.savar_gt_adj,
                path=learner.plots_path,
                name_suffix="transition",
                no_gt=False,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
            )
        else:
            self.plot_adjacency_matrix(
                mat1=adj,
                mat2=gt_dag,
                path=learner.plots_path,
                name_suffix="transition",
                no_gt=learner.no_gt,
                iteration=learner.iteration,
                plot_through_time=learner.plot_params.plot_through_time,
            )

        # plot the weights W for latent models (between the latent Z and the X)
        # hoping that these don't fail due to defaults
        if learner.latent:
            # plot the decoder matrix W
            self.plot_adjacency_matrix_w(adj_w, gt_w, learner.plots_path, "w", learner.no_gt)
            # plot the encoder matrix W_2
            gt_w2 = gt_w
            self.plot_adjacency_matrix_w(adj_w2, gt_w2, learner.plots_path, "encoder_w", learner.no_gt)
            if not learner.no_gt:
                self.plot_adjacency_through_time_w(
                    learner.adj_w_tt, learner.gt_w, learner.iteration, learner.plots_path, "w"
                )
            elif learner.plot_params.savar:
                self.plot_savar_feature_maps(
                    learner,
                    adj_w,
                    coordinates=learner.coordinates,
                    iteration=learner.iteration,
                    plot_through_time=learner.plot_params.plot_through_time,
                    path=learner.plots_path,
                )
            elif learner.plot_params.chirps:
                self.plot_regions_map_by_var(
                    adj_w,
                    learner.coordinates,
                    input_var_shapes,
                    input_var_offsets,
                    learner.iteration,
                    learner.plot_params.plot_through_time,
                    path=learner.plots_path,
                    annotate=True,
                    one_plot=True,
                )
            else:
                self.plot_regions_map(
                    adj_w,
                    learner.coordinates,
                    learner.iteration,
                    learner.plot_params.plot_through_time,
                    path=learner.plots_path,
                    idx_region=None,
                    annotate=True,
                )

    def plot_learned_mixing(self, z, z_hat, w, gt_w, x, path):
        n_first = 5

        for i in range(n_first):
            # plot z_hat vs x
            # find parent of x_i
            j = np.argmax(w[0, i])
            fig = plt.figure()
            fig.suptitle("Mixing Learned vs Ground-truth")
            axes = fig.subplots(nrows=1, ncols=2)

            axes[0].scatter(z_hat[:, j], x[:, 0, i], s=2)
            axes[0].set_title(f"Learned mixing. j={j}, val={w[0, i, j]:.2f}")

            # plot z vs x
            j = np.argmax(gt_w[0, i])
            axes[1].scatter(z[:, j], x[:, 0, i], s=2)
            axes[1].set_title(f"GT mixing. j={j}, val={gt_w[0, i, j]:.2f}")

            plt.savefig(path / f"learned_mixing_x{i}.png")

    def plot_compare_prediction(self, x, x_past, x_hat, coordinates: np.ndarray, path):
        """Plot the predicted x_hat compared to the ground-truth x using Cartopy."""
        fig, axs = plt.subplots(3, 1, subplot_kw={"projection": ccrs.Robinson()}, figsize=(12, 12))

        titles = ["Previous GT", "Ground-truth", "Prediction"]
        data = [x_past, x, x_hat]

        lon = coordinates[:, 0]
        lat = coordinates[:, 1]
        X, Y = np.meshgrid(np.unique(lon), np.unique(lat))

        for ax, title, z in zip(axs, titles, data):
            ax.set_global()
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.LAND, edgecolor="black")
            ax.gridlines(draw_labels=False)

            Z = z.reshape(Y.shape)
            pcm = ax.pcolormesh(X, Y, Z, cmap="RdBu_r", vmin=-3.5, vmax=3.5, transform=ccrs.PlateCarree())
            ax.set_title(title)

        fig.colorbar(pcm, ax=axs, orientation="vertical", shrink=0.7, label="Normalized value")
        plt.suptitle("Ground-truth vs prediction", fontsize=16)
        plt.savefig(path / "prediction.png", format="png")
        plt.close()

    def plot_compare_predictions_regular_grid(
        self,
        x_past: np.ndarray,
        y_true: np.ndarray,
        y_recons: np.ndarray,
        y_hat: np.ndarray,
        sample: int,
        coordinates: np.ndarray,
        path,
        iteration: int,
        valid: str = False,
        plot_through_time: bool = True,
    ):
        """Plot a prediction from the method, the last time step and the ground-truth on a regular grid."""

        if y_true.shape[1] > 1:
            fig, axs = plt.subplots(
                y_true.shape[1],
                4,
                subplot_kw={"projection": ccrs.PlateCarree()},
                layout="constrained",
                figsize=(32, 16),
            )
        else:
            fig, axs = plt.subplots(
                1, 4, subplot_kw={"projection": ccrs.PlateCarree()}, layout="constrained", figsize=(32, 8)
            )
            axs = [axs]

        lon = coordinates[:, 0]
        lat = coordinates[:, 1]
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        for j, ax_row in enumerate(axs):
            for i, ax in enumerate(ax_row):
                ax.set_global()
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, linestyle=":")
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.LAND, edgecolor="black")
                ax.gridlines(draw_labels=False)

                if i == 0:
                    s = ax.pcolormesh(
                        lon_grid,
                        lat_grid,
                        x_past[sample, j, :].reshape(lat.size, lon.size),
                        alpha=1,
                        vmin=-3.5,
                        vmax=3.5,
                        cmap="RdBu_r",
                        transform=ccrs.PlateCarree(),
                    )
                    ax.set_title("Ground-truth t-1")
                elif i == 1:
                    s = ax.pcolormesh(
                        lon_grid,
                        lat_grid,
                        y_true[sample, j, :].reshape(lat.size, lon.size),
                        alpha=1,
                        vmin=-3.5,
                        vmax=3.5,
                        cmap="RdBu_r",
                        transform=ccrs.PlateCarree(),
                    )
                    ax.set_title("Ground truth")
                elif i == 2:
                    s = ax.pcolormesh(
                        lon_grid,
                        lat_grid,
                        y_recons[sample, j, :].reshape(lat.size, lon.size),
                        alpha=1,
                        vmin=-3.5,
                        vmax=3.5,
                        cmap="RdBu_r",
                        transform=ccrs.PlateCarree(),
                    )
                    ax.set_title("Reconstruction")
                elif i == 3:
                    s = ax.pcolormesh(
                        lon_grid,
                        lat_grid,
                        y_hat[sample, j, :].reshape(lat.size, lon.size),
                        alpha=1,
                        vmin=-3.5,
                        vmax=3.5,
                        cmap="RdBu_r",
                        transform=ccrs.PlateCarree(),
                    )
                    ax.set_title("Prediction")

            if j == 0:
                fig.colorbar(s, ax=ax_row[3], label="Normalised skin temperature", orientation="vertical", shrink=1.0)
            elif j == 1:
                fig.colorbar(s, ax=ax_row[3], label="Normalised 2m temperature", orientation="vertical", shrink=1.0)
            elif j == 2:
                fig.colorbar(s, ax=ax_row[3], label="Normalised slp", orientation="vertical", shrink=1.0)
            elif j == 3:
                fig.colorbar(s, ax=ax_row[3], label="Normalised precipitation", orientation="vertical", shrink=1.0)
            elif j == 4:
                fig.colorbar(s, ax=ax_row[3], label="Normalised u-wind", orientation="vertical", shrink=1.0)
            elif j == 5:
                fig.colorbar(s, ax=ax_row[3], label="Normalised v-wind strat", orientation="vertical", shrink=1.0)

        if not valid:
            if plot_through_time:
                fname = f"compare_predictions_{iteration}_sample_{sample}_train.png"
            else:
                fname = "compare_predictions_train.png"
        else:
            if plot_through_time:
                fname = f"compare_predictions_{iteration}_sample_{sample}_valid.png"
            else:
                fname = "compare_predictions_valid.png"

        plt.suptitle("Ground truth last timestep, Ground truth, Reconstruction, and Prediction", fontsize=24)
        plt.savefig(path / fname, format="png")
        plt.close()

    # This plot should allow the plotting of multiple variables now.
    def plot_compare_predictions_icosahedral(
        self,
        x_past: np.ndarray,
        y_true: np.ndarray,
        y_recons: np.ndarray,
        y_hat: np.ndarray,
        sample: int,
        coordinates: np.ndarray,
        path,
        iteration: int,
        valid: str = False,
        plot_through_time: bool = True,
    ):
        """Plot a prediction from the method, the last time step and the ground-truth."""

        if y_true.shape[1] > 1:
            fig, axs = plt.subplots(
                y_true.shape[1], 4, subplot_kw={"projection": ccrs.Robinson()}, layout="constrained", figsize=(32, 16)
            )
        else:
            fig, axs = plt.subplots(
                1, 4, subplot_kw={"projection": ccrs.Robinson()}, layout="constrained", figsize=(32, 8)
            )
            axs = [axs]

        for j, ax_row in enumerate(axs):

            for i, ax in enumerate(ax_row):

                ax.set_global()
                ax.coastlines()
                # Add some map features for context
                ax.add_feature(cfeature.BORDERS, linestyle=":")
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.LAND, edgecolor="black")
                ax.gridlines(draw_labels=False)

                # Unpack coordinates for vectorized scatter plot
                # something like lonlat_vertex_mapping.txt
                lon = coordinates[:, 0]
                lat = coordinates[:, 1]

                # Vectorized scatter plot with color array
                if i == 0:
                    # print('x_past shape:', x_past.shape)
                    s = ax.scatter(
                        x=lon,
                        y=lat,
                        c=x_past[sample, j, :],
                        alpha=1,
                        s=30,
                        vmin=-3.5,
                        vmax=3.5,
                        cmap="RdBu_r",
                        transform=ccrs.PlateCarree(),
                    )
                    ax.set_title("Ground-truth t-1")
                elif i == 1:
                    # print('y shape:', y_true.shape)
                    s = ax.scatter(
                        x=lon,
                        y=lat,
                        c=y_true[sample, j, :],
                        alpha=1,
                        s=30,
                        vmin=-3.5,
                        vmax=3.5,
                        cmap="RdBu_r",
                        transform=ccrs.PlateCarree(),
                    )
                    ax.set_title("Ground truth")
                elif i == 2:
                    # print('y_hat shape:', y_hat.shape)
                    s = ax.scatter(
                        x=lon,
                        y=lat,
                        c=y_recons[sample, j, :],
                        alpha=1,
                        s=30,
                        vmin=-3.5,
                        vmax=3.5,
                        cmap="RdBu_r",
                        transform=ccrs.PlateCarree(),
                    )
                    ax.set_title("Reconstruction")
                elif i == 3:
                    # print('y_recons shape:', y_recons.shape)
                    s = ax.scatter(
                        x=lon,
                        y=lat,
                        c=y_hat[sample, j, :],
                        alpha=1,
                        s=30,
                        vmin=-3.5,
                        vmax=3.5,
                        cmap="RdBu_r",
                        transform=ccrs.PlateCarree(),
                    )
                    ax.set_title("Prediction")

            # add one colorbar for all subplots
            # fig.colorbar(s, ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)

            if j == 0:
                fig.colorbar(
                    s, ax=ax_row[3], label="Normalised skin temperature", orientation="vertical", shrink=1.0
                )  # adjust shrink
            elif j == 1:
                fig.colorbar(s, ax=ax_row[3], label="Normalised 2m temperature", orientation="vertical", shrink=1.0)
            elif j == 2:
                fig.colorbar(s, ax=ax_row[3], label="Normalised slp", orientation="vertical", shrink=1.0)
            elif j == 3:
                fig.colorbar(s, ax=ax_row[3], label="Normalised precipitation", orientation="vertical", shrink=1.0)
            elif j == 4:
                fig.colorbar(s, ax=ax_row[3], label="Normalised u-wind", orientation="vertical", shrink=1.0)
            elif j == 5:
                fig.colorbar(s, ax=ax_row[3], label="Normalised v-wind strat", orientation="vertical", shrink=1.0)

        if not valid:
            if plot_through_time:
                fname = f"compare_predictions_{iteration}_sample_{sample}_train.png"
            else:
                fname = "compare_predictions_train.png"
        else:
            if plot_through_time:
                fname = f"compare_predictions_{iteration}_sample_{sample}_valid.png"
            else:
                fname = "compare_predictions_valid.png"

        plt.suptitle("Ground truth last timestep, Ground truth, Reconstruction, and Prediction", fontsize=24)
        # plt.legend()
        plt.savefig(path / fname, format="png")
        plt.close()

    def plot_compare_regions():
        pass

    # need to fix this plot so it works well for multiple variables

    # NOTE:(seb) trying to extend the plot_regions_map function to plot multiple variables
    def plot_regions_map(
        self,
        w_adj,
        coordinates: np.ndarray,
        iteration: int,
        plot_through_time: bool,
        path,
        idx_region: int = None,
        annotate: bool = False,
        one_plot: bool = False,
    ):
        """Here we extend the plot_regions_map function to plot multiple variables."""

        # find the argmax per row
        idx = np.argmax(w_adj, axis=2)

        # here we want the number of latents PER variable

        d_z = w_adj.shape[2]

        # plot the regions

        colors = plt.cm.rainbow(np.linspace(0, 1, d_z))

        assert coordinates.shape[1] == 2

        # Ensure coordinates are (longitude, latitude)
        if np.max(coordinates[:, 0]) < 91:
            coordinates = coordinates  # already (lon, lat)
        elif np.max(coordinates[:, 0]) > 91:
            coordinates = np.flip(coordinates, axis=1)
        else:
            coordinates = coordinates[:, [1, 0]]  # swap (lat, lon) to (lon, lat)

        if w_adj.shape[0] > 1:
            fig, axs = plt.subplots(
                1, w_adj.shape[0], subplot_kw={"projection": ccrs.Robinson()}, layout="constrained", figsize=(32, 8)
            )
        else:
            fig, axs = plt.subplots(
                1, w_adj.shape[0], subplot_kw={"projection": ccrs.Robinson()}, layout="constrained", figsize=(32, 8)
            )
            axs = [axs]

        for i, ax in enumerate(axs):
            ax.set_global()
            ax.coastlines()
            # Add some map features for context
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.LAND, edgecolor="black")
            ax.gridlines(draw_labels=False)
            alpha = 1.0

            for k, color in zip(range(d_z), colors):
                region = coordinates[idx[i] == k]

                c = np.repeat(np.array([color]), region.shape[0], axis=0)

                if annotate:
                    if np.max(coordinates[:, 0]) > 91:
                        x, y = self.get_centroid(region[:, 1], region[:, 0])
                        ax.scatter(x=region[:, 1], y=region[:, 0], c=c, alpha=alpha, s=20, transform=ccrs.PlateCarree())
                    else:
                        x, y = self.get_centroid(region[:, 0], region[:, 1])
                        ax.scatter(x=region[:, 0], y=region[:, 1], c=c, alpha=alpha, s=20, transform=ccrs.PlateCarree())
                    ax.text(x, y, str(k), transform=ccrs.PlateCarree())

        if idx_region is not None:
            fname = f"spatial_aggregation{idx_region}.png"
        elif plot_through_time:
            fname = f"spatial_aggregation_{iteration}.png"
        elif one_plot:
            fname = "spatial_aggregation_all_clusters.png"
        else:
            fname = "spatial_aggregation.png"

        plt.savefig(path / fname, format="png")
        plt.close()

    # TO REWRITE PROPERLY AND PROPAGATE
    def plot_savar_feature_maps(
        self,
        learner,
        w_adj,
        coordinates: np.ndarray,
        iteration: int,
        plot_through_time: bool,
        path,
    ):

        grid_shape = (learner.lat, learner.lon)

        w_adj = w_adj[0]  # Now w_adj_mean should be (lat*lon, num_latents)
        d_z = w_adj.shape[1]

        # w_adj_mean = self.permute_latents(w_adj_mean, grid_shape)
        # Create a combined plot showing all features
        combined_map_n_rows, combined_map_n_columns = int(np.sqrt(d_z + 1)) + 1, int(np.sqrt(d_z + 1)) + 1
        fig, axs = plt.subplots(
            nrows=combined_map_n_rows,
            ncols=combined_map_n_columns,
            figsize=(combined_map_n_columns * 3, combined_map_n_rows * 3),
        )

        ax = axs.flat[0]
        im = ax.imshow(
            learner.datamodule.savar_gt_noise + learner.datamodule.savar_gt_modes, cmap="viridis"
        )  # , vmin=vmin, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.set_title("Ground-Truth", fontsize="large")
        ax.tick_params(axis="both", labelsize="large")

        for i in range(d_z):
            ax = axs.flat[i + 1]
            feature_data = w_adj[:, i]
            data = feature_data.reshape(grid_shape)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            im = ax.imshow(data, cmap="viridis")  # , vmin=vmin, vmax=vmax)
            # cbar = plt.colorbar(im, cax=cax)
            plt.colorbar(im, cax=cax)
            ax.set_title(f"Feature {i}", fontsize="large")
            ax.tick_params(axis="both", labelsize="large")

        for ax in axs.flat[d_z + 1 :]:
            fig.delaxes(ax)

        fig.tight_layout()

        if plot_through_time:
            fname = f"spatial_aggregation_{iteration}.png"
        else:
            fname = "spatial_aggregation.png"

        plt.savefig(path / fname)
        plt.close()

    def get_centroid(self, xs, ys):
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
            # print(final_x, final_y)
            return final_x, final_y
        else:
            return 0.0, 0.0

    def plot_learning_curves(
        self,
        train_loss: list,
        train_recons: list = None,
        train_kl: list = None,
        valid_loss: list = None,
        valid_recons: list = None,
        valid_kl: list = None,
        best_metrics: dict = None,
        iteration: int = 0,
        plot_through_time: bool = False,
        path="",
    ):
        """Plot the training and validation loss through time
        Args:
          train_loss: training loss
          train_recons: for latent models, the reconstruction part of the loss
          train_kl: for latent models, the Kullback-Leibler part of the loss
          valid_loss: validation loss (on held-out dataset)
          valid_recons: see train_recons
          valid_kl: see train_kl
          iteration: number of iterations
          plot_through_time: if False, overwrite the plot
          path: path where to save the plot
        """
        # remove first steps to avoid really high values
        start = 1
        t_loss = moving_average(train_loss[start:])
        v_loss = moving_average(valid_loss[start:])

        if train_recons is not None:
            t_recons = moving_average(train_recons[start:])
            t_kl = moving_average(train_kl[start:])
            # v_recons = moving_average(valid_recons[10:])
            # v_kl = moving_average(valid_kl[10:])

        plt.plot(v_loss, label="valid loss", color="green")
        if train_recons is not None:
            plt.plot(t_recons, label="tr recons", color="blue")
            plt.axhline(y=best_metrics["recons"], color="blue", linestyle="dotted")
            plt.plot(t_kl, label="tr kl", color="red")
            plt.axhline(y=best_metrics["kl"], color="red", linestyle="dotted")
            plt.plot(t_loss, label="tr loss", color="purple")
            plt.axhline(y=best_metrics["elbo"], color="purple", linestyle="dotted")
        else:
            plt.plot(t_loss, label="tr ELBO")

        if plot_through_time:
            fname = f"loss_{iteration}.png"
        else:
            fname = "loss.png"

        plt.title("Learning curves")
        plt.legend()
        plt.savefig(path / fname, format="png")

    def plot_learning_curves2(
        self,
        losses: list,
        iteration: int = 0,
        plot_through_time: bool = False,
        path="",
        fname="loss_detailed",
        yaxis_log: bool = False,
    ):
        """
        Plot all list present in 'losses'.

        Args:
            losses: contains losses, their name, and style
            iteration: number of iterations
            plot_through_time: if False, overwrite the plot
            path: path where to save the plot
        """
        ax = plt.gca()
        # if fname != "losses":
        if yaxis_log:
            ax.set_yscale("log")

        # compute moving_averages and
        # remove first steps to avoid really high values
        for loss in losses:
            smoothed_loss = moving_average(loss["data"][1:])
            plt.plot(smoothed_loss, label=loss["name"], linestyle=loss["s"])

        if plot_through_time:
            fname = f"{fname}_{iteration}.png"
        else:
            fname = f"{fname}.png"

        plt.title("Learning curves")
        plt.legend()
        plt.savefig(path / fname, format="png")
        plt.close()

    def save_coordinates_and_adjacency_matrices(self, learner):
        """
        Save the coordinates and adjacency matrices, at every plotting iteration.

        Args:
            coordinates: coordinates of the grid
            adj_encoder_w: adjacency matrix between X and Z
            adj_w: adjacency matrix between Z and X
        """
        # if the coordinates file does not exist, save it
        if not os.path.exists(learner.plots_path / "coordinates.npy"):
            np.save(learner.plots_path / "coordinates.npy", learner.coordinates)

        adj_w = learner.model.autoencoder.get_w_decoder().cpu().detach().numpy()
        adj_encoder_w = learner.model.autoencoder.get_w_encoder().cpu().detach().numpy()
        np.save(learner.plots_path / f"adj_encoder_w_{learner.iteration}.npy", adj_encoder_w)
        np.save(learner.plots_path / f"adj_w_{learner.iteration}.npy", adj_w)

    # SH: edited to do plot through time, without changing the function name
    # simply follow the lead of the function above, and try to plot through time.
    def plot_adjacency_matrix(
        self,
        mat1: np.ndarray,
        mat2: np.ndarray,
        path,
        name_suffix: str,
        no_gt: bool = False,
        iteration: int = 0,
        plot_through_time: bool = True,
    ):
        """Plot the adjacency matrices learned and compare it to the ground truth,
        the first dimension of the matrix should be the time (tau)
        Args:
          mat1: learned adjacency matrices
          mat2: ground-truth adjacency matrices
          path: path where to save the plot
          name_suffix: suffix for the name of the plot
          no_gt: if True, does not use the ground-truth graph
        """
        tau = mat1.shape[0]

        subfig_names = [
            f"Learned, latent dimensions = {mat1.shape[1], mat1.shape[2]}",
            "Ground Truth",
            "Difference: Learned - GT",
        ]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle("Adjacency matrices: learned vs ground-truth")

        if no_gt:
            nrows = 1
        else:
            nrows = 3

        if tau == 1:
            axes = fig.subplots(nrows=nrows, ncols=1)
            for row in range(nrows):
                if no_gt:
                    ax = axes
                else:
                    ax = axes[row]
                # axes.set_title(f"t - {i+1}")
                if row == 0:
                    sns.heatmap(
                        mat1[0], ax=ax, cbar=False, vmin=-1, vmax=1, cmap="Blues", xticklabels=False, yticklabels=False
                    )
                elif row == 1:
                    sns.heatmap(
                        mat2[0], ax=ax, cbar=False, vmin=-1, vmax=1, cmap="Blues", xticklabels=False, yticklabels=False
                    )
                elif row == 2:
                    sns.heatmap(
                        mat1[0] - mat2[0],
                        ax=ax,
                        cbar=False,
                        vmin=-1,
                        vmax=1,
                        cmap="Blues",
                        xticklabels=False,
                        yticklabels=False,
                    )

        else:
            subfigs = fig.subfigures(nrows=nrows, ncols=1)
            for row in range(nrows):
                if nrows == 1:
                    subfig = subfigs
                else:
                    subfig = subfigs[row]
                subfig.suptitle(f"{subfig_names[row]}")

                axes = subfig.subplots(nrows=1, ncols=tau)
                for i in range(tau):
                    axes[i].set_title(f"t - {i+1}")
                    if row == 0:
                        sns.heatmap(
                            mat1[tau - i - 1],
                            ax=axes[i],
                            cbar=False,
                            vmin=-1,
                            vmax=1,
                            cmap="Blues",
                            xticklabels=False,
                            yticklabels=False,
                        )
                        # add a horizontal line every 50 columns
                        for j in range(0, mat1.shape[1], 50):
                            axes[i].axhline(y=j, color="black", linewidth=0.4)
                        # add a vertical line every 50 columns
                        for j in range(0, mat1.shape[1], 50):
                            axes[i].axvline(x=j, color="black", linewidth=0.4)

                    elif row == 1:
                        sns.heatmap(
                            mat2[tau - i - 1],
                            ax=axes[i],
                            cbar=False,
                            vmin=-1,
                            vmax=1,
                            cmap="Blues",
                            xticklabels=False,
                            yticklabels=False,
                        )
                    elif row == 2:
                        sns.heatmap(
                            mat1[tau - i - 1] - mat2[tau - i - 1],
                            ax=axes[i],
                            cbar=False,
                            vmin=-1,
                            vmax=1,
                            cmap="Blues",
                            xticklabels=False,
                            yticklabels=False,
                        )

        # new
        if plot_through_time:
            fname = f"adjacency_{name_suffix}_{iteration}.png"
        else:
            fname = f"adjacency_{name_suffix}.png"

        # updated this from before - see jb_causal_emulator branch (27/03/24) if problem:
        # plt.savefig(path / f'adjacency_{name_suffix}.png', format="png")
        plt.savefig(path / fname, format="png")
        plt.close()

    def plot_adjacency_matrix_w(self, mat1: np.ndarray, mat2: np.ndarray, path, name_suffix: str, no_gt: bool = False):
        """Plot the adjacency matrices learned and compare it to the ground truth,
        the first dimension of the matrix should be the features (d)
        Args:
          mat1: learned adjacency matrices
          mat2: ground-truth adjacency matrices
          path: path where to save the plot
          name_suffix: suffix for the name of the plot
          no_gt: if True, does not use ground-truth W
        """
        d = mat1.shape[0]
        subfig_names = ["Learned", "Ground Truth", "Difference: Learned - GT"]

        fig = plt.figure(constrained_layout=True)
        fig.suptitle("Matrices W")

        if no_gt:
            nrows = 1
        else:
            nrows = 3

        if d == 1:
            axes = fig.subplots(nrows=nrows, ncols=1)
            for row in range(nrows):
                if no_gt:
                    ax = axes
                else:
                    ax = axes[row]

                if row == 0:
                    mat = mat1[0]
                elif row == 1:
                    mat = mat2[0]
                else:
                    mat = mat1[0] - mat2[0]

                if mat1[0].size < 100:
                    annotation = True
                else:
                    annotation = False
                sns.heatmap(
                    mat,
                    ax=ax,
                    cbar=False,
                    vmin=-1,
                    vmax=1,
                    annot=annotation,
                    fmt=".5f",
                    cmap="Blues",
                    xticklabels=False,
                    yticklabels=False,
                )

                # if the matrix is small enough, print also the value of each
                # element of W in the heatmap
                # if mat1.size < 500:
                #     for i in range(mat.shape[0]):
                #         for j in range(mat.shape[1]):
                #             text = ax.text(j, i, f"{mat[i, j]:.1f}",
                #                            ha="center", va="center", color="w")

        else:
            subfigs = fig.subfigures(nrows=nrows, ncols=1)

            for row in range(nrows):
                if nrows == 1:
                    subfig = subfigs
                else:
                    subfig = subfigs[row]
                subfig.suptitle(f"{subfig_names[row]}")

                axes = subfig.subplots(nrows=1, ncols=d)
                for i in range(d):
                    axes[i].set_title(f"d = {i}")
                    if row == 0:
                        sns.heatmap(
                            mat1[d - i - 1],
                            ax=axes[i],
                            cbar=False,
                            vmin=-1,
                            vmax=1,
                            cmap="Blues",
                            xticklabels=False,
                            yticklabels=False,
                        )
                    elif row == 1:
                        sns.heatmap(
                            mat2[d - i - 1],
                            ax=axes[i],
                            cbar=False,
                            vmin=-1,
                            vmax=1,
                            cmap="Blues",
                            xticklabels=False,
                            yticklabels=False,
                        )
                    elif row == 2:
                        sns.heatmap(
                            mat1[d - i - 1] - mat2[d - i - 1],
                            ax=axes[i],
                            cbar=False,
                            vmin=-1,
                            vmax=1,
                            cmap="Blues",
                            xticklabels=False,
                            yticklabels=False,
                        )

        plt.savefig(path / f"adjacency_{name_suffix}.png", format="png")
        plt.close()

    def plot_adjacency_through_time(self, w_adj: np.ndarray, gt_dag: np.ndarray, t: int, path, name_suffix: str):
        """Plot the probability of each edges through time up to timestep t
        Args:
          w_adj: weight of edges
          gt_dag: ground-truth DAG
          t: timestep where to stop plotting
          path: path where to save the plot
          name_suffix: suffix for the name of the plot
        """
        taus = w_adj.shape[1]
        d = w_adj.shape[2]  # * w_adj.shape[3]
        w_adj = w_adj.reshape(w_adj.shape[0], taus, d, d)
        fig, ax1 = plt.subplots()

        for tau in range(taus):
            for i in range(d):
                for j in range(d):
                    # plot in green edges that are in the gt_dag
                    # otherwise in red
                    if gt_dag[tau, i, j]:
                        color = "g"
                        zorder = 2
                    else:
                        color = "r"
                        zorder = 1
                    ax1.plot(range(1, t), w_adj[1:t, tau, i, j], color, linewidth=1, zorder=zorder)
        fig.suptitle("Learned adjacencies through time")
        fig.savefig(path / f"adjacency_time_{name_suffix}.png", format="png")
        fig.clf()

    def plot_adjacency_through_time_w(self, w_adj: np.ndarray, gt_dag: np.ndarray, t: int, path, name_suffix: str):
        """Plot the probability of each edges through time up to timestep t
        Args:
          w_adj: weight of edges
          gt_dag: ground-truth DAG
          t: timestep where to stop plotting
          path: path where to save the plot
          name_suffix: suffix for the name of the plot
        """
        tau = w_adj.shape[1]
        dk = w_adj.shape[2]
        dk = w_adj.shape[3]
        # w_adj = w_adj.reshape(w_adj.shape[0], taus, d, d)
        fig, ax1 = plt.subplots()

        for i in range(tau):
            for j in range(dk):
                for k in range(dk):
                    ax1.plot(range(1, t), np.abs(w_adj[1:t, i, j, k] - gt_dag[i, j, k]), linewidth=1)
        fig.suptitle("Learned adjacencies through time")
        fig.savefig(path / f"adjacency_time_{name_suffix}.png", format="png")
        fig.clf()

    def save_mcc_and_assignement(self, exp_path):
        np.save(exp_path / "mcc", np.array(self.mcc))
        np.save(exp_path / "assignments", np.array(self.assignments))
        if len(self.mcc) > 1:
            fig = plt.figure()
            plt.plot(self.mcc)
            plt.title("MCC score through time")
            fig.savefig(exp_path / "mcc.png")
            fig.clf()

    def plot_compare_predictions_by_variable(
        self,
        x_past: np.ndarray,  # (B, T, num_vars, spatial)
        y_true: np.ndarray,  # (B, num_vars, spatial)
        y_recons: np.ndarray,  # (B, num_vars, spatial)
        y_hat: np.ndarray,  # (B, num_vars, spatial)
        sample: int,
        coordinates: np.ndarray,  # (total_spatial, 2)
        input_var_shapes: dict,
        input_var_offsets: list,
        path,
        iteration: int,
        valid: str = False,
    ):
        """
        Plot predictions for all variables using their own spatial grid.

        One row per variable, four columns for [GT t-1, GT, Recons, Pred].
        """

        print(
            f"x_past.shape: {x_past.shape}, y_true.shape: {y_true.shape}, "
            f"y_recons.shape: {y_recons.shape}, y_hat.shape: {y_hat.shape}"
        )
        print(f"sample: {sample}, coordinates.shape: {coordinates.shape}")

        titles = ["Ground-truth t-1", "Ground truth", "Reconstruction", "Prediction"]
        data_sources = [x_past, y_true, y_recons, y_hat]
        n_vars = len(input_var_shapes)

        for timestep in [-2, -1]:
            fig, axs = plt.subplots(
                n_vars,
                4,
                subplot_kw={"projection": ccrs.PlateCarree()},
                layout="constrained",
                figsize=(32, 8 * n_vars),
            )

            if n_vars == 1:
                axs = [axs]  # ensure 2D indexing

            for var_idx, (var, spatial_dim) in enumerate(input_var_shapes.items()):
                offset_start = input_var_offsets[var_idx]
                offset_end = input_var_offsets[var_idx + 1]
                coords = coordinates[offset_start:offset_end]

                lon = np.unique(coords[:, 0])
                lat = np.unique(coords[:, 1])
                lon_grid, lat_grid = np.meshgrid(lon, lat)

                for j in range(4):
                    ax = axs[var_idx][j]
                    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
                    ax.coastlines(resolution="50m")
                    ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
                    ax.add_feature(cfeature.LAND.with_scale("50m"), edgecolor="black")
                    ax.gridlines(draw_labels=False)

                    if j == 0:
                        raw_data = data_sources[j][sample, timestep, 0, :]
                    else:
                        raw_data = data_sources[j][sample, 0, :]

                    data = raw_data[offset_start:offset_end].reshape(lat.size, lon.size)

                    s = ax.pcolormesh(
                        lon_grid,
                        lat_grid,
                        data,
                        alpha=1,
                        vmin=-3.5,
                        vmax=3.5,
                        cmap="RdBu_r",
                        transform=ccrs.PlateCarree(),
                    )
                    ax.set_title(f"{titles[j]} ({var})", fontsize=14)

                # Add colorbar to the final column of each row
                fig.colorbar(s, ax=axs[var_idx][-1], orientation="vertical", shrink=0.8, label=f"Normalised {var}")

            timestep_label = timestep + 365
            fname_prefix = "valid" if valid else "train"
            fname = f"{fname_prefix}_compare_allvars_t{timestep_label}_sample_{sample}_it{iteration}.png"

            plt.suptitle(f"Comparison @ timestep {timestep_label}", fontsize=24)
            plt.savefig(path / fname, format="png")
            plt.close()

    def plot_regions_map_by_var(
        self,
        w_adj,
        coordinates: np.ndarray,
        input_var_shapes: dict,
        input_var_offsets: list,
        iteration: int,
        plot_through_time: bool,
        path,
        annotate: bool = False,
        one_plot: bool = False,
    ):
        """Plot spatial regions (latent clusters) for each variable separately on the same figure, zooming to each
        variable’s extent."""
        print(
            f"input_var_shapes: {input_var_shapes}, input_var_offsets: {input_var_offsets}, w_adj.shape: {w_adj.shape}"
        )
        print(f"coordinates.shape: {coordinates.shape}")

        # Flip coordinates once if necessary
        if np.max(coordinates[:, 0]) > 91:  # latitude in first column
            coordinates = coordinates[:, [1, 0]]  # now (lon, lat)

        d_z = w_adj.shape[2]
        d_vars = len(input_var_shapes)
        latents_per_var = d_z // d_vars
        colors = plt.cm.rainbow(np.linspace(0, 1, latents_per_var))
        idx = np.argmax(w_adj[0], axis=1)  # shape: (spatial,)

        fig, axs = plt.subplots(
            d_vars, 1, subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(10, 6 * d_vars), layout="constrained"
        )
        if d_vars == 1:
            axs = [axs]

        for var_idx, (var, spatial_dim) in enumerate(input_var_shapes.items()):
            offset_start = input_var_offsets[var_idx]
            offset_end = input_var_offsets[var_idx + 1]
            print("offset_start", offset_start, "offset_end", offset_end)
            coords = coordinates[offset_start:offset_end]
            idx_subset = idx[offset_start:offset_end]

            lon_min, lon_max = coords[:, 0].min(), coords[:, 0].max()
            lat_min, lat_max = coords[:, 1].min(), coords[:, 1].max()

            ax = axs[var_idx]
            ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.coastlines(resolution="50m")
            ax.add_feature(cfeature.LAND.with_scale("50m"), edgecolor="black")
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.gridlines(draw_labels=False)

            latent_offset = var_idx * latents_per_var
            for k in range(latents_per_var):
                global_latent_index = latent_offset + k
                region_coords = coords[idx_subset == global_latent_index]
                if region_coords.shape[0] == 0:
                    continue

                ax.scatter(
                    x=region_coords[:, 0],
                    y=region_coords[:, 1],
                    c=[colors[k]] * region_coords.shape[0],
                    s=20,
                    alpha=1.0,
                    transform=ccrs.PlateCarree(),
                )

                if annotate:
                    x_c, y_c = region_coords[:, 0].mean(), region_coords[:, 1].mean()
                    ax.text(x_c, y_c, str(k), transform=ccrs.PlateCarree())

            ax.set_title(f"Latent regions for {var}", fontsize=14)

        # Save file
        fname = "spatial_aggregation"
        if plot_through_time:
            fname += f"_{iteration}"
        if one_plot:
            fname += "_all_clusters"
        fname += ".png"

        plt.savefig(path / fname, format="png")
        plt.close()

    # # Below are functions used for plotting savar results / metrics. Not used yet but could be useful / integrated into the savar pipeline

    # def plot_original_savar(self, path, lon, lat, savar_path):
    #     """Plotting the original savar data."""
    #     data = np.load(f"{savar_path}.npy")

    #     # Get the dimensions
    #     time_steps = data.shape[1]
    #     data_reshaped = data.T.reshape((time_steps, lat, lon))

    #     # Calculate the average over the time axis
    #     avg_data = np.mean(data_reshaped, axis=0)

    #     # Determine the global min and max from the averaged data for consistent color scaling
    #     vmin = np.min(avg_data)
    #     vmax = np.max(avg_data)

    #     fig, ax = plt.subplots(figsize=(lon / 10, lat / 10))
    #     cax = ax.imshow(data_reshaped[0], aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    #     cbar = fig.colorbar(cax, ax=ax)

    #     def animate(i):
    #         cax.set_data(data_reshaped[i])
    #         ax.set_title(f"Time step: {i+1}")
    #         return (cax,)

    #     # Create an animation
    #     ani = animation.FuncAnimation(fig, animate, frames=100, blit=True)

    #     fname = "original_savar_data.gif"
    #     # Save the animation as a video file
    #     ani.save(os.path.join(path, fname), writer="pillow", fps=10)

    #     plt.close()

    # def compute_time_averaged_pixel_error(self, learner, cdsd_data, savar_data, iteration, path):
    #     """
    #     Computes the pixel error between time-averaged SAVAR ground truth and reconstructed CDSD latent variables.

    #     Args:
    #         cdsd_data (numpy.ndarray): CDSD latent variables of shape (1, lon*lat, d_z).
    #         savar_data (numpy.ndarray): SAVAR ground truth data of shape (time_steps, lat, lon).

    #     Returns:
    #         float: The mean squared error between time-averaged SAVAR and reconstructed CDSD.
    #     """
    #     # Step 1: Time-average the SAVAR data over time_steps
    #     savar_avg = np.mean(savar_data, axis=0)  # Shape becomes (lat, lon)

    #     # Step 2: Reshape cdsd_data to (lat, lon, d_z) based on savar spatial dimensions
    #     lat, lon = savar_avg.shape
    #     d_z = cdsd_data.shape[2]

    #     # Assuming lon*lat matches the savar grid
    #     cdsd_reshaped = cdsd_data.reshape(lat, lon, d_z)  # Shape becomes (lat, lon, d_z)

    #     # Step 3: Reconstruct CDSD by summing over the latent dimension (d_z)
    #     cdsd_reconstructed = np.sum(cdsd_reshaped, axis=2)  # Shape becomes (lat, lon)

    #     # Step 4: Compute pixel-wise error (Mean Squared Error)
    #     pixel_error = np.mean((cdsd_reconstructed - savar_avg) ** 2)

    #     print(f"Pixel error: {pixel_error}")

    #     combined_min = min(np.min(cdsd_reconstructed), np.min(savar_avg))
    #     combined_max = max(np.max(cdsd_reconstructed), np.max(savar_avg))

    #     # Step 5: Plot both the reconstructed CDSD data and the time-averaged SAVAR data
    #     fig, axes = plt.subplots(1, 2, figsize=(learner.hp.compute_pixel_figsize_x, learner.hp.compute_pixel_figsize_y))

    #     # Plot the reconstructed CDSD data
    #     im1 = axes[0].imshow(cdsd_reconstructed, cmap="viridis", aspect="auto", vmin=combined_min, vmax=combined_max)
    #     axes[0].set_title(f"Reconstructed CDSD Data (Pixel Error: {pixel_error:.4f})")
    #     axes[0].set_xlabel("Longitude")
    #     axes[0].set_ylabel("Latitude")
    #     plt.colorbar(im1, ax=axes[0], label="Reconstructed Value")

    #     # Plot the time-averaged SAVAR data
    #     im2 = axes[1].imshow(savar_avg, cmap="viridis", aspect="auto", vmin=combined_min, vmax=combined_max)
    #     axes[1].set_title("Time-Averaged SAVAR Data")
    #     axes[1].set_xlabel("Longitude")
    #     axes[1].set_ylabel("Latitude")
    #     plt.colorbar(im2, ax=axes[1], label="SAVAR Value")
    #     plt.tight_layout()

    #     fname = f"cdsd_reconstructed_{iteration}.png"

    #     plt.savefig(os.path.join(path, fname))
    #     plt.close()

    #     return pixel_error

    # def calculate_mcc_with_savar(self, cdsd_data, savar_data):
    #     """
    #     Calculates the Mean Correlation Coefficient (MCC) between discovered latents (CDSD) and ground truth SAVAR data,
    #     where SAVAR data is reshaped and projected into the same number of latents as the CDSD discovered data.

    #     Args:
    #         cdsd_latents (numpy array): Discovered latent variables from CDSD with shape (n_samples, n_latents).
    #         savar_data (numpy array): Ground-truth SAVAR data with shape (time_steps, longitude, latitude).
    #         num_latents (int): The number of latent variables (e.g., 3 in your case).

    #     Returns:
    #         float: The Mean Correlation Coefficient (MCC) between the CDSD latents and projected SAVAR latents.
    #     """
    #     num_latents = cdsd_data.shape[2]

    #     # Reshape SAVAR data from (time_steps, longitude, latitude) to (time_steps, longitude * latitude)
    #     time_steps, lat, lon = savar_data.shape
    #     savar_data_reshaped = savar_data.reshape(time_steps, lon * lat)

    #     # Apply ICA
    #     ica = FastICA(n_components=num_latents)
    #     savar_latents = ica.fit_transform(savar_data_reshaped.T).T
    #     print(savar_latents.shape)

    #     # Now, reshape the latents back into (time_steps, lat, lon, num_latents)
    #     savar_latents_reshaped = savar_latents.reshape(num_latents, lat, lon)

    #     for i in range(num_latents):
    #         plt.figure(figsize=(6, 6))  # Create a new figure for each latent
    #         latent_component = savar_latents_reshaped[i]  # Shape: (lat, lon)
    #         plt.imshow(latent_component, cmap="viridis", aspect="auto")
    #         plt.title(f"Latent {i + 1} after PCA")
    #         plt.colorbar()
    #         plt.show()  # Show each plot separately

    #     # Ensure CDSD latents and SAVAR latents have the same shape
    #     assert cdsd_data.shape == savar_latents.shape, "CDSD and SAVAR latent representations must have the same shape"

    #     # Number of latent variables
    #     n_latents = cdsd_data.shape[1]

    #     # Compute the correlation matrix between each latent variable of CDSD and SAVAR
    #     correlation_matrix = np.corrcoef(cdsd_data, savar_latents, rowvar=False)[:n_latents, n_latents:]

    #     # Use the Hungarian algorithm to find the best matching between CDSD and SAVAR latents
    #     row_ind, col_ind = linear_sum_assignment(-np.abs(correlation_matrix))

    #     # Extract the corresponding correlations
    #     matched_correlations = correlation_matrix[row_ind, col_ind]

    #     # Calculate the Mean Correlation Coefficient (MCC)
    #     mcc = np.mean(np.abs(matched_correlations))

    #     return mcc
