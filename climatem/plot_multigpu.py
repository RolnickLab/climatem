# Collection of plotting functions for plotting the results of experiments.

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
from climatem.metrics import mcc_latent
import torch

import cartopy.crs as ccrs
import cartopy.feature as cfeature

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
        return ret[n - 1:] / n

class Plotter:
    def __init__(self):
        self.mcc = []
        self.assignments = []

    
    def save(self, learner):
        """
        Save all the different metrics. Can then reload them to plot them.
        """

        if learner.latent:
            # save matrix W of the decoder and encoder
            print('Saving the decoder, encoder and graphs.')
            w_decoder = learner.model.module.autoencoder.get_w_decoder().cpu().detach().numpy()
            np.save(os.path.join(learner.hp.exp_path, "w_decoder.npy"), w_decoder)
            w_encoder = learner.model.module.autoencoder.get_w_encoder().cpu().detach().numpy()
            np.save(os.path.join(learner.hp.exp_path, "w_encoder.npy"), w_encoder)

            # save the graphs G
            adj = learner.model.module.get_adj().cpu().detach().numpy()
            np.save(os.path.join(learner.hp.exp_path, "graphs.npy"), adj)

    def load(self, exp_path: str, data_loader):
        # load matrix W of the decoder and encoder
        self.w = np.load(os.path.join(exp_path, "w_decoder.npy"))
        self.w_encoder = np.load(os.path.join(exp_path, "w_encoder.npy"))

        # load adj_tt and adj_w_tt, adjacencies through time
        self.adj_tt = np.load(os.path.join(exp_path, "adj_tt"))
        self.adj_w_tt = np.load(os.path.join(exp_path, "adj_w_tt"))

        # load log-variance of encoder and decoder
        self.logvar_encoder_tt = np.load(os.path.join(exp_path, "logvar_encoder_tt"))
        self.logvar_decoder_tt = np.load(os.path.join(exp_path, "logvar_decoder_tt"))
        self.logvar_transition_tt = np.load(os.path.join(exp_path, "logvar_transition_tt"))

        # load losses and penalties
        self.penalties = {}
        penalties = [{"name": "sparsity", "data": "train_sparsity_reg"},
                     {"name": "tr ortho", "data": "train_ortho_cons"},
                     {"name": "mu ortho", "data": "mu_ortho"}]
        for p in penalties:
            self.penalties[p["data"]] = np.load(os.path.join(exp_path, p["name"]))

        losses = [{"name": "tr ELBO", "data": "train_loss"},
                  {"name": "Recons", "data": "train_recons"},
                  {"name": "KL", "data": "train_kl"},
                  {"name": "val ELBO", "data": "valid_loss"}]
        for loss in losses:
            self.losses[loss["data"]] = np.load(os.path.join(exp_path, loss["name"]))

        # load GT W and graph
        self.gt_w = data_loader.gt_w
        self.gt_graph = data_loader.gt_dag

    def plot(self, learner, save=False):
        """
        Main plotting function.
        Plot the learning curves and
        if the ground-truth is known the adjacency and adjacency through time.
        """
        
        # NOTE:(seb) I am going to save the coordinates here, but this should be moved.
        np.save(os.path.join(learner.hp.exp_path, "coordinates.npy"), learner.coordinates)

        if save:
            self.save(learner)

        # plot learning curves
        if learner.latent:
            
            # NOTE:(seb) adding here capacity to plot the new sparsity constraint!
            losses = [{"name": "sparsity", "data": learner.train_sparsity_reg_list, "s": "-"},
                      {"name": "tr ortho", "data": learner.train_ortho_cons_list, "s": ":"},
                      {"name": "mu ortho", "data": learner.mu_ortho_list, "s": ":"},
                      #{"name": "tr sparsity", "data": learner.train_sparsity_cons_list, "s": ":"},
                      #{"name": "mu sparsity", "data": learner.mu_sparsity_list, "s": ":"},
                      #{"name": "gamma ortho", "data": learner.gamma_ortho_list, "s": ":"},
                      #{"name": "gamma sparsity", "data": learner.gamma_sparsity_list, "s": ":"},
                      ]
            
            
            
            self.plot_learning_curves2(losses=losses,
                                       iteration=learner.iteration,
                                       plot_through_time=learner.hp.plot_through_time,
                                       path=learner.hp.exp_path,
                                       fname="penalties",
                                       yaxis_log=True)
            losses = [{"name": "tr loss", "data": learner.train_loss_list, "s": "-."},
                      {"name": "tr recons", "data": learner.train_recons_list, "s": "-"},
                      {"name": "val recons", "data": learner.valid_recons_list, "s": "-"},
                      {"name": "KL", "data": learner.train_kl_list, "s": "-"},
                      {"name": "val loss", "data": learner.valid_loss_list, "s": "-."},
                      {"name": "tr ELBO", "data": learner.train_elbo_list, "s": "-."},
                      {"name": "val ELBO", "data": learner.valid_elbo_list, "s": "-."}
                      ]
            self.plot_learning_curves2(losses=losses,
                                       iteration=learner.iteration,
                                       plot_through_time=learner.hp.plot_through_time,
                                       path=learner.hp.exp_path,
                                       fname="losses")
            logvar = [{"name": "logvar encoder", "data": learner.logvar_encoder_tt, "s": "-"},
                      {"name": "logvar decoder", "data": learner.logvar_decoder_tt, "s": "-"},
                      {"name": "logvar transition", "data": learner.logvar_transition_tt, "s": "-"}]
            self.plot_learning_curves2(losses=logvar,
                                       iteration=learner.iteration,
                                       plot_through_time=learner.hp.plot_through_time,
                                       path=learner.hp.exp_path,
                                       fname="logvar")
        else:
            self.plot_learning_curves(train_loss=learner.train_loss_list,
                                      valid_loss=learner.valid_loss_list,
                                      iteration=learner.iteration,
                                      plot_through_time=learner.hp.plot_through_time,
                                      path=learner.hp.exp_path)

        # plot the adjacency matrix (learned vs ground-truth)
        adj = learner.model.module.get_adj().cpu().detach().numpy()
        if not learner.no_gt:
            if learner.latent:
                # for latent models, find the right permutation of the latent
                adj_w = learner.model.module.autoencoder.get_w_decoder().cpu().detach().numpy()
                adj_w2 = learner.model.module.autoencoder.get_w_encoder().cpu().detach().numpy()
                # variables using MCC
                if learner.debug_gt_z:
                    gt_dag = learner.gt_dag
                    gt_w = learner.gt_w
                    self.mcc.append(1.)
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
                self.save_mcc_and_assignement(learner.hp.exp_path)

                # draw learned mixing fct vs GT
                if learner.hp.nonlinear_mixing:
                    self.plot_learned_mixing(z, z_hat, adj_w, gt_w, x, learner.hp.exp_path)

            else:
                gt_dag = learner.gt_dag

            self.plot_adjacency_through_time(learner.adj_tt,
                                             gt_dag,
                                             learner.iteration,
                                             learner.hp.exp_path,
                                             'transition')
        else:
            gt_dag = None
            gt_w = None

            # for latent models, find the right permutation of the latent
            adj_w = learner.model.module.autoencoder.get_w_decoder().cpu().detach().numpy()
            adj_w2 = learner.model.module.autoencoder.get_w_encoder().cpu().detach().numpy()

        # this is where this was before, but I have now added the argument names for myself
        self.plot_adjacency_matrix(mat1=adj,
                                   mat2=gt_dag,
                                   path=learner.hp.exp_path,
                                   name_suffix='transition',
                                   no_gt=learner.no_gt,
                                   iteration=learner.iteration,
                                   plot_through_time=learner.hp.plot_through_time)

        # plot the weights W for latent models (between the latent Z and the X)
        # hoping that these don't fail due to defaults
        if learner.latent:
            # plot the decoder matrix W
            self.plot_adjacency_matrix_w(adj_w,
                                         gt_w,
                                         learner.hp.exp_path,
                                         'w',
                                         learner.no_gt)
            # plot the encoder matrix W_2
            # gt_w2 = np.swapaxes(gt_w, 1, 2)
            gt_w2 = gt_w
            self.plot_adjacency_matrix_w(adj_w2,
                                         gt_w2,
                                         learner.hp.exp_path,
                                         'encoder_w',
                                         learner.no_gt)
            if not learner.no_gt:
                self.plot_adjacency_through_time_w(learner.adj_w_tt,
                                                   learner.gt_w,
                                                   learner.iteration,
                                                   learner.hp.exp_path,
                                                   'w')
            else:
                self.plot_regions_map(adj_w,
                                      learner.coordinates,
                                      learner.iteration,
                                      learner.hp.plot_through_time,
                                      path=learner.hp.exp_path,
                                      idx_region=None,
                                      annotate=True,
                                      one_plot=True)

                self.plot_regions_map(adj_w,
                                      learner.coordinates,
                                      learner.iteration,
                                      learner.hp.plot_through_time,
                                      path=learner.hp.exp_path,
                                      idx_region=None,
                                      annotate=True)

                                    
    def plot_sparsity(self, learner, save=False):
        """
        Main plotting function.
        Plot the learning curves and
        if the ground-truth is known the adjacency and adjacency through time.
        """
      
        np.save(os.path.join(learner.hp.exp_path, "coordinates.npy"), learner.coordinates)

        if save:
            self.save(learner)

        # plot learning curves
        if learner.latent:

            self.plot_learning_curves(train_loss=learner.train_loss_list,
                                      train_recons=learner.train_recons_list,
                                      train_kl=learner.train_kl_list,
                                      valid_loss=learner.valid_loss_list,
                                      valid_recons=learner.valid_recons_list,
                                      valid_kl=learner.valid_kl_list,
                                      best_metrics=learner.best_metrics,
                                      iteration=learner.iteration,
                                      plot_through_time=learner.hp.plot_through_time,
                                      path=learner.hp.exp_path)
            # NOTE:(seb) adding here capacity to plot the new sparsity constraint!
            losses = [#{"name": "sparsity", "data": learner.train_sparsity_reg_list, "s": "-"},
                      {"name": "tr ortho", "data": learner.train_ortho_cons_list, "s": ":"},
                      {"name": "mu ortho", "data": learner.mu_ortho_list, "s": ":"},
                      {"name": "tr sparsity", "data": learner.train_sparsity_cons_list, "s": ":"},
                      {"name": "mu sparsity", "data": learner.mu_sparsity_list, "s": ":"},
                      #{"name": "gamma ortho", "data": learner.gamma_ortho_list, "s": ":"},
                      #{"name": "gamma sparsity", "data": learner.gamma_sparsity_list, "s": ":"},
                      ]
            # {"name": "tr acyclic", "data": learner.train_acyclic_cons_list, "s": "-"},
            # {"name": "tr connect", "data": learner.train_connect_reg_list, "s": "-"},
            self.plot_learning_curves2(losses=losses,
                                       iteration=learner.iteration,
                                       plot_through_time=learner.hp.plot_through_time,
                                       path=learner.hp.exp_path,
                                       fname="penalties",
                                       yaxis_log=True)
            losses = [{"name": "tr loss", "data": learner.train_loss_list, "s": "-."},
                      {"name": "tr recons", "data": learner.train_recons_list, "s": "-"},
                      {"name": "val recons", "data": learner.valid_recons_list, "s": "-"},
                      {"name": "KL", "data": learner.train_kl_list, "s": "-"},
                      {"name": "val loss", "data": learner.valid_loss_list, "s": "-."},
                      {"name": "tr ELBO", "data": learner.train_elbo_list, "s": "-."},
                      {"name": "val ELBO", "data": learner.valid_elbo_list, "s": "-."}
                      ]
            self.plot_learning_curves2(losses=losses,
                                       iteration=learner.iteration,
                                       plot_through_time=learner.hp.plot_through_time,
                                       path=learner.hp.exp_path,
                                       fname="losses")
            logvar = [{"name": "logvar encoder", "data": learner.logvar_encoder_tt, "s": "-"},
                      {"name": "logvar decoder", "data": learner.logvar_decoder_tt, "s": "-"},
                      {"name": "logvar transition", "data": learner.logvar_transition_tt, "s": "-"}]
            self.plot_learning_curves2(losses=logvar,
                                       iteration=learner.iteration,
                                       plot_through_time=learner.hp.plot_through_time,
                                       path=learner.hp.exp_path,
                                       fname="logvar")
        else:
            self.plot_learning_curves(train_loss=learner.train_loss_list,
                                      valid_loss=learner.valid_loss_list,
                                      iteration=learner.iteration,
                                      plot_through_time=learner.hp.plot_through_time,
                                      path=learner.hp.exp_path)

        # TODO: plot the prediction vs gt
        # plot_compare_prediction(x, x_hat)

        # plot the adjacency matrix (learned vs ground-truth)
        adj = learner.model.module.get_adj().cpu().detach().numpy()
        if not learner.no_gt:
            if learner.latent:
                # for latent models, find the right permutation of the latent
                adj_w = learner.model.module.autoencoder.get_w_decoder().cpu().detach().numpy()
                adj_w2 = learner.model.module.autoencoder.get_w_encoder().cpu().detach().numpy()
                # variables using MCC
                if learner.debug_gt_z:
                    gt_dag = learner.gt_dag
                    gt_w = learner.gt_w
                    self.mcc.append(1.)
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
                self.save_mcc_and_assignement(learner.hp.exp_path)

                # draw learned mixing fct vs GT
                if learner.hp.nonlinear_mixing:
                    self.plot_learned_mixing(z, z_hat, adj_w, gt_w, x, learner.hp.exp_path)

            else:
                gt_dag = learner.gt_dag

            self.plot_adjacency_through_time(learner.adj_tt,
                                             gt_dag,
                                             learner.iteration,
                                             learner.hp.exp_path,
                                             'transition')
        else:
            gt_dag = None
            gt_w = None

            # for latent models, find the right permutation of the latent
            adj_w = learner.model.module.autoencoder.get_w_decoder().cpu().detach().numpy()
            adj_w2 = learner.model.module.autoencoder.get_w_encoder().cpu().detach().numpy()

        # this is where this was before, but I have now added the argument names for myself
        self.plot_adjacency_matrix(mat1=adj,
                                   mat2=gt_dag,
                                   path=learner.hp.exp_path,
                                   name_suffix='transition',
                                   no_gt=learner.no_gt,
                                   iteration=learner.iteration,
                                   plot_through_time=learner.hp.plot_through_time)

        # plot the weights W for latent models (between the latent Z and the X)
        # hoping that these don't fail due to defaults
        if learner.latent:
            # plot the decoder matrix W
            self.plot_adjacency_matrix_w(adj_w,
                                         gt_w,
                                         learner.hp.exp_path,
                                         'w',
                                         learner.no_gt)
            # plot the encoder matrix W_2
            gt_w2 = gt_w
            self.plot_adjacency_matrix_w(adj_w2,
                                         gt_w2,
                                         learner.hp.exp_path,
                                         'encoder_w',
                                         learner.no_gt)
            if not learner.no_gt:
                self.plot_adjacency_through_time_w(learner.adj_w_tt,
                                                   learner.gt_w,
                                                   learner.iteration,
                                                   learner.hp.exp_path,
                                                   'w')
            else:
                self.plot_regions_map(adj_w,
                                      learner.coordinates,
                                      learner.iteration,
                                      learner.hp.plot_through_time,
                                      path=learner.hp.exp_path,
                                      idx_region=None,
                                      annotate=True)
    
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

            plt.savefig(os.path.join(path, f'learned_mixing_x{i}.png'))
            plt.close()

    def plot_compare_prediction(self, x, x_past, x_hat, coordinates: np.ndarray, path: str):
        """
        Plot the predicted x_hat compared to the ground-truth x
        Args:
            x: ground-truth x (for a specific physical variable)
            x_past: ground-truth x at (t-1)
            x_hat: x predicted by the model
            coordinates: xxx
            path: path where to save the plot
        """

        fig = plt.figure()
        fig.suptitle("Ground-truth vs prediction")

        lat = np.unique(coordinates[:, 0])
        lon = np.unique(coordinates[:, 1])
        X, Y = np.meshgrid(lon, lat)

        for i in range(3):
            if i == 0:
                z = x_past
                axes = fig.add_subplot(311)
                axes.set_title("Previous GT")
            if i == 1:
                z = x
                axes = fig.add_subplot(312)
                axes.set_title("Ground-truth")
            if i == 2:
                z = x_hat
                axes = fig.add_subplot(313)
                axes.set_title("Prediction")

            map = Basemap(projection='robin', lon_0=0)
            map.drawcoastlines()
            map.drawparallels(np.arange(-90, 90, 30), labels=[1, 0, 0, 0])
            # map.drawmeridians(np.arange(map.lonmin, map.lonmax + 30, 60), labels=[0, 0, 0, 1])

            Z = z.reshape(X.shape[0], X.shape[1])

            map.contourf(X, Y, Z, latlon=True)

        # plt.colorbar()
        plt.savefig(os.path.join(path, "prediction.png"), format="png")
        plt.close()


    def plot_compare_predictions_regular_grid(self, x_past: np.ndarray, y_true: np.ndarray, y_recons: np.ndarray, 
                                          y_hat: np.ndarray, sample: int, coordinates: np.ndarray, 
                                          path: str, iteration: int, valid: str = False, plot_through_time: bool = True):
        """
        Plot a prediction from the method, the last time step and the ground-truth on a regular grid.
        """

        if y_true.shape[1] > 1:
            fig, axs = plt.subplots(y_true.shape[1], 4, subplot_kw={'projection': ccrs.PlateCarree()}, layout='constrained', figsize=(32, 16))
        else:
            fig, axs = plt.subplots(1, 4, subplot_kw={'projection': ccrs.PlateCarree()}, layout='constrained', figsize=(32, 8))
            axs = [axs]

        lon = coordinates[:, 0]
        lat = coordinates[:, 1]
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        for j, ax_row in enumerate(axs):
            for i, ax in enumerate(ax_row):
                ax.set_global()
                ax.coastlines()
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.LAND, edgecolor='black')
                ax.gridlines(draw_labels=False)

                if i == 0:
                    s = ax.pcolormesh(lon_grid, lat_grid, x_past[sample, j, :].reshape(lat.size, lon.size), alpha=1, vmin=-3.5, vmax=3.5, cmap="RdBu_r", transform=ccrs.PlateCarree())
                    ax.set_title("Ground-truth t-1")
                elif i == 1:
                    s = ax.pcolormesh(lon_grid, lat_grid, y_true[sample, j, :].reshape(lat.size, lon.size), alpha=1, vmin=-3.5, vmax=3.5, cmap="RdBu_r", transform=ccrs.PlateCarree())
                    ax.set_title("Ground truth")
                elif i == 2:
                    s = ax.pcolormesh(lon_grid, lat_grid, y_recons[sample, j, :].reshape(lat.size, lon.size), alpha=1, vmin=-3.5, vmax=3.5, cmap="RdBu_r", transform=ccrs.PlateCarree())
                    ax.set_title("Reconstruction")
                elif i == 3:
                    s = ax.pcolormesh(lon_grid, lat_grid, y_hat[sample, j, :].reshape(lat.size, lon.size), alpha=1, vmin=-3.5, vmax=3.5, cmap="RdBu_r", transform=ccrs.PlateCarree())
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
        plt.savefig(os.path.join(path, fname), format="png")
        plt.close()


    # This plot should allow the plotting of multiple variables now.
    def plot_compare_predictions_icosahedral(self, x_past: np.ndarray, y_true: np.ndarray, y_recons: np.ndarray, 
                                             y_hat: np.ndarray, sample: int, coordinates: np.ndarray, 
                                             path: str, iteration: int, valid: str = False, plot_through_time: bool = True):
    
        """
        Plot a prediction from the method, the last time step and the ground-truth.
        """

        if y_true.shape[1] > 1:
            fig, axs = plt.subplots(y_true.shape[1], 4, subplot_kw={'projection': ccrs.Robinson()}, layout='constrained', figsize=(32, 16))
        else:
            fig, axs = plt.subplots(1, 4, subplot_kw={'projection': ccrs.Robinson()}, layout='constrained', figsize=(32, 8))
            axs = [axs]



        for j, ax_row in enumerate(axs):

           
            for i, ax in enumerate(ax_row):
                
                ax.set_global()
                ax.coastlines()
                # Add some map features for context
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.LAND, edgecolor='black')
                ax.gridlines(draw_labels=False)

                # Unpack coordinates for vectorized scatter plot
                # something like lonlat_vertex_mapping.txt
                lon = coordinates[:, 0]
                lat = coordinates[:, 1]

                # Vectorized scatter plot with color array   
                if i == 0:
                    #print('x_past shape:', x_past.shape)
                    s = ax.scatter(x=lon, y=lat, c=x_past[sample, j, :], alpha=1, s=30, vmin=-3.5, vmax=3.5, cmap="RdBu_r", transform=ccrs.PlateCarree())
                    ax.set_title("Ground-truth t-1")
                elif i == 1:
                    #print('y shape:', y_true.shape)
                    s = ax.scatter(x=lon, y=lat, c=y_true[sample, j, :], alpha=1, s=30, vmin=-3.5, vmax=3.5, cmap="RdBu_r", transform=ccrs.PlateCarree())
                    ax.set_title("Ground truth")
                elif i == 2:
                    #print('y_hat shape:', y_hat.shape)
                    s = ax.scatter(x=lon, y=lat, c=y_recons[sample, j, :], alpha=1, s=30, vmin=-3.5, vmax=3.5, cmap="RdBu_r", transform=ccrs.PlateCarree())
                    ax.set_title("Reconstruction")
                elif i == 3:
                    #print('y_recons shape:', y_recons.shape)
                    s = ax.scatter(x=lon, y=lat, c=y_hat[sample, j, :], alpha=1, s=30, vmin=-3.5, vmax=3.5, cmap="RdBu_r", transform=ccrs.PlateCarree())
                    ax.set_title("Prediction")

        # add one colorbar for all subplots
        #fig.colorbar(s, ax=axs, orientation='horizontal', fraction=0.05, pad=0.05)
        
            if j == 0:
                fig.colorbar(s, ax=ax_row[3], label="Normalised skin temperature", orientation="vertical", shrink=1.0) # adjust shrink
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
        #plt.legend()
        plt.savefig(os.path.join(path, fname), format="png")
        plt.close()

     
    def plot_compare_regions():
        pass

    # need to fix this plot so it works well for multiple variables

    # NOTE:(seb) trying to extend the plot_regions_map function to plot multiple variables
    def plot_regions_map(self, w_adj, coordinates: np.ndarray, iteration: int,
                            plot_through_time: bool, path: str, idx_region: int = None,
                            annotate: bool = False, one_plot: bool = False):
        """
        Here we extend the plot_regions_map function to plot multiple variables.
        """

        # find the argmax per row
        idx = np.argmax(w_adj, axis=2)
        norms = np.max(w_adj, axis=2)

        # here we want the number of latents PER variable
        
        d_z = w_adj.shape[2]

        # plot the regions

        colors = plt.cm.rainbow(np.linspace(0, 1, d_z))

        # First, I will assert that I have two columns.
        assert coordinates.shape[1] == 2

        # Then, swap the columns if the first column at the moment is the longitude column.
        if np.max(coordinates[:, 0]) > 91:
            coordinates = np.flip(coordinates, axis=1)
        
        #fig, axs = plt.subplots(1, w_adj.shape[0], subplot_kw={'projection': ccrs.Robinson()}, layout='constrained', figsize=(32, 8))

        if w_adj.shape[0] > 1:
            fig, axs = plt.subplots(1, w_adj.shape[0], subplot_kw={'projection': ccrs.Robinson()}, layout='constrained', figsize=(32, 8))
        else:
            fig, axs = plt.subplots(1, w_adj.shape[0], subplot_kw={'projection': ccrs.Robinson()}, layout='constrained', figsize=(32, 8))
            axs = [axs]

        for i, ax in enumerate(axs):
                
                ax.set_global()
                ax.coastlines()
                # Add some map features for context
                ax.add_feature(cfeature.BORDERS, linestyle=':')
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.LAND, edgecolor='black')
                ax.gridlines(draw_labels=False)
    
                # Unpack coordinates for vectorized scatter plot
                # something like lonlat_vertex_mapping.txt
                lon = coordinates[:, 0]
                lat = coordinates[:, 1]
    
                # Vectorized scatter plot with color array
                for k, color in zip(range(d_z), colors):
                    alpha = 1.
    
                    region = coordinates[idx[i] == k]
                    
                    c = np.repeat(np.array([color]), region.shape[0], axis=0)
    
                    ax.scatter(x=region[:, 1], y=region[:, 0], c=c, alpha=alpha, s=20, transform=ccrs.PlateCarree())
    
                    # add number for each region (that are completely in one of the four quadrants)
                    if annotate:
                        x, y = self.get_centroid(region[:, 1], region[:, 0])
                        ax.text(x, y, str(k), transform=ccrs.PlateCarree())
        
        if idx_region is not None:
            fname = f"spatial_aggregation{idx_region}.png"
        elif plot_through_time:
            fname = f"spatial_aggregation_{iteration}.png"
        elif one_plot:
            fname = "spatial_aggregation_all_clusters.png"
        else:
            fname = "spatial_aggregation.png"

        plt.savefig(os.path.join(path, fname),  format="png")
        plt.close()

    def get_centroid(self, xs, ys):
        """
        http://www.geomidpoint.com/example.html
        http://gis.stackexchange.com/questions/6025/find-the-centroid-of-a-cluster-of-points
        """
        sum_x, sum_y, sum_z = 0, 0, 0
        n = float(xs.shape[0])

        if n > 0:
            for (x, y) in zip(xs, ys):
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
            return 0., 0.


    def plot_learning_curves(self, train_loss: list, train_recons: list = None, train_kl: list = None,
                             valid_loss: list = None, valid_recons: list = None,
                             valid_kl: list = None, best_metrics: dict = None, iteration: int = 0,
                             plot_through_time: bool = False, path: str = ""):
        """ Plot the training and validation loss through time
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

        ax = plt.gca()
        # ax.set_ylim([0, 5])
        # ax.set_yscale("log")
        plt.plot(v_loss, label="valid loss", color="green")
        if train_recons is not None:
            plt.plot(t_recons, label="tr recons", color="blue")
            plt.axhline(y=best_metrics["recons"], color='blue', linestyle='dotted')
            plt.plot(t_kl, label="tr kl", color="red")
            plt.axhline(y=best_metrics["kl"], color='red', linestyle='dotted')
            plt.plot(t_loss, label="tr loss", color="purple")
            plt.axhline(y=best_metrics["elbo"], color='purple', linestyle='dotted')
            # plt.plot(v_recons, label="val recons")
            # plt.plot(v_kl, label="val kl")
        else:
            plt.plot(t_loss, label="tr ELBO")

        if plot_through_time:
            fname = f"loss_{iteration}.png"
        else:
            fname = "loss.png"

        plt.title("Learning curves")
        plt.legend()
        # NOTE:(seb) making sure we don't save the loss here...
        #plt.savefig(os.path.join(path, fname), format="png")
        plt.close()

    def plot_learning_curves2(self, losses: list, iteration: int = 0, plot_through_time: bool = False,
                              path: str = "", fname="loss_detailed", yaxis_log: bool = False):
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
        plt.savefig(os.path.join(path, fname), format="png")
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
        if not os.path.exists(os.path.join(learner.hp.exp_path, "coordinates.npy")):
            np.save(os.path.join(learner.hp.exp_path, "coordinates.npy"), learner.coordinates)

        adj_w = learner.model.module.autoencoder.get_w_decoder().cpu().detach().numpy()
        adj_encoder_w = learner.model.module.autoencoder.get_w_encoder().cpu().detach().numpy()
        np.save(os.path.join(learner.hp.exp_path, f"adj_encoder_w_{learner.iteration}.npy"), adj_encoder_w)
        np.save(os.path.join(learner.hp.exp_path, f"adj_w_{learner.iteration}.npy"), adj_w)


    
    # SH: edited to do plot through time, without changing the function name
    # simply follow the lead of the function above, and try to plot through time.
    def plot_adjacency_matrix(self, mat1: np.ndarray, mat2: np.ndarray,  
                              path: str, name_suffix: str, no_gt: bool = False,
                              iteration: int = 0, plot_through_time: bool = True):
        """ Plot the adjacency matrices learned and compare it to the ground truth,
        the first dimension of the matrix should be the time (tau)
        Args:
          mat1: learned adjacency matrices
          mat2: ground-truth adjacency matrices
          path: path where to save the plot
          name_suffix: suffix for the name of the plot
          no_gt: if True, does not use the ground-truth graph
        """
        tau = mat1.shape[0]
        
        subfig_names = [f"Learned, latent dimensions = {mat1.shape[1], mat1.shape[2]}", "Ground Truth", "Difference: Learned - GT"]

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
                    sns.heatmap(mat1[0], ax=ax, cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)
                elif row == 1:
                    sns.heatmap(mat2[0], ax=ax, cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)
                elif row == 2:
                    sns.heatmap(mat1[0] - mat2[0], ax=ax, cbar=False, vmin=-1, vmax=1,
                                cmap="Blues", xticklabels=False, yticklabels=False)

        else:
            subfigs = fig.subfigures(nrows=nrows, ncols=1)
            for row in range(nrows):
                if nrows == 1:
                    subfig = subfigs
                else:
                    subfig = subfigs[row]
                subfig.suptitle(f'{subfig_names[row]}')

                axes = subfig.subplots(nrows=1, ncols=tau)
                for i in range(tau):
                    axes[i].set_title(f"t - {i+1}")
                    if row == 0:
                        sns.heatmap(mat1[tau - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                    cmap="Blues", xticklabels=False, yticklabels=False)
                        # add a horizontal line every 50 columns
                        for j in range(0, mat1.shape[1], 50):
                            axes[i].axhline(y=j, color='black', linewidth=0.4)
                        # add a vertical line every 50 columns
                        for j in range(0, mat1.shape[1], 50):
                            axes[i].axvline(x=j, color='black', linewidth=0.4)
                        
                    elif row == 1:
                        sns.heatmap(mat2[tau - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                    cmap="Blues", xticklabels=False, yticklabels=False)
                    elif row == 2:
                        sns.heatmap(mat1[tau - i - 1] - mat2[tau - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                    cmap="Blues", xticklabels=False, yticklabels=False)

        # new
        if plot_through_time:
            fname = f"adjacency_{name_suffix}_{iteration}.png"
        else:
            fname = f"adjacency_{name_suffix}.png"   
        
        # updated this from before - see jb_causal_emulator branch (27/03/24) if problem:
        # plt.savefig(os.path.join(path, f'adjacency_{name_suffix}.png'), format="png")
        plt.savefig(os.path.join(path, fname), format="png")
        plt.close()

    def plot_adjacency_matrix_w(self, mat1: np.ndarray, mat2: np.ndarray, path: str,
                                name_suffix: str, no_gt: bool = False):
        """ Plot the adjacency matrices learned and compare it to the ground truth,
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
                sns.heatmap(mat, ax=ax, cbar=False, vmin=-1, vmax=1,
                            annot=annotation, fmt=".5f", cmap="Blues",
                            xticklabels=False, yticklabels=False)

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
                subfig.suptitle(f'{subfig_names[row]}')

                axes = subfig.subplots(nrows=1, ncols=d)
                for i in range(d):
                    axes[i].set_title(f"d = {i}")
                    if row == 0:
                        sns.heatmap(mat1[d - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                    cmap="Blues", xticklabels=False, yticklabels=False)
                    elif row == 1:
                        sns.heatmap(mat2[d - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                    cmap="Blues", xticklabels=False, yticklabels=False)
                    elif row == 2:
                        sns.heatmap(mat1[d - i - 1] - mat2[d - i - 1], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                    cmap="Blues", xticklabels=False, yticklabels=False)

        plt.savefig(os.path.join(path, f'adjacency_{name_suffix}.png'), format="png")
        plt.close()

    def plot_adjacency_through_time(self, w_adj: np.ndarray, gt_dag: np.ndarray, t: int,
                                    path: str, name_suffix: str):
        """ Plot the probability of each edges through time up to timestep t
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
                        color = 'g'
                        zorder = 2
                    else:
                        color = 'r'
                        zorder = 1
                    ax1.plot(range(1, t), w_adj[1:t, tau, i, j], color, linewidth=1, zorder=zorder)
        fig.suptitle("Learned adjacencies through time")
        fig.savefig(os.path.join(path, f'adjacency_time_{name_suffix}.png'), format="png")
        fig.clf()

    def plot_adjacency_through_time_w(self, w_adj: np.ndarray, gt_dag: np.ndarray, t: int,
                                      path: str, name_suffix: str):
        """ Plot the probability of each edges through time up to timestep t
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
        fig.savefig(os.path.join(path, f'adjacency_time_{name_suffix}.png'), format="png")
        fig.clf()

    def save_mcc_and_assignement(self, exp_path):
        np.save(os.path.join(exp_path, "mcc"), np.array(self.mcc))
        np.save(os.path.join(exp_path, "assignments"), np.array(self.assignments))
        if len(self.mcc) > 1:
            fig = plt.figure()
            plt.plot(self.mcc)
            plt.title("MCC score through time")
            fig.savefig(os.path.join(exp_path, 'mcc.png'))
            fig.clf()


