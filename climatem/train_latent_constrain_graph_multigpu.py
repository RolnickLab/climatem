# Adapting to do training across multiple GPUs with huggingface accelerate.

import os

import numpy as np
import torch
import torch.distributions as dist

# we use accelerate for distributed training
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from geopy import distance
from torch.profiler import ProfilerActivity

from climatem.dag_optim import compute_dag_constraint
from climatem.plot_multigpu import Plotter
from climatem.prox import monkey_patch_RMSprop
from climatem.utils import ALM

# Using Accelerator, not wandb now
# import wandb


# here is the combo:

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs], log_with="wandb")


# set profiler manually for now


class TrainingLatent:
    def __init__(self, model, datamodule, hp, best_metrics, d, profiler=False, profiler_path="./log"):
        # TODO: do we want to have the profiler as an argument? Maybe not, but useful to speed up the code
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = accelerator.device
        self.model = model
        # NOTE: here we move the model to gpu if it is available
        self.model.to(self.device)
        self.datamodule = datamodule
        self.data_loader_train = iter(datamodule.train_dataloader())
        self.data_loader_val = iter(datamodule.val_dataloader())
        self.coordinates = datamodule.coordinates
        self.hp = hp
        self.best_metrics = best_metrics

        self.latent = hp.latent
        self.no_gt = hp.no_gt
        self.debug_gt_z = hp.debug_gt_z
        self.d_z = hp.d_z
        self.no_w_constraint = hp.no_w_constraint

        self.d = d
        self.patience = hp.patience
        self.best_valid_loss = np.inf
        self.batch_size = datamodule.hparams.batch_size
        self.tau = hp.tau  # Here, 5 by default in both but we want to pass this as an argument
        self.d_x = hp.d_x
        self.instantaneous = hp.instantaneous

        self.patience_freq = 50
        self.iteration = 1
        self.logging_iter = 0
        self.converged = False
        self.thresholded = False
        self.ended = False

        self.profiler = profiler
        self.profiler_path = profiler_path

        # collection of lists to store relevant metrics
        self.train_loss_list = []
        self.train_elbo_list = []
        self.train_recons_list = []
        self.train_kl_list = []
        self.train_sparsity_reg_list = []
        self.train_connect_reg_list = []
        self.train_ortho_cons_list = []
        self.train_ortho_vector_cons_list = []
        self.train_acyclic_cons_list = []
        self.mu_ortho_list = []
        self.gamma_ortho_list = []
        self.h_ortho_list = []

        # add the crps
        self.train_crps_loss_list = []

        # add the spectral loss
        self.train_spectral_loss_list = []

        # add the temporal spectral loss
        self.train_temporal_spectral_loss_list = []

        self.train_sparsity_cons_list = []
        self.mu_sparsity_list = []
        self.gamma_sparsity_list = []

        self.valid_loss_list = []
        self.valid_elbo_list = []
        self.valid_recons_list = []
        self.valid_kl_list = []
        self.valid_sparsity_reg_list = []
        self.valid_connect_reg_list = []
        self.valid_ortho_cons_list = []
        self.valid_ortho_vector_cons_list = []
        self.valid_acyclic_cons_list = []
        self.valid_sparsity_cons_list = []

        self.plotter = Plotter()

        # I think this is just initialising a tensor of zeroes to store results in
        if self.instantaneous:
            self.adj_tt = torch.zeros(
                [int(self.hp.max_iteration / self.hp.valid_freq), self.tau + 1, self.d * self.d_z, self.d * self.d_z]
            )
        else:
            self.adj_tt = torch.zeros(
                [int(self.hp.max_iteration / self.hp.valid_freq), self.tau, self.d * self.d_z, self.d * self.d_z]
            )
        if not self.no_gt:
            self.adj_w_tt = torch.zeros([int(self.hp.max_iteration / self.hp.valid_freq), self.d, self.d_x, self.d_z])
        self.logvar_encoder_tt = []
        self.logvar_decoder_tt = []
        self.logvar_transition_tt = []

        # self.model.mask.fix(self.gt_dag)

        # optimizer
        if hp.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=hp.lr)
        elif hp.optimizer == "rmsprop":
            monkey_patch_RMSprop(torch.optim.RMSprop)
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=hp.lr)
        else:
            raise NotImplementedError(f"optimizer {hp.optimizer} is not implemented")
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=hp.lr_scheduler_epochs, gamma=hp.lr_scheduler_gamma
        )

        # prepare the model, optimizer, data loader, and scheduler using Accerate for distributed training
        print("Preparing all the models here!")
        self.model, self.optimizer, self.data_loader_train, self.scheduler = accelerator.prepare(
            self.model, self.optimizer, self.data_loader_train, self.scheduler
        )

        # compute constraint normalization
        with torch.no_grad():
            d = model.d * model.d_z
            full_adjacency = torch.ones((d, d)) - torch.eye(d)
            self.acyclic_constraint_normalization = compute_dag_constraint(full_adjacency).item()

            if self.latent:
                self.ortho_normalization = self.d_x * self.d_z
                self.sparsity_normalization = self.tau * self.d_z * self.d_z

    def train_with_QPM(self):
        """
        Optimize a problem under constraint using the Augmented Lagragian method (or QPM).

        We train in 3 phases: first with ALM, then until
        the likelihood remain stable, then continue after thresholding
        the adjacency matrix
        """

        # Pre-Accelerate - start a new wandb run to track this script
        # wandb.init(
        # set the wandb project where this run will be logged
        # please alter this project, and set the name to something appropriate for your experiments
        #    project="test-gpu-code-wandb",
        #    name=f"var_{self.datamodule.hparams.in_var_ids}_scenarios_{self.datamodule.hparams.train_scenarios}_climatemodel_{self.datamodule.hparams.train_models}_historical_years_{self.datamodule.hparams.train_historical_years}_ssp_years_{self.datamodule.hparams.train_years}_aggregate_{self.datamodule.num_months_aggregated}_tau_{self.tau}_z_{self.d_z}_nonlinearmixing_{self.model.nonlinear_mixing}_batchsize_{self.batch_size}_orthomuinit_{self.hp.ortho_mu_init}_orthoh_{self.hp.ortho_h_threshold}_sparsitymuinit_{self.hp.sparsity_mu_init}_sparsityupperthreshold_{self.hp.sparsity_upper_threshold}_fixed_{self.hp.fixed}_seasonality_removal_{self.datamodule.hparams.seasonality_removal}_num_ensembles_{self.datamodule.hparams.num_ensembles}",
        # )

        config = self.hp

        accelerator.init_trackers(
            "test-gpu-code-wandb",
            config=config,
            init_kwargs={
                "wandb": {
                    "name": f"var_{self.datamodule.hparams.in_var_ids}_scenarios_{self.datamodule.hparams.train_scenarios}_climatemodel_{self.datamodule.hparams.train_models}_historical_years_{self.datamodule.hparams.train_historical_years}_ssp_years_{self.datamodule.hparams.train_years}_aggregate_{self.datamodule.num_months_aggregated}_tau_{self.tau}_z_{self.d_z}_nonlinearmixing_{self.hp.nonlinear_mixing}_batchsize_{self.batch_size}_orthomuinit_{self.hp.ortho_mu_init}_orthoh_{self.hp.ortho_h_threshold}_sparsitymuinit_{self.hp.sparsity_mu_init}_sparsityupperthreshold_{self.hp.sparsity_upper_threshold}_fixed_{self.hp.fixed}_seasonality_removal_{self.datamodule.hparams.seasonality_removal}_num_ensembles_{self.datamodule.hparams.num_ensembles}_crpscoef_{self.hp.crps_coeff}_spcoef_{self.hp.spectral_coeff}_tempspcoef_{self.hp.temporal_spectral_coeff}"
                }
            },
        )

        # initialize ALM/QPM for orthogonality and acyclicity constraints
        self.ALM_ortho = ALM(
            self.hp.ortho_mu_init,
            self.hp.ortho_mu_mult_factor,
            self.hp.ortho_omega_gamma,
            self.hp.ortho_omega_mu,
            self.hp.ortho_h_threshold,
            self.hp.ortho_min_iter_convergence,
            dim_gamma=(self.d_z, self.d_z),
        )

        # NOTE:(seb) adding the sparsity constraint here, with different dimensions as sparsity is one dimensional
        self.ALM_sparsity = ALM(
            self.hp.sparsity_mu_init,
            self.hp.sparsity_mu_mult_factor,
            self.hp.sparsity_omega_gamma,
            self.hp.sparsity_omega_mu,
            self.hp.sparsity_h_threshold,
            self.hp.sparsity_min_iter_convergence,
            dim_gamma=(1, 1),
        )

        if self.instantaneous:
            # here we add the acyclicity constraint if the instantaneous connections are interesting
            self.QPM_acyclic = ALM(
                self.hp.acyclic_mu_init,
                self.hp.acyclic_mu_mult_factor,
                self.hp.acyclic_omega_gamma,
                self.hp.acyclic_omega_mu,
                self.hp.acyclic_h_threshold,
                self.hp.acyclic_min_iter_convergence,
                dim_gamma=(1, 1),
            )

        if self.profiler:

            # NOTE:(seb) profiler here.
            # we should have this function elsewhere. It is rarely used.
            def trace_handler(p):
                print("Printing profiler key averages from trace handler!")
                output_cpu = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
                output_cuda = p.key_averages().table(sort_by="cuda_time_total", row_limit=20)
                print(output_cpu)
                print(output_cuda)
                p.export_chrome_trace(f"/home/mila/s/sebastian.hickman/scratch/profiling/{str(p.step_num)}.json")

            prof = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=5, warmup=5, active=1, repeat=1),
                # using the torch tensorboard handler
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "/home/mila/s/sebastian.hickman/scratch/profiling"
                ),
                # on_trace_ready=torch.profiler.export_chrome_trace(self.profiler_path),
                # on_trace_ready=trace_handler,
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
            )
            prof.start()
            # print out the output of the profiler

        while self.iteration < self.hp.max_iteration and not self.ended:

            # train and valid step
            self.train_step()
            self.scheduler.step()
            if self.profiler:
                prof.step()

            if self.iteration % self.hp.valid_freq == 0:
                self.logging_iter += 1
                x, y, y_pred = self.valid_step()
                self.log_losses()

                # log these metrics to wandb every print_freq...
                #  multiple metrics here...
                if self.iteration % (self.hp.print_freq) == 0:
                    # altered to use the accelerator.log function
                    accelerator.log(
                        {
                            "kl_train": self.train_kl,
                            "loss_train": self.train_loss,
                            "recons_train": self.train_recons,
                            "kl_valid": self.valid_kl,
                            "loss_valid": self.valid_loss,
                            "recons_valid": self.valid_recons,
                            "nll_train": self.train_nll,
                            "nll_valid": self.valid_nll,
                            "mae_recons_train": self.train_mae_recons,
                            "mae_pred_train": self.train_mae_pred,
                            "mae_persistence_train": self.train_mae_persistence,
                            "mae_recons_valid": self.val_mae_recons,
                            "mae_pred_valid": self.val_mae_pred,
                            "mse_recons_train": self.train_mse_recons,
                            "mse_pred_train": self.train_mse_pred,
                            "mse_recons_valid": self.val_mse_recons,
                            "mse_pred_valid": self.val_mse_pred,
                            "var_original_train": self.train_var_original,
                            "var_recons_train": self.train_var_recons,
                            "var_pred_train": self.train_var_pred,
                            "var_original_valid": self.val_var_original,
                            "var_recons_val": self.val_var_recons,
                            "var_pred_valid": self.val_var_pred,
                            "mae_recons_valid_1": self.val_mae_recons_1,
                            "mae_recons_valid_2": self.val_mae_recons_2,
                            "mae_recons_valid_3": self.val_mae_recons_3,
                            "mae_recons_valid_4": self.val_mae_recons_4,
                            "mae_pred_valid_1": self.val_mae_pred_1,
                            "mae_pred_valid_2": self.val_mae_pred_2,
                            "mae_pred_valid_3": self.val_mae_pred_3,
                            "mae_pred_valid_4": self.val_mae_pred_4,
                            "mae_recons_train_1": self.train_mae_recons_1,
                            "mae_recons_train_2": self.train_mae_recons_2,
                            "mae_recons_train_3": self.train_mae_recons_3,
                            "mae_recons_train_4": self.train_mae_recons_4,
                            "mae_pred_train_1": self.train_mae_pred_1,
                            "mae_pred_train_2": self.train_mae_pred_2,
                            "mae_pred_train_3": self.train_mae_pred_3,
                            "mae_pred_train_4": self.train_mae_pred_4,
                            "spectral_loss_train": self.train_spectral_loss,
                            "temporal_spectral_loss_train": self.train_temporal_spectral_loss,
                            "crps_loss_train": self.train_crps_loss,
                        }
                    )

                else:
                    accelerator.log(
                        {
                            "kl_train": self.train_kl,
                            "loss_train": self.train_loss,
                            "recons_train": self.train_recons,
                            "kl_valid": self.valid_kl,
                            "loss_valid": self.valid_loss,
                            "recons_valid": self.valid_recons,
                        }
                    )

                # print and plot losses
                if self.iteration % (self.hp.print_freq) == 0:
                    self.print_results()

            if self.logging_iter > 0 and self.iteration % (self.hp.plot_freq) == 0:
                print(f"Plotting Iteration {self.iteration}")
                # NOTE:(seb), updated to use plot_sparsity to log the sparsity constraint
                self.plotter.plot_sparsity(self)
                # trying to save coords and adjacency matrices
                self.plotter.save_coordinates_and_adjacency_matrices(self)
                # NOTE:(seb) also saving the PyTorch model, saving the state dict:
                torch.save(self.model.state_dict(), f"{self.hp.exp_path}/model.pth")

                # try to use the accelerator.save function here
                accelerator.save_state(output_dir=self.hp.exp_path)

                # NOTE:(seb) also saving the optimizer state here:
                # torch.save(self.optimizer.state_dict(), self.hp.exp_path + '/optimizer.pth')

            if not self.converged:

                # train with penalty method
                # NOTE: here valid_freq is critical for updating the parameters of the ALM method!
                # this is easy to miss - perhaps we should implement another parameter for this.
                if self.iteration % self.hp.valid_freq == 0:
                    self.ALM_ortho.update(self.iteration, self.valid_ortho_vector_cons_list, self.valid_loss_list)
                    # updating ALM_sparsity here
                    self.ALM_sparsity.update(self.iteration, self.valid_sparsity_cons_list, self.valid_loss_list)

                    # This iteration value should be explored.
                    if self.iteration > 1000:
                        if not self.no_w_constraint:
                            ortho_converged = self.ALM_ortho.has_converged
                            sparsity_converged = self.ALM_sparsity.has_converged
                        else:
                            self.converged = True
                    else:
                        ortho_converged = False
                        sparsity_converged = False

                    # if has_increased_mu then reinitialize the optimizer?
                    if self.ALM_ortho.has_increased_mu:
                        if self.hp.optimizer == "sgd":
                            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hp.lr)
                        elif self.hp.optimizer == "rmsprop":
                            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.hp.lr)

                    # Repeat for sparsity constraint?
                    if self.ALM_sparsity.has_increased_mu:
                        if self.hp.optimizer == "sgd":
                            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hp.lr)
                        elif self.hp.optimizer == "rmsprop":
                            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.hp.lr)

                    if self.instantaneous:
                        self.QPM_acyclic.update(self.iteration, self.valid_acyclic_cons_list, self.valid_loss_list)
                        acyclic_converged = self.QPM_acyclic.has_converged
                        # TODO: add optimizer reinit
                        if self.QPM_acyclic.has_increased_mu:
                            if self.hp.optimizer == "sgd":
                                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hp.lr)
                            elif self.hp.optimizer == "rmsprop":
                                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.hp.lr)
                        self.converged = ortho_converged & acyclic_converged
                    else:
                        # NOTE:(seb) trying to adapt this to do sparsity too
                        # self.converged = ortho_converged
                        self.converged = ortho_converged & sparsity_converged
            else:
                # continue training without penalty method
                if not self.thresholded and self.iteration % self.patience_freq == 0:
                    # self.plotter.plot(self, save=True)
                    if not self.has_patience(self.hp.patience, self.valid_loss):
                        self.threshold()
                        self.patience = self.hp.patience_post_thresh
                        self.best_valid_loss = np.inf
                        # self.plotter.plot(self, save=True)
                # continue training after thresholding
                else:
                    if self.iteration % self.patience_freq == 0:
                        # self.plotter.plot(self, save=True)
                        if not self.has_patience(self.hp.patience_post_thresh, self.valid_loss):
                            self.ended = True

            self.iteration += 1

            # might want this for the profiler:
            # if self.profiler:
            #    prof.step()

        if self.iteration >= self.hp.max_iteration:
            self.threshold()

        # final plotting and printing
        self.plotter.plot_sparsity(self, save=True)
        self.print_results()

        # wandb.finish()
        accelerator.end_training()

        valid_loss = {
            "valid_loss": self.valid_loss,
            "best_valid_loss": self.best_valid_loss,
            "valid_loss1": -self.valid_loss_list[-1],
            "valid_loss2": -self.valid_loss_list[-2],
            "valid_loss3": -self.valid_loss_list[-3],
            "valid_loss4": -self.valid_loss_list[-4],
            "valid_loss5": -self.valid_loss_list[-5],
            "valid_neg_elbo": self.valid_nll,
            "valid_recons": self.valid_recons,
            "valid_kl": self.valid_kl,
            "valid_sparsity_reg": self.valid_sparsity_reg,
            "valid_ortho_cons": torch.sum(self.valid_ortho_cons).item(),
            "valid_sparsity_cons": self.valid_sparsity_cons,
        }

        # I guess this is just making sure...
        if self.profiler:
            # prof.export_chrome_trace("./log/trace.json")
            prof.stop()

        return valid_loss

    def train_step(self):

        self.model.train()

        # sample data

        # TODO: send data to gpu in the initialization and sample afterwards
        # x, y = next(self.data_loader_train) #.sample(self.batch_size, valid=False)

        try:
            x, y = next(self.data_loader_train)
            x = torch.nan_to_num(x)
            y = torch.nan_to_num(y)
        except StopIteration:
            self.data_loader_train = iter(self.datamodule.train_dataloader())
            x, y = next(self.data_loader_train)
            x = torch.nan_to_num(x)
            y = torch.nan_to_num(y)

        y = y[:, 0]
        z = None

        # NOTE:(seb) note that this makes the reconstruction
        nll, recons, kl, y_pred_recons = self.get_nll(x, y, z)

        # also make the proper prediction, not the reconstruction as we do above
        # we have to take care here to make sure that we have the right tensors with requires_grad
        y_pred, y_spare, z_spare, pz_mu, pz_std = self.model.module.predict(x, y)
        px_mu, px_std = self.model.module.predict_pxmu_pxstd(x, y)

        # compute regularisations (sparsity and connectivity)
        sparsity_reg = self.get_regularisation()
        connect_reg = torch.tensor([0.0])
        if self.hp.latent and self.hp.reg_coeff_connect > 0:
            # TODO: might be interesting to explore this
            connect_reg = self.connectivity_reg()

        # compute constraints (acyclicity and orthogonality)
        h_acyclic = torch.tensor([0.0])
        if self.instantaneous and not self.converged:
            h_acyclic = self.get_acyclicity_violation()
        # if self.hp.reg_coeff_connect:
        h_ortho = self.get_ortho_violation(self.model.module.autoencoder.get_w_decoder())

        # Compute a sparsity constraint here, with a lower (that doesn't end up being relevant) and upper threshold
        h_sparsity = self.get_sparsity_violation(lower_threshold=0.05, upper_threshold=self.hp.sparsity_upper_threshold)

        # compute total loss - here we are removing the sparsity regularisation as we are using the constraint here.
        loss = nll + connect_reg  # + sparsity_reg

        # Here we add the constraints to the loss.
        if not self.no_w_constraint:
            loss = loss + torch.sum(self.ALM_ortho.gamma @ h_ortho) + 0.5 * self.ALM_ortho.mu * torch.sum(h_ortho**2)
        if self.instantaneous:
            loss = loss + 0.5 * self.QPM_acyclic.mu * h_acyclic**2

        # Add sparsity constraint.
        loss = loss + self.ALM_sparsity.gamma * h_sparsity + 0.5 * self.ALM_sparsity.mu * h_sparsity**2

        # need to be superbly careful here that we are really using predictions, not the reconstruction
        crps = self.get_crps_loss(y, px_mu, px_std)
        spectral_loss = self.get_spectral_loss(y, y_pred)
        temporal_spectral_loss = self.get_temporal_spectral_loss(x, y, y_pred)

        # add the spectral loss to the loss
        loss = (
            loss
            + self.hp.crps_coeff * crps
            + self.hp.spectral_coeff * spectral_loss
            + self.hp.temporal_spectral_coeff * temporal_spectral_loss
        )

        # backprop
        # mask_prev = self.model.mask.param.clone()
        self.optimizer.zero_grad()

        # NOTE:(seb), multi-GPU here we use accelerator
        # loss.backward()
        accelerator.backward(loss)

        _, _ = self.optimizer.step() if self.hp.optimizer == "rmsprop" else self.optimizer.step(), self.hp.lr

        # projection of the gradient for w
        if self.model.module.autoencoder.use_grad_project and not self.no_w_constraint:
            with torch.no_grad():
                self.model.module.autoencoder.get_w_decoder().clamp_(min=0.0)
            assert torch.min(self.model.module.autoencoder.get_w_decoder()) >= 0.0

        self.train_loss = loss.item()
        self.train_nll = nll.item()
        self.train_recons = recons.item()
        self.train_kl = kl.item()
        self.train_sparsity_reg = sparsity_reg.item()
        self.train_connect_reg = connect_reg.item()
        self.train_ortho_cons = h_ortho  # .detach()
        self.train_acyclic_cons = h_acyclic  # .item() # errors with .item() as not tensor

        # adding the sparsity constraint to the logs
        self.train_sparsity_cons = h_sparsity  # .detach()

        # adding the crps loss to the logs
        self.train_crps_loss = crps.item()

        # adding the spectral loss to the logs
        self.train_spectral_loss = spectral_loss.item()

        # adding the temporal spectral loss to the logs
        self.train_temporal_spectral_loss = temporal_spectral_loss.item()

        # NOTE: here we have the saving, prediction, and analysis of some metrics, which comes at every print_freq
        # This can be cut if we want faster training...

        if self.iteration % self.hp.print_freq == 0:

            np.save(f"{self.hp.exp_path}/x_true_recons_train.npy", x.cpu().detach().numpy())
            np.save(f"{self.hp.exp_path}/y_true_recons_train.npy", y.cpu().detach().numpy())
            np.save(f"{self.hp.exp_path}/y_pred_recons_train.npy", y_pred_recons.cpu().detach().numpy())

            # also carry out autoregressive predictions
            mse, smape, y_original, y_original_pred, y_original_recons, x_original = (
                self.autoregress_prediction_original(valid=False, timesteps=10)
            )

            self.train_mae_recons = torch.mean(torch.abs(y_original_recons - y_original)).item()
            self.train_mae_pred = torch.mean(torch.abs(y_original_pred - y_original)).item()
            self.train_mae_persistence = torch.mean(torch.abs(y_original - x_original[:, -1, :, :])).item()

            self.train_mse_recons = torch.mean((y_original_recons - y_original) ** 2).item()
            self.train_mse_pred = torch.mean((y_original_pred - y_original) ** 2).item()
            self.train_mse_persistence = torch.mean((y_original - x_original[:, -1, :, :]) ** 2).item()

            # NOTE:(seb) including some additional metrics
            # include the variance of the predictions
            self.train_var_original = torch.var(y_original)
            self.train_var_recons = torch.var(y_original_recons)
            self.train_var_pred = torch.var(y_original_pred)

            # including per variable metrics, for when we train in the 4 variable case.
            if self.d == 4:
                self.train_mae_recons_1 = torch.mean(torch.abs(y_original_recons[:, 0] - y_original[:, 0])).item()
                self.train_mae_recons_2 = torch.mean(torch.abs(y_original_recons[:, 1] - y_original[:, 1])).item()
                self.train_mae_recons_3 = torch.mean(torch.abs(y_original_recons[:, 2] - y_original[:, 2])).item()
                self.train_mae_recons_4 = torch.mean(torch.abs(y_original_recons[:, 3] - y_original[:, 3])).item()

                self.train_mae_pred_1 = torch.mean(torch.abs(y_original_pred[:, 0] - y_original[:, 0])).item()
                self.train_mae_pred_2 = torch.mean(torch.abs(y_original_pred[:, 1] - y_original[:, 1])).item()
                self.train_mae_pred_3 = torch.mean(torch.abs(y_original_pred[:, 2] - y_original[:, 2])).item()
                self.train_mae_pred_4 = torch.mean(torch.abs(y_original_pred[:, 3] - y_original[:, 3])).item()
            else:
                self.train_mae_recons_1 = 0
                self.train_mae_recons_2 = 0
                self.train_mae_recons_3 = 0
                self.train_mae_recons_4 = 0

                self.train_mae_pred_1 = 0
                self.train_mae_pred_2 = 0
                self.train_mae_pred_3 = 0
                self.train_mae_pred_4 = 0

            # Need a better way to do this
            script_dir = os.path.dirname(__file__)
            rel_path = "vertex_lonlat_mapping.txt"
            abs_path_coords = os.path.join(script_dir, rel_path)
            coordinates = np.loadtxt(abs_path_coords)
            coordinates = coordinates[:, 1:]

            # choose a random integer in self.batch_size, setting a seed for this
            np.random.seed(0)

            # sample = np.random.randint(0, self.batch_size)

            # NOTE:(seb)  these samples have been changed to be different for each plot
            # Plotting the predictions for three different samples, including the reconstructions and the true values
            # if the shape of the data is icosahedral, we can plot like this:
            if self.d == 1 or self.d == 4:
                self.plotter.plot_compare_predictions_icosahedral(
                    x_past=x_original[:, -1, :, :].cpu().detach().numpy(),
                    y_true=y_original.cpu().detach().numpy(),
                    y_recons=y_original_recons.cpu().detach().numpy(),
                    y_hat=y_original_pred.cpu().detach().numpy(),
                    sample=np.random.randint(0, self.batch_size),
                    coordinates=self.coordinates,
                    path=self.hp.exp_path,
                    iteration=self.iteration,
                    valid=False,
                    plot_through_time=True,
                )

                self.plotter.plot_compare_predictions_icosahedral(
                    x_past=x_original[:, -1, :, :].cpu().detach().numpy(),
                    y_true=y_original.cpu().detach().numpy(),
                    y_recons=y_original_recons.cpu().detach().numpy(),
                    y_hat=y_original_pred.cpu().detach().numpy(),
                    sample=np.random.randint(0, self.batch_size),
                    coordinates=coordinates,
                    path=self.hp.exp_path,
                    iteration=self.iteration,
                    valid=False,
                    plot_through_time=True,
                )

                self.plotter.plot_compare_predictions_icosahedral(
                    x_past=x_original[:, -1, :, :].cpu().detach().numpy(),
                    y_true=y_original.cpu().detach().numpy(),
                    y_recons=y_original_recons.cpu().detach().numpy(),
                    y_hat=y_original_pred.cpu().detach().numpy(),
                    sample=np.random.randint(0, self.batch_size),
                    coordinates=coordinates,
                    path=self.hp.exp_path,
                    iteration=self.iteration,
                    valid=False,
                    plot_through_time=True,
                )
            else:
                print("Not plotting predictions.")
                # self.plotter.plot_compare_prediction(x_past=x_original[0, -1, :, :].cpu().detach().numpy(),
                #                                      x=y_original[0, 0].cpu().detach().numpy(),
                #                                      #y_recons=y_original_recons.cpu().detach().numpy(),
                #                                      x_hat=y_original_pred[0, 0].cpu().detach().numpy(),
                #                                      coordinates=self.coordinates,
                #                                      path=self.hp.exp_path,
                #                                      )

        # note that this has been changed to y_pred_recons
        return x, y, y_pred_recons

    # Validation step here.
    def valid_step(self):
        self.model.eval()

        with torch.no_grad():
            # sample data
            try:
                x, y = next(self.data_loader_val)
                x = torch.nan_to_num(x)
                y = torch.nan_to_num(y)
            except StopIteration:
                self.data_loader_val = iter(self.datamodule.val_dataloader())
                x, y = next(self.data_loader_val)
                x = torch.nan_to_num(x)
                y = torch.nan_to_num(y)

            # x, y = next(self.data_loader_val) #.sample(self.data_loader_val.n_valid - self.data_loader_val.tau, valid=True) #Check they have these features
            y = y[:, 0]
            # NOTE: sh, z here is if we have a ground truth
            z = None

            # print('doing get_nll in validation')
            nll, recons, kl, y_pred = self.get_nll(x, y, z)

            # compute regularisations (sparsity and connectivity)
            sparsity_reg = self.get_regularisation()
            connect_reg = torch.tensor([0.0])
            if self.hp.latent and self.hp.reg_coeff_connect > 0:
                # what is happening here between connectivity_reg and connectivity_reg_complete? See below.
                connect_reg = self.connectivity_reg()

            # compute constraints (acyclicity and orthogonality)
            h_acyclic = torch.tensor([0.0])
            # h_ortho = torch.tensor([0.])
            if self.instantaneous and not self.converged:
                h_acyclic = self.get_acyclicity_violation()
            h_ortho = self.get_ortho_violation(self.model.module.autoencoder.get_w_decoder())

            # NOTE:(seb) compute sparsity constraint 9/5/2024
            h_sparsity = self.get_sparsity_violation(
                lower_threshold=0.05, upper_threshold=self.hp.sparsity_upper_threshold
            )

            # compute total loss
            loss = nll + connect_reg  # + sparsity_reg - for now we are removing the sparsity regularisation

            # NOTE: ignore the constraints loss for saving the loss of the validation data. We are basically just interested in the nll.
            # loss = loss + self.ALM_ortho.gamma * h_ortho + \
            #     0.5 * self.ALM_ortho.mu * h_ortho ** 2

            # if self.instantaneous:
            #    loss = loss + 0.5 * self.QPM_acyclic.mu * h_acyclic ** 2

            self.valid_loss = loss.item()
            self.valid_nll = nll.item()
            self.valid_recons = recons.item()
            self.valid_kl = kl.item()
            self.valid_sparsity_reg = sparsity_reg.item()
            self.valid_ortho_cons = h_ortho  # .detach()
            self.valid_connect_reg = connect_reg.item()
            self.valid_acyclic_cons = h_acyclic  # .item() # NOTE:(seb) errors with .item() as not tensor

            # adding the sparsity constraint to the logs
            self.valid_sparsity_cons = h_sparsity  # .detach()

        # NOTE: here we have the saving, prediction, and analysis of some metrics, which comes at every print_freq
        # This can be cut if we want faster training...

        if self.iteration % self.hp.print_freq == 0:

            np.save(f"{self.hp.exp_path}/x_true_recons_val.npy", x.cpu().detach().numpy())
            np.save(f"{self.hp.exp_path}/y_true_recons_val.npy", y.cpu().detach().numpy())
            np.save(f"{self.hp.exp_path}/y_pred_recons_val.npy", y_pred.cpu().detach().numpy())

            mse, smape, y_original, y_original_pred, y_original_recons, x_original = (
                self.autoregress_prediction_original(valid=True, timesteps=10)
            )

            # print all the shapes of these

            self.val_mae_recons = torch.mean(torch.abs(y_original_recons - y_original)).item()
            self.val_mae_pred = torch.mean(torch.abs(y_original_pred - y_original)).item()
            self.val_mae_persistence = torch.mean(torch.abs(y_original - x_original[:, -1, :, :])).item()

            self.val_mse_recons = torch.mean((y_original_recons - y_original) ** 2).item()
            self.val_mse_pred = torch.mean((y_original_pred - y_original) ** 2).item()
            self.val_mse_persistence = torch.mean((y_original - x_original[:, -1, :, :]) ** 2).item()

            # NOTE:(seb) including some additional metrics
            # include the variance of the predictions
            self.val_var_original = torch.var(y_original)
            self.val_var_recons = torch.var(y_original_recons)
            self.val_var_pred = torch.var(y_original_pred)

            # NOTE:(seb) including some per variable metrics, specifically for the 4 variable case

            if self.d == 4:
                self.val_mae_recons_1 = torch.mean(torch.abs(y_original_recons[:, 0] - y_original[:, 0])).item()
                self.val_mae_recons_2 = torch.mean(torch.abs(y_original_recons[:, 1] - y_original[:, 1])).item()
                self.val_mae_recons_3 = torch.mean(torch.abs(y_original_recons[:, 2] - y_original[:, 2])).item()
                self.val_mae_recons_4 = torch.mean(torch.abs(y_original_recons[:, 3] - y_original[:, 3])).item()

                self.val_mae_pred_1 = torch.mean(torch.abs(y_original_pred[:, 0] - y_original[:, 0])).item()
                self.val_mae_pred_2 = torch.mean(torch.abs(y_original_pred[:, 1] - y_original[:, 1])).item()
                self.val_mae_pred_3 = torch.mean(torch.abs(y_original_pred[:, 2] - y_original[:, 2])).item()
                self.val_mae_pred_4 = torch.mean(torch.abs(y_original_pred[:, 3] - y_original[:, 3])).item()
            else:
                self.val_mae_recons_1 = 0
                self.val_mae_recons_2 = 0
                self.val_mae_recons_3 = 0
                self.val_mae_recons_4 = 0

                self.val_mae_pred_1 = 0
                self.val_mae_pred_2 = 0
                self.val_mae_pred_3 = 0
                self.val_mae_pred_4 = 0

            # also plot a comparison of the past true, true, reconstructed and the predicted values for the validation data
            # self.plotter.plot_compare_predictions_icosahedral(self, lots of arguments! save=True)

            # should probably have this somewhere else...
            script_dir = os.path.dirname(__file__)
            rel_path = "vertex_lonlat_mapping.txt"
            abs_path_coords = os.path.join(script_dir, rel_path)
            coordinates = np.loadtxt(abs_path_coords)
            coordinates = coordinates[:, 1:]

            # NOTE:(seb) changed to have random selection for each plot

            if self.d == 1 or self.d == 4:
                self.plotter.plot_compare_predictions_icosahedral(
                    x_past=x_original[:, -1, :, :].cpu().detach().numpy(),
                    y_true=y_original.cpu().detach().numpy(),
                    y_recons=y_original_recons.cpu().detach().numpy(),
                    y_hat=y_original_pred.cpu().detach().numpy(),
                    sample=np.random.randint(0, self.batch_size),
                    coordinates=coordinates,
                    path=self.hp.exp_path,
                    iteration=self.iteration,
                    valid=True,
                    plot_through_time=True,
                )

                self.plotter.plot_compare_predictions_icosahedral(
                    x_past=x_original[:, -1, :, :].cpu().detach().numpy(),
                    y_true=y_original.cpu().detach().numpy(),
                    y_recons=y_original_recons.cpu().detach().numpy(),
                    y_hat=y_original_pred.cpu().detach().numpy(),
                    sample=np.random.randint(0, self.batch_size),
                    coordinates=coordinates,
                    path=self.hp.exp_path,
                    iteration=self.iteration,
                    valid=True,
                    plot_through_time=True,
                )

                self.plotter.plot_compare_predictions_icosahedral(
                    x_past=x_original[:, -1, :, :].cpu().detach().numpy(),
                    y_true=y_original.cpu().detach().numpy(),
                    y_recons=y_original_recons.cpu().detach().numpy(),
                    y_hat=y_original_pred.cpu().detach().numpy(),
                    sample=np.random.randint(0, self.batch_size),
                    coordinates=coordinates,
                    path=self.hp.exp_path,
                    iteration=self.iteration,
                    valid=True,
                    plot_through_time=True,
                )

        return x, y, y_pred

    def has_patience(self, patience_init, valid_loss):
        """Check if the validation loss has not improved for 'patience' steps."""
        if self.patience > 0:
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.patience = patience_init
                print(f"Best valid loss: {self.best_valid_loss}")
            else:
                self.patience -= 1
            return True
        else:
            return False

    def threshold(self):
        """
        Consider that the graph has been found.

        Convert it to a binary graph and fix it.
        """
        with torch.no_grad():
            thresholded_adj = (self.model.module.get_adj() > 0.5).type(torch.Tensor)
            self.model.module.mask.fix(thresholded_adj)
        self.thresholded = True
        print("Thresholding ================")

    def log_losses(self):
        """Append in lists values of the losses and more."""
        # train
        self.train_loss_list.append(-self.train_loss)
        self.train_recons_list.append(self.train_recons)
        self.train_kl_list.append(self.train_kl)

        # here note that train_ortho_cons_list is a torch.sum...
        self.train_sparsity_reg_list.append(self.train_sparsity_reg)
        self.train_connect_reg_list.append(self.train_connect_reg)
        self.train_ortho_cons_list.append(torch.sum(self.train_ortho_cons))
        self.train_ortho_vector_cons_list.append(self.train_ortho_cons)
        # make this torch.sum?
        self.train_acyclic_cons_list.append(self.train_acyclic_cons)

        # valid
        self.valid_loss_list.append(-self.valid_loss)
        self.valid_recons_list.append(self.valid_recons)
        self.valid_kl_list.append(self.valid_kl)

        # here note that valid_ortho_cons_list is a torch.sum...
        self.valid_sparsity_reg_list.append(self.valid_sparsity_reg)
        self.valid_connect_reg_list.append(self.valid_connect_reg)
        self.valid_ortho_cons_list.append(torch.sum(self.valid_ortho_cons))
        self.valid_ortho_vector_cons_list.append(self.valid_ortho_cons)

        self.valid_acyclic_cons_list.append(self.valid_acyclic_cons)

        self.mu_ortho_list.append(self.ALM_ortho.mu)
        self.gamma_ortho_list.append(self.ALM_ortho.gamma)

        # NOTE:(seb) sparsity constraint lists here.
        self.train_sparsity_cons_list.append(self.train_sparsity_cons)
        self.valid_sparsity_cons_list.append(self.valid_sparsity_cons)

        # adding crps
        self.train_crps_loss_list.append(self.train_crps_loss)

        # adding spectral loss
        self.train_spectral_loss_list.append(self.train_spectral_loss)

        # adding temporal spectral loss
        self.train_temporal_spectral_loss_list.append(self.train_temporal_spectral_loss)

        self.mu_sparsity_list.append(self.ALM_sparsity.mu)
        self.gamma_sparsity_list.append(self.ALM_sparsity.gamma)

        self.adj_tt[int(self.iteration / self.hp.valid_freq)] = self.model.module.get_adj()  # .cpu().detach().numpy()
        w = self.model.module.autoencoder.get_w_decoder()  # .cpu().detach().numpy()
        if not self.no_gt:
            self.adj_w_tt[int(self.iteration / self.hp.valid_freq)] = w

        # here we just plot the first element of the logvar_decoder and logvar_encoder
        self.logvar_decoder_tt.append(self.model.module.autoencoder.logvar_decoder[0].item())
        self.logvar_encoder_tt.append(self.model.module.autoencoder.logvar_encoder[0].item())
        self.logvar_transition_tt.append(self.model.module.transition_model.logvar[0, 0].item())

    def print_results(self):
        """Print values of many variable: losses, constraint violation, etc.
        at the frequency self.hp.print_freq"""

        # NOTE:(seb) here plotting the terms that contribute to the loss that I care about.
        # print("****************************************************************************************")
        # print("What is the loss, the NLL, the reconstruction, the KL, the sparsity reg, the connect reg, the ortho cons, the acyclic cons, the sparsity cons?")
        # print("****************************************************************************************")
        # print("The loss is:", self.train_loss)
        # print("The NLL is:", self.train_nll)
        # print("The reconstruction is:", self.train_recons)
        # print("The KL is:", self.train_kl)

        # print("The torch.sum(self.ALM_ortho.gamma @ h_ortho) is:", torch.sum(self.ALM_ortho.gamma @ self.train_ortho_cons))
        # print("The 0.5 * self.ALM_ortho.mu * torch.sum(h_ortho ** 2) is:", 0.5 * self.ALM_ortho.mu * torch.sum(self.train_ortho_cons ** 2))

        # print("The self.ALM_sparsity.gamma * h_sparsity is:", self.ALM_sparsity.gamma * self.train_sparsity_cons)
        # print("The 0.5 * self.ALM_sparsity.mu * h_sparsity**2 is:", (0.5 * self.ALM_sparsity.mu * self.train_sparsity_cons**2))

        print("****************************************************************************************")
        # print("What are the actual values of the constraints?")
        # print("The connect reg is:", self.train_connect_reg)
        # print("The sparsity reg is:", self.train_sparsity_reg)
        # print("The ortho cons is:", self.train_ortho_cons)
        # print("The acyclic cons is:", self.train_acyclic_cons)
        # print("The sparsity cons is:", self.train_sparsity_cons)
        # print("****************************************************************************************")

    def get_nll(self, x, y, z=None) -> torch.Tensor:

        # this is just running the forward pass of LatentTSDCD...
        elbo, recons, kl, pred = self.model(x, y, z, self.iteration)

        # print('what is len(self.model(arg)) with arguments', len(self.model(x, y, z, self.iteration)))
        return -elbo, recons, kl, pred

    def get_regularisation(self) -> float:
        if self.iteration > self.hp.schedule_reg:
            adj = self.model.module.get_adj()
            reg = self.hp.reg_coeff * torch.norm(adj, p=1)
            # reg /= adj.numel()
        else:
            reg = torch.tensor([0.0])

        return reg

    def get_acyclicity_violation(self) -> torch.Tensor:
        if self.iteration > 0:
            adj = self.model.module.get_adj()[-1].view(self.d * self.d_z, self.d * self.d_z)
            h = compute_dag_constraint(adj) / self.acyclic_constraint_normalization
        else:
            h = torch.tensor([0.0])

        assert torch.is_tensor(h)

        return h

    def get_ortho_violation(self, w: torch.Tensor) -> float:

        if self.iteration > self.hp.schedule_ortho:
            # constraint = torch.tensor([0.])
            k = w.size(2)
            # for i in range(w.size(0)):
            #     constraint = constraint + torch.norm(w[i].T @ w[i] - torch.eye(k), p=2)
            i = 0
            # constraint = torch.norm(w[i].T @ w[i] - torch.eye(k), p=2, dim=1)
            constraint = w[i].T @ w[i] - torch.eye(k)
            # print('What is the ortho constraint shape:', constraint.shape)
            h = constraint / self.ortho_normalization
        else:
            h = torch.tensor([0.0])

        assert torch.is_tensor(h)

        return h

    # NOTE Adding the number of causal links as a constraint, rather than having it as a penalty as in CDSD originally
    # NOTE Previously we did model.get_adj() as an argument. I am changing this to just be self...
    # more like get_regularisation, which is what we want to copy closely.

    def get_sparsity_violation(self, lower_threshold, upper_threshold) -> float:
        """
        Calculate the number of causal links in the adjacency matrix, and constrain this to be less than a certain
        number.

        Threshold is the fraction of causal links, e.g. 0.1, 0.3
        """
        if self.iteration > self.hp.schedule_sparsity:

            # first get the adj
            adj = self.model.module.get_adj()

            # NOTE:(seb) try p=2
            sum_of_connections = torch.norm(adj, p=1) / self.sparsity_normalization
            # print('constraint value, before I subtract a threshold from it:', sum_of_connections)

            # If the sum_of_connections is greater than the upper threshold, then we have a violation
            if sum_of_connections > upper_threshold:
                constraint = sum_of_connections - upper_threshold

            # If the constraint is less than the lower threshold, then we also have a violation
            elif sum_of_connections < lower_threshold:
                constraint = lower_threshold - sum_of_connections

            # Otherwise, there is no penalty due to the constraint:
            else:
                constraint = torch.tensor([0.0])

            # print('constraint value, after I subtract a threshold, or whatever:', constraint)

            # NOTE:(seb) - here we implement an inequality constraint rather than a forced equality
            h = torch.max(constraint, torch.tensor([0.0]))

        else:
            h = torch.tensor([0.0])

        assert torch.is_tensor(h)

        return h

    def _normpdf(self, x):
        """Probability density function of a univariate standard Gaussian distribution with zero mean and unit
        variance."""
        return (1.0 / torch.sqrt(torch.tensor(2.0 * torch.pi))) * torch.exp(torch.tensor(-(x * x) / 2.0))

    def get_crps_loss(self, y, mu, sigma):
        """
        Calculate the CRPS loss between the true values and the predicted values. We need to extract the parameters of
        the Gaussian of the model. I am going to start by just taking the parameters of all the Gaussians for the
        observations first...

        I think better would actually be to produce an ensemble from the latent variable distributions, and then calculate the CRPS loss from this ensemble.

        I would quite like to do this on future timesteps too.

        Args:
            y: torch.Tensor, the true values
            mu: torch.Tensor, the mean of the Gaussians
            sigma: torch.Tensor, the standard deviation of the Gaussians
        """

        # gaussian_dist = torch.distributions.Normal(mu, sigma)

        y = y
        mu = mu
        sigma = sigma

        # standardised y
        sy = (y - mu) / sigma

        forecast_dist = dist.Normal(0, 1)

        # some precomputations to speed up the gradient
        pdf = self._normpdf(sy)
        cdf = forecast_dist.cdf(sy)

        pi_inv = 1.0 / torch.sqrt(torch.tensor([np.pi]))

        # calculate the CRPS
        crps = sigma * (sy * (2.0 * cdf - 1.0) + 2.0 * pdf - pi_inv)

        # add together all the CRPS values and divide by the number of samples
        crps = torch.sum(crps) / y.size(0)

        return crps

    def get_spectral_loss(self, y_true, y_pred):
        """
        Calculate the spectral loss between the true values and the predicted values. We need to calculate the spectra
        of thhe true values and the predicted values, and then determine an appropriate metric to compare them.

        There are a lot of design choices here that may not make a lot of sense.
        Averaging across batches? Square of the difference? Absolute value of the difference?

        Separating out the contributions of the different variables? All unclear.

        I might actually want to log this, so that the loss is not just dominated by the very low frequency, high power components.

        I should be setting some kind of limit at which I do this here - I am still not sure if it is an upper or lower bound that is the right threshold on the power spectrum.

        Args:
            y: torch.Tensor, the true values
            y_pred: torch.Tensor, the predicted values
        """

        # assert that y_true has 3 dimensions
        assert y_true.dim() == 3
        assert y_pred.dim() == 3

        if y_true.size(-1) == 96 * 144:

            y_true = torch.reshape(y_true, (y_true.size(0), y_true.size(1), 96, 144))
            y_pred = torch.reshape(y_pred, (y_pred.size(0), y_pred.size(1), 96, 144))

            # NOTE:(seb) here we are doing a zonal spectrum!
            # calculate the spectra of the true values
            # note we calculate the spectra across space, and then take the mean across the batch
            fft_true = torch.mean(torch.abs(torch.fft.rfft(y_true[:, :, :], dim=3)), dim=0)
            # calculate the spectra of the predicted values
            fft_pred = torch.mean(torch.abs(torch.fft.rfft(y_pred[:, :, :], dim=3)), dim=0)

        elif y_true.size(-1) == 6250:

            y_true = y_true
            y_pred = y_pred

            # NOTE:(seb) here we are doing a very dodgy spectral loss, since there is no spatial ordering to the unstructured grid and hence array...beware...
            # calculate the spectra of the true values
            # note we calculate the spectra across space, and then take the mean across the batch
            fft_true = torch.mean(torch.abs(torch.fft.rfft(y_true[:, :, :], dim=2)), dim=0)
            # calculate the spectra of the predicted values
            fft_pred = torch.mean(torch.abs(torch.fft.rfft(y_pred[:, :, :], dim=2)), dim=0)
        else:
            raise ValueError("The size of the input is a surprise, and should be addressed here.")

        # Calculate the power spectrum
        spectral_loss = torch.abs(fft_pred - fft_true)

        spectral_loss = torch.mean(spectral_loss[..., :])
        # print('what is the shape of the spectral loss?', spectral_loss)

        return spectral_loss

    def get_temporal_spectral_loss(self, x, y_true, y_pred):
        """
        Calculate the temporal spectra (frequency domain) of the true values compared to the predicted values. This
        needs to look at the power spectra through time per grid cell of predicted and true values.

        Args:
            x: torch.Tensor, the input values, past timesteps
            y_true: torch.Tensor, the true value of the timestep we predict
            y_pred: torch.Tensor, the predicted values of the timestep we predict
        """

        # unsqueeze y_true and y_pred along the time axis, so that they go from (batch_size, num_vars, coords) to (batch_size, 1, num_vars, coords)
        # where coords can be 6250, icosahedral, or 96, 144 in the case where we still have regular data
        y_true = y_true.unsqueeze(1)
        y_pred = y_pred.unsqueeze(1)

        # concatenate x and y_true along the time axis
        obs = torch.cat((x, y_true), dim=1)
        pred = torch.cat((x, y_pred), dim=1)

        # calculate the spectra of the true values along the time dimension, and then take the mean across the batch
        fft_true = torch.mean(torch.abs(torch.fft.rfft(obs, dim=1)), dim=0)
        # calculate the spectra of the predicted values along the time dimension, and then take the mean across the batch
        fft_pred = torch.mean(torch.abs(torch.fft.rfft(pred, dim=1)), dim=0)

        # Calculate the power spectrum

        # compute the distance between the losses...
        temporal_spectral_loss = torch.abs(fft_pred - fft_true)

        # the shape here is (time/2 + 1, num_vars, coords)

        # average across all frequencies, variables and coordinates...
        temporal_spectral_loss = torch.mean(temporal_spectral_loss[:, :, :])

        return temporal_spectral_loss

    def connectivity_reg_complete(self):
        """
        Calculate the connectivity constraint, ie the sum of all the distances.

        inside each clusters.
        Not used yet - could be interesting :)
        """
        c = torch.tensor([0.0])
        w = self.model.module.autoencoder.get_w_encoder()
        d = self.data.distances
        for i in self.d:
            for k in self.d_z:
                c = c + torch.sum(torch.outer(w[i, :, k], w[i, :, k]) * d)
        return self.hp.reg_coeff_connect * c

    def connectivity_reg(self, ratio: float = 0.0005):
        """Calculate a connectivity regularisation only on a subsample of the complete data."""
        c = torch.tensor([0.0])
        w = self.model.module.autoencoder.get_w_encoder()
        n = int(self.d_x * ratio)
        points = np.random.choice(np.arange(self.d_x), n)

        if n <= 1:
            raise ValueError(
                "You should use a higher value for the ratio of \
                             considered points for the connectivity constraint"
            )

        # fixed here to remove self.data.coordinates, now coordinates is direct
        for d in range(self.d):
            for k in range(self.d_z):
                for i, c1 in enumerate(self.coordinates[points]):
                    for j, c2 in enumerate(self.coordinates[points]):
                        if i > j:
                            dist = distance.geodesic(c1, c2).km
                            c = c + w[d, i, k] * w[d, j, k] * dist
        return self.hp.reg_coeff_connect * c

    # Here I am going to add some functions which will seek to save predictions and true values
    # And also to complete autoregressive rollout every so often...
    # I am going to add this to the train_step and valid_step functions.

    # NOTE:(seb) rewrite so that we make the iters again!
    def autoregress_prediction_original(self, valid: bool = False, timesteps: int = 120):
        """
        Calculate the MSE and SMAPE between X_{t+1} and X_hat_{t+1}. We also do an autoregressive lead out for set
        number of timesteps, with a default of 120 timesteps of rollout.

        Args:
            valid: bool, whether we are operating on the validation data
            timesteps: int, the number of timesteps to predict into the future autoregressively
        """

        self.model.eval()

        if not valid:

            # Make the iterator again, since otherwise we have iterated through it already...
            train_dataloader = iter(self.datamodule.train_dataloader())
            x, y = next(train_dataloader)

            # NOTE:(seb) what we had before, 29/5/24:
            # x, y = next(self.data_loader_train)

            x = torch.nan_to_num(x)
            y = torch.nan_to_num(y)
            y = y[:, 0]
            z = None

            # print("First up, I will do the reconstruction effort")
            # NOTE:(seb) I need to just estimate whether the model is doing better using the nll!
            nll, recons, kl, y_pred_recons = self.get_nll(x, y, z)

            # ensure these are correct
            y_pred, y, z, pz_mu, pz_std = self.model.module.predict(x, y)

            # Here we predict, but taking 100 samples from the latents
            # TODO: make this into an argument
            samples_from_xs, samples_from_zs, y = self.model.module.predict_sample(x, y, 100)

            # make a copy of y_pred, which is a tensor
            x_original = x.clone().detach()
            y_original = y.clone().detach()
            y_original_pred = y_pred.clone().detach()
            y_original_recons = y_pred_recons.clone().detach()

            # save these original values, x_original, y_orginal, y_original_pred
            np.save(os.path.join(self.hp.exp_path, "train_x_ar_0.npy"), x_original.detach().cpu().numpy())
            np.save(os.path.join(self.hp.exp_path, "train_y_ar_0.npy"), y_original.detach().cpu().numpy())
            np.save(os.path.join(self.hp.exp_path, "train_y_pred_ar_0.npy"), y_original_pred.detach().cpu().numpy())
            np.save(os.path.join(self.hp.exp_path, "train_y_recons_0.npy"), y_original_recons.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "train_encoded_z_ar_0.npy"), z.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "train_pz_mu_ar_0.npy"), pz_mu.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "train_pz_std_ar_0.npy"), pz_std.detach().cpu().numpy())

            # saving the samples
            np.save(os.path.join(self.hp.exp_path, "train_samples_from_xs.npy"), samples_from_xs.detach().cpu().numpy())
            np.save(os.path.join(self.hp.exp_path, "train_samples_from_zs.npy"), samples_from_zs.detach().cpu().numpy())

            # Now doing the autoregressive rollout...
            # TODO: implement the autoregressive rollout and also take samples
            for i in range(1, timesteps):

                # assert that x_original and x are the same
                if i == 1:
                    assert torch.allclose(x_original, x)

                # remove the first timestep, so now we have (tau - 1) timesteps,
                # then append the prediction
                x = x[:, 1:, :, :]

                x = torch.cat([x, y_pred.unsqueeze(1)], dim=1)

                # then predict the next timestep
                # y at this point is pointless!!!
                y_pred, y, z, pz_mu, pz_std = self.model.module.predict(x, y)

                assert i != 0

                np.save(os.path.join(self.hp.exp_path, f"train_x_ar_{i}.npy"), x.detach().cpu().numpy())
                np.save(os.path.join(self.hp.exp_path, f"train_y_ar_{i}.npy"), y.detach().cpu().numpy())
                np.save(os.path.join(self.hp.exp_path, f"train_y_pred_ar_{i}.npy"), y_pred.detach().cpu().numpy())
                # np.save(os.path.join(self.hp.exp_path, f"train_encoded_z_ar_{i}.npy"), z.detach().cpu().numpy())
                # np.save(os.path.join(self.hp.exp_path, f"train_pz_mu_ar_{i}.npy"), pz_mu.detach().cpu().numpy())
                # np.save(os.path.join(self.hp.exp_path, f"train_pz_std_ar_{i}.npy"), pz_std.detach().cpu().numpy())

                # saving the samples here:
                # np.save(os.path.join(self.hp.exp_path, f"train_samples_from_xs_{i}.npy"), samples_from_xs.detach().cpu().numpy())
                # np.save(os.path.join(self.hp.exp_path, f"train_samples_from_zs_{i}.npy"), samples_from_zs.detach().cpu().numpy())

        else:

            # bs = np.min([self.data.n_valid, 1000])
            # Make the iterator again
            val_dataloader = iter(self.datamodule.val_dataloader())
            x, y = next(val_dataloader)

            # old, using existing dataloader
            # x, y = next(self.data_loader_val)

            y = torch.nan_to_num(y)
            x = torch.nan_to_num(x)
            y = y[:, 0]
            z = None

            # print("First up, I will do the reconstruction effort")
            nll, recons, kl, y_pred_recons = self.get_nll(x, y, z)

            # swap
            y_pred, y, z, pz_mu, pz_std = self.model.module.predict(x, y)

            # predict and take 100 samples too
            samples_from_xs, samples_from_zs, y = self.model.module.predict_sample(x, y, 100)

            # make a copy of y_pred, which is a tensor
            x_original = x.clone().detach()
            y_original = y.clone().detach()
            y_original_pred = y_pred.clone().detach()
            y_original_recons = y_pred_recons.clone().detach()

            # saving these
            np.save(os.path.join(self.hp.exp_path, "val_x_ar_0.npy"), x_original.detach().cpu().numpy())
            np.save(os.path.join(self.hp.exp_path, "val_y_ar_0.npy"), y_original.detach().cpu().numpy())
            np.save(os.path.join(self.hp.exp_path, "val_y_pred_ar_0.npy"), y_original_pred.detach().cpu().numpy())
            np.save(os.path.join(self.hp.exp_path, "val_y_recons_0.npy"), y_original_recons.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "val_encoded_z_ar_0.npy"), z.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "val_pz_mu_ar_0.npy"), pz_mu.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "val_pz_std_ar_0.npy"), pz_std.detach().cpu().numpy())

            # saving the samples
            np.save(os.path.join(self.hp.exp_path, "val_samples_from_xs.npy"), samples_from_xs.detach().cpu().numpy())
            np.save(os.path.join(self.hp.exp_path, "val_samples_from_zs.npy"), samples_from_zs.detach().cpu().numpy())

            for i in range(1, timesteps):

                # remove the first timestep, so now we have (tau - 1) timesteps
                x = x[:, 1:, :, :]
                x = torch.cat([x, y_pred.unsqueeze(1)], dim=1)

                # then predict the next timestep
                # y at this point is not being updated!!!
                y_pred, y, z, pz_mu, pz_std = self.model.module.predict(x, y)

                np.save(os.path.join(self.hp.exp_path, f"val_x_ar_{i}.npy"), x.detach().cpu().numpy())
                np.save(os.path.join(self.hp.exp_path, f"val_y_ar_{i}.npy"), y.detach().cpu().numpy())
                np.save(os.path.join(self.hp.exp_path, f"val_y_pred_ar_{i}.npy"), y_pred.detach().cpu().numpy())
                # np.save(os.path.join(self.hp.exp_path, f"val_encoded_z_ar_{i}.npy"), z.detach().cpu().numpy())
                # np.save(os.path.join(self.hp.exp_path, f"val_pz_mu_ar_{i}.npy"), pz_mu.detach().cpu().numpy())
                # np.save(os.path.join(self.hp.exp_path, f"val_pz_std_ar_{i}.npy"), pz_std.detach().cpu().numpy())

            # not finished, probably need to add some metrics here.
            # what are the shapes here?

        with torch.no_grad():

            # NOTE: just looking at some metrics...

            # I guess there are different MAEs that we can calculate here.
            # mae1 = torch.mean(torch.abs(y_original - y_original_pred))
            # print('Overall MAE:', mae1)

            # do the same for the MSE
            # mse1 = torch.mean((y_original - y_original_pred) ** 2)
            # print('Overall MSE:', mse1)

            # check
            mse = torch.mean(torch.sum(0.5 * (y_original - y_original_pred) ** 2, dim=2))
            # print("MSE:", mse)
            # print("MSE shape:", mse.shape)

            smape = torch.mean(
                torch.sum(2 * (y_original - y_original_pred).abs() / (y_original.abs() + y_original_pred.abs()), dim=2)
            )
            # print("SMAPE:", smape)
            # print()

        return mse.item(), smape.item(), y_original, y_original_pred, y_original_recons, x_original
