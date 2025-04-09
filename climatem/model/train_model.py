# Adapting to do training across multiple GPUs with huggingface accelerate.
import numpy as np
import torch
import torch.distributions as dist

# we use accelerate for distributed training
from geopy import distance

# from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import ProfilerActivity

from climatem.model.dag_optim import compute_dag_constraint
from climatem.model.prox import monkey_patch_RMSprop
from climatem.model.utils import ALM
from climatem.plotting.plot_model_output import Plotter


class TrainingLatent:
    def __init__(
        self,
        model,
        datamodule,
        exp_params,
        gt_params,
        model_params,
        train_params,
        optim_params,
        plot_params,
        save_path,
        plots_path,
        best_metrics,
        d,
        accelerator,
        wandbname="unspecified",
        profiler=False,
        profiler_path="./log",
    ):
        # TODO: do we want to have the profiler as an argument? Maybe not, but useful to speed up the code
        self.accelerator = accelerator
        self.model = model
        self.model.to(accelerator.device)
        self.datamodule = datamodule
        self.data_loader_train = iter(datamodule.train_dataloader(accelerator=accelerator))
        self.data_loader_val = iter(datamodule.val_dataloader())
        self.coordinates = datamodule.coordinates
        self.exp_params = exp_params
        self.train_params = train_params
        self.optim_params = optim_params
        self.coefs_scheduler_spectra = (
            None
            if optim_params.scheduler_spectra is None
            else np.linspace(0, 1, len(optim_params.scheduler_spectra), endpoint=True)
        )

        self.plot_params = plot_params
        self.best_metrics = best_metrics
        self.save_path = save_path
        self.plots_path = plots_path
        self.wandbname = wandbname

        self.latent = exp_params.latent
        self.no_gt = gt_params.no_gt
        self.debug_gt_z = gt_params.debug_gt_z
        self.d_z = exp_params.d_z
        self.no_w_constraint = model_params.no_w_constraint

        self.d = d
        self.patience = train_params.patience
        self.best_valid_loss = np.inf

        self.batch_size = datamodule.hparams.batch_size
        self.tau = exp_params.tau
        self.future_timesteps = exp_params.future_timesteps
        self.d_x = exp_params.d_x
        self.lat = exp_params.lat
        self.lon = exp_params.lon
        self.instantaneous = model_params.instantaneous

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

        self.best_spatial_spectra_score = None

        self.plotter = Plotter()

        # if MULTI_GPU:
        #     print("I am using multiple GPUs!!")
        #     # setup_ddp()
        #     # DistributedSampler
        #     self.model = DDP(self.model)

        # I think this is just initialising a tensor of zeroes to store results in
        if self.instantaneous:
            self.adj_tt = torch.zeros(
                [
                    int(self.train_params.max_iteration / self.train_params.valid_freq),
                    self.tau + 1,
                    self.d * self.d_z,
                    self.d * self.d_z,
                ]
            )
        else:
            self.adj_tt = torch.zeros(
                [
                    int(self.train_params.max_iteration / self.train_params.valid_freq),
                    self.tau,
                    self.d * self.d_z,
                    self.d * self.d_z,
                ]
            )
        if not self.no_gt:
            self.adj_w_tt = torch.zeros(
                [int(self.train_params.max_iteration / self.train_params.valid_freq), self.d, self.d_x, self.d_z]
            )
        self.logvar_encoder_tt = []
        self.logvar_decoder_tt = []
        self.logvar_transition_tt = []

        # self.model.mask.fix(self.gt_dag)

        # optimizer
        if self.optim_params.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=self.train_params.lr)
        elif self.optim_params.optimizer == "rmsprop":
            monkey_patch_RMSprop(torch.optim.RMSprop)
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=self.train_params.lr)
        else:
            raise NotImplementedError(f"optimizer {self.optim_params.optimizer} is not implemented")
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.train_params.lr_scheduler_epochs, gamma=self.train_params.lr_scheduler_gamma
        )

        # prepare the model, optimizer, data loader, and scheduler using Accerate for distributed training
        print("Preparing all the models here!")
        self.data_loader_train, self.model, self.optimizer, self.scheduler = accelerator.prepare(
            self.data_loader_train, self.model, self.optimizer, self.scheduler
        )

        # compute constraint normalization
        with torch.no_grad():
            d = model.d * model.d_z
            full_adjacency = torch.ones((d, d)) - torch.eye(d)
            self.acyclic_constraint_normalization = compute_dag_constraint(full_adjacency).item()

            if self.latent:
                self.ortho_normalization = self.d_x * self.d_z
                self.sparsity_normalization = self.tau * self.d_z * self.d_z

    def train_with_QPM(self):  # noqa: C901
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
        #    name=...
        # # )

        # print("what is the cuda device count?", torch.cuda.device_count())
        # print("MULTI GPU?", MULTI_GPU)

        # TODO: Why config here?
        # config = self.hp
        self.accelerator.init_trackers(
            "gpu-code-wandb",
            # config=config,
            init_kwargs={"wandb": {"name": self.wandbname}},
        )

        # initialize ALM/QPM for orthogonality and acyclicity constraints
        self.ALM_ortho = ALM(
            self.optim_params.ortho_mu_init,
            self.optim_params.ortho_mu_mult_factor,
            self.optim_params.ortho_omega_gamma,
            self.optim_params.ortho_omega_mu,
            self.optim_params.ortho_h_threshold,
            self.optim_params.ortho_min_iter_convergence,
            dim_gamma=(self.d_z, self.d_z),
        )

        self.ALM_sparsity = ALM(
            self.optim_params.sparsity_mu_init,
            self.optim_params.sparsity_mu_mult_factor,
            self.optim_params.sparsity_omega_gamma,
            self.optim_params.sparsity_omega_mu,
            self.optim_params.sparsity_h_threshold,
            self.optim_params.sparsity_min_iter_convergence,
            dim_gamma=(1, 1),
        )

        if self.instantaneous:
            # here we add the acyclicity constraint if the instantaneous connections are interesting
            self.QPM_acyclic = ALM(
                self.optim_params.acyclic_mu_init,
                self.optim_params.acyclic_mu_mult_factor,
                self.optim_params.acyclic_omega_gamma,
                self.optim_params.acyclic_omega_mu,
                self.optim_params.acyclic_h_threshold,
                self.optim_params.acyclic_min_iter_convergence,
                dim_gamma=(1, 1),
            )

        if self.profiler:

            # we should have this function elsewhere. It is rarely used.
            def trace_handler(p):
                print("Printing profiler key averages from trace handler!")
                output_cpu = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
                output_cuda = p.key_averages().table(sort_by="cuda_time_total", row_limit=20)
                print(output_cpu)
                print(output_cuda)

            prof = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=5, warmup=5, active=1, repeat=1),
                # using the torch tensorboard handler
                # on_trace_ready=torch.profiler.export_chrome_trace(self.profiler_path),
                # on_trace_ready=trace_handler,
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
            )
            prof.start()
            # print out the output of the profiler

        while self.iteration < self.train_params.max_iteration and not self.ended:

            # train and valid step
            # HERE MODIFY train_step()
            self.train_step()
            self.scheduler.step()
            if self.profiler:
                prof.step()

            if self.iteration % self.train_params.valid_freq == 0:
                self.logging_iter += 1
                # HERE MODIFY valid_step()
                self.valid_step()
                self.log_losses()

                # log these metrics to wandb every print_freq...
                #  multiple metrics here...
                if self.iteration % (self.plot_params.print_freq) == 0:
                    # altered to use the accelerator.log function
                    self.accelerator.log(
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
                    self.accelerator.log(
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
                # TODO : the plotting frrequency is hard to control and unintuitive... update the code here
                if self.iteration % (self.plot_params.print_freq) == 0:
                    self.print_results()

            if self.logging_iter > 0 and self.iteration % (self.plot_params.plot_freq) == 0:
                print(f"Plotting Iteration {self.iteration}")
                self.plotter.plot_sparsity(self)
                # trying to save coords and adjacency matrices
                # Todo propagate the path!
                if not self.plot_params.savar:
                    self.plotter.save_coordinates_and_adjacency_matrices(self)
                torch.save(self.model.state_dict(), self.save_path / "model.pth")

                # try to use the accelerator.save function here
                self.accelerator.save_state(output_dir=self.save_path)

            if not self.converged:

                # train with penalty method
                # NOTE: here valid_freq is critical for updating the parameters of the ALM method!
                # this is easy to miss - perhaps we should implement another parameter for this.
                if self.iteration % self.train_params.valid_freq == 0:
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
                        if self.optim_params.optimizer == "sgd":
                            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.train_params.lr)
                        elif self.optim_params.optimizer == "rmsprop":
                            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.train_params.lr)

                    # Repeat for sparsity constraint?
                    if self.ALM_sparsity.has_increased_mu:
                        if self.optim_params.optimizer == "sgd":
                            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.train_params.lr)
                        elif self.optim_params.optimizer == "rmsprop":
                            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.train_params.lr)

                    if self.instantaneous:
                        self.QPM_acyclic.update(self.iteration, self.valid_acyclic_cons_list, self.valid_loss_list)
                        acyclic_converged = self.QPM_acyclic.has_converged
                        # TODO: add optimizer reinit
                        if self.QPM_acyclic.has_increased_mu:
                            if self.optim_params.optimizer == "sgd":
                                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.train_params.lr)
                            elif self.optim_params.optimizer == "rmsprop":
                                self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.train_params.lr)
                        self.converged = ortho_converged & acyclic_converged
                    else:
                        # self.converged = ortho_converged
                        self.converged = ortho_converged & sparsity_converged
            else:
                # continue training without penalty method
                if not self.thresholded and self.iteration % self.patience_freq == 0:
                    # self.plotter.plot(self, save=True)
                    if not self.has_patience(self.train_params.patience, self.valid_loss):
                        self.threshold()
                        self.patience = self.train_params.patience_post_thresh
                        self.best_valid_loss = np.inf
                        # self.plotter.plot(self, save=True)
                # continue training after thresholding
                else:
                    if self.iteration % self.patience_freq == 0:
                        # self.plotter.plot(self, save=True)
                        if not self.has_patience(self.train_params.patience_post_thresh, self.valid_loss):
                            self.ended = True

            self.iteration += 1

            # might want this for the profiler:
            # if self.profiler:
            #    prof.step()

        if self.iteration >= self.train_params.max_iteration:
            self.threshold()

        # final plotting and printing
        self.plotter.plot_sparsity(self, save=True)
        self.print_results()

        # wandb.finish()
        self.accelerator.end_training()

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
            print(f"x.shape {x.shape}")
            print(f"y.shape {y.shape}")
            x = torch.nan_to_num(x)
            y = torch.nan_to_num(y)
        except StopIteration:
            self.data_loader_train = iter(self.datamodule.train_dataloader(accelerator=self.accelerator))
            x, y = next(self.data_loader_train)
            x = torch.nan_to_num(x)
            y = torch.nan_to_num(y)

        # y = y[:, 0]
        z = None
        x_bis = torch.clone(x)
        y_pred_all = torch.clone(y)
        nll = 0
        recons = 0
        kl = 0

        # also make the proper prediction, not the reconstruction as we do above
        # With multiple future timesteps we append the prediction to x and compute the nll of next timestep etc..
        # We add to the loss the sum multiplied by the decay in future timesteps
        # we have to take care here to make sure that we have the right tensors with requires_grad
        for k in range(self.future_timesteps):
            nll_bis, recons_bis, kl_bis, y_pred_recons = self.get_nll(x_bis, y[:, k], z)
            nll += (self.optim_params.loss_decay_future_timesteps**k) * nll_bis
            recons += (self.optim_params.loss_decay_future_timesteps**k) * recons_bis
            kl += (self.optim_params.loss_decay_future_timesteps**k) * kl_bis
            y_pred, y_spare, z_spare, pz_mu, pz_std = self.model.predict(x_bis, y[:, k])
            y_pred_all[:, k] = y_pred
            x_bis = torch.cat((x_bis[:, 1:], y_pred.unsqueeze(1)), dim=1)
            print(f"y_pred_recons shape {y_pred_recons.shape}")
        del x_bis, y_pred, nll_bis, recons_bis, kl_bis

        print(f"y_pred_all.shape {y_pred_all.shape}")
        print(f"y.shape {y.shape}")
        assert y.shape == y_pred_all.shape

        # compute regularisations constraints/penalties (sparsity and connectivity)
        if self.optim_params.use_sparsity_constraint:
            h_sparsity = self.get_sparsity_violation(
                lower_threshold=0.05, upper_threshold=self.optim_params.sparsity_upper_threshold
            )
            sparsity_reg = self.ALM_sparsity.gamma * h_sparsity + 0.5 * self.ALM_sparsity.mu * h_sparsity**2
        else:
            sparsity_reg = self.get_regularisation()
        connect_reg = torch.tensor([0.0])
        if self.exp_params.latent and self.optim_params.reg_coeff_connect > 0:
            # TODO: might be interesting to explore this
            connect_reg = self.connectivity_reg()

        # compute constraints (acyclicity and orthogonality)
        h_acyclic = torch.tensor([0.0])
        if self.instantaneous and not self.converged:
            h_acyclic = self.get_acyclicity_violation()
        h_ortho = self.get_ortho_violation(self.model.autoencoder.get_w_decoder())

        # compute total loss - here we are removing the sparsity regularisation as we are using the constraint here.
        loss = nll + connect_reg + sparsity_reg
        if not self.no_w_constraint:
            loss = loss + torch.sum(self.ALM_ortho.gamma @ h_ortho) + 0.5 * self.ALM_ortho.mu * torch.sum(h_ortho**2)
        if self.instantaneous:
            loss = loss + 0.5 * self.QPM_acyclic.mu * h_acyclic**2

        # need to be superbly careful here that we are really using predictions, not the reconstruction
        # I was hoping to do this with no_grad, but I do actually need it for the crps loss.
        crps = 0
        spectral_loss = 0
        for k in range(self.future_timesteps):
            px_mu, px_std = self.model.predict_pxmu_pxstd(torch.cat((x[:, k:], y_pred_all[:, :k]), dim=1), y[:, k])
            crps += (self.optim_params.loss_decay_future_timesteps**k) * self.get_crps_loss(y[:, k], px_mu, px_std)
            spectral_loss += (self.optim_params.loss_decay_future_timesteps**k) * self.get_spatial_spectral_loss(
                y[:, k], y_pred_all[:, k], take_log=True
            )

        temporal_spectral_loss = self.get_temporal_spectral_loss(x, y, y_pred_all)

        # add the spectral loss to the loss
        if self.optim_params.scheduler_spectra is None:
            loss = (
                loss
                + self.optim_params.crps_coeff * crps
                + self.optim_params.spectral_coeff * spectral_loss
                + self.optim_params.temporal_spectral_coeff * temporal_spectral_loss
            )
        else:
            print(
                f"scheduling spectrum coefficient at iterations {self.optim_params.scheduler_spectra} at coefficients {self.coefs_scheduler_spectra}!!"
            )
            coef = 0
            update_coef = False
            for new_coef, iter_schedule in zip(self.coefs_scheduler_spectra, self.optim_params.scheduler_spectra):
                update_coef = self.iteration >= iter_schedule and not update_coef
                if update_coef:
                    coef = new_coef
                if self.iteration == iter_schedule:
                    print(f"Updating spectral coefficient to {coef} at iteration {self.iteration}!!")
            loss = (
                loss
                + self.optim_params.crps_coeff * crps
                + coef
                * (
                    self.optim_params.spectral_coeff * spectral_loss
                    + self.optim_params.temporal_spectral_coeff * temporal_spectral_loss
                )
            )

        # backprop
        # mask_prev = self.model.mask.param.clone()
        # as recommended by https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
        # self.optimizer.zero_grad()
        self.optimizer.zero_grad(set_to_none=True)

        # loss.backward()
        self.accelerator.backward(loss)

        _, _ = (
            self.optimizer.step() if self.optim_params.optimizer == "rmsprop" else self.optimizer.step()
        ), self.train_params.lr

        # projection of the gradient for w
        if self.model.autoencoder.use_grad_project and not self.no_w_constraint:
            with torch.no_grad():
                self.model.autoencoder.get_w_decoder().clamp_(min=0.0)
            assert torch.min(self.model.autoencoder.get_w_decoder()) >= 0.0

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

        if self.iteration % self.plot_params.print_freq == 0:

            np.save(self.save_path / "x_true_recons_train.npy", x.cpu().detach().numpy())
            np.save(self.save_path / "y_true_recons_train.npy", y.cpu().detach().numpy())
            np.save(self.save_path / "y_pred_recons_train.npy", y_pred_all.cpu().detach().numpy())

            # also carry out autoregressive predictions
            mse, smape, y_original, y_original_pred, y_original_recons, x_original = (
                self.autoregress_prediction_original(valid=False, timesteps=10)
            )

            # also try the particle filtering approach
            # final_particles = self.particle_filter(x_original, y_original, num_particles=100, timesteps=120)

            self.train_mae_recons = torch.mean(torch.abs(y_original_recons - y_original)).item()
            self.train_mae_pred = torch.mean(torch.abs(y_original_pred - y_original)).item()
            self.train_mae_persistence = torch.mean(torch.abs(y_original - x_original[:, -1, :, :])).item()

            self.train_mse_recons = torch.mean((y_original_recons - y_original) ** 2).item()
            self.train_mse_pred = torch.mean((y_original_pred - y_original) ** 2).item()
            self.train_mse_persistence = torch.mean((y_original - x_original[:, -1, :, :]) ** 2).item()

            # include the variance of the predictions
            self.train_var_original = torch.var(y_original)
            self.train_var_recons = torch.var(y_original_recons)
            self.train_var_pred = torch.var(y_original_pred)

            # including per variable metrics, for when we train in the 4 variable case.
            if self.d == 3:
                self.train_mae_recons_1 = torch.mean(torch.abs(y_original_recons[:, 0] - y_original[:, 0])).item()
                self.train_mae_recons_2 = torch.mean(torch.abs(y_original_recons[:, 1] - y_original[:, 1])).item()
                self.train_mae_recons_3 = torch.mean(torch.abs(y_original_recons[:, 2] - y_original[:, 2])).item()
                self.train_mae_recons_4 = 0

                self.train_mae_pred_1 = torch.mean(torch.abs(y_original_pred[:, 0] - y_original[:, 0])).item()
                self.train_mae_pred_2 = torch.mean(torch.abs(y_original_pred[:, 1] - y_original[:, 1])).item()
                self.train_mae_pred_3 = torch.mean(torch.abs(y_original_pred[:, 2] - y_original[:, 2])).item()
                self.train_mae_pred_4 = 0
            else:
                self.train_mae_recons_1 = 0
                self.train_mae_recons_2 = 0
                self.train_mae_recons_3 = 0
                self.train_mae_recons_4 = 0

                self.train_mae_pred_1 = 0
                self.train_mae_pred_2 = 0
                self.train_mae_pred_3 = 0
                self.train_mae_pred_4 = 0

            # choose a random integer in self.batch_size, setting a seed for this
            np.random.seed(0)

            # sample = np.random.randint(0, self.batch_size)

            # Plotting the predictions for three different samples, including the reconstructions and the true values
            # if the shape of the data is icosahedral, we can plot like this:
            if not self.plot_params.savar and (self.d == 1 or self.d == 2 or self.d == 3 or self.d == 4):
                self.plotter.plot_compare_predictions_icosahedral(
                    x_past=x_original[:, -1, :, :].cpu().detach().numpy(),
                    y_true=y_original.cpu().detach().numpy(),
                    y_recons=y_original_recons.cpu().detach().numpy(),
                    y_hat=y_original_pred.cpu().detach().numpy(),
                    sample=np.random.randint(0, self.batch_size),
                    coordinates=self.coordinates,
                    path=self.plots_path,
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
                    coordinates=self.coordinates,
                    path=self.plots_path,
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
                    coordinates=self.coordinates,
                    path=self.plots_path,
                    iteration=self.iteration,
                    valid=False,
                    plot_through_time=True,
                )
            else:
                print("Not plotting predictions.")

        # note that this has been changed to y_pred_recons
        # return x, y, y_pred_all

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

            # y = y[:, 0]
            # NOTE: sh, z here is if we have a ground truth
            z = None
            x_bis = torch.clone(x)
            y_pred_all = torch.clone(y)
            nll = 0
            recons = 0
            kl = 0

            # also make the proper prediction, not the reconstruction as we do above
            # With multiple future timesteps we append the prediction to x and compute the nll of next timestep etc..
            # We add to the loss the sum multiplied by the decay in future timesteps
            # we have to take care here to make sure that we have the right tensors with requires_grad
            for k in range(self.future_timesteps):
                nll_bis, recons_bis, kl_bis, y_pred_recons = self.get_nll(x_bis, y[:, k], z)
                nll += (self.optim_params.loss_decay_future_timesteps**k) * nll_bis
                recons += (self.optim_params.loss_decay_future_timesteps**k) * recons_bis
                kl += (self.optim_params.loss_decay_future_timesteps**k) * kl_bis
                y_pred, y_spare, z_spare, pz_mu, pz_std = self.model.predict(x_bis, y[:, k])
                y_pred_all[:, k] = y_pred
                x_bis = torch.cat((x_bis[:, 1:], y_pred.unsqueeze(1)), dim=1)
                print(f"y_pred_recons shape {y_pred_recons.shape}")
            del x_bis, y_pred, nll_bis, recons_bis, kl_bis

            # compute regularisations (sparsity and connectivity)
            sparsity_reg = self.get_regularisation()
            connect_reg = torch.tensor([0.0])
            if self.exp_params.latent and self.optim_params.reg_coeff_connect > 0:
                # what is happening here between connectivity_reg and connectivity_reg_complete? See below.
                connect_reg = self.connectivity_reg()

            # compute constraints (acyclicity and orthogonality)
            h_acyclic = torch.tensor([0.0])
            # h_ortho = torch.tensor([0.])
            if self.instantaneous and not self.converged:
                h_acyclic = self.get_acyclicity_violation()
            h_ortho = self.get_ortho_violation(self.model.autoencoder.get_w_decoder())

            h_sparsity = self.get_sparsity_violation(
                lower_threshold=0.05, upper_threshold=self.optim_params.sparsity_upper_threshold
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
            self.valid_acyclic_cons = h_acyclic  # .item()

            # adding the sparsity constraint to the logs
            self.valid_sparsity_cons = h_sparsity  # .detach()

        # NOTE: here we have the saving, prediction, and analysis of some metrics, which comes at every print_freq
        # This can be cut if we want faster training...

        if self.iteration % self.plot_params.print_freq == 0:

            np.save(self.save_path / "x_true_recons_val.npy", x.cpu().detach().numpy())
            np.save(self.save_path / "y_true_recons_val.npy", y.cpu().detach().numpy())
            np.save(self.save_path / "y_pred_recons_val.npy", y_pred_all.cpu().detach().numpy())

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

            # include the variance of the predictions
            self.val_var_original = torch.var(y_original)
            self.val_var_recons = torch.var(y_original_recons)
            self.val_var_pred = torch.var(y_original_pred)

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

            if not self.plot_params.savar and (self.d == 1 or self.d == 2 or self.d == 3 or self.d == 4):
                self.plotter.plot_compare_predictions_icosahedral(
                    x_past=x_original[:, -1, :, :].cpu().detach().numpy(),
                    y_true=y_original.cpu().detach().numpy(),
                    y_recons=y_original_recons.cpu().detach().numpy(),
                    y_hat=y_original_pred.cpu().detach().numpy(),
                    sample=np.random.randint(0, self.batch_size),
                    coordinates=self.coordinates,
                    path=self.plots_path,
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
                    coordinates=self.coordinates,
                    path=self.plots_path,
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
                    coordinates=self.coordinates,
                    path=self.plots_path,
                    iteration=self.iteration,
                    valid=True,
                    plot_through_time=True,
                )

        # return x, y, y_pred_all

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
            thresholded_adj = (self.model.get_adj() > 0.5).type(torch.Tensor)
            self.model.mask.fix(thresholded_adj)
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

        self.adj_tt[int(self.iteration / self.train_params.valid_freq)] = (
            self.model.get_adj()
        )  # .cpu().detach().numpy()
        w = self.model.autoencoder.get_w_decoder()  # .cpu().detach().numpy()
        if not self.no_gt:
            self.adj_w_tt[int(self.iteration / self.train_params.valid_freq)] = w

        # here we just plot the first element of the logvar_decoder and logvar_encoder
        self.logvar_decoder_tt.append(self.model.autoencoder.logvar_decoder[0].item())
        self.logvar_encoder_tt.append(self.model.autoencoder.logvar_encoder[0].item())
        self.logvar_transition_tt.append(self.model.transition_model.logvar[0, 0].item())

    def print_results(self):
        """Print values of many variable: losses, constraint violation, etc.
        at the frequency self.plot_params.print_freq"""

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
        elbo, recons, kl, preds = self.model(x, y, z, self.iteration)
        # print('what is len(self.model(arg)) with arguments', len(self.model(x, y, z, self.iteration)))
        return -elbo, recons, kl, preds

    def get_regularisation(self) -> float:
        if self.iteration > self.optim_params.schedule_reg:
            adj = self.model.get_adj()
            reg = self.optim_params.reg_coeff * torch.norm(adj, p=1)
            # reg /= adj.numel()
        else:
            reg = torch.tensor([0.0])

        return reg

    def get_acyclicity_violation(self) -> torch.Tensor:
        if self.iteration > 0:
            adj = self.model.get_adj()[-1].view(self.d * self.d_z, self.d * self.d_z)
            h = compute_dag_constraint(adj) / self.acyclic_constraint_normalization
        else:
            h = torch.tensor([0.0])

        assert torch.is_tensor(h)

        return h

    def get_ortho_violation(self, w: torch.Tensor) -> float:

        if self.iteration > self.optim_params.schedule_ortho:
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
        if self.iteration > self.optim_params.schedule_sparsity:

            # first get the adj
            adj = self.model.get_adj()

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

    def get_spatial_spectral_loss(self, y_true, y_pred, take_log=True):
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
        assert y_true.dim() == 3
        assert y_pred.dim() == 3

        if y_true.size(-1) == self.lat * self.lon:

            y_true = torch.reshape(y_true, (y_true.size(0), y_true.size(1), self.lat, self.lon))
            y_pred = torch.reshape(y_pred, (y_pred.size(0), y_pred.size(1), self.lat, self.lon))

            # calculate the spectra of the true values
            # note we calculate the spectra across space, and then take the mean across the batch
            fft_true = torch.mean(torch.abs(torch.fft.rfft(y_true, dim=3)), dim=0)
            # calculate the spectra of the predicted values
            fft_pred = torch.mean(torch.abs(torch.fft.rfft(y_pred, dim=3)), dim=0)

        elif y_true.size(-1) == self.d_x:

            y_true = y_true
            y_pred = y_pred

            # calculate the spectra of the true values
            # note we calculate the spectra across space, and then take the mean across the batch
            fft_true = torch.mean(torch.abs(torch.fft.rfft(y_true, dim=2)), dim=0)
            # calculate the spectra of the predicted values
            fft_pred = torch.mean(torch.abs(torch.fft.rfft(y_pred, dim=2)), dim=0)
        else:
            raise ValueError("The size of the input is a surprise, and should be addressed here.")

        if take_log:
            fft_true = torch.log(fft_true)
            fft_pred = torch.log(fft_pred)

        # Calculate the power spectrum
        spectral_loss = torch.abs(fft_pred - fft_true)
        if self.optim_params.fraction_highest_wavenumbers is not None:
            spectral_loss = spectral_loss[
                :, round(self.optim_params.fraction_highest_wavenumbers * fft_true.shape[1]) :
            ]
        if self.optim_params.fraction_lowest_wavenumbers is not None:
            spectral_loss = spectral_loss[:, : round(self.optim_params.fraction_lowest_wavenumbers * fft_true.shape[1])]

        spectral_loss = torch.mean(spectral_loss)
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
        temporal_spectral_loss = torch.mean(temporal_spectral_loss)

        return temporal_spectral_loss

    def connectivity_reg_complete(self):
        """
        Calculate the connectivity constraint, ie the sum of all the distances.

        inside each clusters.
        Not used yet - could be interesting :)
        """
        c = torch.tensor([0.0])
        w = self.model.autoencoder.get_w_encoder()
        d = self.data.distances
        for i in self.d:
            for k in self.d_z:
                c = c + torch.sum(torch.outer(w[i, :, k], w[i, :, k]) * d)
        return self.optim_params.reg_coeff_connect * c

    def connectivity_reg(self, ratio: float = 0.0005):
        """Calculate a connectivity regularisation only on a subsample of the complete data."""
        c = torch.tensor([0.0])
        w = self.model.autoencoder.get_w_encoder()
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
        return self.optim_params.reg_coeff_connect * c

    # Here I am going to add some functions which will seek to save predictions and true values
    # And also to complete autoregressive rollout every so often...
    # I am going to add this to the train_step and valid_step functions.

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

            # make an empty list to store the predictions
            predictions = []

            # Make the iterator again, since otherwise we have iterated through it already...
            train_dataloader = iter(self.datamodule.train_dataloader(accelerator=self.accelerator))
            x, y = next(train_dataloader)

            x = torch.nan_to_num(x)
            y = torch.nan_to_num(y)
            y = y[:, 0]
            z = None

            # print("First up, I will do the reconstruction effort")
            nll, recons, kl, y_pred_recons = self.get_nll(x, y, z)

            # ensure these are correct
            with torch.no_grad():
                y_pred, y, z, pz_mu, pz_std = self.model.predict(x, y)

                # Here we predict, but taking 100 samples from the latents
                # TODO: make this into an argument
                samples_from_xs, samples_from_zs, y = self.model.predict_sample(x, y, 10)

            # append the first prediction
            predictions.append(y_pred)

            # make a copy of y_pred, which is a tensor
            x_original = x.clone().detach()
            y_original = y.clone().detach()
            y_original_pred = y_pred.clone().detach()
            y_original_recons = y_pred_recons.clone().detach()

            # save these original values, x_original, y_orginal, y_original_pred
            np.save(self.save_path / "train_x_ar_0.npy", x_original.detach().cpu().numpy())
            np.save(self.save_path / "train_y_ar_0.npy", y_original.detach().cpu().numpy())
            np.save(self.save_path / "train_y_pred_ar_0.npy", y_original_pred.detach().cpu().numpy())
            np.save(self.save_path / "train_y_recons_0.npy", y_original_recons.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "train_encoded_z_ar_0.npy"), z.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "train_pz_mu_ar_0.npy"), pz_mu.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "train_pz_std_ar_0.npy"), pz_std.detach().cpu().numpy())

            # saving the samples
            np.save(self.save_path / "train_samples_from_xs.npy", samples_from_xs.detach().cpu().numpy())
            np.save(self.save_path / "train_samples_from_zs.npy", samples_from_zs.detach().cpu().numpy())

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
                with torch.no_grad():
                    y_pred, y, z, pz_mu, pz_std = self.model.predict(x, y)

                # append the prediction
                predictions.append(y_pred)

                assert i != 0

                np.save(self.save_path / f"train_x_ar_{i}.npy", x.detach().cpu().numpy())
                np.save(self.save_path / f"train_y_ar_{i}.npy", y.detach().cpu().numpy())
                np.save(self.save_path / f"train_y_pred_ar_{i}.npy", y_pred.detach().cpu().numpy())
                # np.save(os.path.join(self.hp.exp_path, f"train_encoded_z_ar_{i}.npy"), z.detach().cpu().numpy())
                # np.save(os.path.join(self.hp.exp_path, f"train_pz_mu_ar_{i}.npy"), pz_mu.detach().cpu().numpy())
                # np.save(os.path.join(self.hp.exp_path, f"train_pz_std_ar_{i}.npy"), pz_std.detach().cpu().numpy())

                # saving the samples here:
                # np.save(os.path.join(self.hp.exp_path, f"train_samples_from_xs_{i}.npy"), samples_from_xs.detach().cpu().numpy())
                # np.save(os.path.join(self.hp.exp_path, f"train_samples_from_zs_{i}.npy"), samples_from_zs.detach().cpu().numpy())

            # at the end of this for loop, make the prediction a tensor

            predictions = torch.stack(predictions, dim=1)
            # the resulting shape of this tensor is (batch_size, timesteps, num_vars, coords)

            print("What is the shape of the predictions, once I made it into a tensor?", predictions.shape)

            # then calculate the mean of the predictions along the timesteps
            y_pred_mean = torch.mean(predictions, dim=1)
            # calculate the variance of the predictions along the timesteps dimension
            y_pred_var = torch.var(predictions, dim=1)
            print("What is the shape of the mean of the predictions?", y_pred_mean.shape)
            print("What is the shape of the variance of the predictions?", y_pred_var.shape)

            # take the mean of the predictions along the batch and coordinates dimension:
            print(
                "What is the shape when I try to take the mean across the batch and coordinates:",
                torch.mean(y_pred_mean, dim=(0, 2)),
            )

            # Ok, well done me. Now actually, what I want to do is to compare the spatial spectra of the true values and the predicted values.
            # I will do this by calculating the spatial spectra of the true values and the predicted values, and then calculating a score between them.
            # This is a measure of how well the model is predicting the spatial spectra of the true values.

            # here I calculate the spatial spectra across the coordinates, then I average across the batch and across the timesteps
            fft_true = torch.mean(torch.abs(torch.fft.rfft(x_original[:, :, :, :], dim=3)), dim=(0, 1))

            # calculate the average spatial spectra of the individual predicted fields - I think this below is wrong
            fft_pred = torch.mean(torch.abs(torch.fft.rfft(predictions[:, :, :, :], dim=3)), dim=(0, 1))

            # calculate the difference between the true and predicted spatial spectra
            spatial_spectra_score = torch.abs(fft_pred - fft_true)
            # take the mean across the frequencies, the 1st dimension
            spatial_spectra_score = torch.mean(spatial_spectra_score, dim=1)

            print("Spatial spectra score, lower is better...should be a spectra for each var", spatial_spectra_score)

            # if this spatial_spectra_score is the lowest we have seen, then save the predictions
            if self.best_spatial_spectra_score is None:
                self.best_spatial_spectra_score = spatial_spectra_score

            # assert that self.best_spatial_spectra_score is not None
            assert self.best_spatial_spectra_score is not None

            # check if every element of spatial_spectra_score is less than the best_spatial_spectra_score:
            print(torch.all(spatial_spectra_score < self.best_spatial_spectra_score))

            print("new score", spatial_spectra_score)
            print("previous best score", self.best_spatial_spectra_score)

            if torch.all(spatial_spectra_score < self.best_spatial_spectra_score):
                print("The spatial spectra score is the best we have seen for all variables, I am in the if.")

                self.best_spatial_spectra_score = spatial_spectra_score
                print(f"Best spatial spectra score: {self.best_spatial_spectra_score}")

                # save the model in its current state
                print("Saving the model, since the spatial spectra score is the best we have seen for all variables.")
                torch.save(self.model.state_dict(), self.save_path / "best_model_for_average_spectra.pth")

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
            with torch.no_grad():
                y_pred, y, z, pz_mu, pz_std = self.model.predict(x, y)

                # predict and take 100 samples too
                samples_from_xs, samples_from_zs, y = self.model.predict_sample(x, y, 100)

            # make a copy of y_pred, which is a tensor
            x_original = x.clone().detach()
            y_original = y.clone().detach()
            y_original_pred = y_pred.clone().detach()
            y_original_recons = y_pred_recons.clone().detach()

            # saving these
            np.save(self.save_path / "val_x_ar_0.npy", x_original.detach().cpu().numpy())
            np.save(self.save_path / "val_y_ar_0.npy", y_original.detach().cpu().numpy())
            np.save(self.save_path / "val_y_pred_ar_0.npy", y_original_pred.detach().cpu().numpy())
            np.save(self.save_path / "val_y_recons_0.npy", y_original_recons.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "val_encoded_z_ar_0.npy"), z.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "val_pz_mu_ar_0.npy"), pz_mu.detach().cpu().numpy())
            # np.save(os.path.join(self.hp.exp_path, "val_pz_std_ar_0.npy"), pz_std.detach().cpu().numpy())

            # saving the samples
            np.save(self.save_path / "val_samples_from_xs.npy", samples_from_xs.detach().cpu().numpy())
            np.save(self.save_path / "val_samples_from_zs.npy", samples_from_zs.detach().cpu().numpy())

            for i in range(1, timesteps):

                # remove the first timestep, so now we have (tau - 1) timesteps

                x = x[:, 1:, :, :]
                x = torch.cat([x, y_pred.unsqueeze(1)], dim=1)

                with torch.no_grad():
                    # then predict the next timestep
                    y_pred, y, z, pz_mu, pz_std = self.model.predict(x, y)

                np.save(self.save_path / f"val_x_ar_{i}.npy", x.detach().cpu().numpy())
                np.save(self.save_path / f"val_y_ar_{i}.npy", y.detach().cpu().numpy())
                np.save(self.save_path / f"val_y_pred_ar_{i}.npy", y_pred.detach().cpu().numpy())
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

    def score_the_samples_for_spatial_spectra(self, y_true, y_pred_samples, num_samples=100):
        """
        Calculate the spatial spectra of the true values and the predicted values, and then calculate a score between
        them. This is a measure of how well the model is predicting the spatial spectra of the true values.

        Args:
            true_values: torch.Tensor, observed values in a batch
            y_pred: torch.Tensor, a selection of predicted values
            num_samples: int, the number of samples that have been taken from the model
        """

        # calculate the average spatial spectra of the true values, averaging across the batch
        print("y_true shape:", y_true.shape)
        fft_true = torch.mean(torch.abs(torch.fft.rfft(y_true, dim=3)), dim=0)
        # calculate the average spatial spectra of the individual predicted fields - I think this below is wrong
        print("y_pred shape:", y_pred_samples.shape)
        fft_pred = torch.mean(torch.abs(torch.fft.rfft(y_pred_samples, dim=3)), dim=0)

        # extend fft_true so it is the same value but extended to the same shape as fft_pred
        fft_true = fft_true.repeat(num_samples, 1, 1)

        # calculate the difference between the true and predicted spatial spectra
        spatial_spectra_score = torch.abs(fft_pred - fft_true)

        # then normalise all the values of spatial_spectra_score by the maximum value
        # this is to make sure that the score is between 0 and 1
        spatial_spectra_score = spatial_spectra_score / torch.max(spatial_spectra_score)

        # the do 1 - score to give the score to be increasing...
        spatial_spectra_score = 1 - spatial_spectra_score

        # score = ...

        return spatial_spectra_score

    def particle_filter(self, x, y, num_particles, timesteps=120):
        """Implement a particle filter to make a set of autoregressive predictions, where each created sample is
        evaluated by some score, and we do a particle filter to select only best samples to continue the autoregressive
        rollout."""

        particles = torch.randn(num_particles)
        weights = torch.ones(num_particles) / num_particles

        for _ in range(timesteps):
            # Prediction
            # make all the new predictions, taking samples from the latents
            _, samples_from_zs, y = self.model.predict_sample(x, y, 100)

            # then calculate the score of each of the samples
            # Update the weights, where we want the weights to increase as the score improves
            new_weights = weights * self.score_the_samples_for_spatial_spectra(y, samples_from_zs)
            new_weights /= new_weights.sum()

            # Resampling (e.g., systematic resampling)
            indices = torch.multinomial(new_weights, num_particles, replacement=True)
            selected_samples = samples_from_zs[indices]
            weights = torch.ones(num_particles) / num_particles

            # alternative here for rejection sampling, where we only keep the best samples
            # indices = torch.argsort(new_weights, descending=True)
            # particles = samples_from_zs[indices[:num_particles]]
            # weights = torch.ones(num_particles) / num_particles

            # store these selected_samples
            np.save(self.save_path / f"selected_samples_{_}.npy", selected_samples.detach().cpu().numpy())

            # then we are going to be passing the selected samples to the next timestep, so we need to make the input again
            # first drop the first value of x, then
            x = x[:, 1:, :, :]

            # then we need to append the selected samples to x, along the right axis
            x = torch.cat([x, selected_samples.unsqueeze(1)], dim=1)

            # then we are going back to the top of the loop

        return particles
