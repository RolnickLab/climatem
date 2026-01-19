import numpy as np
import torch
import torch.distributions as dist

from climatem.model.prox import monkey_patch_RMSprop
from climatem.model.utils import ALM
from climatem.plotting.plot_model_output import Plotter


class TrainingFluxnet:
    def __init__(
        self,
        model,
        datamodule,
        train_params,
        optim_params,
        plot_params,
        save_path,
        plots_path,
        accelerator,
        wandbname="unspecified",
    ):

        self.accelerator = accelerator
        self.model = model
        self.model.to(accelerator.device)
        self.datamodule = datamodule
        self.data_loader_train = iter(datamodule.train_dataloader)
        self.data_loader_val = iter(datamodule.val_dataloader)
        self.train_params = train_params
        self.optim_params = optim_params
        self.plot_params = plot_params
        self.save_path = save_path
        self.plots_path = plots_path
        self.wandbname = wandbname

        self.d_x = self.model.d_x
        self.d_z = self.model.d_z
        self.total_d = self.model.total_d

        self.patience_freq = 50
        self.patience = 1000
        self.best_valid_loss = np.inf
        self.iteration = 1
        self.logging_iter = 0
        self.converged = False
        self.thresholded = False
        self.ended = False

        # collection of lists to store relevant metrics
        self.train_loss_list = []
        self.train_sparsity_reg_list = []
        self.train_acyclic_cons_list = []
        self.train_sparsity_cons_list = []
        self.train_transition_var_list = []
        self.mu_sparsity_list = []
        self.gamma_sparsity_list = []
        self.mu_acyclic_list = []

        self.valid_loss_list = []
        self.valid_recons_list = []
        self.valid_sparsity_reg_list = []
        self.valid_acyclic_cons_list = []
        self.valid_sparsity_cons_list = []
        self.valid_transition_var_list = []

        self.logvar_transition_tt = []

        self.plotter = Plotter()

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
            self.acyclic_constraint_normalization = self.get_acyclicity_normalization().item()
            self.sparsity_normalization = self.total_d * self.total_d

    def train_with_QPM(self):  # noqa: C901
        """
        Optimize a problem under constraint using the Augmented Lagragian method (or QPM).

        We train in 3 phases: first with ALM, then until
        the likelihood remain stable, then continue after thresholding
        the adjacency matrix
        """

        self.accelerator.init_trackers(
            "gpu-code-wandb",
            # config=config,
            init_kwargs={"wandb": {"name": self.wandbname}},
        )

        self.ALM_sparsity = ALM(
            self.optim_params.sparsity_mu_init,
            self.optim_params.sparsity_mu_mult_factor,
            self.optim_params.sparsity_omega_gamma,
            self.optim_params.sparsity_omega_mu,
            self.optim_params.sparsity_h_threshold,
            self.optim_params.sparsity_min_iter_convergence,
        )

        self.ALM_acyclic = ALM(
            self.optim_params.acyclic_mu_init,
            self.optim_params.acyclic_mu_mult_factor,
            self.optim_params.acyclic_omega_gamma,
            self.optim_params.acyclic_omega_mu,
            self.optim_params.acyclic_h_threshold / self.acyclic_constraint_normalization,
            self.optim_params.acyclic_min_iter_convergence,
            # dim_gamma=(1,),
        )

        while self.iteration < self.train_params.max_iteration and not self.ended:

            # train and valid step
            # HERE MODIFY train_step()
            self.train_step()
            self.scheduler.step()

            if self.iteration % self.train_params.valid_freq == 0:
                self.logging_iter += 1
                self.valid_step()
                self.log_losses()

                if self.iteration % (self.plot_params.print_freq) == 0:
                    # altered to use the accelerator.log function
                    self.accelerator.log(
                        {
                            "loss_train": self.train_loss,
                            "recons_train": self.train_recons,
                            "loss_valid": self.valid_loss,
                            "recons_valid": self.valid_recons,
                        }
                    )

                else:
                    self.accelerator.log(
                        {
                            "loss_train": self.train_loss,
                            "recons_train": self.train_recons,
                            "loss_valid": self.valid_loss,
                            "recons_valid": self.valid_recons,
                        }
                    )

            if self.logging_iter > 0 and self.iteration % (self.plot_params.plot_freq) == 0:
                print(f"Plotting Iteration {self.iteration}")
                self.plotter.plot_sparsity_control(self)
                # trying to save coords and adjacency matrices
                # Todo propagate the path!
                torch.save(self.model.state_dict(), self.save_path / "model.pth")

                # try to use the accelerator.save function here
                self.accelerator.save_state(output_dir=self.save_path)

            if not self.converged:
                if self.iteration % self.train_params.valid_freq == 0:

                    self.ALM_sparsity.update(self.iteration, self.valid_sparsity_cons_list, self.valid_loss_list)
                    if self.iteration > 1000:
                        sparsity_converged = self.ALM_sparsity.has_converged
                    else:
                        sparsity_converged = False

                    self.ALM_acyclic.update(self.iteration, self.valid_acyclic_cons_list, self.valid_loss_list)
                    acyclic_converged = self.ALM_acyclic.has_converged

                    if self.ALM_acyclic.has_increased_mu or self.ALM_sparsity.has_increased_mu:
                        if self.optim_params.optimizer == "sgd":
                            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.train_params.lr)
                        elif self.optim_params.optimizer == "rmsprop":
                            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.train_params.lr)

                    self.converged = acyclic_converged & sparsity_converged

            else:
                # continue training without penalty method
                if not self.thresholded and self.iteration % self.patience_freq == 0:
                    if not self.has_patience(self.train_params.patience, self.valid_loss):
                        self.threshold()
                        self.patience = self.train_params.patience_post_thresh
                        self.best_valid_loss = np.inf
                # continue training after thresholding
                else:
                    if self.iteration % self.patience_freq == 0:
                        if not self.has_patience(self.train_params.patience_post_thresh, self.valid_loss):
                            self.ended = True

            self.iteration += 1

        if self.iteration >= self.train_params.max_iteration:
            self.threshold()

        # final plotting and printing
        self.plotter.plot_sparsity_control(self, save=True)

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
            "valid_recons": self.valid_recons,
            "valid_sparsity_reg": self.valid_sparsity_reg,
            "valid_sparsity_cons": self.valid_sparsity_cons,
        }

        return valid_loss

    def train_step(self):  # noqa: C901

        self.model.train()

        try:
            x = next(self.data_loader_train)
            y = x[:, : self.d_z]
            x = x[:, self.d_z :]
            x = torch.nan_to_num(x)
            y = torch.nan_to_num(y)
        except StopIteration:
            self.data_loader_train = iter(self.datamodule.train_dataloader)
            x = next(self.data_loader_train)
            y = x[:, : self.d_z]
            x = x[:, self.d_z :]
            x = torch.nan_to_num(x)
            y = torch.nan_to_num(y)

        recons, y_pred_recons, y_std_recons = self.get_nll(x, y)

        assert y_pred_recons.shape[0] == x.shape[0]
        assert y_pred_recons.shape[1] == self.total_d

        # compute regularisations constraints/penalties (sparsity and connectivity)
        if self.optim_params.use_sparsity_constraint:
            h_sparsity = self.get_sparsity_violation(
                lower_threshold=0.05, upper_threshold=self.optim_params.sparsity_upper_threshold
            )
            sparsity_reg = self.ALM_sparsity.gamma * h_sparsity + 0.5 * self.ALM_sparsity.mu * h_sparsity**2
            if self.optim_params.binarize_transition and h_sparsity == 0:
                h_variance = self.adj_transition_variance()
                sparsity_reg = self.ALM_sparsity.gamma * h_variance + 0.5 * self.ALM_sparsity.mu * h_variance**2
        else:
            sparsity_reg = self.get_regularisation()

        # compute constraints (acyclicity and orthogonality)
        h_acyclic = torch.as_tensor([0.0])
        if not self.converged:
            h_acyclic = self.get_acyclicity_violation()
        acyclicity_reg = self.ALM_acyclic.gamma * h_acyclic + 0.5 * self.ALM_acyclic.mu * h_acyclic**2
        # sparsity_first_threshold

        loss = -recons + acyclicity_reg + sparsity_reg

        loss += self.optim_params.crps_coeff * self.get_crps_loss(x, y, y_pred_recons, y_std_recons)

        # if self.optim_params.sparsity_first_threshold is not None and h_sparsity>self.optim_params.sparsity_first_threshold:
        #     loss = - recons + sparsity_reg
        # elif self.optim_params.update_sparsity_after_acyclicity and h_acyclic < self.optim_params.acyclic_h_threshold:
        #     loss = - recons + sparsity_reg
        # elif self.optim_params.update_sparsity_after_acyclicity:
        # # compute total loss - here we are removing the sparsity regularisation as we are using the constraint here.
        #     loss = - recons + acyclicity_reg
        # else:
        #     loss = - recons + acyclicity_reg + sparsity_reg

        # backprop
        # mask_prev = self.model.mask.param.clone()
        # as recommended by https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#use-parameter-grad-none-instead-of-model-zero-grad-or-optimizer-zero-grad
        # self.optimizer.zero_grad()
        self.optimizer.zero_grad(set_to_none=True)
        # loss.backward()
        self.accelerator.backward(loss)
        # for name, param in self.model.named_parameters():
        #     if param.grad is not None and torch.isnan(param.grad).any():
        _, _ = (
            self.optimizer.step() if self.optim_params.optimizer == "rmsprop" else self.optimizer.step()
        ), self.train_params.lr

        self.train_loss = loss.item()
        self.train_recons = recons.item()
        self.train_sparsity_reg = sparsity_reg.item()
        self.train_acyclic_cons = h_acyclic.item()  # errors with .item() as it is a tensor

        # adding the sparsity constraint to the logs
        self.train_sparsity_cons = h_sparsity.item()  # .detach()
        self.train_transition_var = self.adj_transition_variance().item()

        # # NOTE: here we have the saving, prediction, and analysis of some metrics, which comes at every print_freq
        # # This can be cut if we want faster training...
        # print(f"[GPU] Peak allocated: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        # print(f"[GPU] Currently allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # Validation step here.
    def valid_step(self):
        self.model.eval()

        with torch.no_grad():
            # sample data
            try:
                x = next(self.data_loader_val)
                y = x[:, : self.d_z]
                x = x[:, self.d_z :]
                x = torch.nan_to_num(x)
                y = torch.nan_to_num(y)
            except StopIteration:
                self.data_loader_val = iter(self.datamodule.val_dataloader)
                x = next(self.data_loader_val)
                y = x[:, : self.d_z]
                x = x[:, self.d_z :]
                x = torch.nan_to_num(x)
                y = torch.nan_to_num(y)

            recons, y_pred_recons, y_std_recons = self.get_nll(x, y)

            # compute regularisations constraints/penalties (sparsity and connectivity)
            if self.optim_params.use_sparsity_constraint:
                h_sparsity = self.get_sparsity_violation(
                    lower_threshold=0.05, upper_threshold=self.optim_params.sparsity_upper_threshold
                )
                sparsity_reg = self.ALM_sparsity.gamma * h_sparsity + 0.5 * self.ALM_sparsity.mu * h_sparsity**2
                if self.optim_params.binarize_transition and h_sparsity == 0:
                    h_variance = self.adj_transition_variance()
                    sparsity_reg = self.ALM_sparsity.gamma * h_variance + 0.5 * self.ALM_sparsity.mu * h_variance**2
            else:
                sparsity_reg = self.get_regularisation()

            # compute constraints (acyclicity and orthogonality)
            h_acyclic = torch.as_tensor([0.0])
            if not self.converged:
                h_acyclic = self.get_acyclicity_violation()
            acyclicity_reg = self.ALM_acyclic.gamma * h_acyclic + 0.5 * self.ALM_acyclic.mu * h_acyclic**2
            # sparsity_first_threshold

            loss = -recons + acyclicity_reg + sparsity_reg

            loss += self.optim_params.crps_coeff * self.get_crps_loss(x, y, y_pred_recons, y_std_recons)

            # if self.optim_params.sparsity_first_threshold is not None and h_sparsity>self.optim_params.sparsity_first_threshold:
            #     loss = - recons + sparsity_reg
            # elif self.optim_params.update_sparsity_after_acyclicity and h_acyclic < self.optim_params.acyclic_h_threshold:
            #     loss = - recons + sparsity_reg
            # elif self.optim_params.update_sparsity_after_acyclicity:
            # # compute total loss - here we are removing the sparsity regularisation as we are using the constraint here.
            #     loss = - recons + acyclicity_reg
            # else:
            #     loss = - recons + acyclicity_reg + sparsity_reg

            self.valid_loss = loss.item()
            self.valid_recons = recons.item()
            self.valid_sparsity_reg = sparsity_reg.item()
            self.valid_acyclic_cons = h_acyclic.item()
            self.valid_sparsity_cons = h_sparsity.item()
            self.valid_transition_var = self.adj_transition_variance().item()

    def threshold(self):
        """
        Consider that the graph has been found.

        Convert it to a binary graph and fix it.
        """
        with torch.no_grad():
            thresholded_adj = (self.model.get_adj() > 0.5).type(torch.Tensor)
            self.model.mask = thresholded_adj
        self.thresholded = True
        print("Thresholding ================")

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

    def get_acyclicity_normalization(self) -> torch.Tensor:
        full_adj = torch.ones((self.d_x, self.d_x)) - torch.eye(self.d_x)
        return torch.trace(torch.linalg.matrix_exp(full_adj)) - self.d_x

    def log_losses(self):
        """Append in lists values of the losses and more."""
        # train
        self.train_loss_list.append(-self.train_loss)

        self.train_sparsity_reg_list.append(self.train_sparsity_reg)
        self.train_acyclic_cons_list.append(self.train_acyclic_cons)

        # valid
        self.valid_loss_list.append(-self.valid_loss)
        self.valid_recons_list.append(self.valid_recons)
        self.valid_sparsity_reg_list.append(self.valid_sparsity_reg)
        self.valid_acyclic_cons_list.append(self.valid_acyclic_cons)

        self.train_sparsity_cons_list.append(self.train_sparsity_cons)
        self.train_transition_var_list.append(self.train_transition_var)
        self.valid_sparsity_cons_list.append(self.valid_sparsity_cons)
        self.valid_transition_var_list.append(self.valid_transition_var)

        self.mu_sparsity_list.append(self.ALM_sparsity.mu)
        self.gamma_sparsity_list.append(self.ALM_sparsity.gamma)

        self.mu_acyclic_list.append(self.ALM_acyclic.mu)

        self.logvar_transition_tt.append(self.model.transition_model.logvar[0, 0].item())

    def get_nll(self, x, y) -> torch.Tensor:

        # this is just running the forward pass of LatentTSDCD...
        recons, px_mu, px_std = self.model(x, y)
        # print('what is len(self.model(arg)) with arguments', len(self.model(x, y, z, self.iteration)))
        return recons, px_mu, px_std

    def get_regularisation(self) -> float:
        if self.iteration > self.optim_params.schedule_reg:
            adj = self.model.get_adj()
            reg = self.optim_params.reg_coeff * torch.norm(adj, p=1)
            # reg /= adj.numel()
        else:
            reg = torch.as_tensor([0.0])

        return reg

    def get_acyclicity_violation(self) -> torch.Tensor:
        if self.iteration > 0:
            adj = self.model.get_adj().view(self.total_d, self.total_d)[self.d_z :][:, self.d_z :]
            h = (torch.trace(torch.linalg.matrix_exp(adj)) - self.d_x) / self.acyclic_constraint_normalization
        else:
            h = torch.as_tensor([0.0])

        assert torch.is_tensor(h)

        return h

    def adj_transition_variance(self) -> float:
        adj = self.model.get_adj()
        h = torch.norm(adj - torch.square(adj), p=1) / self.sparsity_normalization
        assert torch.is_tensor(h)
        return h

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

            # If the sum_of_connections is greater than the upper threshold, then we have a violation
            if sum_of_connections > upper_threshold:
                constraint = sum_of_connections - upper_threshold

            # If the constraint is less than the lower threshold, then we also have a violation
            elif sum_of_connections < lower_threshold:
                constraint = lower_threshold - sum_of_connections

            # Otherwise, there is no penalty due to the constraint:
            else:
                constraint = torch.as_tensor([0.0])

            # print('constraint value, after I subtract a threshold, or whatever:', constraint)

            h = torch.max(constraint, torch.as_tensor([0.0]))

        else:
            h = torch.as_tensor([0.0])

        assert torch.is_tensor(h)

        return h

    def get_crps_loss(self, x, z, mu, sigma):
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

        y = torch.cat((z, x), dim=1)
        mu = mu
        sigma = sigma

        sy = (y - mu) / sigma
        forecast_dist = dist.Normal(0, 1)
        pdf = self._normpdf(sy)
        cdf = forecast_dist.cdf(sy)

        pi_inv = 1.0 / torch.sqrt(torch.as_tensor(torch.pi))

        # calculate the CRPS
        crps = sigma * (sy * (2.0 * cdf - 1.0) + 2.0 * pdf - pi_inv)

        # add together all the CRPS values and divide by the number of samples
        crps = torch.sum(crps) / y.size(0)

        return crps

    def _normpdf(self, x):
        """Probability density function of a univariate standard Gaussian distribution with zero mean and unit
        variance."""
        return (1.0 / torch.sqrt(torch.as_tensor(2.0 * torch.pi))) * torch.exp(-torch.square(x) / 2.0)
