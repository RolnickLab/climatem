# Adapted from the original code for CDSD, Brouillard et al., 2024.
# Hierachical Model (x-> z-> z_global) or (x-> z1-> z2)
from collections import OrderedDict
from math import pi

import torch
import torch.distributions as distr
import torch.nn as nn
from torch.distributions import Distribution

euler_mascheroni = 0.57721566490153286060


class Mask(nn.Module):
    def __init__(
        self,
        d: int,
        d_x: int,
        tau: int,
        latent: bool,
        instantaneous: bool,
        fixed: bool = False,
        fixed_output_fraction: float = 1.0,
        nodiag: bool = False,
    ):
        super().__init__()

        self.d = d
        self.d_x = d_x
        self.tau = tau
        self.latent = latent
        self.instantaneous = instantaneous
        self.fixed = fixed
        self.fixed_output_fraction = fixed_output_fraction
        # Here we can just set what we want the output to be.
        self.fixed_output = None
        self.uniform = distr.uniform.Uniform(0, 1)

        # Here we could change how the mask is instantiated in the causal graph.
        if self.latent:
            if not nodiag:
                self.param = nn.Parameter(torch.ones((self.tau, d * d_x, d * d_x)) * 5)
                self.fixed_mask = torch.ones_like(self.param)
            else:
                param = torch.ones((self.tau, d * d_x, d * d_x))
                param[:, torch.arange(d * d_x), torch.arange(d * d_x)] = -1
                self.param = nn.Parameter(param * 5)
                self.fixed_mask = torch.ones_like(self.param)
                self.fixed_mask[:, torch.arange(self.fixed_mask.size(1)), torch.arange(self.fixed_mask.size(2))] = 0
            if self.instantaneous:
                # TODO: G[0] or G[-1]
                self.fixed_mask[-1, torch.arange(self.fixed_mask.size(1)), torch.arange(self.fixed_mask.size(2))] = 0
        else:
            if self.instantaneous:
                # initialize mask as log(mask_ij) = 1
                self.param = nn.Parameter(torch.ones((self.tau, d, d, d_x)) * 5)
                self.fixed_mask = torch.ones_like(self.param)
                # set diagonal 0 for G_t0
                self.fixed_mask[-1, torch.arange(self.fixed_mask.size(1)), torch.arange(self.fixed_mask.size(2))] = 0
                # TODO: set neighbors to 0
                # self.fixed_mask[:, :, :, d_x] = 0
            else:
                # initialize mask as log(mask_ij) = 1
                self.param = nn.Parameter(torch.ones((tau, d, d, d_x)) * 5)
                self.fixed_mask = torch.ones_like(self.param)

    def forward(self, b: int, tau: float = 1) -> torch.Tensor:
        """
        :param b: batch size
        :param tau: temperature constant for sampling
        """

        if not self.fixed:
            adj = gumbel_sigmoid(self.param, self.uniform, b, tau=tau)
            adj = adj * self.fixed_mask
            return adj
        else:
            # Here we declare we have a fixed output, and we can do something with it here.
            # What we are doing here is setting the number of ones in the mask to be fixed_output_fraction
            if self.fixed_output is None:
                # We are using a fixed mask of 1s, or a fraction of 1s.
                # Set a seed so we can keep the same fixed mask.
                torch.manual_seed(353)
                num_elements = self.tau * self.d_x * self.d_x
                num_ones = int(num_elements * self.fixed_output_fraction)

                # overwrite the fixed mask here
                self.fixed_mask = torch.zeros((self.tau, self.d_x, self.d_x))

                # here we are just selecting a random number of ones in the mask.
                indices = torch.multinomial(torch.ones(num_elements), num_ones, replacement=False)
                # Convert linear indices to 3D indices
                (
                    i,
                    j,
                    k,
                ) = torch.unravel_index(indices, (self.tau, self.d_x, self.d_x))
                self.fixed_mask[i, j, k] = 1

                return self.fixed_mask.repeat(b, 1, 1, 1)

            else:
                # here I am specifically setting the fixed_output to be the fixed_output
                # I set that in the __init__ function, and I can set it to whatever I want, of the right shape.
                return self.fixed_output.repeat(b, 1, 1, 1)

    def get_proba(self) -> torch.Tensor:
        if not self.fixed:
            return torch.sigmoid(self.param) * self.fixed_mask
        elif self.fixed_output is None:
            # changing to return fixed mask...
            return self.fixed_mask
        else:
            assert self.fixed_output is not None
            return self.fixed_output

    def fix(self, fixed_output):
        self.fixed_output = fixed_output
        self.fixed = True


class MixingMask(nn.Module):
    def __init__(self, d: int, d_x: int, d_z: int, gt_mask=None):
        super().__init__()
        if gt_mask is not None:
            self.param = (gt_mask > 0) * 10.0
        else:
            self.param = nn.Parameter(torch.ones(d, d_x, d_z) * 5)

    def forward(self, batch_size):
        param = self.param.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        mask = nn.functional.gumbel_softmax(param, tau=1)
        return mask


def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)


def gumbel_sigmoid(log_alpha, uniform, bs, tau=1):
    shape = tuple([bs] + list(log_alpha.size()))
    logistic_noise = sample_logistic(shape, uniform)

    return torch.sigmoid((log_alpha + logistic_noise) / tau)


class MLP(nn.Module):
    def __init__(self, num_layers: int, num_hidden: int, num_input: int, num_output: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_input = num_input
        self.num_output = num_output
        self.use_grad_project = True

        module_dict = OrderedDict()

        # create model layer by layer
        in_features = num_input
        out_features = num_hidden
        if num_layers == 0:
            out_features = num_output

        module_dict["lin0"] = nn.Linear(in_features, out_features)

        for layer in range(num_layers):
            in_features = num_hidden
            out_features = num_hidden

            if layer == num_layers - 1:
                out_features = num_output

            module_dict[f"nonlin{layer}"] = nn.LeakyReLU()
            module_dict[f"lin{layer+1}"] = nn.Linear(in_features, out_features)

        self.model = nn.Sequential(module_dict)

    def forward(self, x) -> torch.Tensor:
        return self.model(x)


class HierarchicalLatentTSDCD(nn.Module):
    """Differentiable Causal Discovery for time series with latent variables."""

    def __init__(
        self,
        num_layers: int,
        num_hidden: int,
        num_input: int,
        num_output: int,
        num_layers_mixing: int,
        num_hidden_mixing: int,
        position_embedding_dim: int,
        transition_param_sharing: bool,
        position_embedding_transition: int,
        coeff_kl: float,
        distr_z0: str,
        distr_encoder: str,
        distr_transition: str,
        distr_decoder: str,
        d: int,  # Number of variables
        d_x: int,  # Dimension of observations
        d_z: int,  # Dimension of latent space
        d_z_global: int,  # Dimension of global latent space, fixed as 1
        tau: int,  # Number of timesteps as input
        instantaneous: bool,
        nonlinear_mixing: bool,
        nonlinear_dynamics: bool,
        nonlinear_global_dynamics: bool,
        tied_w: bool = False,
        fixed: bool = False,
        fixed_output_fraction: float = 1.0,
        gev_learn_xi: bool = False,
        use_exogenous: bool = False,
        use_forced_latents: bool = False,
        d_y_co2: int = 0,
        d_y_aerosol: int = 0,
        d_y_ch4: int = 0,
        d_y_so2: int = 0,
        map_aerosol_to_climate=False,
        forcing_mse: bool = False,
        **args,
    ):
        """
        Args:
            num_layers: number of layers of each MLP
            num_hidden: number of hidden units of each MLP
            num_input: number of inputs of each MLP
            num_output: number of inputs of each MLP
            num_layer_mixing: number of layer for the autoencoder
            num_hidden_mixing: number of hidden units for the autoencoder
            coeff_kl: coefficient of the KL term

            distr_z0: distribution of the first z (gaussian)
            distr_encoder: distribution parametrized by the encoder (gaussian)
            distr_transition: distribution parametrized by the transition model (gaussian)
            distr_decoder: distribution parametrized by the decoder (gaussian)

            d: number of features
            d_x: number of grid locations
            d_z: number of latent variables
            d_z_global: number of global latent variables
            tau: size of the timewindow
            instantaneous: if True, models instantaneous connections

            no_gt: if True, do not use any ground-truth data (useful with realworld dataset)
            debug_gt_graph: if True, set the masks to the ground-truth graphes (gt_graph)
            debug_gt_z: if True, use directly the ground-truth z (gt_z sampled with the data)
            debug_gt_w: if True, set the matrices W to the ground-truth W (gt_w)
            gt_graph: Ground-truth graphes, only used if debug_gt_graph is True
            gt_w: Ground-truth W, only used if debug_gt_w is True

            # including the option for a fixed causal graph for experiments
            fixed: if True, fix the mask (in simple case to all ones)
            fixed_output_fraction: fraction of ones in the fixed
            gev_learn_xi: if True, GEV will take learned xi
        """
        super().__init__()

        # nn encoder hyperparameters
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_input = num_input
        self.num_output = num_output
        self.num_layers_mixing = num_layers_mixing
        self.num_hidden_mixing = num_hidden_mixing
        self.position_embedding_dim = position_embedding_dim
        self.transition_param_sharing = transition_param_sharing
        self.position_embedding_transition = position_embedding_transition
        self.coeff_kl = coeff_kl

        self.d = d
        self.d_x = d_x
        self.d_z = d_z
        self.d_z2 = d_z_global
        self.tau = tau
        self.instantaneous = instantaneous
        self.nonlinear_mixing = nonlinear_mixing
        self.nonlinear_dynamics = nonlinear_dynamics
        self.nonlinear_global_dynamics = nonlinear_global_dynamics
        self.tied_w = tied_w
        self.fixed = fixed
        self.fixed_output_fraction = fixed_output_fraction
        self.gev_learn_xi = gev_learn_xi

        if self.instantaneous:
            self.total_tau = tau + 1
        else:
            self.total_tau = tau

        # if self.no_gt:
        #     self.gt_w = None
        #     self.gt_graph = None
        # else:
        #     self.gt_w = torch.as_tensor(gt_w).double()
        #     self.gt_graph = torch.as_tensor(gt_graph).double()

        if distr_z0 == "gaussian":
            self.distr_z0 = torch.normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        if distr_transition == "gaussian":
            # use distr.normal.Normal so that we can sample from these distributions
            self.distr_transition = distr.normal.Normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        if distr_encoder == "gaussian":
            self.distr_encoder = torch.normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        if distr_decoder == "gev":
            self.distr_decoder = GEVDistribution

            if gev_learn_xi:
                # Learn a xi for each variable/grid point (or customize shape as needed)
                self.xi = nn.Parameter(torch.zeros(d, d_x))  # shape matches px_mu
            else:
                # Use fixed xi (e.g., Gumbel limit)
                self.xi = torch.tensor(0.0)

            self.gev_learn_xi = gev_learn_xi

        elif distr_decoder == "gaussian":
            self.distr_decoder = distr.normal.Normal
        else:
            raise NotImplementedError(f"Decoder distribution '{distr_decoder}' is not implemented.")

        # self.encoder_decoder = EncoderDecoder(self.d, self.d_x, self.d_z, self.nonlinear_mixing, 4, 1, self.debug_gt_w, self.gt_w, self.tied_w)
        if self.nonlinear_mixing:
            print("NON-LINEAR MIXING")
            # NOTE:(seb) using the noloop version of non-linear here to make it much faster.
            # Local autoencoder, encode z from x
            self.autoencoder = NonLinearAutoEncoderUniqueMLP_noloop(
                d,
                d_x,
                d_z,
                self.num_hidden_mixing,
                self.num_layers_mixing,
                tied=tied_w,
                embedding_dim=self.position_embedding_dim,
                gt_w=None,
            )
            # Global autoencoder, encode z_global from z
            self.autoencoder_global = NonLinearAutoEncoderUniqueMLP_noloop(
                d,
                d_z,
                self.d_z2,
                self.num_hidden_mixing,
                self.num_layers_mixing,
                tied=tied_w,
                embedding_dim=self.position_embedding_dim,
                gt_w=None,
            )
        else:
            # print('Using linear mixing')
            print("LINEAR MIXING")
            self.autoencoder = LinearAutoEncoder(d, d_x, d_z, tied=tied_w)
            self.autoencoder_global = LinearAutoEncoder(d, d_z, self.d_z2, tied=tied_w)

        # if debug_gt_w:
        #     self.decoder.w = gt_w
        self.use_exogenous = use_exogenous
        self.use_forced_latents = use_forced_latents
        d_y_co2 = d_y_co2 if self.use_exogenous else 0
        d_y_ch4 = d_y_ch4 if self.use_exogenous else 0
        d_y_aerosol = d_y_aerosol if self.use_exogenous else 0
        d_y_so2 = d_y_so2 if self.use_exogenous else 0

        self.d_y_aerosol = d_y_aerosol
        self.d_y_so2 = d_y_so2

        self.n_forced_latents_aerosol = 0
        self.n_forced_latents_so2 = 0

        self.forcing_order = ("co2", "ch4", "aerosol", "so2")
        if self.transition_param_sharing:
            # Local transition model, transit from z^{<t} and z2^{t}
            self.transition_model = TransitionModelParamSharing(
                self.d,
                self.d_z,
                self.d_z2,
                self.total_tau,
                self.nonlinear_dynamics,
                self.num_layers,
                self.num_hidden,
                self.num_output,
                self.position_embedding_transition,
                d_y_aerosol=d_y_aerosol,
                d_y_so2=d_y_so2,
                local_impact=map_aerosol_to_climate,
            )
        else:
            self.transition_model = TransitionModel(
                self.d,
                self.d_z,
                self.d_z2,
                self.total_tau,
                self.nonlinear_dynamics,
                self.num_layers,
                self.num_hidden,
                self.num_output,
                d_y_aerosol=d_y_aerosol,
                d_y_so2=d_y_so2,
                local_impact=map_aerosol_to_climate,
            )
        # Global transition model, single latent: no mask, no parameter-sharing
        self.transition_model_global = TransitionModelGlobal(
            self.d,
            self.d_z2,
            self.total_tau,
            self.nonlinear_global_dynamics,
            self.num_layers,
            self.num_hidden,
            self.num_output,
            d_y_co2=d_y_co2,
            d_y_ch4=d_y_ch4,
        )
        # print("We are setting the Mask here.")
        self.mask = Mask(
            d,
            d_z,
            self.total_tau,
            instantaneous=instantaneous,
            latent=True,
            fixed=fixed,
            fixed_output_fraction=fixed_output_fraction,
        )
        # if self.debug_gt_graph:
        #     if self.instantaneous:
        #         self.mask.fix(self.gt_graph)
        #     else:
        #         self.mask.fix(self.gt_graph[:-1])

        self.map_aerosol_to_climate = map_aerosol_to_climate
        self.gmst_mse = forcing_mse  # we used alias for gmst_mse

    def get_adj(self):
        """
        Returns: Matrices of the probabilities from which the masks linking the
        latent variables are sampled
        """
        return self.mask.get_proba()

    def encode(self, x, y):
        """Encode X and Y into latent variables Z."""
        b = x.size(0)
        z = torch.zeros(b, self.tau + 1, self.d, self.d_z)
        mu = torch.zeros(b, self.d, self.d_z)
        std = torch.zeros(b, self.d, self.d_z)

        # sample Zs

        # TODO: Can we remove this for loop?
        for i in range(self.d):
            # TODO: Can we remove this for loop?
            for t in range(self.tau):
                # q_mu, q_logvar = self.encoder_decoder(x[:, t, i], i, encoder=True)  # torch.matmul(self.W, x)
                q_mu, q_logvar = self.autoencoder(x[:, t, i], i, encode=True)
                # reparam trick - here we sample from a Gaussian...every time
                q_std = torch.exp(0.5 * q_logvar)
                z[:, t, i] = q_mu + q_std * self.distr_encoder(0, 1, size=q_mu.size())

            # q_mu, q_logvar = self.encoder_decoder(y[:, i], i, encoder=True)  # torch.matmul(self.W, x)

            q_mu, q_logvar = self.autoencoder(y[:, i], i, encode=True)
            q_std = torch.exp(0.5 * q_logvar)

            # # e.g. z[:, -2, i]
            # all_z_except_last = z[:, :-1, i].clone()
            # penultimate_z = z[:, -2, i].clone()

            # assert torch.mean(z[:, -1, i]) == 0.0

            # carry on
            z[:, -1, i] = q_mu + q_std * self.distr_encoder(0, 1, size=q_mu.size())
            # assert torch.all(penultimate_z == z[:, -2, i])
            # assert torch.all(all_z_except_last == z[:, :-1, i])

            mu[:, i] = q_mu
            std[:, i] = q_std

        return z, mu, std

    def encode_global(self, z):
        """
        Encode Z into latent variables Z2 (higher level).
        Args:
            z (Tensor): Shape (B, T, D_z), where T = tau + 1 corresponds to the
                temporal window [t - tau, ..., t - 1, t].

        Returns:
            z2 (Tensor): Shape (B, T, D_z), higher-level latent representation
                inferred from z at each time step.
            mu (Tensor): Shape (B, D_z), mean of the approximate posterior
                q(z2_t | z_t) at the final time step t.
            std (Tensor): Shape (B, D_z), standard deviation of the approximate
                posterior q(z2_t | z_t) at the final time step t.
        """
        b = z.size(0)
        z2 = torch.zeros(b, self.tau + 1, self.d, self.d_z2)
        mu = torch.zeros(b, self.d, self.d_z2)
        std = torch.zeros(b, self.d, self.d_z2)

        for i in range(self.d):
            for t in range(self.tau + 1):
                q_mu, q_logvar = self.autoencoder_global(z[:, t, i], i, encode=True)  # q_mu: shape (b, d_z)
                # reparam trick - here we sample from a Gaussian...every time
                q_std = torch.exp(0.5 * q_logvar)
                z2[:, t, i] = q_mu + q_std * self.distr_encoder(0, 1, size=q_mu.size())
            mu[:, i] = q_mu
            std[:, i] = q_std
        return z2, mu, std

    def transition_global(self, z2, y_co2=None, y_ch4=None):
        """
        Transition model for z2, fully conencted mask
        Args:
            z2 (Tensor): Shape (B, T, D_z2), where T = tau corresponds to the
                temporal window [t - tau, ..., t - 1].
            y_co2 (Tensor): Shape (B, 1) at time T=t

        Returns:
            mu (Tensor): Shape (B, D_z2), mean of the predictive distribution
                p(z2_t | z2_{<t}).
            std (Tensor): Shape (B, D_z2), standard deviation of the predictive
                distribution p(z2_t | z2_{<t}).
        """
        b = z2.size(0)
        mu = torch.zeros(b, self.d, self.d_z2)
        std = torch.zeros(b, self.d, self.d_z2)
        # transition2 doesn't consider parameter sharing because only a single latent
        # TODO Can we remove this for loop

        for i in range(self.d):
            pz_params = torch.zeros(b, self.d_z2, 1)
            for k in range(self.d_z2):
                pz_params[:, k] = self.transition_model_global(
                    z2[:, :, i][:, :, :, None], i, k, y_co2=y_co2, y_ch4=y_ch4
                )
            mu[:, i] = pz_params[:, :, 0]
            mu[:, i] = pz_params[:, :, 0]
            std[:, i] = torch.exp(0.5 * self.transition_model_global.logvar[i])

        return mu, std

    def transition(self, z, z_global, mask, y_aerosol=None, y_so2=None):
        """
        Transition model for z (sparse mask). z transit from its history and current z_global
        Args:
            z (Tensor): Shape (B, T, D_z), where T = tau corresponds to the
                temporal window [t - tau, ..., t - 1].
            z_global (Tensor): Shape (B, 1, D_z2), at the time step t.

        Returns:
            mu (Tensor): Shape (B, D_z), mean of the predictive distribution
                p(z_t | z_{<t}, z2_t).
            std (Tensor): Shape (B, D_z), standard deviation of the predictive
                distribution p(z_t | z_{<t}, z2_t).
        """
        b = z.size(0)
        mu = torch.zeros(b, self.d, self.d_z)
        std = torch.zeros(b, self.d, self.d_z)

        # TODO Can we remove this for loop
        # for d>1, z needs to reshaped to z (b, K, d*K),
        # so that for each target var, we can consider the impact from all other source variables
        for i in range(self.d):
            decoder_w = None

            if self.map_aerosol_to_climate and self.use_exogenous:
                decoder_w = self.autoencoder.get_w_decoder()[i]  # (1, dx, dz)

            if self.transition_param_sharing:
                pz_params = self.transition_model(
                    z[:, :, i][:, :, :, None],
                    z_global,
                    mask[:, :, i * self.d_z : (i + 1) * self.d_z],
                    i,
                    y_aerosol=y_aerosol,
                    y_so2=y_so2,
                    local_mapping=decoder_w,
                )
            else:
                pz_params = torch.zeros(b, self.d_z, 1)
                for k in range(self.d_z):
                    pz_params[:, k] = self.transition_model(
                        z[:, :, i][:, :, :, None],
                        z_global,
                        mask[:, :, i * self.d_z + k],
                        i,
                        k,
                        y_aerosol=y_aerosol,
                        y_so2=y_so2,
                        local_mapping=decoder_w,
                    )
            mu[:, i] = pz_params[:, :, 0]
            std[:, i] = torch.exp(0.5 * self.transition_model.logvar[i])

        # print("This is giving us the pz_mu and pz_std that we use later.")
        return mu, std

    def decode(self, z):
        """Decode x from z."""

        mu = torch.zeros(z.size(0), self.d, self.d_x)
        std = torch.zeros(z.size(0), self.d, self.d_x)

        # TODO: Can we remove this for loop
        for i in range(self.d):
            px_mu, px_logvar = self.autoencoder(z[:, i], i, encode=False)
            if px_mu.ndim == mu.ndim:  # In case of linear mixing with one variable, second dimension is too much
                # Check that linear autoencoder corresponds to PF when multi varia/bles
                px_mu = px_mu.squeeze()

            mu[:, i] = px_mu
            std[:, i] = torch.exp(0.5 * px_logvar)

        return mu, std

    def forward(self, x, y, gt_z, iteration, xi=None, y_co2=None, y_aerosol=None, y_ch4=None, y_so2=None):

        if iteration == 1:
            print(f"shape x: {x.shape}, y: {y.shape}")
            if y_co2 is not None and y_aerosol is not None:
                print(f"y_co2: {y_co2.shape}")  # (b, tau+1, 1)
                print(f"y_aerosol: {y_aerosol.shape}")  # (b, tau+1, 3072)
            if y_ch4 is not None and y_so2 is not None:
                print(f"y_ch4: {y_ch4.shape}")  # (b, tau+1, 1)
                print(f"y_so2: {y_so2.shape}")  # (b, tau+1, 3072)

        b = x.size(0)

        # sample Zs (based on X)
        z, q_mu_y, q_std_y = self.encode(x, y)
        z2, q_mu_z2, q_std_z2 = self.encode_global(z)
        # if self.debug_gt_z:
        #     z = gt_z

        # get params of the global transition model p(z2^t | z2^{<t})
        # get params of the transition model p(z^t | z^{<t}, z2^t)
        mask = self.mask(b)

        if self.use_exogenous:
            # Take the last step forcing as condition
            if y_aerosol is not None:
                y_aerosol = y_aerosol[:, -1]
            if y_so2 is not None:
                y_so2 = y_so2[:, -1]
            if y_co2 is not None:
                y_co2 = y_co2[:, -1]
            if y_ch4 is not None:
                y_ch4 = y_ch4[:, -1]

        if self.instantaneous:
            pz2_mu, pz2_std = self.transition_global(z2[:, :-1].clone(), y_co2=y_co2, y_ch4=y_ch4)
            pz_mu, pz_std = self.transition(z.clone(), z2[:, -1], mask, y_aerosol=y_aerosol, y_so2=y_so2)
        else:
            pz2_mu, pz2_std = self.transition_global(
                z2[:, :-1].clone(), y_co2=y_co2, y_ch4=y_ch4
            )  # pz2_mu of shape (b, d, d_z2)
            """
            Q: pz_mu, pz_std = self.transition(z[:, :-1].clone(), pz2_mu, mask) ?
            """
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), z2[:, -1], mask, y_aerosol=y_aerosol, y_so2=y_so2)
        # get params from decoder p(x^t | z^t)
        # we pass only the last z to the decoder, to get xs.

        px_mu, px_std = self.decode(z[:, -1])

        # set distribution with obtained parameters
        if self.distr_decoder.__name__ == "GEVDistribution":
            xi = self.xi.unsqueeze(0).expand_as(px_mu) if self.gev_learn_xi else torch.full_like(px_mu, self.xi)
            px_distr = self.distr_decoder(px_mu, px_std, xi)
            eps = 1e-6
            q_std_y_safe = q_std_y.clamp(min=eps)
            pz_std_safe = pz_std.clamp(min=eps)
            kl_raw_z = (
                0.5 * (torch.log(pz_std_safe**2) - torch.log(q_std_y_safe**2))
                + 0.5 * (q_std_y_safe**2 + (q_mu_y - pz_mu) ** 2) / pz_std_safe**2
                - 0.5
            )
            # TO DO: KL_Z2 term
        else:
            px_distr = self.distr_decoder(px_mu, px_std)
            recons = torch.mean(torch.sum(px_distr.log_prob(y), dim=[1, 2]))
            # compute the KL, the reconstruction and the ELBO
            kl_raw_z = (
                0.5 * (torch.log(pz_std**2) - torch.log(q_std_y**2))
                + 0.5 * (q_std_y**2 + (q_mu_y - pz_mu) ** 2) / pz_std**2
                - 0.5
            )
            kl_raw_z2 = (
                0.5 * (torch.log(pz2_std**2) - torch.log(q_std_z2**2))
                + 0.5 * (q_std_z2**2 + (q_mu_z2 - pz2_mu) ** 2) / pz2_std**2
                - 0.5
            )

        kl_local = torch.sum(kl_raw_z, dim=[2]).mean()
        kl_global = torch.sum(kl_raw_z2, dim=[2]).mean()

        kl = (kl_local + kl_global) * self.coeff_kl
        assert kl >= 0, f"KL={kl} has to be >= 0"

        elbo = recons - kl

        gmst_loss = torch.mean((pz2_mu.mean(dim=-1) - y.mean(dim=-1)) ** 2) + torch.mean(
            (px_mu.mean(dim=-1) - y.mean(dim=-1)) ** 2
        )

        forcing_recons_losses = OrderedDict((name, torch.tensor(0.0, device=x.device)) for name in self.forcing_order)
        forcing_recons_losses["gmst_loss"] = gmst_loss
        forcing_recons_losses["z2_sparse_loss"] = torch.tensor(0.0, device=x.device)

        # Place-holder for compatibility with use_forced_latents version
        return (
            elbo,
            recons,
            kl,
            px_mu,
            0,
            0,
            None,
            forcing_recons_losses,
        )  # any additional forcing recons losses goes her

    def predict_pxmu_pxstd(self, x, y, y_co2=None, y_aerosol=None, y_ch4=None, y_so2=None):

        # NOTE: this one was working fine for the CRPS loss because I was not using no_grad...
        # I need to keep the grads if I am going to add to the loss

        b = x.size(0)

        # sample Zs (based on X)
        z, _, _ = self.encode(x, y)
        z2, _, _ = self.encode_global(z)

        # get params of the global transition model p(z2^t | z2^{<t})
        # get params of the transition model p(z^t | z^{<t}, z2^t)
        mask = self.mask(b)

        if self.use_exogenous:
            # Take the last step forcing as condition
            if y_aerosol is not None:
                y_aerosol = y_aerosol[:, -1]
            if y_so2 is not None:
                y_so2 = y_so2[:, -1]
            if y_co2 is not None:
                y_co2 = y_co2[:, -1]
            if y_ch4 is not None:
                y_ch4 = y_ch4[:, -1]

        if self.instantaneous:
            z2_mu, pz2_std = self.transition_global(z2[:, :-1].clone(), y_co2=y_co2, y_ch4=y_ch4)
            pz_mu, _ = self.transition(z.clone(), z2_mu, mask, y_aerosol=y_aerosol, y_so2=y_so2)
        else:
            pz2_mu, _ = self.transition_global(z2[:, :-1].clone(), y_co2=y_co2, y_ch4=y_ch4)  # (b,1,d_z2)
            pz_mu, _ = self.transition(z[:, :-1].clone(), pz2_mu, mask, y_aerosol=y_aerosol, y_so2=y_so2)

        # get params from decoder p(x^t | z^t)
        # we pass only the last z to the decoder, to get xs.
        px_mu, px_std = self.decode(pz_mu)

        return px_mu, px_std

    def predict(self, x, y, y_co2=None, y_aerosol=None, y_ch4=None, y_so2=None):

        # Use no grad to speed it up! But I need to keep the grads if I am going to add to the loss.

        """
        This is the prediction function for the model.

        We want to take past time steps and predict the next time step, not to reconstruct the past time steps.
        """
        b = x.size(0)

        # NOTE: we are not using y here. We encode using both x and y,
        # but then we discard the latents from the y encoding.

        z, _, _ = self.encode(x, y)
        z2, _, _ = self.encode_global(z)

        if self.use_exogenous:
            # Take the last step forcing as condition
            if y_aerosol is not None:
                y_aerosol = y_aerosol[:, -1]
            if y_so2 is not None:
                y_so2 = y_so2[:, -1]
            if y_co2 is not None:
                y_co2 = y_co2[:, -1]
            if y_ch4 is not None:
                y_ch4 = y_ch4[:, -1]

        mask = self.mask(b)
        if self.instantaneous:
            pz2_mu, _ = self.transition_global(z2[:, :-1].clone(), y_co2=y_co2, y_ch4=y_ch4)
            pz_mu, pz_std = self.transition(z.clone(), pz2_mu.clone(), mask, y_aerosol=y_aerosol, y_so2=y_so2)
        else:
            pz2_mu, pz2_std = self.transition_global(z2[:, :-1].clone(), y_co2=y_co2, y_ch4=y_ch4)  # (b,1,d_z2)
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), pz2_mu.clone(), mask, y_aerosol=y_aerosol, y_so2=y_so2)

        # decode
        px_mu, _ = self.decode(pz_mu)

        return px_mu, y, z, pz_mu, pz_std  # , pz2_mu

    def predict_counterfactual(
        self, x, y, counterfactual_z_index, counterfactual_z_value, y_co2=None, y_aerosol=None, y_ch4=None, y_so2=None
    ):

        # Use no grad to speed it up! But I need to keep the grads if I am going to add to the loss.

        """
        This is the prediction function for the model.

        We want to take past time steps and predict the next time step, not to reconstruct the past time steps.
        """

        b = x.size(0)

        z, _, _ = self.encode(x, y)
        z2, _, _ = self.encode_global(z)

        if self.use_exogenous:
            # Take the last step forcing as condition
            if y_aerosol is not None:
                y_aerosol = y_aerosol[:, -1]
            if y_so2 is not None:
                y_so2 = y_so2[:, -1]
            if y_co2 is not None:
                y_co2 = y_co2[:, -1]
            if y_ch4 is not None:
                y_ch4 = y_ch4[:, -1]

        print("This is the shape of the latents that we are going to intervene on.", z.shape)
        print(
            "Here is where we are going to intervene on the latents, and the value.",
            counterfactual_z_index,
            counterfactual_z_value,
        )

        assert torch.all(z[:, 4, :, :] == z[:, -2, :, :])

        # here we are going to intervene on the latents
        # BEFORE we pass them through the transition model.
        # we want to intervene on the final (non-instantaneous) latent variable.
        # we also intervene on only the first variable
        # we also intervene on all batch members

        z[:, -2, 0, counterfactual_z_index] = counterfactual_z_value

        print("This is e.g. the new value of the latents after intervention.", z[0, -2, 0, counterfactual_z_index])
        assert torch.all(z[:, -2, 0, counterfactual_z_index] == counterfactual_z_value)

        mask = self.mask(b)

        if self.instantaneous:
            pz2_mu, _ = self.transition_global(z2[:, :-1].clone(), y_co2=y_co2, y_ch4=y_ch4)
            pz_mu, pz_std = self.transition(z.clone(), pz2_mu, mask, y_aerosol=y_aerosol, y_so2=y_so2)
        else:
            pz2_mu, _ = self.transition_global(z2[:, :-1].clone(), y_co2=y_co2, y_ch4=y_ch4)  # (b,1,d_z2)
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), pz2_mu, mask, y_aerosol=y_aerosol, y_so2=y_so2)

        # decode
        px_mu, _ = self.decode(pz_mu)

        return px_mu, y, z, pz_mu, pz_std

    def predict_sample(self, x, y, num_samples, y_co2=None, y_aerosol=None, y_ch4=None, y_so2=None):
        """
        This is a prediction function for the model, but where we take samples from the Gaussians of the latents.

        Note this function also returns the option where we sample from the decoders, but of course these samples are
        just chequerboards and not very interesting.

        I can use no_grad here, because I am not going to be using the gradients for anything.
        """

        b = x.size(0)

        with torch.no_grad():
            # sample Zs (based on X)
            z, _, _ = self.encode(x, y)
            z2, _, _ = self.encode_global(z)

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)

            if self.use_exogenous:
                # Take the last step forcing as condition
                if y_aerosol is not None:
                    y_aerosol = y_aerosol[:, -1]
                if y_so2 is not None:
                    y_so2 = y_so2[:, -1]
                if y_co2 is not None:
                    y_co2 = y_co2[:, -1]
                if y_ch4 is not None:
                    y_ch4 = y_ch4[:, -1]

            if self.instantaneous:
                pz2_mu, _ = self.transition_global(z2[:, :-1].clone(), y_co2=y_co2, y_ch4=y_ch4)
                pz_mu, pz_std = self.transition(z.clone(), pz2_mu, mask, y_aerosol=y_aerosol, y_so2=y_so2)
            else:
                pz2_mu, _ = self.transition_global(z2[:, :-1].clone(), y_co2=y_co2, y_ch4=y_ch4)  # (b,1,d_z2)
                pz_mu, pz_std = self.transition(z[:, :-1].clone(), pz2_mu, mask, y_aerosol=y_aerosol, y_so2=y_so2)

            # here I am taking the approach of sampling from the Z distributions, and then decoding.
            samples_from_zs = torch.zeros(num_samples, b, self.d, self.d_x)
            z_samples = torch.zeros(num_samples, b, self.d, self.d_z)

            # TODO: Remove this for loop
            for i in range(num_samples):
                z_samples[i] = self.distr_transition(pz_mu, pz_std).sample()
                samples_from_zs[i], _ = self.decode(z_samples[i])

                # some_decoded_samples_mu, some_decoded_samples_std = self.decode(z_samples[i])

                # samples_from_zs[i] = some_decoded_samples_mu

            # decode
            px_mu, px_std = self.decode(pz_mu)

            # here we decode from pz_mu, and then sample from the distribution over xs.
            # note this will simply give us chequerboards.
            samples_from_xs = torch.zeros(num_samples, b, self.d, self.d_x)

            # TODO: Remove this for loop
            for i in range(num_samples):
                if self.distr_decoder.__name__ == "GEVDistribution":
                    xi = self.xi
                    if isinstance(xi, torch.Tensor) and xi.ndim < px_mu.ndim:
                        xi = xi.expand_as(px_mu)  # ensure broadcast shape
                    samples_from_xs[i] = self.distr_decoder(px_mu, px_std, xi).sample()
                else:
                    samples_from_xs[i] = self.distr_decoder(px_mu, px_std).sample()

            del z_samples

        return samples_from_xs, samples_from_zs, y
        # return px_mu, y, z, pz_mu, pz_std

    def predict_sample_bayesianfiltering(
        self, x, y, num_samples, with_zs_logprob: bool = False, y_co2=None, y_aerosol=None, y_ch4=None, y_so2=None
    ):
        """
        This is a prediction function for the model, but where we take samples from the Gaussians of the latents.

        Note this function also returns the option where we sample from the decoders, but of course these samples are
        just chequerboards and not very interesting.

        I can use no_grad here, because I am not going to be using the gradients for anything.
        """

        b = x.size(0)

        with torch.no_grad():
            # sample Zs (based on X)
            z, q_mu_y, q_std_y = self.encode(x, y)
            z2, q_mu_z2, q_std_z2 = self.encode_global(z)

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)
            if self.use_exogenous:
                # Take the last step forcing as condition
                if y_aerosol is not None:
                    y_aerosol = y_aerosol[:, -1]
                if y_so2 is not None:
                    y_so2 = y_so2[:, -1]
                if y_co2 is not None:
                    y_co2 = y_co2[:, -1]
                if y_ch4 is not None:
                    y_ch4 = y_ch4[:, -1]

            if self.instantaneous:
                pz2_mu, pz2_std = self.transition_global(
                    z2[:, :-1].clone(), y_co2=y_co2.repeat(b, 1), y_ch4=y_ch4.repeat(b, 1)
                )
                pz_mu, pz_std = self.transition(
                    z.clone(), pz2_mu, mask, y_aerosol=y_aerosol.repeat(b, 1), y_so2=y_so2.repeat(b, 1)
                )
            else:
                pz2_mu, pz2_std = self.transition_global(
                    z2[:, :-1].clone(), y_co2=y_co2.repeat(b, 1), y_ch4=y_ch4.repeat(b, 1)
                )  # (b,1,d_z2)
                pz_mu, pz_std = self.transition(
                    z[:, :-1].clone(), pz2_mu, mask, y_aerosol=y_aerosol.repeat(b, 1), y_so2=y_so2.repeat(b, 1)
                )

            # here I am taking the approach of sampling from the Z distributions, and then decoding.
            #             samples_from_zs = torch.zeros(num_samples, b, self.d, self.d_x)
            #             z_samples = torch.zeros(num_samples, b, self.d, self.d_z)
            #             if with_zs_logprob:
            #                 z_samples_logprob = torch.zeros(num_samples, b, self.d, self.d_z)

            #             print(f"FOR LOOP MODEL num_samples {num_samples}")
            #             print(f"z_samples.shape {z_samples.shape}")
            #             print(f"pz_mu.shape {pz_mu.shape}")
            #             print(f"pz_std.shape {pz_std.shape}")
            dim = pz_mu.ndim
            new_shape = [num_samples]
            for k in range(dim):
                new_shape.append(1)
            z_samples = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).sample()
            #             for i in trange(num_samples):
            #                 #TODO: remove this FOR loop
            #                 z_samples[i] = self.distr_transition(pz_mu, pz_std).sample()
            #                 print(f"z_samples[i].shape {z_samples[i].shape}")

            if with_zs_logprob:

                z_samples_logprob = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).log_prob(
                    z_samples
                )
                # GMST of each decoded particle

                # self.distr_transition(pz_mu, pz_std).log_prob(z_samples[i]) gives log probability
            samples_from_zs, some_decoded_samples_std = self.decode(
                z_samples.reshape(z_samples.size(0) * z_samples.size(1), z_samples.size(2), z_samples.size(3))
            )
            samples_from_zs = samples_from_zs.reshape(z_samples.size(0), z_samples.size(1), z_samples.size(2), self.d_x)
            # some_decoded_samples_mu, some_decoded_samples_std = self.decode(z_samples[i])

            # samples_from_zs[i] = some_decoded_samples_mu

            # decode
            px_mu, px_std = self.decode(pz_mu.unsqueeze(2))
            px_mu = px_mu.squeeze(2)
            px_std = px_std.squeeze(2)

            dim = px_mu.ndim
            new_shape = [num_samples]
            for k in range(dim):
                new_shape.append(1)
            # here we decode from pz_mu, and then sample from the distribution over xs.
            # note this will simply give us chequerboards.
            samples_from_xs = torch.zeros(num_samples, b, self.d, self.d_x)

            #             for i in range(num_samples):
            samples_from_xs = self.distr_decoder(px_mu.repeat(new_shape), px_std.repeat(new_shape)).sample()

        if with_zs_logprob:
            return samples_from_xs, samples_from_zs, y, z_samples_logprob, pz2_mu
        return samples_from_xs, samples_from_zs, y
        # return px_mu, y, z, pz_mu, pz_std

    def get_kl(self, mu1, sigma1, mu2, sigma2) -> float:
        """
        KL between two multivariate Gaussian Q and P.

        Here, Q is spherical and P is diagonal
        """
        kl = 0.5 * (
            torch.log(torch.prod(sigma2, dim=1) / torch.prod(sigma1, dim=1))
            + torch.sum(sigma1 / sigma2, dim=1)
            - self.d_z
            + torch.einsum("bd, bd -> b", (mu2 - mu1) * (1 / sigma2), mu2 - mu1)
        )
        # kl = 0.5 * (torch.log(torch.prod(sigma2, dim=1) / sigma1 ** self.d_z) +
        #             torch.sum(sigma1 / sigma2, dim=1) - self.d_z +
        #             torch.einsum('bd, bd -> b', (mu2 - mu1) * (1 / sigma2), mu2 - mu1))
        if torch.sum(kl) < 0:
            __import__("ipdb").set_trace()
            print(sigma2**self.d_z)
            print(torch.prod(sigma1, dim=1))
            print(torch.sum(torch.log(sigma2**self.d_z / torch.prod(sigma1, dim=1))))
            print(torch.sum(torch.sum(sigma1 / sigma2, dim=1)))
            # print(torch.sum(torch.einsum('bd, bd -> b', (mu2 - mu1) * (1 / s_p), mu2 - mu1)))

        return torch.sum(kl)


class LinearAutoEncoder(nn.Module):
    def __init__(self, d, d_x, d_z, tied):
        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.tied = tied
        unif = (1 - 0.1) * torch.rand(size=(d, d_x, d_z)) + 0.1
        self.w = nn.Parameter(unif / torch.as_tensor(d_z))
        if not tied:
            unif = (1 - 0.1) * torch.rand(size=(d, d_z, d_x)) + 0.1
            self.w_encoder = nn.Parameter(unif / torch.as_tensor(d_x))

        # self.logvar_encoder = nn.Parameter(torch.ones(d) * -1)
        # self.logvar_decoder = nn.Parameter(torch.ones(d) * -1)
        self.logvar_encoder = nn.Parameter(torch.ones(d_z) * -1)
        self.logvar_decoder = nn.Parameter(torch.ones(d_x) * -1)

    def get_w_encoder(self):
        if self.tied:
            return torch.transpose(self.w, 1, 2)
        else:
            return self.w_encoder

    def get_w_decoder(self):
        return self.w

    def encode(self, x, i):
        if self.tied:
            w = self.w[i].T
        else:
            w = self.w_encoder[i]
        mu = torch.matmul(x, w.T)
        return mu, self.logvar_encoder

    def decode(self, z, i):
        w = self.w[i]
        mu = torch.matmul(z, w.T)
        return mu, self.logvar_decoder

    def forward(self, x, i, encode: bool = False):
        if encode:
            return self.encode(x, i)
        else:
            return self.decode(x, i)


class NonLinearAutoEncoder(nn.Module):
    def __init__(self, d, d_x, d_z, num_hidden, num_layer, tied, gt_w=None):
        super().__init__()
        self.d_x = d_x
        self.d_z = d_z
        self.tied = tied
        self.use_grad_project = True

        unif = (1 - 0.4) * torch.rand(size=(d, d_x, d_z)) + 0.2
        self.w = nn.Parameter(unif / torch.as_tensor(d_z))
        if not tied:
            unif = (1 - 0.4) * torch.rand(size=(d, d_z, d_x)) + 0.2
            self.w_encoder = nn.Parameter(unif / torch.as_tensor(d_x))

        # self.logvar_encoder = nn.Parameter(torch.ones(d) * -1)
        # self.logvar_decoder = nn.Parameter(torch.ones(d) * -1)
        self.logvar_encoder = nn.Parameter(torch.ones(d_z) * -1)
        self.logvar_decoder = nn.Parameter(torch.ones(d_x) * -1)

    def get_w_encoder(self):
        if self.tied:
            return torch.transpose(self.w, 1, 2)
        return self.w_encoder

    def get_w_decoder(self):
        return self.w

    def get_encode_mask(self):
        if self.tied:
            return torch.transpose(self.w, 1, 2)
        return self.w_encoder

    def select_encoder_mask(self, mask, i, j):
        return mask[i, j]

    def get_decode_mask(self):
        return self.w

    def select_decoder_mask(self, mask, i, j):
        return mask[i, j]


class NonLinearAutoEncoderUniqueMLP_noloop(NonLinearAutoEncoder):

    def __init__(
        self,
        d,
        d_x,
        d_z,
        num_hidden,
        num_layer,
        tied,
        embedding_dim,
        gt_w=None,
    ):
        super().__init__(d, d_x, d_z, num_hidden, num_layer, tied, gt_w)
        self.embedding_encoder = nn.Embedding(d_z, embedding_dim)
        self.encoder = MLP(num_layer, num_hidden, d_x + embedding_dim, 1)  # embedding_dim_encoding

        self.decoder = MLP(num_layer, num_hidden, d_z + embedding_dim, 1)
        self.embedding_decoder = nn.Embedding(d_x, embedding_dim)

    def encode(self, x, i):

        mask = super().get_encode_mask()
        mu = torch.zeros((x.shape[0], self.d_z), device=x.device)

        j_values = torch.arange(self.d_z, device=x.device).expand(
            x.shape[0], -1
        )  # create a 2D tensor with shape (x.shape[0], self.d_z) # Is this batch size * d_z? or is d_z here the dimn of observations?

        # For each latent, create an embedding of dimension 100
        embedded_x = self.embedding_encoder(j_values)  # size b * d_z * embedding_dim

        # for each latent, select the locations it is mapped to
        mask_ = super().select_encoder_mask(mask, i, j_values)  # mask[i, j_values]

        # Could I reduce the memory usage of this?
        # each location create a lask in latents b * d_z * d_x
        # Then concatenate in the last axis (d_x) with the embedding of the latents?
        # x_ = mask_ * x.unsqueeze(1)
        x_ = torch.cat(
            (mask_ * x.unsqueeze(1), embedded_x), dim=2
        )  # expand dimensions of x for broadcasting - looks good.

        del embedded_x
        del mask_
        # Global encoding: dim=1, the encoded feature dim will be squeezed, so use squeeze(-1) to keep the feature dimension!
        mu = self.encoder(x_).squeeze(-1)

        return mu, self.logvar_encoder

    def decode(self, z, i):

        mask = super().get_decode_mask()  # (1, dx, dz)

        mu = torch.zeros((z.shape[0], self.d_x), device=z.device)

        # Create a tensor of shape (z.shape[0], self.d_x) where each row is a sequence from 0 to self.d_x
        j_values = torch.arange(self.d_x, device=z.device).expand(z.shape[0], -1)

        # Embed all j_values at once
        embedded_z = self.embedding_decoder(j_values)

        # Select all decoder masks at once
        mask_ = super().select_decoder_mask(mask, i, j_values)

        if z.ndim < mask_.ndim:
            z_expanded = z.unsqueeze(1).expand(-1, self.d_x, -1)
        else:
            z_expanded = z.expand(-1, self.d_x, -1)
        z_expanded_copy = z_expanded.clone()
        z_expanded_copy.mul_(mask_)
        z_expanded_copy.unsqueeze(2)

        z_ = torch.cat((z_expanded_copy, embedded_z), dim=2)

        del z_expanded
        del z_expanded_copy

        # Apply the decoder to all z_ at once and squeeze the result
        mu = self.decoder(z_).squeeze()

        return mu, self.logvar_decoder

    def forward(self, x, i, encode: bool = False):
        if encode:
            return self.encode(x, i)
        else:
            return self.decode(x, i)


class TransitionModelGlobal(nn.Module):
    """Models the transitions between the latent variables Z2 with neural networks."""

    def __init__(
        self,
        d: int,
        d_z: int,
        tau: int,
        nonlinear_dynamics: bool,
        num_layers: int,
        num_hidden: int,
        num_output: int = 2,
        d_y_co2: int = 0,
        d_y_ch4: int = 0,
    ):
        """
        Args:
            d: number of features
            d_z: number of latent variables
            tau: size of the timewindow
            num_layers: number of layers for the neural networks
            num_hidden: number of hidden units
            num_output: number of outputs
        """
        super().__init__()
        self.d = d  # number of variables
        self.d_z = d_z
        self.tau = tau
        output_var = False

        # initialize NNs
        self.nonlinear_dynamics = nonlinear_dynamics
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        if output_var:
            self.num_output = num_output
        else:
            self.num_output = 1
            # self.logvar = torch.ones(1)  * 0. # nn.Parameter(torch.ones(d) * 0.1)
            # self.logvar = nn.Parameter(torch.ones(d) * -4)
            self.logvar = nn.Parameter(torch.ones(d, d_z) * -4)
        num_forcing = 0
        if d_y_co2 > 0:
            num_forcing += 1
        if d_y_ch4 > 0:
            num_forcing += 1
        if self.nonlinear_dynamics:
            print("NON LINEAR GLOBAL DYNAMICS")
            self.nn = nn.ModuleList(
                MLP(num_layers, num_hidden, d * d_z * tau + num_forcing, self.num_output) for i in range(d * d_z)
            )
        else:
            print("LINEAR GLOBAL DYNAMICS")
            self.nn = nn.ModuleList(MLP(0, 0, d * d_z * tau + num_forcing, self.num_output) for i in range(d * d_z))
        self.use_exogenous = (d_y_co2 + d_y_ch4) > 0

    def forward(self, z, i, k, y_co2=None, y_ch4=None):
        """
        Returns the params of N(z_t | z_{<t}) for a specific feature i and latent variable k.

        z: (b, tau, d_z, 1)
        y_co2: optional condition at t, shape should be (b, 1)
        y_ch4: optional condition at t, shape should be (b, 1)
        """
        batch_size = z.size(0)

        flat_z = z.view(batch_size, -1)

        components = [flat_z]

        if self.use_exogenous:
            if y_co2 is not None:
                components.append(y_co2.view(batch_size, -1))

            if y_ch4 is not None:
                components.append(y_ch4.view(batch_size, -1))

        conditioned_input = torch.cat(components, dim=-1)  # ([b, d_z + d_z2 + n_global_forcings])

        param_z = self.nn[i * self.d_z + k](conditioned_input)

        return param_z


class TransitionModel(nn.Module):
    """Models the transitions between the latent variables Z with neural networks."""

    def __init__(
        self,
        d: int,
        d_z: int,
        d_z_global: int,
        tau: int,
        nonlinear_dynamics: bool,
        num_layers: int,
        num_hidden: int,
        num_output: int = 2,
        d_y_aerosol: int = 0,
        d_y_so2: int = 0,
        local_impact: bool = False,
    ):
        """
        Args:
            d: number of features
            d_z: number of latent variables
            d_z_global: number of global latent variable
            tau: size of the timewindow
            num_layers: number of layers for the neural networks
            num_hidden: number of hidden units
            num_output: number of outputs
        """
        super().__init__()
        self.d = d  # number of variables
        self.d_z = d_z
        self.d_z2 = d_z_global
        self.tau = tau
        output_var = False

        # initialize NNs
        self.nonlinear_dynamics = nonlinear_dynamics
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        input_dim = (d * d_z * tau) + (d * self.d_z2 * 1)
        if output_var:
            self.num_output = num_output
        else:
            self.num_output = 1
            # self.logvar = torch.ones(1)  * 0. # nn.Parameter(torch.ones(d) * 0.1)
            # self.logvar = nn.Parameter(torch.ones(d) * -4)
            self.logvar = nn.Parameter(torch.ones(d, d_z) * -4)
        forcing_dim = 0
        if self.nonlinear_dynamics:
            print("NON LINEAR DYNAMICS")
            if d_y_aerosol > 0:
                if self.local_impact:
                    self.aerosol_nn = nn.ModuleList(
                        MLP(num_layers, num_hidden, d_y_aerosol, self.num_output) for _ in range(d_z)
                    )
                else:
                    self.aerosol_nn = MLP(num_layers, num_hidden, d_y_aerosol, self.num_output)
                forcing_dim += self.num_output

            if d_y_so2 > 0:
                if self.local_impact:
                    self.so2_nn = nn.ModuleList(
                        MLP(num_layers, num_hidden, d_y_so2, self.num_output) for _ in range(d_z)
                    )
                else:
                    self.so2_nn = MLP(num_layers, num_hidden, d_y_so2, self.num_output)
                forcing_dim += self.num_output
            self.nn = nn.ModuleList(
                MLP(num_layers, num_hidden, input_dim + forcing_dim, self.num_output) for i in range(d * d_z)
            )
        else:
            print("LINEAR DYNAMICS")
            if d_y_aerosol > 0:
                # should we keep this nn for each location?
                if self.local_impact:
                    self.aerosol_nn = nn.ModuleList(MLP(0, 0, d_y_aerosol, self.num_output) for _ in range(d_z))
                else:
                    self.aerosol_nn = MLP(0, 0, d_y_aerosol, self.num_output)
                forcing_dim += self.num_output

            if d_y_so2 > 0:
                if self.local_impact:
                    self.so2_nn = nn.ModuleList(MLP(0, 0, d_y_so2, self.num_output) for _ in range(d_z))
                else:
                    self.so2_nn = MLP(0, 0, d_y_so2, self.num_output)
                forcing_dim += self.num_output
            self.nn = nn.ModuleList(MLP(0, 0, input_dim + forcing_dim, self.num_output) for i in range(d * d_z))
        # self.nn = MLP(num_layers, num_hidden, d * k * k, self.num_output)
        self.use_exogenous = (d_y_aerosol + d_y_so2) > 0
        self.local_impact = local_impact

    def forward(self, z, z_global, mask, i, k, y_aerosol=None, y_so2=None, local_mapping=None):
        """
        Returns the params of N(z_t | z_{<t}, z2_t) for a specific feature i and latent variable k NN(G_{tau-1} *
        z_{t-1}, ..., G_{tau-k} * z_{t-k})

        Input: z: (b, tau, d, dz) - History of z1 (t-tau, ...t-2, t-1)
        mask: (b, tau, d, dz) - Adjacency mask for z1
        z2: (b, 1, d, dz2) - Current step of z2 (t)
        y_aerosol: (b, 1, d, dz2)  - Current step of spatial/local forcing (t)
        """
        batch_size = z.size(0)
        z = z.view(mask.size())

        # 1. Process z1 history (t-tau, ...t-2, t-1): apply mask and flatten
        # (b, tau, d, dz) -> (b, tau * d * dz)

        masked_z_history = (mask * z).view(batch_size, -1)

        # 2. Process current z2 (t): flatten without masking
        # (b, 1, d, dz2) -> (b, 1*d * dz2)
        flat_z2 = z_global.view(batch_size, -1)
        components = [masked_z_history, flat_z2]

        if self.use_exogenous:
            if self.local_impact:
                if local_mapping is None:
                    raise ValueError("local_impact=True requires local_mapping, but got None.")

                W = local_mapping  # (dx, dz)

                if y_aerosol is not None:
                    aerosol_input = y_aerosol * W[:, k].unsqueeze(0)
                    # (B, dx)

                    z_aerosol = self.aerosol_nn[k](aerosol_input)
                    # (B, forcing_output_dim)

                    components.append(z_aerosol)

                if y_so2 is not None:
                    so2_input = y_so2 * W[:, k].unsqueeze(0)
                    # (B, dx)

                    z_so2 = self.so2_nn[k](so2_input)
                    # (B, forcing_output_dim)

                    components.append(z_so2)

            else:
                if y_aerosol is not None:
                    z_aerosol = self.aerosol_nn(y_aerosol)
                    components.append(z_aerosol)

                if y_so2 is not None:
                    z_so2 = self.so2_nn(y_so2)
                    components.append(z_so2)
        # 3. Concatenate along the feature dimension
        # Result shape: (b, (tau*d*dz) + (1*d*dz2) + forcing_dims)
        combined_input = torch.cat(components, dim=1)

        # 4. Predict params for N(z1_t | z1_{<t}, z2_t)
        param_z = self.nn[i * self.d_z + k](combined_input)

        return param_z


class TransitionModelParamSharing(nn.Module):
    """Models the transitions between the latent variables Z with neural networks."""

    # Attempt at parameter sharing in the transition model

    def __init__(
        self,
        d: int,
        d_z: int,
        d_z_global: int,
        tau: int,
        nonlinear_dynamics: bool,
        num_layers: int,
        num_hidden: int,
        num_output: int = 2,
        embedding_dim: int = 100,
        d_y_aerosol: int = 0,
        d_y_so2: int = 0,
        local_impact: bool = False,
    ):
        """
        Args:
            d: number of features
            d_z: number of latent variables
            tau: size of the timewindow
            num_layers: number of layers for the neural networks
            num_hidden: number of hidden units
            num_output: number of outputs
        """

        super().__init__()
        self.d = d  # number of variables
        self.d_z = d_z
        self.d_z2 = d_z_global
        self.tau = tau
        output_var = False

        # initialize NNs
        self.nonlinear_dynamics = nonlinear_dynamics
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.embedding_dim = embedding_dim
        self.embedding_transition = nn.Embedding(d_z, embedding_dim)
        input_dim = (d * d_z * tau) + (d * self.d_z2 * 1)
        if output_var:
            self.num_output = num_output
        else:
            self.num_output = 1
            # self.logvar = torch.ones(1)  * 0. # nn.Parameter(torch.ones(d) * 0.1)
            # self.logvar = nn.Parameter(torch.ones(d) * -4)
            self.logvar = nn.Parameter(torch.ones(d, d_z) * -4)
        input_dim = (d * d_z * tau) + (d * self.d_z2 * 1)
        forcing_dim = 0

        if self.nonlinear_dynamics:
            print("NON LINEAR DYNAMICS")
            forcing_num_layers = num_layers
            forcing_num_hidden = num_hidden
            transition_num_layers = num_layers
            transition_num_hidden = num_hidden
        else:
            print("LINEAR DYNAMICS")
            forcing_num_layers = 0
            forcing_num_hidden = 0
            transition_num_layers = 0
            transition_num_hidden = 0

        if d_y_so2 > 0:
            so2_input_dim = d_y_so2 + embedding_dim if local_impact else d_y_so2
            self.so2_nn = MLP(
                forcing_num_layers,
                forcing_num_hidden,
                so2_input_dim,
                self.num_output,
            )
            forcing_dim += self.num_output

        if d_y_aerosol > 0:
            aerosol_input_dim = d_y_aerosol + embedding_dim if local_impact else d_y_aerosol
            self.aerosol_nn = MLP(
                forcing_num_layers,
                forcing_num_hidden,
                aerosol_input_dim,
                self.num_output,
            )
            forcing_dim += self.num_output

        self.nn = nn.ModuleList(
            MLP(
                transition_num_layers,
                transition_num_hidden,
                input_dim + embedding_dim + forcing_dim,
                self.num_output,
            )
            for _ in range(d)
        )
        self.local_impact = local_impact
        # self.nn = MLP(num_layers, num_hidden, d * k * k, self.num_output)
        self.use_exogenous = (d_y_aerosol + d_y_so2) > 0

    def forward(self, z, z_global, mask, i, y_aerosol=None, y_so2=None, local_mapping=None):
        """
        Returns the params of N(z_t | z_{<t}, z2_t) for a specific feature i and latent variable k NN(G_{tau-1} *
        z_{t-1}, ..., G_{tau-k} * z_{t-k})

        Predict transition parameters for all climate target latents of variable i.
        Shape convention:
            z: (B, tau, N, 1)
                N is the number of source latents (parents).
            mask: (B, tau, K, N)
                K is the number of target climate latents (e.g., climate).
                N is the number of source latents (parents, e.g., climate+forcings).
                mask[b, tau, k, n] controls whether source latent n at lag tau-t can influence target climate latent k.
        Output:
            param_z: (B, K, output_dim)
            param_z[:, k] contains the transition parameters for target climate latent k.

        But for this implementation, we only consider K=N, allow the transition among climate latents while put last step forcing as condition.
        """
        batch_size = z.shape[0]
        device = z.device
        K = mask.shape[-2]
        # One learned embedding per target latent k. This tells the shared MLP
        # which target latent it is predicting while sharing parameters across K.
        j_values = torch.arange(K, device=device).expand(batch_size, -1)
        embedded_z = self.embedding_transition(j_values)  # (B, K, embedding_dim)
        # Apply the causal mask.
        # z:                (B, tau, N, 1)
        # z.transpose:      (B, tau, 1, N)
        # mask:             (B, tau, K, N)
        #
        # Broadcasting gives:  masked_z[b, t, k, n] = mask[b, t, k, n] * z[b, t, 0, n]
        # For each target k, this keeps only the parent latents n that are allowed to influence k at lag tau-t.
        masked_z = mask * z.transpose(-1, -2)  # (B, tau, K, N)
        # Problem with previous implementation:
        # We should apply the k-th row of the mask to the parents of latent k, but this applies the k-th column of the mask to k's parents
        # The broadcast can work if K=N, but the semantics are mixed up (see attached screenshot).
        # masked_z = (mask * z).transpose(3, 2).reshape((batch_size, -1, self.d_z)).transpose(2, 1)
        # Arrange one row per target latent:
        # The target dimension K is preserved. The time and source dimensions become the feature vector used to predict each target.
        masked_z = masked_z.permute(0, 2, 1, 3).reshape(
            batch_size, K, -1
        )  # (B, tau, K, N) -> (B, K, tau, N)-> (B, K, tau * N)

        flat_z2 = z_global.view(batch_size, -1)
        flat_z2_expanded = flat_z2.unsqueeze(1).expand(-1, K, -1)
        components = [masked_z, embedded_z, flat_z2_expanded]

        # Process the last step spatial forcings

        if self.use_exogenous:
            if self.local_impact and local_mapping is not None:
                W = local_mapping.squeeze(0)  # usually (dx, dz)

                if y_aerosol is not None:
                    aerosol_input = y_aerosol.unsqueeze(1) * W.T.unsqueeze(0)  # (B, K, d_y_aerosol)
                    aerosol_input = torch.cat([aerosol_input, embedded_z], dim=-1)
                    z_aerosol = self.aerosol_nn(aerosol_input)  # (B, K, num_output)
                    components.append(z_aerosol)

                if y_so2 is not None:
                    so2_input = y_so2.unsqueeze(1) * W.T.unsqueeze(0)  # (B, K, d_y_so2)
                    so2_input = torch.cat([so2_input, embedded_z], dim=-1)
                    z_so2 = self.so2_nn(so2_input)  # (B, K, num_output)
                    components.append(z_so2)

            else:
                if y_aerosol is not None:
                    z_aerosol = self.aerosol_nn(y_aerosol)
                    z_aerosol = z_aerosol.unsqueeze(1).expand(-1, K, -1)
                    components.append(z_aerosol)

                if y_so2 is not None:
                    z_so2 = self.so2_nn(y_so2)
                    z_so2 = z_so2.unsqueeze(1).expand(-1, K, -1)
                    components.append(z_so2)

        z_ = torch.cat(components, dim=2)
        param_z = self.nn[i](z_)

        return param_z
        #  Resulting shape (b, K, tau * N + embed_dim + d_z_global + n_forcings)


class GEVDistribution(Distribution):
    arg_constraints = {}
    has_rsample = False
    support = torch.distributions.constraints.real

    def __init__(self, mu, sigma, xi, validate_args=None):
        """
        Generalized Extreme Value (GEV) distribution.

        Args:
            mu: location parameter
            sigma: scale parameter (must be > 0)
            xi: shape parameter
        """
        self.mu = mu
        self.sigma = sigma
        self.xi = xi
        batch_shape = torch.broadcast_shapes(mu.shape, sigma.shape, xi.shape)
        super().__init__(batch_shape, validate_args=validate_args)

    def _standardized(self, value):
        """Transform to standardized variable z = (x - mu)/sigma"""
        return (value - self.mu) / self.sigma

    def log_prob(self, value):
        eps = 1e-6
        z = self._standardized(value)  # (value - mu) / sigma
        z = z.clamp(min=-1e4, max=1e4)  # avoid overflow

        sigma = self.sigma.clamp(min=eps)
        xi = self.xi
        xi_safe = xi.clone().clamp(min=-1e2, max=1e2)

        t = (1 + xi_safe * z).clamp(min=eps, max=1e6)  # stability in log/pow

        gumbel_mask = xi.abs() < eps
        log_pdf_gumbel = -z - torch.exp(-z.clamp(min=-100, max=100)) - torch.log(sigma)

        if torch.all(gumbel_mask):
            return log_pdf_gumbel

        elif torch.all(~gumbel_mask):
            inv_xi = (1 / xi_safe).clamp(min=-1e2, max=1e2)
            logt = torch.log(t)
            pow_term = torch.nan_to_num(t.pow(-inv_xi), nan=1e3, posinf=1e3, neginf=1e3)
            log_pdf_gev = -((1 + inv_xi) * logt) - pow_term - torch.log(sigma)
            return log_pdf_gev

        else:
            log_pdf = torch.empty_like(log_pdf_gumbel)

            # Fill Gumbel values
            log_pdf[gumbel_mask] = log_pdf_gumbel[gumbel_mask]

            # GEV values
            gev_mask = ~gumbel_mask
            xi_gev = xi_safe[gev_mask]
            sigma_gev = sigma[gev_mask]
            z_gev = z[gev_mask]

            t_gev = (1 + xi_gev * z_gev).clamp(min=eps, max=1e6)
            inv_xi_gev = (1 / xi_gev).clamp(min=-1e2, max=1e2)

            logt_gev = torch.log(t_gev)
            pow_term = torch.nan_to_num(t_gev.pow(-inv_xi_gev), nan=1e3, posinf=1e3, neginf=1e3)

            log_pdf_gev = -((1 + inv_xi_gev) * logt_gev) - pow_term - torch.log(sigma_gev)
            log_pdf[gev_mask] = log_pdf_gev

            if torch.isnan(log_pdf).any():
                print("[NaN DETECTED] in GEV log_prob!")

            return log_pdf

    def sample(self, sample_shape=torch.Size()):
        """Inverse transform sampling from the GEV distribution."""
        u = torch.rand(sample_shape + self.mu.shape, device=self.mu.device).clamp(1e-6, 1 - 1e-6)

        if torch.any(self.xi.abs() < 1e-8):
            # Gumbel case
            return self.mu - self.sigma * torch.log(-torch.log(u))
        else:
            return self.mu + self.sigma * ((-torch.log(u)).pow(-self.xi) - 1) / self.xi

    def mean(self):
        """Return mean if defined (xi < 1)"""
        # mu = location parameter
        # sigma = scale parameter
        # xi = shape parameter
        # gamma = gamma function
        # hardcodes the Euler–Mascheroni constant, which is the mean of the Gumbel distribution — the special case of GEV when ξ = 0.
        if torch.any(self.xi >= 1):
            # xi values are ≥ 1
            return torch.tensor(float("nan"), device=self.mu.device)
        if torch.all(self.xi.abs() < 1e-8):
            # xi is approximately zero, this returns the Gumbel mean
            return self.mu + self.sigma * euler_mascheroni
        else:
            # general GEV cases where 0 < xi < 1,
            return torch.tensor(float("nan"), device=self.mu.device)

    def variance(self):
        """Return variance if defined (xi < 0.5)"""
        if torch.any(self.xi >= 0.5):
            return torch.tensor(float("nan"), device=self.mu.device)
        if torch.all(self.xi.abs() < 1e-8):
            # closed-form variance of the Gumbel distribution
            return (pi**2 / 6) * self.sigma**2
        else:
            # 0 < xi < 0.5 — currently not implemented
            return torch.tensor(float("nan"), device=self.mu.device)


if __name__ == "__main__":

    device = "cuda:0"
    var = ["ts"]
    tau = 5
    d_x = 9
    d = len(var)
    future_time_steps = 1
    num_input = d * tau
    model = HierarchicalLatentTSDCD(
        num_layers=2,
        num_hidden=8,
        num_input=num_input,
        num_output=2,
        num_layers_mixing=2,
        num_hidden_mixing=16,
        position_embedding_dim=100,
        transition_param_sharing=True,
        position_embedding_transition=45,
        coeff_kl=1,
        d=d,
        # Here, everything hardcoded to gaussian because GEV leads to Nan... TBD
        distr_z0="gaussian",
        distr_encoder="gaussian",
        distr_transition="gaussian",
        distr_decoder="gaussian",
        d_x=d_x,
        d_z=4,
        d_z_global=1,
        tau=tau,
        instantaneous=False,
        nonlinear_dynamics=True,
        nonlinear_global_dynamics=True,
        nonlinear_mixing=True,
        tied_w=False,
        fixed=False,
        d_y_co2=1,
        d_y_aerosol=d_x,
        d_y_ch4=1,
        d_y_so2=d_x,
        map_aerosol_to_climate=False,
        use_exogenous=True,
    )
    # model = model.to(device)
    batch_size = 1
    x = torch.randn(batch_size, tau, 1, d_x)
    y = torch.randn(batch_size, future_time_steps, d_x)

    y_co2 = y.mean(dim=-1, keepdim=True)  # .to(device)
    y_aerosol = torch.randn(batch_size, tau + future_time_steps, d_x)  # .to(device)
    y_ch4 = torch.randn(batch_size, tau + future_time_steps, 1)  # .to(device)
    y_so2 = torch.randn(batch_size, tau + future_time_steps, d_x)  # .to(device)

    y_co2_pert = y_co2.clone() + torch.randn(batch_size, tau + 1, 1)
    y_ch4_pert = y_ch4.clone() + 1
    y_so2_pert = y_so2.clone() + 1
    y_aerosol_pert = y_aerosol.clone() + 1

    model.predict_sample_bayesianfiltering(x, y, 2, y_co2=y_co2, y_aerosol=y_aerosol, y_ch4=y_ch4, y_so2=y_so2)

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # optimizer.zero_grad()

    # for i in range(10):
    #     elbo,recons,kl,px_mu,_,_,_, forcing_recons_loss = model(x, y, gt_z=None, iteration=i, y_co2=y_co2, y_aerosol=y_aerosol, y_ch4=y_ch4, y_so2=y_so2)
    #     gmst_loss = forcing_recons_loss["gmst_loss"]
    #     print(
    #         f"{i}: elbo {elbo.item():.4f}, recons {recons.item():.4f}, gmst_loss {gmst_loss.item():.4f}"
    #     )
    #     loss = -elbo + gmst_loss*1000
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    # print(f"Forward: {px_mu[0]}")
    # print(f"Ground truth: {y[0]}")

    # model.eval()
    # torch.manual_seed(0)

    # with torch.no_grad():

    #     z_base, _, _ = model.encode(x, y)

    #     zg_base, _, _ = model.encode_global(z_base)

    #     pz2_base, _ = model.transition_global(z2=zg_base[:, :-1], y_co2=y_co2[:,-1], y_ch4=y_ch4[:,-1])
    #     pz2_pert, _ = model.transition_global(z2=zg_base[:, :-1], y_co2=y_co2_pert[:,-1], y_ch4=y_ch4[:,-1])

    #     print("pz2 diff:",
    #         (pz2_base - pz2_pert).abs().max().item())

    #     mask = model.mask(x.size(0))
    #     # z[:, :-1].clone(), z2[:, -1]
    #     pz_base, _ = model.transition(z_base[:, :-1], z_global=pz2_base, mask=mask, y_aerosol=y_aerosol[:,-1], y_so2=y_so2[:,-1])
    #     pz_pert, _ = model.transition(z_base[:, :-1], z_global=pz2_pert, mask=mask, y_aerosol=y_aerosol[:,-1], y_so2=y_so2[:,-1])

    #     print("climate pz diff:",
    #         (pz_base - pz_pert).abs().max().item())

    #     px_base, _ = model.decode(pz_base)
    #     px_pert, _ = model.decode(pz_pert)

    #     print("px diff:", (px_base - px_pert).abs().max().item())
    #     W = model.autoencoder.get_w_decoder()[0, :, :4]
    #     W_strength = W.abs().mean(dim=0)

    #     latent_diff = (pz_base - pz_pert).abs().mean(dim=(0, 1))

    #     print("latent_diff:", latent_diff)
    #     print("W_strength:", W_strength)
    #     print("estimated contribution:", latent_diff * W_strength)

    # with torch.no_grad():
    #     px_mu, y, z, pz_mu, pz_std = model.predict(x, y, y_co2=y_co2, y_aerosol=y_aerosol, y_ch4=y_ch4, y_so2=y_so2)

    #     print(f"Prediction: {px_mu[0]}")
    #     px_mu, y, z, pz_mu, pz_std = model.predict_counterfactual(x, y, 1, 0.1, y_co2=y_co2, y_aerosol=y_aerosol, y_ch4=y_ch4, y_so2=y_so2)

    #     print(f"predict_counterfactual: {px_mu[0]}")
    #     samples_from_xs, samples_from_zs, y = model.predict_sample(x, y, 2, y_co2=y_co2, y_aerosol=y_aerosol, y_ch4=y_ch4, y_so2=y_so2)

    #     print(samples_from_xs.shape)
    #     print(f"predict_sample: {samples_from_xs[0]}")
    #     samples_from_xs, samples_from_zs, y = model.predict_sample_bayesianfiltering(x, y, 2, y_co2=y_co2, y_aerosol=y_aerosol, y_ch4=y_ch4, y_so2=y_so2)

    #     print(f"predict_sample_bayesianfiltering: {samples_from_xs[0]}")
