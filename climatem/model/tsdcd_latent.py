"""
Core causal discovery model with latent variables (LatentTSDCD).

This module implements latent causal graph learning with an encoder/decoder architecture and a learnable causal mask.
The model discovers temporal causal structure among latent variables from observed time series, using differentiable
structure learning with Gumbel-softmax relaxation and acyclicity constraints.

Adapted from the original code for CDSD (Brouillard et al., 2024):     "Causal Discovery with Score-based methods for
time series with latent     confounders."
"""

from collections import OrderedDict
from math import pi

import torch
import torch.distributions as distr
import torch.nn as nn
from torch.distributions import Distribution

from climatem.utils import get_logger

logger = get_logger(__name__)

# Euler-Mascheroni constant, used in the Gumbel distribution CDF for Gumbel-softmax reparameterization
euler_mascheroni = 0.57721566490153286060


class Mask(nn.Module):
    """
    Learnable causal graph adjacency matrix for differentiable structure learning.

    Parameterizes the causal graph via sigmoid (or Gumbel-softmax) over learnable
    logits. During training, edges are sampled stochastically; at evaluation,
    the sigmoid probabilities can be thresholded to obtain a binary graph.

    Attributes:
        param: Learnable logit tensor of shape ``(tau, d*d_x, d*d_x)`` where
            ``tau`` is the number of time lags, and each ``(d*d_x, d*d_x)``
            slice is a source-to-target adjacency matrix over all
            (variable, latent) pairs.
        drawhard: If True, uses the straight-through estimator to produce
            binary mask values in the forward pass while allowing gradient
            flow through the soft Gumbel-sigmoid in the backward pass.
        fixed_output_fraction: Fraction of mask entries that are fixed (not
            learned). When ``fixed=True``, this controls the density of the
            random fixed mask.
    """

    def __init__(
        self,
        d: int,
        d_x: int,
        tau: int,
        latent: bool,
        instantaneous: bool,
        drawhard: bool,
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
        self.drawhard = drawhard
        self.fixed = fixed
        self.fixed_output_fraction = fixed_output_fraction
        # Here we can just set what we want the output to be.
        self.fixed_output = None
        self.uniform = distr.uniform.Uniform(0, 1)

        # Here we could change how the mask is instantiated in the causal graph.
        if self.latent:
            if not nodiag:
                # Initializes logits to 5 so sigmoid(5) ~ 0.993, meaning all edges
                # start as "present" and are pruned during training via sparsity penalty.
                self.param = nn.Parameter(torch.ones((self.tau, d * d_x, d * d_x)) * 5)
                self.fixed_mask = torch.ones_like(self.param)
            else:
                param = torch.ones((self.tau, d * d_x, d * d_x))
                param[:, torch.arange(d * d_x), torch.arange(d * d_x)] = -1
                # Initializes logits to 5 so sigmoid(5) ~ 0.993 (all edges "present");
                # diagonal entries are set to -5 so sigmoid(-5) ~ 0.007 (self-loops suppressed).
                self.param = nn.Parameter(param * 5)
                self.fixed_mask = torch.ones_like(self.param)
                self.fixed_mask[:, torch.arange(self.fixed_mask.size(1)), torch.arange(self.fixed_mask.size(2))] = 0
            if self.instantaneous:
                # TODO: G[0] or G[-1]
                self.fixed_mask[-1, torch.arange(self.fixed_mask.size(1)), torch.arange(self.fixed_mask.size(2))] = 0
        else:
            if self.instantaneous:
                # Logits initialized to 5: sigmoid(5) ~ 0.993, all edges start "present".
                self.param = nn.Parameter(torch.ones((self.tau, d, d, d_x)) * 5)
                self.fixed_mask = torch.ones_like(self.param)
                # set diagonal 0 for G_t0
                self.fixed_mask[-1, torch.arange(self.fixed_mask.size(1)), torch.arange(self.fixed_mask.size(2))] = 0
                # TODO: set neighbors to 0
                # self.fixed_mask[:, :, :, d_x] = 0
            else:
                # Logits initialized to 5: sigmoid(5) ~ 0.993, all edges start "present".
                self.param = nn.Parameter(torch.ones((tau, d, d, d_x)) * 5)
                self.fixed_mask = torch.ones_like(self.param)

    def forward(self, b: int, tau: float = 1) -> torch.Tensor:
        """
        :param b: batch size
        :param tau: temperature constant for sampling
        """

        if not self.fixed:
            adj = gumbel_sigmoid(self.param, self.uniform, b, tau=tau, hard=self.drawhard)
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
            # Clamps ground-truth edges to high logit value (10.0) so
            # sigmoid(10) ~ 1.0, effectively fixing known edges as present.
            self.param = (gt_mask > 0) * 10.0
        else:
            # Logits initialized to 5: sigmoid(5) ~ 0.993, all edges start "present".
            self.param = nn.Parameter(torch.ones(d, d_x, d_z) * 5)

    def forward(self, batch_size):
        param = self.param.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        mask = nn.functional.gumbel_softmax(param, tau=1, hard=False)
        return mask


def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)


def gumbel_sigmoid(log_alpha, uniform, bs, tau=1, hard=False):
    shape = tuple([bs] + list(log_alpha.size()))
    logistic_noise = sample_logistic(shape, uniform)

    y_soft = torch.sigmoid((log_alpha + logistic_noise) / tau)

    if hard:
        # 0.5 threshold: binarization cutoff for the sigmoid mask (straight-through estimator)
        y_hard = (y_soft > 0.5).type(torch.Tensor)

        # This weird line does two things:
        #   1) at forward, we get a hard sample.
        #   2) at backward, we differentiate the gumbel sigmoid
        y = y_hard.detach() - y_soft.detach() + y_soft

    else:
        y = y_soft
    return y


class MLP(nn.Module):
    def __init__(self, num_layers: int, num_hidden: int, num_input: int, num_output: int):
        super().__init__()
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.num_input = num_input
        self.num_output = num_output

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


class LatentTSDCD(nn.Module):
    """
    Differentiable Causal Discovery for time series with latent variables.

    Implements the LatentTSDCD architecture: an encoder maps observations to latent
    variables, a learnable causal mask (Gumbel-sigmoid) parameterizes the temporal
    causal graph, a transition model predicts future latents conditioned on masked
    past latents, and a decoder reconstructs observations from latents.

    Variable name glossary (used throughout forward / loss computation):
        px_mu  -- predicted x mean (decoder output), shape (batch, d, d_x)
        px_std -- predicted x std  (decoder output), shape (batch, d, d_x)
        pz_mu  -- predicted z mean (latent dynamics / transition output), shape (batch, d, d_z)
        pz_std -- predicted z std  (latent dynamics / transition output), shape (batch, d, d_z)
        q_mu_y -- variational posterior mean for the target y, shape (batch, d, d_z)
        q_std_y -- variational posterior std for the target y, shape (batch, d, d_z)
        qz_mu  -- variational posterior mean for z (alias used in some contexts)
    """

    def __init__(
        self,
        num_layers: int,
        num_hidden: int,
        num_input: int,
        num_output: int,
        num_layers_mixing: int,
        num_hidden_mixing: int,
        position_embedding_dim: int,
        reduce_encoding_pos_dim: bool,
        coeff_kl: float,
        distr_z0: str,
        distr_encoder: str,
        distr_transition: str,
        distr_decoder: str,
        d: int,
        d_x: int,
        d_z: int,
        tau: int,
        instantaneous: bool,
        nonlinear_mixing: bool,
        nonlinear_dynamics: bool,
        hard_gumbel: bool,
        no_gt: bool,
        debug_gt_graph: bool,
        debug_gt_z: bool,
        debug_gt_w: bool,
        gt_graph: torch.tensor = None,
        gt_w: torch.tensor = None,
        tied_w: bool = False,
        fixed: bool = False,
        fixed_output_fraction: float = 1.0,
        gev_learn_xi: bool = False,
        use_exogenous: bool = False,
        d_y_co2: int = 0,
        d_y_aerosol: int = 0,
        use_forced_latents: bool = False,
        n_forced_latents_co2: int = 1,
        n_forced_latents_aerosol: int = 2,
        forcing_arch: str = "baseline",
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
            tau: size of the timewindow
            instantaneous: if True, models instantaneous connections
            hard_gumbel: if True, use hard sampling for the masks

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
            use_exogenous: if True, condition on exogenous forcings (CO2 + aerosols)
            d_y_co2: dimension of CO2 forcing (typically 1 for global)
            d_y_aerosol: dimension of aerosol forcing (typically d_x for local)
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
        self.reduce_encoding_pos_dim = reduce_encoding_pos_dim
        self.coeff_kl = coeff_kl

        self.d = d
        self.d_x = d_x
        self.d_z = d_z
        self.tau = tau
        self.instantaneous = instantaneous
        self.nonlinear_mixing = nonlinear_mixing
        self.nonlinear_dynamics = nonlinear_dynamics
        self.hard_gumbel = hard_gumbel
        self.no_gt = no_gt
        self.debug_gt_graph = debug_gt_graph
        self.debug_gt_z = debug_gt_z
        self.debug_gt_w = debug_gt_w
        self.tied_w = tied_w
        self.fixed = fixed
        self.fixed_output_fraction = fixed_output_fraction
        self.gev_learn_xi = gev_learn_xi
        self.use_exogenous = use_exogenous
        self.d_y_co2 = d_y_co2
        self.d_y_aerosol = d_y_aerosol
        self.use_forced_latents = use_forced_latents
        self.n_forced_latents_co2 = n_forced_latents_co2
        self.n_forced_latents_aerosol = n_forced_latents_aerosol
        self.forcing_arch = forcing_arch
        self._forcing_arch_logged = False
        if self.forcing_arch == "predefined" and self.use_forced_latents:
            raise ValueError(
                "forcing_arch='predefined' requires use_forced_latents=False and d_z to include only climate latents."
            )

        if self.instantaneous:
            self.total_tau = tau + 1
        else:
            self.total_tau = tau

        if self.no_gt:
            self.gt_w = None
            self.gt_graph = None
        else:
            self.gt_w = torch.as_tensor(gt_w).double()
            self.gt_graph = torch.as_tensor(gt_graph).double()

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
        # MLP conditioning dims: only non-zero when use_exogenous (raw forcings concatenated to MLP inputs)
        d_y_co2_cond = self.d_y_co2 if self.use_exogenous else 0
        d_y_aerosol_cond = self.d_y_aerosol if self.use_exogenous else 0
        # Forcing encoder/decoder spatial dims: need real dims whenever use_forced_latents,
        # independent of whether raw forcings are used as MLP conditioning (use_exogenous).
        d_y_co2_spatial = self.d_y_co2 if self.use_forced_latents else 0
        d_y_aerosol_spatial = self.d_y_aerosol if self.use_forced_latents else 0

        if self.nonlinear_mixing:
            logger.info("NON-LINEAR MIXING")
            # NOTE:(seb) using the noloop version of non-linear here to make it much faster.
            self.autoencoder = NonLinearAutoEncoderUniqueMLP_noloop(
                d,
                d_x,
                d_z,
                self.num_hidden_mixing,
                self.num_layers_mixing,
                use_gumbel_mask=False,
                tied=tied_w,
                embedding_dim=self.position_embedding_dim,
                reduce_encoding_pos_dim=self.reduce_encoding_pos_dim,
                gt_w=None,
                d_y_co2=d_y_co2_cond,
                d_y_aerosol=d_y_aerosol_cond,
                use_forced_latents=self.use_forced_latents,
                n_forced_latents_co2=self.n_forced_latents_co2,
                n_forced_latents_aerosol=self.n_forced_latents_aerosol,
                d_y_co2_spatial=d_y_co2_spatial,
                d_y_aerosol_spatial=d_y_aerosol_spatial,
            )
        else:
            # print('Using linear mixing')
            logger.info("LINEAR MIXING")
            self.autoencoder = LinearAutoEncoder(
                d,
                d_x,
                d_z,
                tied=tied_w,
                d_y_co2=d_y_co2_cond,
                d_y_aerosol=d_y_aerosol_cond,
                use_forced_latents=self.use_forced_latents,
                n_forced_latents_co2=self.n_forced_latents_co2,
                n_forced_latents_aerosol=self.n_forced_latents_aerosol,
                d_y_co2_spatial=d_y_co2_spatial,
                d_y_aerosol_spatial=d_y_aerosol_spatial,
            )

        if debug_gt_w:
            self.decoder.w = gt_w

        self.transition_model = TransitionModelParamSharing(
            self.d,
            self.d_z,
            self.total_tau,
            self.nonlinear_dynamics,
            self.num_layers,
            self.num_hidden,
            self.num_output,
            self.position_embedding_dim,
            d_y_co2=self.d_y_co2 if self.use_exogenous else 0,
            d_y_aerosol=self.d_y_aerosol if self.use_exogenous else 0,
        )

        # print("We are setting the Mask here.")
        self.mask = Mask(
            d,
            d_z,
            self.total_tau,
            instantaneous=instantaneous,
            latent=True,
            drawhard=hard_gumbel,
            fixed=fixed,
            fixed_output_fraction=fixed_output_fraction,
        )
        if self.debug_gt_graph:
            if self.instantaneous:
                self.mask.fix(self.gt_graph)
            else:
                self.mask.fix(self.gt_graph[:-1])

    def get_adj(self):
        """
        Returns: Matrices of the probabilities from which the masks linking the
        latent variables are sampled
        """
        return self.mask.get_proba()

    def encode(self, x, y, y_co2=None, y_aerosol=None):
        """
        Encode observations X (history) and Y (target) into latent variables Z.

        Args:
            x: Historical observations, shape (batch, tau, d, d_x).
            y: Target observation, shape (batch, d, d_x).
            y_co2: Optional CO2 forcing, shape (batch, tau+1, d_y_co2) or (batch, d_y_co2).
            y_aerosol: Optional aerosol forcing, shape (batch, tau+1, d_y_aerosol) or (batch, d_y_aerosol).

        Returns:
            z: Latent variables, shape (batch, tau+1, d, d_z).
            mu: Variational posterior mean for the target timestep, shape (batch, d, d_z).
            std: Variational posterior std for the target timestep, shape (batch, d, d_z).
        """
        b = x.size(0)  # batch size
        z = torch.zeros(b, self.tau + 1, self.d, self.d_z, device=x.device)
        mu = torch.zeros(b, self.d, self.d_z, device=x.device)
        std = torch.zeros(b, self.d, self.d_z, device=x.device)

        # Handle forced latents if enabled
        if self.use_forced_latents and y_co2 is not None and y_aerosol is not None:
            # Calculate number of climate latents
            n_climate_latents = self.d_z - self.n_forced_latents_co2 - self.n_forced_latents_aerosol

            # y_co2 shape: (batch_size, tau+1, spatial_dim) - NOW SPATIAL like aerosol!
            # y_aerosol shape: (batch_size, tau+1, spatial_dim)
            # We need to process forcings at each timestep separately

            # For SAVAR, d=1, so we only iterate once over i
            for i in range(self.d):
                # Encode climate latents from observations for all timesteps
                for t in range(self.tau):
                    # Extract forcing at timestep t
                    y_co2_t = y_co2[:, t] if y_co2.dim() == 3 else y_co2  # Handle both temporal and single timestep
                    y_aerosol_t = y_aerosol[:, t] if y_aerosol.dim() == 3 else y_aerosol

                    # Encode forcings for this timestep
                    z_forced_t, _, _ = self.autoencoder.encode_forcings(y_co2_t, y_aerosol_t)

                    if self.use_exogenous:
                        q_mu, q_logvar = self.autoencoder(
                            x[:, t, i], i, encode=True, forcing_co2=y_co2_t, forcing_aerosol=y_aerosol_t
                        )
                    else:
                        q_mu, q_logvar = self.autoencoder(x[:, t, i], i, encode=True)

                    q_std = torch.exp(0.5 * q_logvar)
                    # Only encode to climate latents
                    z[:, t, i, :n_climate_latents] = q_mu[:, :n_climate_latents] + q_std[
                        :n_climate_latents
                    ] * self.distr_encoder(0, 1, size=(b, n_climate_latents))
                    # Fill forced latents with timestep-specific forcings
                    z[:, t, i, n_climate_latents:] = z_forced_t

                # Encode the target timestep (y) using final forcing timestep
                y_co2_target = y_co2[:, -1] if y_co2.dim() == 3 else y_co2
                y_aerosol_target = y_aerosol[:, -1] if y_aerosol.dim() == 3 else y_aerosol

                z_forced_target, mu_forced, std_forced = self.autoencoder.encode_forcings(
                    y_co2_target, y_aerosol_target
                )

                if self.use_exogenous:
                    q_mu, q_logvar = self.autoencoder(
                        y[:, i], i, encode=True, forcing_co2=y_co2_target, forcing_aerosol=y_aerosol_target
                    )
                else:
                    q_mu, q_logvar = self.autoencoder(y[:, i], i, encode=True)

                q_std = torch.exp(0.5 * q_logvar)
                # Only encode climate latents
                z[:, -1, i, :n_climate_latents] = q_mu[:, :n_climate_latents] + q_std[
                    :n_climate_latents
                ] * self.distr_encoder(0, 1, size=(b, n_climate_latents))
                # Fill forced latents with target timestep forcings
                z[:, -1, i, n_climate_latents:] = z_forced_target

                # Store full mu and std (including forced latents from target timestep)
                mu[:, i, :n_climate_latents] = q_mu[:, :n_climate_latents]
                mu[:, i, n_climate_latents:] = mu_forced
                std[:, i, :n_climate_latents] = q_std[:n_climate_latents]
                std[:, i, n_climate_latents:] = std_forced

        else:
            # Original encoding path (all latents from observations)
            for i in range(self.d):
                for t in range(self.tau):
                    if self.use_exogenous and y_co2 is not None and y_aerosol is not None:
                        q_mu, q_logvar = self.autoencoder(
                            x[:, t, i], i, encode=True, forcing_co2=y_co2, forcing_aerosol=y_aerosol
                        )
                    else:
                        q_mu, q_logvar = self.autoencoder(x[:, t, i], i, encode=True)

                    q_std = torch.exp(0.5 * q_logvar)
                    z[:, t, i] = q_mu + q_std * self.distr_encoder(0, 1, size=q_mu.size())

                if self.use_exogenous and y_co2 is not None and y_aerosol is not None:
                    q_mu, q_logvar = self.autoencoder(
                        y[:, i], i, encode=True, forcing_co2=y_co2, forcing_aerosol=y_aerosol
                    )
                else:
                    q_mu, q_logvar = self.autoencoder(y[:, i], i, encode=True)

                q_std = torch.exp(0.5 * q_logvar)
                z[:, -1, i] = q_mu + q_std * self.distr_encoder(0, 1, size=q_mu.size())

                mu[:, i] = q_mu
                std[:, i] = q_std

        return z, mu, std

    def transition(self, z, mask, y_co2=None, y_aerosol=None):
        """Compute latent dynamics: predict next-step latent distribution p(z^t | z^{<t}).

        Args:
            z: Past latent variables, shape (batch, tau, d, d_z) or (batch, tau+1, d, d_z).
            mask: Sampled causal mask, shape (batch, tau, d*d_z, d*d_z).
            y_co2: Optional CO2 forcing.
            y_aerosol: Optional aerosol forcing.

        Returns:
            mu: Predicted latent mean (pz_mu), shape (batch, d, d_z).
            std: Predicted latent std (pz_std), shape (batch, d, d_z).
        """
        b = z.size(0)  # batch size
        mu = torch.zeros(b, self.d, self.d_z)  # pz_mu to be filled
        std = torch.zeros(b, self.d, self.d_z)  # pz_std to be filled

        # print("What is the shape of the mus and stds that we are going to fill up?", mu.shape, std.shape)

        # learning conditional variance
        # for i in range(self.d):
        #     pz_params = torch.zeros(b, self.d_z, 2)
        #     for k in range(self.d_z):
        #         pz_params[:, k] = self.transition_model(z, mask[:, :, i * self.d_z + k], i, k)
        #     mu[:, i] = pz_params[:, :, 0]
        #     std[:, i] = torch.exp(0.5 * pz_params[:, :, 1])

        # TODO Can we remove this for loop
        for i in range(self.d):
            pz_params = torch.zeros(b, self.d_z, 1)
            # print("This is pz_params shape, before we fill it up with a for loop, where the 2nd dimension is filled with the result of the transition model.", pz_params.shape)
            # for k in range(self.d_z):
            # print("Doing the transition, and this is the k at the moment.", k)
            # print("**************************************************")
            # print("What is the shape of the mask?", mask.shape)
            # print("What is the shape of mask[:, :, i * self.d_z + k]?", mask[:, :, i * self.d_z + k].shape)
            # print("THIS DEFINES THE MASK THAT IS USED TO PRODUCE A PARTICULAR LATENT, Z_k.")
            if self.use_exogenous and y_co2 is not None and y_aerosol is not None:
                # Extract last timestep if temporal forcing (shape: batch, tau+1, dim)
                # TransitionModel expects 2D forcing (batch, dim)
                y_co2_t = y_co2[:, -1] if y_co2.dim() == 3 else y_co2
                y_aerosol_t = y_aerosol[:, -1] if y_aerosol.dim() == 3 else y_aerosol

                pz_params = self.transition_model(
                    z,
                    mask[:, :, i * self.d_z : (i + 1) * self.d_z],
                    i,
                    forcing_co2=y_co2_t,
                    forcing_aerosol=y_aerosol_t,
                )

            else:

                pz_params = self.transition_model(z, mask[:, :, i * self.d_z : (i + 1) * self.d_z], i)

            # print("Note here that mu[:, i] is the same as pz_params[:, :, 0], once we have filled up pz_params [:, k] wise, with each k being a forward pass.")
            # print("What is the shape of mu[:, i] and std[:, i]?", mu[:, i].shape, std[:, i].shape)
            # print("What is the shape of pz_params[:, :, 0]?", pz_params[:, :, 0].shape)
            mu[:, i] = pz_params[:, :, 0]
            std[:, i] = torch.exp(0.5 * self.transition_model.logvar[i])

        # print("This is giving us the pz_mu and pz_std that we use later.")
        return mu, std

    def decode(self, z, y_co2=None, y_aerosol=None):
        """
        Decode latent variables z to observations.

        When use_forced_latents=True, only climate latents are used for observation
        reconstruction. Forcing latents are excluded from the observation decoder
        (they are only used in the causal transition model and forcing decoders).
        Raw forcing fields (y_co2, y_aerosol) are still used as conditioning.

        Args:
            z: Latent variables, shape (batch, d, d_z).
            y_co2: Raw CO2 forcing field for conditioning, shape (batch, d_y_co2) or (batch, tau+1, d_y_co2).
            y_aerosol: Raw aerosol forcing field for conditioning, shape (batch, d_y_aerosol) or (batch, tau+1, d_y_aerosol).

        Returns:
            mu: Predicted observation mean (px_mu), shape (batch, d, d_x).
            std: Predicted observation std (px_std), shape (batch, d, d_x).
        """
        mu = torch.zeros(z.size(0), self.d, self.d_x)  # px_mu to be filled
        std = torch.zeros(z.size(0), self.d, self.d_x)  # px_std to be filled

        # Only use climate latents for observation decoding (forcing latents excluded)
        if self.use_forced_latents:
            n_climate = self.d_z - self.n_forced_latents_co2 - self.n_forced_latents_aerosol
            z_for_decode = z[:, :, :n_climate]  # Shape: (batch, d, n_climate)
        else:
            z_for_decode = z

        # TODO: Can we remove this for loop
        for i in range(self.d):
            if self.use_exogenous and y_co2 is not None and y_aerosol is not None:
                # Extract last timestep if temporal forcing (shape: batch, tau+1, dim)
                # Autoencoder expects 2D forcing (batch, dim)
                y_co2_t = y_co2[:, -1] if y_co2.dim() == 3 else y_co2
                y_aerosol_t = y_aerosol[:, -1] if y_aerosol.dim() == 3 else y_aerosol

                # Pass only climate latents to decoder, but keep raw forcing as conditioning
                px_mu, px_logvar = self.autoencoder(
                    z_for_decode[:, i], i, encode=False, forcing_co2=y_co2_t, forcing_aerosol=y_aerosol_t
                )
            else:
                px_mu, px_logvar = self.autoencoder(z_for_decode[:, i], i, encode=False)

            if px_mu.ndim == mu.ndim:  # In case of linear mixing with one variable, second dimension is too much
                px_mu = px_mu.squeeze()

            mu[:, i] = px_mu
            std[:, i] = torch.exp(0.5 * px_logvar)

        return mu, std

    def forward(self, x, y, gt_z, iteration, xi=None, y_co2=None, y_aerosol=None):
        """Full forward pass: encode, transition, decode, and compute ELBO.

        Args:
            x: Historical observations, shape (batch, tau, d, d_x).
            y: Target observation, shape (batch, d, d_x).
            gt_z: Ground-truth latents (used only when debug_gt_z=True).
            iteration: Current training iteration (unused in forward, passed for API consistency).
            xi: Optional GEV shape parameter override.
            y_co2: Optional CO2 forcing.
            y_aerosol: Optional aerosol forcing.

        Returns:
            Tuple of (elbo, recons, kl, px_mu, forcing_recons_loss_co2,
            forcing_recons_loss_aerosol, encoded_forcing_mu).
        """
        b = x.size(0)  # batch size

        # sample Zs (based on X)

        z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol)

        # Store encoded forcing latent means for supervision loss (if using forced latents)
        encoded_forcing_mu = None
        if self.use_forced_latents and y_co2 is not None and y_aerosol is not None:
            # Extract encoded forcing latent means from q_mu_y
            n_climate_latents = self.d_z - self.n_forced_latents_co2 - self.n_forced_latents_aerosol
            # q_mu_y shape: (batch, d, d_z), we want forcing latents from first feature dimension
            encoded_forcing_mu = q_mu_y[:, 0, n_climate_latents:]  # Shape: (batch, n_forced_latents_total)

        if self.debug_gt_z:
            z = gt_z

        # get params of the transition model p(z^t | z^{<t})
        mask = self.mask(b)
        if self.instantaneous:
            pz_mu, pz_std = self.transition(z.clone(), mask, y_co2, y_aerosol)
        else:
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask, y_co2, y_aerosol)

        # get params from decoder p(x^t | z^t)
        # we pass only the last z to the decoder, to get xs.

        px_mu, px_std = self.decode(z[:, -1], y_co2, y_aerosol)

        # set distribution with obtained parameters
        if self.distr_decoder.__name__ == "GEVDistribution":
            xi = self.xi.unsqueeze(0).expand_as(px_mu) if self.gev_learn_xi else torch.full_like(px_mu, self.xi)
            px_distr = self.distr_decoder(px_mu, px_std, xi)
            eps = 1e-6
            q_std_y_safe = q_std_y.clamp(min=eps)
            pz_std_safe = pz_std.clamp(min=eps)
            kl_raw = (
                0.5 * (torch.log(pz_std_safe**2) - torch.log(q_std_y_safe**2))
                + 0.5 * (q_std_y_safe**2 + (q_mu_y - pz_mu) ** 2) / pz_std_safe**2
                - 0.5
            )
        else:
            px_distr = self.distr_decoder(px_mu, px_std)
            recons = torch.mean(torch.sum(px_distr.log_prob(y), dim=[1, 2]))
            # compute the KL, the reconstruction and the ELBO
            # kl = distr.kl_divergence(q, p).mean()
            kl_raw = (
                0.5 * (torch.log(pz_std**2) - torch.log(q_std_y**2))
                + 0.5 * (q_std_y**2 + (q_mu_y - pz_mu) ** 2) / pz_std**2
                - 0.5
            )

        kl = torch.sum(kl_raw, dim=[2]).mean()
        # kl = torch.sum(0.5 * (torch.log(pz_std**2) - torch.log(q_std_y**2)) + 0.5 *
        # (q_std_y**2 + (q_mu_y - pz_mu) ** 2) / pz_std**2 - 0.5, dim=[1, 2]).mean()
        assert kl >= 0, f"KL={kl} has to be >= 0"

        elbo = recons - self.coeff_kl * kl

        # Compute forcing reconstruction losses
        forcing_recons_loss_co2 = torch.tensor(0.0, device=x.device)
        forcing_recons_loss_aerosol = torch.tensor(0.0, device=x.device)

        if self.use_forced_latents and y_co2 is not None and y_aerosol is not None:
            # Extract forcing latents from z (last timestep, first feature dimension, forcing latent indices)
            n_climate_latents = self.d_z - self.n_forced_latents_co2 - self.n_forced_latents_aerosol
            forcing_arch = getattr(self, "forcing_arch", "baseline")
            if forcing_arch == "baseline":
                if not self._forcing_arch_logged:
                    logger.info("[ForcingArch] Using forcing_arch='baseline' (encoded forced latents)")
                    self._forcing_arch_logged = True
                z_forced_target = z[:, -1, 0, n_climate_latents:]  # Shape: (batch, n_forced_latents_total)
                # Decode forcing latents back to forcing space
                forcing_co2_recons, forcing_aerosol_recons = self.autoencoder.decode_forcings(z_forced_target)
            elif forcing_arch == "transitioned":
                if not self._forcing_arch_logged:
                    logger.info("[ForcingArch] Using forcing_arch='transitioned' (pz_mu forced latents)")
                    self._forcing_arch_logged = True
                # Use transitioned latents (pz_mu) for forcing reconstruction
                z_forced_target = pz_mu[:, 0, n_climate_latents:]  # Shape: (batch, n_forced_latents_total)
                forcing_co2_recons, forcing_aerosol_recons = self.autoencoder.decode_forcings(z_forced_target)
            elif forcing_arch == "predefined":
                if not self._forcing_arch_logged:
                    logger.info("[ForcingArch] Using forcing_arch='predefined' (no forcing reconstruction)")
                    self._forcing_arch_logged = True
                # No forcing reconstruction in predefined conditioning mode
                forcing_co2_recons, forcing_aerosol_recons = None, None
            else:
                raise ValueError(f"Unknown forcing_arch='{forcing_arch}'")

            if forcing_arch != "predefined":
                # Get target forcings (last timestep)
                y_co2_target = y_co2[:, -1] if y_co2.dim() == 3 else y_co2
                y_aerosol_target = y_aerosol[:, -1] if y_aerosol.dim() == 3 else y_aerosol

                # Compute MSE reconstruction losses
                forcing_recons_loss_co2 = torch.mean((forcing_co2_recons - y_co2_target) ** 2)
                forcing_recons_loss_aerosol = torch.mean((forcing_aerosol_recons - y_aerosol_target) ** 2)

        return (
            elbo,
            recons,
            kl,
            px_mu,
            forcing_recons_loss_co2,
            forcing_recons_loss_aerosol,
            encoded_forcing_mu,
        )

    # def predict(self, x, y):
    #    b = x.size(0)

    #    with torch.no_grad():
    #        # sample Zs (based on X)
    #        z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol)
    #
    #        # get params of the transition model p(z^t | z^{<t})
    #        mask = self.mask(b)
    #        pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask, y_co2, y_aerosol)
    #        px_mu, px_std = self.decode(pz_mu)
    #    return px_mu, y

    def predict_pxmu_pxstd(self, x, y, y_co2=None, y_aerosol=None):

        # NOTE: this one was working fine for the CRPS loss because I was not using no_grad...
        # I need to keep the grads if I am going to add to the loss

        b = x.size(0)

        # sample Zs (based on X)
        z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol)

        # get params of the transition model p(z^t | z^{<t})
        mask = self.mask(b)
        if self.instantaneous:
            pz_mu, pz_std = self.transition(z.clone(), mask, y_co2, y_aerosol)
        else:
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask, y_co2, y_aerosol)

        # get params from decoder p(x^t | z^t)
        # we pass only the last z to the decoder, to get xs.
        px_mu, px_std = self.decode(pz_mu, y_co2, y_aerosol)

        return px_mu, px_std

    def predict(self, x, y, y_co2=None, y_aerosol=None):

        # Use no grad to speed it up! But I need to keep the grads if I am going to add to the loss.

        """
        This is the prediction function for the model.

        We want to take past time steps and predict the next time step, not to reconstruct the past time steps.
        """
        b = x.size(0)

        # NOTE: we are not using y here. We encode using both x and y,
        # but then we discard the latents from the y encoding.

        z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol)

        mask = self.mask(b)

        if self.instantaneous:
            pz_mu, pz_std = self.transition(z.clone(), mask, y_co2, y_aerosol)
        else:
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask, y_co2, y_aerosol)

        # decode
        px_mu, px_std = self.decode(pz_mu, y_co2, y_aerosol)

        return px_mu, y, z, pz_mu, pz_std

    def predict_counterfactual(self, x, y, counterfactual_z_index, counterfactual_z_value, y_co2=None, y_aerosol=None):

        # Use no grad to speed it up! But I need to keep the grads if I am going to add to the loss.

        """
        This is the prediction function for the model.

        We want to take past time steps and predict the next time step, not to reconstruct the past time steps.
        """

        b = x.size(0)

        z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol)

        logger.debug("This is the shape of the latents that we are going to intervene on. %s", z.shape)
        logger.debug(
            "Here is where we are going to intervene on the latents, and the value. %s %s",
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

        logger.debug(
            "This is e.g. the new value of the latents after intervention. %s", z[0, -2, 0, counterfactual_z_index]
        )
        assert torch.all(z[:, -2, 0, counterfactual_z_index] == counterfactual_z_value)

        mask = self.mask(b)

        if self.instantaneous:
            pz_mu, pz_std = self.transition(z.clone(), mask, y_co2, y_aerosol)
        else:
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask, y_co2, y_aerosol)

        # decode
        px_mu, px_std = self.decode(pz_mu, y_co2, y_aerosol)

        return px_mu, y, z, pz_mu, pz_std

    def predict_sample(self, x, y, num_samples, y_co2=None, y_aerosol=None):
        """
        This is a prediction function for the model, but where we take samples from the Gaussians of the latents.

        Note this function also returns the option where we sample from the decoders, but of course these samples are
        just chequerboards and not very interesting.

        I can use no_grad here, because I am not going to be using the gradients for anything.
        """

        b = x.size(0)

        with torch.no_grad():
            # sample Zs (based on X)
            z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol)

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)

            if self.instantaneous:
                pz_mu, pz_std = self.transition(z.clone(), mask, y_co2, y_aerosol)
            else:
                pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask, y_co2, y_aerosol)

            # here I am taking the approach of sampling from the Z distributions, and then decoding.
            samples_from_zs = torch.zeros(num_samples, b, self.d, self.d_x)
            z_samples = torch.zeros(num_samples, b, self.d, self.d_z)

            # TODO: Remove this for loop
            for i in range(num_samples):
                z_samples[i] = self.distr_transition(pz_mu, pz_std).sample()
                samples_from_zs[i], some_decoded_samples_std = self.decode(z_samples[i], y_co2, y_aerosol)

                # some_decoded_samples_mu, some_decoded_samples_std = self.decode(z_samples[i])

                # samples_from_zs[i] = some_decoded_samples_mu

            # decode
            px_mu, px_std = self.decode(pz_mu, y_co2, y_aerosol)

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
        self, x, y, num_samples, with_zs_logprob: bool = False, y_co2=None, y_aerosol=None
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
            z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol)

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)

            if self.instantaneous:
                pz_mu, pz_std = self.transition(z.clone(), mask, y_co2, y_aerosol)
            else:
                pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask, y_co2, y_aerosol)

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

                # self.distr_transition(pz_mu, pz_std).log_prob(z_samples[i]) gives log probability
            samples_from_zs, some_decoded_samples_std = self.decode(
                z_samples.reshape(z_samples.size(0) * z_samples.size(1), z_samples.size(2), z_samples.size(3)),
                y_co2,
                y_aerosol,
            )
            samples_from_zs = samples_from_zs.reshape(z_samples.size(0), z_samples.size(1), z_samples.size(2), self.d_x)
            # some_decoded_samples_mu, some_decoded_samples_std = self.decode(z_samples[i])

            # samples_from_zs[i] = some_decoded_samples_mu

            # decode
            px_mu, px_std = self.decode(pz_mu.unsqueeze(1), y_co2, y_aerosol)
            px_mu = px_mu.squeeze(1)
            px_std = px_std.squeeze(1)

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
            return samples_from_xs, samples_from_zs, y, z_samples_logprob
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
            logger.debug("sigma2**d_z: %s", sigma2**self.d_z)
            logger.debug("prod(sigma1): %s", torch.prod(sigma1, dim=1))
            logger.debug(
                "sum(log(sigma2**d_z / prod(sigma1))): %s",
                torch.sum(torch.log(sigma2**self.d_z / torch.prod(sigma1, dim=1))),
            )
            logger.debug("sum(sum(sigma1 / sigma2)): %s", torch.sum(torch.sum(sigma1 / sigma2, dim=1)))
            # print(torch.sum(torch.einsum('bd, bd -> b', (mu2 - mu1) * (1 / s_p), mu2 - mu1)))

        return torch.sum(kl)


class LinearAutoEncoder(nn.Module):
    def __init__(
        self,
        d,
        d_x,
        d_z,
        tied,
        d_y_co2=0,
        d_y_aerosol=0,
        use_forced_latents=False,
        n_forced_latents_co2=1,
        n_forced_latents_aerosol=4,
        d_y_co2_spatial=None,
        d_y_aerosol_spatial=None,
    ):
        super().__init__()
        self.d_y_co2 = d_y_co2
        self.d_y_aerosol = d_y_aerosol
        # Spatial dims for forcing encoders/decoders (independent of MLP conditioning dims).
        # When use_exogenous=False but use_forced_latents=True, d_y_co2 may be 0
        # (no MLP conditioning) while d_y_co2_spatial carries the real spatial dimension.
        self.d_y_co2_spatial = d_y_co2_spatial if d_y_co2_spatial is not None else d_y_co2
        self.d_y_aerosol_spatial = d_y_aerosol_spatial if d_y_aerosol_spatial is not None else d_y_aerosol
        self.d_x = d_x
        self.d_z = d_z
        self.tied = tied
        self.use_grad_project = True
        self.use_forced_latents = use_forced_latents
        self.n_forced_latents_co2 = n_forced_latents_co2
        self.n_forced_latents_aerosol = n_forced_latents_aerosol

        unif = (1 - 0.1) * torch.rand(size=(d, d_x, d_z)) + 0.1
        self.w = nn.Parameter(unif / torch.as_tensor(d_z))
        if not tied:
            unif = (1 - 0.1) * torch.rand(size=(d, d_z, d_x)) + 0.1
            self.w_encoder = nn.Parameter(unif / torch.as_tensor(d_x))

        self.logvar_encoder = nn.Parameter(torch.ones(d_z) * -1)
        self.logvar_decoder = nn.Parameter(torch.ones(d_x) * -1)

        if use_forced_latents:
            self.n_climate_latents = d_z - n_forced_latents_co2 - n_forced_latents_aerosol

            # CO2 forcing encoder: linear projection (uses real spatial dim, not MLP conditioning dim)
            self.co2_forcing_encoder_mu = nn.Linear(self.d_y_co2_spatial, n_forced_latents_co2)
            self.co2_forcing_encoder_logvar = nn.Parameter(torch.ones(n_forced_latents_co2) * -1)

            # Aerosol forcing encoder: linear projection
            self.aerosol_forcing_encoder_mu = nn.Linear(self.d_y_aerosol_spatial, n_forced_latents_aerosol)
            self.aerosol_forcing_encoder_logvar = nn.Parameter(torch.ones(n_forced_latents_aerosol) * -1)

            # Forcing decoder spatial weights
            self.w_co2 = nn.Parameter(torch.randn(self.d_y_co2_spatial, n_forced_latents_co2))
            self.w_aerosol = nn.Parameter(torch.randn(self.d_y_aerosol_spatial, n_forced_latents_aerosol))

    def get_w_encoder(self):
        if self.tied:
            return torch.transpose(self.w, 1, 2)
        else:
            return self.w_encoder

    def get_w_decoder(self):
        return self.w

    def get_w_co2(self):
        if self.use_forced_latents:
            return self.w_co2.detach()
        return None

    def get_w_aerosol(self):
        if self.use_forced_latents:
            return self.w_aerosol.detach()
        return None

    def encode_forcings(self, forcing_co2, forcing_aerosol):
        """Encode forcings into latent representations using linear projections."""
        batch_size = forcing_co2.shape[0]
        device = forcing_co2.device

        co2_mu = self.co2_forcing_encoder_mu(forcing_co2)
        co2_std = torch.exp(0.5 * self.co2_forcing_encoder_logvar).expand(batch_size, -1).to(device)

        aerosol_mu = self.aerosol_forcing_encoder_mu(forcing_aerosol)
        aerosol_std = torch.exp(0.5 * self.aerosol_forcing_encoder_logvar).expand(batch_size, -1).to(device)

        co2_z = co2_mu + co2_std * torch.randn_like(co2_std)
        aerosol_z = aerosol_mu + aerosol_std * torch.randn_like(aerosol_std)

        z_forced = torch.cat([co2_z, aerosol_z], dim=1)
        mu_forced = torch.cat([co2_mu, aerosol_mu], dim=1)
        std_forced = torch.cat([co2_std, aerosol_std], dim=1)

        return z_forced, mu_forced, std_forced

    def decode_co2_forcing(self, z_co2):
        """Decode CO2 forcing latents using spatial weights."""
        z_expanded = z_co2.unsqueeze(1).expand(-1, self.d_y_co2_spatial, -1)
        z_masked = z_expanded * self.w_co2.unsqueeze(0)
        return z_masked.sum(dim=-1)

    def decode_aerosol_forcing(self, z_aerosol):
        """Decode aerosol forcing latents using spatial weights."""
        z_expanded = z_aerosol.unsqueeze(1).expand(-1, self.d_y_aerosol_spatial, -1)
        z_masked = z_expanded * self.w_aerosol.unsqueeze(0)
        return z_masked.sum(dim=-1)

    def decode_forcings(self, z_forced_latents):
        """Decode forcing latents back to forcing space for reconstruction loss."""
        co2_latents = z_forced_latents[:, : self.n_forced_latents_co2]
        aerosol_latents = z_forced_latents[:, self.n_forced_latents_co2 :]
        return self.decode_co2_forcing(co2_latents), self.decode_aerosol_forcing(aerosol_latents)

    def encode(self, x, i, forcing_co2=None, forcing_aerosol=None):
        if self.tied:
            w = self.w[i].T
        else:
            w = self.w_encoder[i]
        mu = torch.matmul(x, w.T)
        return mu, self.logvar_encoder

    def decode(self, z, i, forcing_co2=None, forcing_aerosol=None):
        w = self.w[i]
        # When using forced latents, z only contains climate latents (sliced by caller).
        # Use only the corresponding climate columns of w.
        if self.use_forced_latents and z.shape[-1] < w.shape[-1]:
            w = w[:, : z.shape[-1]]
        mu = torch.matmul(z, w.T)
        return mu, self.logvar_decoder

    def forward(self, x, i, encode: bool = False, forcing_co2=None, forcing_aerosol=None):
        if encode:
            return self.encode(x, i, forcing_co2, forcing_aerosol)
        else:
            return self.decode(x, i, forcing_co2, forcing_aerosol)


class NonLinearAutoEncoder(nn.Module):
    def __init__(self, d, d_x, d_z, num_hidden, num_layer, use_gumbel_mask, tied, gt_w=None):
        super().__init__()
        if use_gumbel_mask:
            self.use_grad_project = False
        else:
            self.use_grad_project = True
        self.d_x = d_x
        self.d_z = d_z
        self.tied = tied
        self.use_gumbel_mask = use_gumbel_mask

        if self.use_gumbel_mask:
            self.mask = MixingMask(d, d_x, d_z, gt_w)
            if not tied:
                self.mask_encoder = MixingMask(d, d_x, d_z, gt_w)
        else:
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
        if self.use_gumbel_mask:
            if self.tied:
                return torch.transpose(self.mask.param, 1, 2)
            else:
                return torch.transpose(self.mask_encoder.param, 1, 2)
        else:
            if self.tied:
                return torch.transpose(self.w, 1, 2)
                # return self.w
            else:
                return self.w_encoder

    def get_w_decoder(self):
        if self.use_gumbel_mask:
            return self.mask.param
        else:
            return self.w

    def get_w_co2(self):
        """Get CO2 forcing decoder spatial weights."""
        if self.use_forced_latents:
            return self.w_co2.detach()
        else:
            return None

    def get_w_aerosol(self):
        """Get aerosol forcing decoder spatial weights."""
        if self.use_forced_latents:
            return self.w_aerosol.detach()
        else:
            return None

    def get_encode_mask(self, bs_size: int):
        if self.use_gumbel_mask:
            if self.tied:
                sampled_mask = self.mask(bs_size)
            else:
                sampled_mask = self.mask_encoder(bs_size)
        else:
            if self.tied:
                return torch.transpose(self.w, 1, 2)
            else:
                return self.w_encoder
        return sampled_mask

    def select_encoder_mask(self, mask, i, j):
        if self.use_gumbel_mask:
            mask = mask[:, i, :, j]
        else:
            mask = mask[i, j]
        return mask

    def get_decode_mask(self, bs_size: int):
        if self.use_gumbel_mask:
            sampled_mask = self.mask(bs_size)
            # size: bs, dx, dz, 1
        else:
            sampled_mask = self.w
            # size: dx, dz, 1

        return sampled_mask

    def select_decoder_mask(self, mask, i, j):
        if self.use_gumbel_mask:
            mask = mask[:, i, j]
        else:
            mask = mask[i, j]
        return mask


class NonLinearAutoEncoderUniqueMLP_noloop(NonLinearAutoEncoder):

    # TODO(dev): Investigate embedding naming confusion. After analysis, this is NOT a bug:
    # - embedding_encoder is nn.Embedding(d_z, dim): embeds latent indices (0..d_z-1) so the
    #   shared encoder MLP can distinguish which latent variable it is computing.
    # - embedding_decoder is nn.Embedding(d_x, dim): embeds spatial indices (0..d_x-1) so the
    #   shared decoder MLP can distinguish which spatial location it is reconstructing.
    # The names are misleading (embedding_encoder indexes latents, not observations), but
    # the usage in encode() and decode() is correct and intentional by design.

    def __init__(
        self,
        d,
        d_x,
        d_z,
        num_hidden,
        num_layer,
        use_gumbel_mask,
        tied,
        embedding_dim,
        reduce_encoding_pos_dim,
        gt_w=None,
        d_y_co2=0,
        d_y_aerosol=0,
        use_forced_latents=False,
        n_forced_latents_co2=1,
        n_forced_latents_aerosol=4,
        d_y_co2_spatial=None,
        d_y_aerosol_spatial=None,
    ):
        super().__init__(d, d_x, d_z, num_hidden, num_layer, use_gumbel_mask, tied, gt_w)
        self.d_y_co2 = d_y_co2
        self.d_y_aerosol = d_y_aerosol
        # Spatial dims for forcing encoders/decoders (independent of MLP conditioning dims).
        # When use_exogenous=False but use_forced_latents=True, d_y_co2 may be 0
        # (no MLP conditioning) while d_y_co2_spatial carries the real spatial dimension.
        self.d_y_co2_spatial = d_y_co2_spatial if d_y_co2_spatial is not None else d_y_co2
        self.d_y_aerosol_spatial = d_y_aerosol_spatial if d_y_aerosol_spatial is not None else d_y_aerosol
        self.use_forced_latents = use_forced_latents
        self.n_forced_latents_co2 = n_forced_latents_co2
        self.n_forced_latents_aerosol = n_forced_latents_aerosol

        # embedding_dim_encoding = d_z // 10
        if not reduce_encoding_pos_dim:
            self.embedding_encoder = nn.Embedding(d_z, embedding_dim)
            self.encoder = MLP(
                num_layer, num_hidden, d_x + embedding_dim + d_y_co2 + d_y_aerosol, 1
            )  # embedding_dim_encoding
        else:
            self.embedding_encoder = nn.Embedding(d_z, embedding_dim // 10)
            self.encoder = MLP(
                num_layer, num_hidden, d_x + embedding_dim // 10 + d_y_co2 + d_y_aerosol, 1
            )  # embedding_dim_encoding
        # self.encoder = MLP(num_layer, num_hidden, d_x + embedding_dim, 1)
        # self.embedding_encoder = nn.Embedding(d_z, embedding_dim)

        self.embedding_decoder = nn.Embedding(d_x, embedding_dim)

        # Create climate-only decoder if using forced latents
        if use_forced_latents:
            # Climate-only decoder: only climate latents go through this decoder
            n_climate_latents = d_z - n_forced_latents_co2 - n_forced_latents_aerosol
            self.n_climate_latents = n_climate_latents
            self.decoder = MLP(num_layer, num_hidden, n_climate_latents + embedding_dim + d_y_co2 + d_y_aerosol, 1)
        else:
            # Original decoder: all latents
            self.decoder = MLP(num_layer, num_hidden, d_z + embedding_dim + d_y_co2 + d_y_aerosol, 1)
            self.n_climate_latents = d_z

        # Add forcing encoders if using forced latents (use real spatial dims, not MLP conditioning dims)
        if self.use_forced_latents:
            # CO2 forcing encoder: maps spatial CO2 to n_forced_latents_co2 latent means + logvars
            self.co2_forcing_encoder_mu = MLP(num_layer, num_hidden, self.d_y_co2_spatial, n_forced_latents_co2)
            self.co2_forcing_encoder_logvar = nn.Parameter(torch.ones(n_forced_latents_co2) * -1)

            # Aerosol forcing encoder: maps spatial aerosols to n_forced_latents_aerosol latent means + logvars
            self.aerosol_forcing_encoder_mu = MLP(
                num_layer, num_hidden, self.d_y_aerosol_spatial, n_forced_latents_aerosol
            )
            self.aerosol_forcing_encoder_logvar = nn.Parameter(torch.ones(n_forced_latents_aerosol) * -1)

            # Forcing decoder weights: spatial mask (like climate decoder)
            self.w_co2 = nn.Parameter(torch.randn(self.d_y_co2_spatial, n_forced_latents_co2))
            self.w_aerosol = nn.Parameter(torch.randn(self.d_y_aerosol_spatial, n_forced_latents_aerosol))

    def encode_forcings(self, forcing_co2, forcing_aerosol):
        """
        Encode forcings directly into latent representations.

        Args:
            forcing_co2: CO2 forcing, shape (batch_size, d_y_co2)
            forcing_aerosol: Aerosol forcing, shape (batch_size, d_y_aerosol)

        Returns:
            z_forced: Forced latents, shape (batch_size, n_forced_latents_co2 + n_forced_latents_aerosol)
            mu_forced: Means of forced latents
            std_forced: Stds of forced latents
        """
        batch_size = forcing_co2.shape[0]
        device = forcing_co2.device

        # Encode CO2 forcing
        co2_mu = self.co2_forcing_encoder_mu(forcing_co2)  # (batch_size, n_forced_latents_co2)
        co2_std = torch.exp(0.5 * self.co2_forcing_encoder_logvar).expand(batch_size, -1).to(device)

        # Encode aerosol forcing
        aerosol_mu = self.aerosol_forcing_encoder_mu(forcing_aerosol)  # (batch_size, n_forced_latents_aerosol)
        aerosol_std = torch.exp(0.5 * self.aerosol_forcing_encoder_logvar).expand(batch_size, -1).to(device)

        # Sample forced latents using reparameterization trick
        co2_z = co2_mu + co2_std * torch.randn_like(co2_std)
        aerosol_z = aerosol_mu + aerosol_std * torch.randn_like(aerosol_std)

        # Concatenate forced latents
        z_forced = torch.cat([co2_z, aerosol_z], dim=1)
        mu_forced = torch.cat([co2_mu, aerosol_mu], dim=1)
        std_forced = torch.cat([co2_std, aerosol_std], dim=1)

        return z_forced, mu_forced, std_forced

    def decode_co2_forcing(self, z_co2):
        """
        Decode CO2 forcing latents using spatial mask (like climate decoder).

        Args:
            z_co2: CO2 latents, shape (batch, n_forced_latents_co2)  # (batch, 1)

        Returns:
            co2_recons: Reconstructed CO2 field, shape (batch, d_y_co2)  # (batch, 400)
        """
        # z_co2: (batch, 1)
        # self.w_co2: (400, 1)
        # Output: (batch, 400)

        # Expand z to (batch, 400, 1)
        z_expanded = z_co2.unsqueeze(1).expand(-1, self.d_y_co2_spatial, -1)

        # Apply mask: element-wise multiply with w_co2
        z_masked = z_expanded * self.w_co2.unsqueeze(0)  # (batch, 400, 1)

        # Sum over latents (linear combination)
        co2_recons = z_masked.sum(dim=-1)  # (batch, 400)

        return co2_recons

    def decode_aerosol_forcing(self, z_aerosol):
        """
        Decode aerosol forcing latents using spatial mask (like climate decoder).

        Args:
            z_aerosol: Aerosol latents, shape (batch, n_forced_latents_aerosol)  # (batch, 4)

        Returns:
            aerosol_recons: Reconstructed aerosol field, shape (batch, d_y_aerosol)  # (batch, 400)
        """
        # z_aerosol: (batch, 4)
        # self.w_aerosol: (400, 4)
        # Output: (batch, 400)

        # Expand z to (batch, 400, 4)
        z_expanded = z_aerosol.unsqueeze(1).expand(-1, self.d_y_aerosol_spatial, -1)

        # Apply mask: element-wise multiply with w_aerosol
        z_masked = z_expanded * self.w_aerosol.unsqueeze(0)  # (batch, 400, 4)

        # Sum over latents (linear combination)
        aerosol_recons = z_masked.sum(dim=-1)  # (batch, 400)

        return aerosol_recons

    def decode_forcings(self, z_forced_latents):
        """
        Decode forcing latents back to forcing space for reconstruction loss.

        Args:
            z_forced_latents: Forced latents, shape (batch_size, n_forced_latents_co2 + n_forced_latents_aerosol)

        Returns:
            forcing_co2_recons: Reconstructed CO2 forcing, shape (batch_size, d_y_co2)
            forcing_aerosol_recons: Reconstructed aerosol forcing, shape (batch_size, d_y_aerosol)
        """
        # Split forced latents into CO2 and aerosol components
        co2_latents = z_forced_latents[:, : self.n_forced_latents_co2]
        aerosol_latents = z_forced_latents[:, self.n_forced_latents_co2 :]

        # Decode to forcing space using spatial mask decoders
        forcing_co2_recons = self.decode_co2_forcing(co2_latents)
        forcing_aerosol_recons = self.decode_aerosol_forcing(aerosol_latents)

        return forcing_co2_recons, forcing_aerosol_recons

    def encode(self, x, i, forcing_co2=None, forcing_aerosol=None):

        mask = super().get_encode_mask(x.shape[0])
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
        if forcing_co2 is not None and forcing_aerosol is not None:
            forcing_co2_expanded = forcing_co2.unsqueeze(1).expand(-1, self.d_z, -1)
            forcing_aerosol_expanded = forcing_aerosol.unsqueeze(1).expand(-1, self.d_z, -1)
            x_ = torch.cat((mask_ * x.unsqueeze(1), embedded_x, forcing_co2_expanded, forcing_aerosol_expanded), dim=2)
        else:
            x_ = torch.cat(
                (mask_ * x.unsqueeze(1), embedded_x), dim=2
            )  # expand dimensions of x for broadcasting - looks good.

        del embedded_x
        del mask_

        mu = self.encoder(x_).squeeze()

        return mu, self.logvar_encoder

    def decode(self, z, i, forcing_co2=None, forcing_aerosol=None):
        """
        Decode latents to observations.

        When use_forced_latents=True, this method expects ONLY climate latents
        (the caller should slice z to only include climate latents).
        Raw forcing fields (forcing_co2, forcing_aerosol) are still used as
        conditioning inputs to the MLP but don't go through the learnable mask.

        Args:
            z: Latent variables. If use_forced_latents, should be climate latents only
               with shape (batch, n_climate_latents). Otherwise (batch, d_z).
            i: Feature index
            forcing_co2: Raw CO2 forcing field for conditioning (NOT forcing latent!)
            forcing_aerosol: Raw aerosol forcing field for conditioning (NOT forcing latent!)
        """
        mask = super().get_decode_mask(z.shape[0])
        mu = torch.zeros((z.shape[0], self.d_x), device=z.device)

        # Create a tensor of shape (z.shape[0], self.d_x) where each row is a sequence from 0 to self.d_x
        j_values = torch.arange(self.d_x, device=z.device).expand(z.shape[0], -1)

        # Embed all j_values at once
        embedded_z = self.embedding_decoder(j_values)

        # Select decoder masks - only use climate portion of mask
        mask_ = super().select_decoder_mask(mask, i, j_values)

        # Only use climate latents (first n_climate columns of mask)
        n_climate = self.n_climate_latents
        if mask_.dim() == 3:
            # mask_ shape: (batch, d_x, d_z) -> slice to (batch, d_x, n_climate)
            mask_climate = mask_[:, :, :n_climate]
        else:
            # mask_ shape: (d_x, d_z) -> slice to (d_x, n_climate)
            mask_climate = mask_[:, :n_climate]

        if z.ndim < mask_climate.ndim:
            z_expanded = z.unsqueeze(1).expand(-1, self.d_x, -1)
        else:
            z_expanded = z.expand(-1, self.d_x, -1)
        z_expanded_copy = z_expanded.clone()
        z_expanded_copy.mul_(mask_climate)

        # Raw forcing fields as conditioning (correct design - doesn't go through mask)
        if forcing_co2 is not None and forcing_aerosol is not None:
            forcing_co2_expanded = forcing_co2.unsqueeze(1).expand(-1, self.d_x, -1)
            forcing_aerosol_expanded = forcing_aerosol.unsqueeze(1).expand(-1, self.d_x, -1)
            z_ = torch.cat((z_expanded_copy, embedded_z, forcing_co2_expanded, forcing_aerosol_expanded), dim=2)
        else:
            z_ = torch.cat((z_expanded_copy, embedded_z), dim=2)

        del z_expanded
        del z_expanded_copy

        # Apply the decoder to all z_ at once and squeeze the result
        mu = self.decoder(z_).squeeze()

        return mu, self.logvar_decoder

    def forward(self, x, i, encode: bool = False, forcing_co2=None, forcing_aerosol=None):
        if encode:
            return self.encode(x, i, forcing_co2, forcing_aerosol)
        else:
            return self.decode(x, i, forcing_co2, forcing_aerosol)


class TransitionModel(nn.Module):
    """Models the transitions between the latent variables Z with neural networks."""

    def __init__(
        self,
        d: int,
        d_z: int,
        tau: int,
        nonlinear_dynamics: bool,
        num_layers: int,
        num_hidden: int,
        num_output: int = 2,
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
        if self.nonlinear_dynamics:
            logger.info("NON LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(num_layers, num_hidden, d * d_z * tau, self.num_output) for i in range(d * d_z))
        else:
            logger.info("LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(0, 0, d * d_z * tau, self.num_output) for i in range(d * d_z))
        # self.nn = MLP(num_layers, num_hidden, d * k * k, self.num_output)

    def forward(self, z, mask, i, k):
        """Returns the params of N(z_t | z_{<t}) for a specific feature i and latent variable k NN(G_{tau-1} * z_{t-1},
        ..., G_{tau-k} * z_{t-k})"""

        # works well for original prediction when we do not do instantaneous! :)
        # e.g.     val_mse, val_smape = prediction_original(trainer, True)

        # t_total = torch.max(self.tau, z_past.size(1))  # TODO: find right dim
        # param_z = torch.zeros(z_past.size(0), 2)

        # print("In the forward of the transition model, and trying to ascertain which way the information flows through the mask.")
        # print("The mask is of size: ", mask.size())
        # print("The z is of size: ", z.size())

        # print the unique values and their counts in mask:
        # print("The unique values of mask are: ", torch.unique(mask))
        # print("The counts of the unique values of mask are: ", torch.unique(mask, return_counts=True))

        # print the first few elements of z

        z = z.view(mask.size())

        # print("The z is now, after z.view() of size: ", z.size())

        # print("what is mask * z shape? ", (mask * z).size())

        masked_z = (mask * z).view(z.size(0), -1)

        # print("mask * z is of size: ", (mask * z).size())
        # print("The masked_z is of size: ", masked_z.size())

        # print the first few elements of masked_z
        # print("The first few elements of masked_z are: ", masked_z[0, :10])

        # print all the unique values of masked_z, and the number of unique values.
        # print("The unique values of masked_z are: ", torch.unique(masked_z))

        # count the number of very small values in masked_z
        # print("The number of very small values in masked_z are: ", torch.sum(masked_z < 0.0001))

        # print("What is i, self_d_z, k? ", i, self.d_z, k)
        # print("What is i * self.d_z + k? ", i * self.d_z + k)
        # print("What is self.nn[i * self.d_z + k]?", self.nn[i * self.d_z + k])

        param_z = self.nn[i * self.d_z + k](masked_z)

        # print("What is the shape of param_z?", param_z.size())

        # param_z = self.nn(masked_z)

        return param_z


class TransitionModelParamSharing(nn.Module):
    """Models the transitions between the latent variables Z with neural networks."""

    # Attempt at parameter sharing in the transition model

    def __init__(
        self,
        d: int,
        d_z: int,
        tau: int,
        nonlinear_dynamics: bool,
        num_layers: int,
        num_hidden: int,
        num_output: int = 2,
        embedding_dim: int = 100,
        d_y_co2: int = 0,
        d_y_aerosol: int = 0,
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
        self.d_y_co2 = d_y_co2
        self.d_y_aerosol = d_y_aerosol
        self.d = d  # number of variables
        self.d_z = d_z
        self.tau = tau
        output_var = False

        # initialize NNs
        self.nonlinear_dynamics = nonlinear_dynamics
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.embedding_dim = embedding_dim
        self.embedding_transition = nn.Embedding(d_z, embedding_dim)

        if output_var:
            self.num_output = num_output
        else:
            self.num_output = 1
            # self.logvar = torch.ones(1)  * 0. # nn.Parameter(torch.ones(d) * 0.1)
            # self.logvar = nn.Parameter(torch.ones(d) * -4)
            self.logvar = nn.Parameter(torch.ones(d, d_z) * -4)
        if self.nonlinear_dynamics:
            logger.info("NON LINEAR DYNAMICS")
            self.nn = nn.ModuleList(
                MLP(num_layers, num_hidden, d * d_z * tau + embedding_dim + d_y_co2 + d_y_aerosol, self.num_output)
                for i in range(d)
            )
        else:
            logger.info("LINEAR DYNAMICS")
            self.nn = nn.ModuleList(
                MLP(0, 0, d * d_z * tau + embedding_dim + d_y_co2 + d_y_aerosol, self.num_output) for i in range(d)
            )
        # self.nn = MLP(num_layers, num_hidden, d * k * k, self.num_output)

    def forward(self, z, mask, i, forcing_co2=None, forcing_aerosol=None):
        """Returns the params of N(z_t | z_{<t}) for a specific feature i and latent variable k NN(G_{tau-1} * z_{t-1},
        ..., G_{tau-k} * z_{t-k})"""

        j_values = torch.arange(self.d_z, device=z.device).expand(
            z.shape[0], -1
        )  # create a 2D tensor with shape (x.shape[0], self.d_z)
        embedded_z = self.embedding_transition(j_values)
        masked_z = (mask * z).transpose(3, 2).reshape((z.shape[0], -1, self.d_z)).transpose(2, 1)
        if forcing_co2 is not None and forcing_aerosol is not None:
            forcing_co2_expanded = forcing_co2.unsqueeze(1).expand(-1, self.d_z, -1)
            forcing_aerosol_expanded = forcing_aerosol.unsqueeze(1).expand(-1, self.d_z, -1)
            z_ = torch.cat((masked_z, embedded_z, forcing_co2_expanded, forcing_aerosol_expanded), dim=2)
        else:
            z_ = torch.cat((masked_z, embedded_z), dim=2)

        param_z = self.nn[i](z_)

        del embedded_z
        del masked_z
        del z_

        # print("What is the shape of param_z?", param_z.size())

        # param_z = self.nn(masked_z)

        return param_z


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
                logger.warning("[NaN DETECTED] in GEV log_prob!")

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
