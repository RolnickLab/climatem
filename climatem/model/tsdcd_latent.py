"""
Core causal discovery model with latent variables (LatentTSDCD).

This module implements latent causal graph learning with an encoder/decoder architecture and a learnable causal mask.
The model discovers temporal causal structure among latent variables from observed time series, using differentiable
structure learning with Gumbel-softmax relaxation and acyclicity constraints.

Adapted from the original code for CDSD (Brouillard et al., 2024):     "Causal Discovery with Score-based methods for
time series with latent     confounders."

update:
1. no forcing condition in encoder and decoder (remove use_extrogenous choice), so that the forcing and climate are encoded and decoded separately
but I preserved the choice of encode climate into n_climate + n_forcing choice and decode z_[climate, forcing] to accomodate with use_forced_latents
2. Add nonlinear MLP for the forcing decoding MLP(zw)
3. Return forcing distribution instead of the deterministic forcings
4. adapt class Mask to allow the instantous effect from forcing to climate
5. Reconstruction loss of forcings adapt from MSE to distributional loss (p(f)|y_f)
5.1 7/28 temporaly used MSE for forcings again
"""

# Hierachical Model (x-> z-> z_global) or (x-> z1-> z2)
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
        instantaneous_forcing: bool,
        fixed: bool = False,
        fixed_output_fraction: float = 1.0,
        nodiag: bool = False,
        n_climate: int = 4,
        n_exclude_global_forcing: int = 0,  # NEW: #global forcing latents excluded from forcing-to-climate path.
    ):
        super().__init__()

        self.d = d
        self.d_x = d_x
        self.tau = tau
        self.latent = latent
        self.instantaneous = instantaneous
        self.instantaneous_forcing = instantaneous_forcing
        self.fixed = fixed
        self.fixed_output_fraction = fixed_output_fraction
        # Here we can just set what we want the output to be.
        self.fixed_output = None
        self.uniform = distr.uniform.Uniform(0, 1)

        # Here we could change how the mask is instantiated in the causal graph.
        if self.latent:
            if self.instantaneous:
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
                    # set diagnoal elements to 0
                    self.fixed_mask[:, torch.arange(self.fixed_mask.size(1)), torch.arange(self.fixed_mask.size(2))] = 0
                self.fixed_mask[-1, torch.arange(self.fixed_mask.size(1)), torch.arange(self.fixed_mask.size(2))] = 0
            else:
                # if climate has no instantaneous effect:
                # if forcings has instantaneous effect to climate
                # for mask shape (b,t, z, z) and z shape (b,t, z, 1) : masked_z[b,t,i,j]=mask[b,t,i,j]⋅z[b,t,i], i is the source, j is the target
                # to confirm -1 is current step?
                n_steps = self.tau + 1  # tau + current
                # param (time, source, target)?
                param = torch.ones((n_steps, d * d_x, d * d_x)) * 5
                fixed_mask = torch.zeros_like(param)

                # 1. climate(past) -> climate future，but can't be impacted by the current climate
                fixed_mask[:-1, :n_climate, :n_climate] = 1

                # 2. any time step forcings -> climate
                # if n_exclude_global_forcing !=0, then we only consider spatial_forcings -> climate
                fixed_mask[:, :n_climate, n_climate + n_exclude_global_forcing :] = 1  # b,t, target, source
                if not self.instantaneous_forcing:
                    # turn off current focings -> climate
                    fixed_mask[-1, :n_climate, n_climate:] = 0  # b,t, target, source
                    # 3. who can impact forcing? -> we don't model the impact among forcings
                    # fixed_mask[:, :, external] = 1
                if nodiag:
                    idx = torch.arange(d * d_x)
                    param[:, idx, idx] = -5
                    fixed_mask[:, idx, idx] = 0

                param[fixed_mask == 0] = -5
                self.param = nn.Parameter(param)
                self.fixed_mask = fixed_mask

        else:
            #  the new logic for nonlatent is not implemneted
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
        self.n_climate = n_climate

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
            n_steps = self.tau + 1 if self.instantaneous_forcing else self.tau
            if self.fixed_output is None:
                # We are using a fixed mask of 1s, or a fraction of 1s.
                # Set a seed so we can keep the same fixed mask.
                torch.manual_seed(353)
                num_elements = n_steps * self.d_x * self.d_x
                num_ones = int(num_elements * self.fixed_output_fraction)

                # overwrite the fixed mask here
                self.fixed_mask = torch.zeros((n_steps, self.d_x, self.d_x))

                # here we are just selecting a random number of ones in the mask.
                indices = torch.multinomial(torch.ones(num_elements), num_ones, replacement=False)
                # Convert linear indices to 3D indices
                (
                    i,
                    j,
                    k,
                ) = torch.unravel_index(indices, (n_steps, self.d_x, self.d_x))
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


class LinearAutoEncoder(nn.Module):
    """For linear autoencoder, I will fix as only accept two forcings, as the real four forings will not use the linear
    version."""

    def __init__(
        self,
        d,
        d_x,
        d_z,
        tied,
        use_forced_latents=False,
        n_forced_latents_co2=1,
        n_forced_latents_aerosol=4,
        n_forced_latents_ch4=0,
        n_forced_latents_so2=0,
        d_y_co2_spatial=0,
        d_y_aerosol_spatial=0,
        d_y_ch4_spatial=0,
        d_y_so2_spatial=0,
        forcing_mse=False,
    ):
        super().__init__()
        # Spatial dims for forcing encoders/decoders (independent of MLP conditioning dims).
        # When use_exogenous=False but use_forced_latents=True, d_y_co2 may be 0
        # (no MLP conditioning) while d_y_co2_spatial carries the real spatial dimension.
        self.d_y_co2_spatial = d_y_co2_spatial
        self.d_y_aerosol_spatial = d_y_aerosol_spatial
        self.d_x = d_x
        self.d_z = d_z
        self.tied = tied
        self.use_grad_project = True
        self.use_forced_latents = use_forced_latents
        self.n_forced_latents_co2 = n_forced_latents_co2
        self.n_forced_latents_aerosol = n_forced_latents_aerosol
        # I don't notice a big difference when using the dz=climate +forcing or only climate
        # n_latents_climate = d_z-n_forced_latents_co2-n_forced_latents_aerosol
        # unif = (1 - 0.1) * torch.rand(size=(d, d_x, d_z)) + 0.1
        # self.w = nn.Parameter(unif / torch.as_tensor(d_z))
        # if not tied:
        #     unif = (1 - 0.1) * torch.rand(size=(d, d_z, d_x)) + 0.1
        #     self.w_encoder = nn.Parameter(unif / torch.as_tensor(d_x))

        # self.logvar_encoder = nn.Parameter(torch.ones(d_z) * -1)
        # self.logvar_decoder = nn.Parameter(torch.ones(d_x) * -1)

        if use_forced_latents:
            n_climate = d_z - n_forced_latents_co2 - n_forced_latents_aerosol
        else:
            n_climate = d_z
        self.n_climate = n_climate

        unif = (1 - 0.1) * torch.rand(size=(d, d_x, d_z)) + 0.1
        self.w = nn.Parameter(unif / torch.as_tensor(d_z))
        if not tied:
            unif = (1 - 0.1) * torch.rand(size=(d, d_z, d_x)) + 0.1
            self.w_encoder = nn.Parameter(unif / torch.as_tensor(d_x))

        self.logvar_encoder = nn.Parameter(torch.ones(d_z) * -1)
        self.logvar_decoder = nn.Parameter(torch.ones(d_x) * -1)
        if use_forced_latents:
            # Decoder weights, same style as x: (d, observed_dim, latent_dim)
            unif = (1 - 0.1) * torch.rand(size=(self.d_y_co2_spatial, n_forced_latents_co2)) + 0.1
            self.w_co2 = nn.Parameter(unif / torch.as_tensor(n_forced_latents_co2))

            unif = (1 - 0.1) * torch.rand(size=(self.d_y_aerosol_spatial, n_forced_latents_aerosol)) + 0.1
            self.w_aerosol = nn.Parameter(unif / torch.as_tensor(n_forced_latents_aerosol))

            if not tied:
                unif = (1 - 0.1) * torch.rand(size=(n_forced_latents_co2, self.d_y_co2_spatial)) + 0.1
                self.w_co2_encoder = nn.Parameter(unif / torch.as_tensor(self.d_y_co2_spatial))

                unif = (1 - 0.1) * torch.rand(size=(n_forced_latents_aerosol, self.d_y_aerosol_spatial)) + 0.1
                self.w_aerosol_encoder = nn.Parameter(unif / torch.as_tensor(self.d_y_aerosol_spatial))

            self.logvar_co2_encoder = nn.Parameter(torch.ones(n_forced_latents_co2) * -1)
            self.logvar_aerosol_encoder = nn.Parameter(torch.ones(n_forced_latents_aerosol) * -1)

            self.logvar_co2_decoder = nn.Parameter(torch.ones(self.d_y_co2_spatial) * -1)
            self.logvar_aerosol_decoder = nn.Parameter(torch.ones(self.d_y_aerosol_spatial) * -1)
        self.forcing_mse = forcing_mse

    def get_w_encoder(self):
        if self.tied:
            return torch.transpose(self.w, 1, 2)
        else:
            return self.w_encoder

    def get_w_decoder(self):
        # return self.w

        w_climate = torch.relu(self.w[..., : self.n_climate])
        w_forcing = self.w[..., self.n_climate :]
        return torch.cat([w_climate, w_forcing], dim=-1)

    def get_w_co2_encoder(self):
        if self.tied:
            return torch.transpose(self.w_co2, 0, 1)
        return self.w_co2_encoder

    def get_w_aerosol_encoder(self):
        if self.tied:
            return torch.transpose(self.w_aerosol, 0, 1)
        return self.w_aerosol_encoder

    def get_w_co2(self):
        return self.w_co2 if self.use_forced_latents else None

    def get_w_aerosol(self):
        return self.w_aerosol if self.use_forced_latents else None

    def encode_forcings(self, forcing_dict):
        forcing_co2 = forcing_dict["co2"]
        forcing_aerosol = forcing_dict["aerosol"]
        batch_size = forcing_co2.shape[0]
        device = forcing_co2.device

        w_co2 = self.get_w_co2_encoder()
        w_aerosol = self.get_w_aerosol_encoder()

        co2_mu = torch.matmul(forcing_co2, w_co2.T)
        aerosol_mu = torch.matmul(forcing_aerosol, w_aerosol.T)

        co2_std = torch.exp(0.5 * self.logvar_co2_encoder).expand(batch_size, -1).to(device)
        aerosol_std = torch.exp(0.5 * self.logvar_aerosol_encoder).expand(batch_size, -1).to(device)
        #  From probalistic to determinsitic
        if self.forcing_mse:
            co2_z = co2_mu
            aerosol_z = aerosol_mu
        else:
            co2_z = co2_mu + co2_std * torch.randn_like(co2_std)
            aerosol_z = aerosol_mu + aerosol_std * torch.randn_like(aerosol_std)

        z_forced = torch.cat([co2_z, aerosol_z], dim=1)
        mu_forced = torch.cat([co2_mu, aerosol_mu], dim=1)
        std_forced = torch.cat([co2_std, aerosol_std], dim=1)

        return z_forced, mu_forced, std_forced

    def decode_co2_forcing(self, z_co2):
        w = self.w_co2
        mu = torch.matmul(z_co2, w.T)
        return mu, self.logvar_co2_decoder

    def decode_aerosol_forcing(self, z_aerosol):
        """Decode aerosol forcing latents using spatial weights."""
        w = self.w_aerosol
        mu = torch.matmul(z_aerosol, w.T)
        return mu, self.logvar_aerosol_decoder

    def decode_forcings(self, z_forced_latents):
        outputs = {}
        co2_latents = z_forced_latents[:, : self.n_forced_latents_co2]
        aerosol_latents = z_forced_latents[:, self.n_forced_latents_co2 :]

        co2_mu, co2_logvar = self.decode_co2_forcing(co2_latents)
        aerosol_mu, aerosol_logvar = self.decode_aerosol_forcing(aerosol_latents)

        outputs["co2_mu"] = co2_mu
        outputs["co2_logvar"] = co2_logvar
        outputs["aerosol_mu"] = aerosol_mu
        outputs["aerosol_logvar"] = aerosol_logvar
        return outputs

    def encode(self, x, i):
        if self.tied:
            w = self.w[i].T
        else:
            w = self.w_encoder[i]
        mu = torch.matmul(x, w.T)
        return mu, self.logvar_encoder

    def decode(self, z, i):
        w = self.w[i]
        # When using forced latents, z only contains climate latents (sliced by caller).
        # Use only the corresponding climate columns of w.
        if self.use_forced_latents and z.shape[-1] < w.shape[-1]:
            w = w[:, : z.shape[-1]]
            w = torch.relu(w)
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

        unif = (1 - 0.1) * torch.rand(size=(d, d_x, d_z)) + 0.1
        self.w = nn.Parameter(unif / torch.as_tensor(d_z))
        if not tied:
            unif = (1 - 0.1) * torch.rand(size=(d, d_z, d_x)) + 0.1
            self.w_encoder = nn.Parameter(unif / torch.as_tensor(d_x))

        self.logvar_encoder = nn.Parameter(torch.ones(d_z) * -1)
        self.logvar_decoder = nn.Parameter(torch.ones(d_x) * -1)

    def get_w_encoder(self):
        if self.tied:
            return torch.transpose(self.w, 1, 2)
        return self.w_encoder

    def get_w_decoder(self):
        return self.w

    def get_w_forcings(self, foricng_name):
        if self.use_forced_latents:
            return self.forcing_mask[foricng_name]
        else:
            return None

    def get_w_co2(self):
        return self.get_w_forcings("co2")

    def get_w_aerosol(self):
        return self.get_w_forcings("aerosol")

    def get_w_so2(self):
        return self.get_w_forcings("so2")

    def get_encode_mask(self):
        if self.tied:
            return torch.transpose(self.w, 1, 2)
        return self.w_encoder

    def select_encoder_mask(self, mask, i, j):
        return mask

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
        reduce_encoding_pos_dim=False,
        gt_w=None,
        use_forced_latents=False,
        n_forced_latents_co2=1,
        n_forced_latents_aerosol=4,
        n_forced_latents_ch4=0,
        n_forced_latents_so2=0,
        d_y_co2_spatial=0,
        d_y_aerosol_spatial=0,
        d_y_ch4_spatial=0,
        d_y_so2_spatial=0,
        forcing_mse=False,
    ):
        super().__init__(d, d_x, d_z, num_hidden, num_layer, tied, gt_w)
        self.reduce_encoding_pos_dim = reduce_encoding_pos_dim
        self.use_forced_latents = use_forced_latents

        # ============================================================
        # Latent dimensions
        # ============================================================
        self.forcing_latent_dims = {
            "co2": n_forced_latents_co2,
            "ch4": n_forced_latents_ch4,
            "so2": n_forced_latents_so2,
            "aerosol": n_forced_latents_aerosol,
        }

        self.forcing_spatial_dims = {
            "co2": d_y_co2_spatial,
            "ch4": d_y_ch4_spatial,
            "so2": d_y_so2_spatial,
            "aerosol": d_y_aerosol_spatial,
        }
        # ============================================================
        # Encoder
        # ============================================================

        if reduce_encoding_pos_dim:
            pos_dim = embedding_dim // 10
        else:
            pos_dim = embedding_dim
        self.embedding_encoder = nn.Embedding(d_z, pos_dim)
        self.encoder = MLP(num_layer, num_hidden, d_x + pos_dim, 1)
        self.embedding_decoder = nn.Embedding(d_x, embedding_dim)

        # ============================================================
        # Climate decoder
        # ============================================================
        if use_forced_latents:
            forced_dim = sum(self.forcing_latent_dims.values())
            self.n_climate_latents = d_z - forced_dim
        else:
            self.n_climate_latents = d_z

        self.decoder = MLP(num_layer, num_hidden, self.n_climate_latents + embedding_dim, 1)

        # ============================================================
        # Forcing modules
        # ============================================================

        if use_forced_latents:
            self._build_forcing_modules(num_layer, num_hidden)
        self.forcing_mse = forcing_mse

    def _build_forcing_modules(self, num_layer, num_hidden):
        """Create forcing encoders and decoders."""
        self.forcing_encoder_mu = nn.ModuleDict()
        self.forcing_decoder = nn.ModuleDict()

        self.forcing_logvar_encoder = nn.ParameterDict()
        self.forcing_logvar_decoder = nn.ParameterDict()

        self.forcing_mask = nn.ParameterDict()

        for name in self.forcing_latent_dims:

            latent_dim = self.forcing_latent_dims[name]
            spatial_dim = self.forcing_spatial_dims[name]

            if latent_dim <= 0 or spatial_dim <= 0:
                continue

            # encoder
            self.forcing_encoder_mu[name] = MLP(num_layer, num_hidden, spatial_dim, latent_dim)
            self.forcing_logvar_encoder[name] = nn.Parameter(torch.ones(latent_dim) * -1)

            # decoder
            self.forcing_decoder[name] = MLP(num_layer, num_hidden, latent_dim, 1)
            self.forcing_mask[name] = nn.Parameter(torch.randn(spatial_dim, latent_dim))
            self.forcing_logvar_decoder[name] = nn.Parameter(torch.ones(spatial_dim) * -1)

    # ================================================================
    # Forcing encoder helper
    # ================================================================
    def _encode_and_sample(self, forcing, name):
        batch_size = forcing.shape[0]
        mu = self.forcing_encoder_mu[name](forcing)
        std = torch.exp(0.5 * self.forcing_logvar_encoder[name]).expand(batch_size, -1)
        if self.forcing_mse:
            z = mu
        else:
            z = mu + std * torch.randn_like(std)
        return z, mu, std

    # ================================================================
    # Encode all forcings
    # ================================================================
    def encode_forcings(self, forcing_dict):

        z_list = []
        mu_list = []
        std_list = []

        for name, forcing in forcing_dict.items():
            if forcing is None or name not in self.forcing_encoder_mu:
                continue
            z, mu, std = self._encode_and_sample(forcing, name)

            z_list.append(z)
            mu_list.append(mu)
            std_list.append(std)

        if not z_list:
            raise ValueError("No active forcing tensors were provided for forced-latent encoding.")

        z_forced = torch.cat(z_list, dim=1)
        mu_forced = torch.cat(mu_list, dim=1)
        std_forced = torch.cat(std_list, dim=1)

        return (z_forced, mu_forced, std_forced)

    # ================================================================
    # Decode one forcing
    # ================================================================

    def _decode_one_forcing(self, z, name):
        spatial_dim = self.forcing_spatial_dims[name]
        z_expand = z.unsqueeze(1).expand(-1, spatial_dim, -1)
        z_mask = z_expand * self.forcing_mask[name].unsqueeze(0)
        reconstruction = self.forcing_decoder[name](z_mask).squeeze(-1)
        logvar = self.forcing_logvar_decoder[name]
        return reconstruction, logvar

    # ================================================================
    # Decode all forcings
    # ================================================================

    def decode_forcings(self, z_forced_latents):
        outputs = {}
        start = 0

        for name, latent_dim in self.forcing_latent_dims.items():
            if latent_dim <= 0 or name not in self.forcing_decoder:
                continue

            z = z_forced_latents[:, start : start + latent_dim]
            start += latent_dim
            mu, logvar = self._decode_one_forcing(z, name)

            outputs[f"{name}_mu"] = mu
            outputs[f"{name}_logvar"] = logvar

        return outputs

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

        mu = self.encoder(x_).squeeze(-1)  # if squeeze(): the batch dim is removed when bs=1

        return mu, self.logvar_encoder

    def decode(self, z, i):
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
        mask = super().get_decode_mask()
        mu = torch.zeros((z.shape[0], self.d_x), device=z.device)

        # Create a tensor of shape (z.shape[0], self.d_x) where each row is a sequence from 0 to self.d_x
        j_values = torch.arange(self.d_x, device=z.device).expand(z.shape[0], -1)

        # Embed all j_values at once
        embedded_z = self.embedding_decoder(j_values)

        # Select decoder masks - only use climate portion of mask
        mask_ = super().select_decoder_mask(mask, i, j_values)

        # Only use climate latents (first n_climate columns of mask)
        # The slicing of mask is performed here
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
        z_ = torch.cat((z_expanded_copy, embedded_z), dim=2)

        del z_expanded
        del z_expanded_copy

        # Apply the decoder to all z_ at once and squeeze the result
        mu = self.decoder(z_).squeeze(-1)

        return mu, self.logvar_decoder

    def forward(self, x, i, encode: bool = False):
        if encode:
            return self.encode(x, i)
        else:
            return self.decode(x, i)


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
        d_z_global: int = 0,
        instantaneous=False,
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
        # if instantaneous, the input tau has already been +1 (total_tau)；
        # if not instantaneous， the input total_tau is 5, but the transition may consider current forcings to climate, so we need to +1
        n_step = tau if instantaneous else tau + 1
        input_dim = d * d_z * n_step + (d * self.d_z2 * 1)

        if output_var:
            self.num_output = num_output
        else:
            self.num_output = 1
            # self.logvar = torch.ones(1)  * 0. # nn.Parameter(torch.ones(d) * 0.1)
            # self.logvar = nn.Parameter(torch.ones(d) * -4)
            self.logvar = nn.Parameter(torch.ones(d, d_z) * -4)
        if self.nonlinear_dynamics:
            logger.info("NON LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(num_layers, num_hidden, input_dim, self.num_output) for i in range(d * d_z))
        else:
            logger.info("LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(0, 0, input_dim, self.num_output) for i in range(d * d_z))
        # self.nn = MLP(num_layers, num_hidden, d * k * k, self.num_output)

    def forward(self, z, mask, i, k, z_forcings=None, z_global=None):
        """Returns the params of N(z_t | z_{<t}) for a specific feature i and latent variable k NN(G_{tau-1} * z_{t-1},
        ..., G_{tau-k} * z_{t-k})"""

        # works well for original prediction when we do not do instantaneous! :)
        # e.g.     val_mse, val_smape = prediction_original(trainer, True)

        # t_total = torch.max(self.tau, z_past.size(1))  # TODO: find right dim
        # param_z = torch.zeros(z_past.size(0), 2)

        # print("In the forward of the transition model, and trying to ascertain which way the information flows through the mask.")
        # print("The mask is of size: ", mask.size())
        # print("The z is of size: ", z.size()) [256, tau, dz, 1])

        # print the unique values and their counts in mask:
        # print("The unique values of mask are: ", torch.unique(mask))
        # print("The counts of the unique values of mask are: ", torch.unique(mask, return_counts=True))

        # print the first few elements of z
        batch_size = z.size(0)
        z_forcings = torch.cat(z_forcings, dim=-2)
        total_z_for_mask = torch.cat([z, z_forcings], dim=-2)  # ([b, tau+1, 7, 1])

        total_z_for_mask = total_z_for_mask.view(mask.size())

        # print("The z is now, after z.view() of size: ", z.size()) [256, tau, dz])

        # print("what is mask * z shape? ", (mask * z).size())

        masked_z = (mask * total_z_for_mask).view(batch_size, -1)

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

        if z_global is not None:
            # 2. Process current z2 (t): flatten without masking
            # (b, 1, d, dz2) -> (b, 1*d * dz2)
            flat_z2 = z_global.view(batch_size, -1)

            # 3. Concatenate along the feature dimension
            # Result shape: (b, (tau*d*dz) + (1*d*dz2))
            masked_z = torch.cat([masked_z, flat_z2], dim=1)

        # 4. Predict params for N(z1_t | z1_{<t}, z2_t)

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
        d_z_global: int = 0,
        instantaneous=False,
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
        self.d_z2 = d_z_global
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
        # tau for all latents and one more time step for forcings
        n_step = tau if instantaneous else tau + 1
        input_dim = d * d_z * n_step + (d * self.d_z2 * 1)
        if self.nonlinear_dynamics:
            logger.info("NON LINEAR DYNAMICS")
            self.nn = nn.ModuleList(
                MLP(num_layers, num_hidden, input_dim + embedding_dim, self.num_output) for i in range(d)
            )
        else:
            logger.info("LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(0, 0, input_dim + embedding_dim, self.num_output) for i in range(d))
        self.instantaneous = instantaneous

    def forward(self, z_climate, mask, i, z_forcings=None, z_global=None):
        """Returns the params of N(z_t | z_{<t}) for a specific feature i and latent variable k NN(G_{tau-1} * z_{t-1},
        ..., G_{tau-k} * z_{t-k})"""
        # z_climate:(b,tau+1,dz_climate, 1),forcing_co2(b,tau+1,dz_climate, 1), :(b,tau+1,dz_climate, 1)
        batch_size = z_climate.shape[0]
        j_values = torch.arange(self.d_z, device=z_climate.device).expand(
            batch_size, -1
        )  # create a 2D tensor with shape (x.shape[0], self.d_z)
        embedded_z = self.embedding_transition(j_values)  # [1, 7, embed_dim]
        # z_forcings is a list of forcings
        z_forcings = torch.cat(z_forcings, dim=-2)
        total_z_for_mask = torch.cat([z_climate, z_forcings], dim=-2)

        # total_z_for_mask([b, tau+1, 7, 1]) mask [1, tau+1, 7, 7]) # 7=climate_z + forcing_co2_z + forcing_aerosol_z
        # masked_z = (mask * total_z_for_mask).transpose(3, 2).reshape((batch_size, -1, self.d_z)).transpose(2, 1) #[b, 7, (tau+1)*d_z]
        masked_z = mask * total_z_for_mask.transpose(-1, -2)  # b,t, tar, sou
        masked_z = masked_z.permute(0, 2, 1, 3).reshape(batch_size, self.d_z, -1)  # b, tar, t*sou
        components = [masked_z, embedded_z]  # ()
        # else, if instantaneous, we directly allow the full connectivity among all latents at all time steps
        # leave the logic for global forcing, probably we will need to interact before the forcing is considered
        if z_global is not None:
            flat_z2 = z_global.view(batch_size, -1)
            components.append(flat_z2.unsqueeze(1).expand(-1, self.d_z, -1))

        z_ = torch.cat(components, dim=-1)

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
        tau: int,  # Number of timesteps as input
        instantaneous: bool,
        instantaneous_forcing: bool,
        nonlinear_mixing: bool,
        nonlinear_dynamics: bool,
        # no_gt: bool,
        # debug_gt_graph: bool,
        # debug_gt_z: bool,
        # debug_gt_w: bool,
        # gt_graph: torch.tensor = None,
        # gt_w: torch.tensor = None,
        tied_w: bool = False,
        reduce_encoding_pos_dim: bool = False,
        fixed: bool = False,
        fixed_output_fraction: float = 1.0,
        gev_learn_xi: bool = False,
        use_exogenous: bool = False,
        d_y_co2: int = 0,
        d_y_aerosol: int = 0,
        d_y_ch4: int = 0,
        d_y_so2: int = 0,
        use_forced_latents: bool = False,
        n_forced_latents_co2: int = 1,
        n_forced_latents_aerosol: int = 2,
        n_forced_latents_ch4: int = 0,
        n_forced_latents_so2: int = 0,
        forcing_arch: str = "baseline",
        map_aerosol_to_climate: bool = False,
        forcing_mse: bool = False,
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
        self.transition_param_sharing = transition_param_sharing
        self.position_embedding_transition = position_embedding_transition
        self.coeff_kl = coeff_kl

        self.d = d
        self.d_x = d_x
        self.d_z = d_z
        self.tau = tau
        self.instantaneous = instantaneous
        self.nonlinear_mixing = nonlinear_mixing
        self.nonlinear_dynamics = nonlinear_dynamics
        # self.no_gt = no_gt
        # self.debug_gt_graph = debug_gt_graph
        # self.debug_gt_z = debug_gt_z
        # self.debug_gt_w = debug_gt_w
        self.tied_w = tied_w
        self.fixed = fixed
        self.fixed_output_fraction = fixed_output_fraction
        self.gev_learn_xi = gev_learn_xi
        self.use_exogenous = use_exogenous
        self.d_y_co2 = d_y_co2
        self.d_y_aerosol = d_y_aerosol
        self.d_y_ch4 = d_y_ch4
        self.d_y_so2 = d_y_so2
        self.use_forced_latents = use_forced_latents
        self.n_forced_latents_co2 = n_forced_latents_co2
        self.n_forced_latents_aerosol = n_forced_latents_aerosol
        self.n_forced_latents_ch4 = n_forced_latents_ch4
        self.n_forced_latents_so2 = n_forced_latents_so2
        self.forcing_arch = forcing_arch
        self.map_aerosol_to_climate = map_aerosol_to_climate
        self.forcing_mse = forcing_mse
        self._forcing_arch_logged = False
        # The latents in z will be ordered by self.forcing_order
        self.forcing_order = ("co2", "ch4", "aerosol", "so2")
        self.forcing_latent_dims = OrderedDict(
            (
                ("co2", self.n_forced_latents_co2),
                ("ch4", self.n_forced_latents_ch4),
                ("aerosol", self.n_forced_latents_aerosol),
                ("so2", self.n_forced_latents_so2),
            )
        )
        self.forcing_spatial_dims = OrderedDict(
            (
                ("co2", self.d_y_co2),
                ("ch4", self.d_y_ch4),
                ("aerosol", self.d_y_aerosol),
                ("so2", self.d_y_so2),
            )
        )
        self.n_forced_latents_total = sum(self.forcing_latent_dims.values()) if self.use_forced_latents else 0
        self.n_climate_latents = self.d_z - self.n_forced_latents_total
        if self.use_forced_latents and self.n_climate_latents <= 0:
            raise ValueError(f"d_z={self.d_z} must be larger than total forced latents={self.n_forced_latents_total}.")
        has_extra_forcings = self.n_forced_latents_ch4 > 0 or self.n_forced_latents_so2 > 0
        if self.use_forced_latents and has_extra_forcings and not self.nonlinear_mixing:
            raise ValueError("Four-forcing forced latents require nonlinear_mixing=True.")
        if self.forcing_arch == "predefined" and self.use_forced_latents:
            raise ValueError(
                "forcing_arch='predefined' requires use_forced_latents=False and d_z to include only climate latents."
            )

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

        # Forcing encoder/decoder spatial dims: need real dims whenever use_forced_latents,
        # independent of whether raw forcings are used as MLP conditioning (use_exogenous).
        d_y_co2_spatial = self.forcing_spatial_dims["co2"] if self.use_forced_latents else 0
        d_y_ch4_spatial = self.forcing_spatial_dims["ch4"] if self.use_forced_latents else 0
        d_y_aerosol_spatial = self.forcing_spatial_dims["aerosol"] if self.use_forced_latents else 0
        d_y_so2_spatial = self.forcing_spatial_dims["so2"] if self.use_forced_latents else 0

        if self.nonlinear_mixing:
            logger.info("NON-LINEAR MIXING")
            # NOTE:(seb) using the noloop version of non-linear here to make it much faster.
            self.autoencoder = NonLinearAutoEncoderUniqueMLP_noloop(
                d,
                d_x,
                d_z,
                self.num_hidden_mixing,
                self.num_layers_mixing,
                tied=tied_w,
                embedding_dim=self.position_embedding_dim,
                reduce_encoding_pos_dim=self.reduce_encoding_pos_dim,
                gt_w=None,
                use_forced_latents=self.use_forced_latents,
                n_forced_latents_co2=self.n_forced_latents_co2,
                n_forced_latents_aerosol=self.n_forced_latents_aerosol,
                n_forced_latents_ch4=self.n_forced_latents_ch4,
                n_forced_latents_so2=self.n_forced_latents_so2,
                d_y_co2_spatial=d_y_co2_spatial,
                d_y_aerosol_spatial=d_y_aerosol_spatial,
                d_y_ch4_spatial=d_y_ch4_spatial,
                d_y_so2_spatial=d_y_so2_spatial,
                forcing_mse=forcing_mse,
            )
        else:
            # print('Using linear mixing')
            logger.info("LINEAR MIXING")
            self.autoencoder = LinearAutoEncoder(
                d,
                d_x,
                d_z,
                tied=tied_w,
                use_forced_latents=self.use_forced_latents,
                n_forced_latents_co2=self.n_forced_latents_co2,
                n_forced_latents_aerosol=self.n_forced_latents_aerosol,
                n_forced_latents_ch4=self.n_forced_latents_ch4,
                n_forced_latents_so2=self.n_forced_latents_so2,
                d_y_co2_spatial=d_y_co2_spatial,
                d_y_aerosol_spatial=d_y_aerosol_spatial,
                d_y_ch4_spatial=d_y_ch4_spatial,
                d_y_so2_spatial=d_y_so2_spatial,
                forcing_mse=forcing_mse,
            )

        if self.transition_param_sharing:
            self.transition_model = TransitionModelParamSharing(
                self.d,
                self.d_z,
                self.total_tau,
                self.nonlinear_dynamics,
                self.num_layers,
                self.num_hidden,
                self.num_output,
                self.position_embedding_dim,
                instantaneous=instantaneous,
            )
        else:
            self.transition_model = TransitionModel(
                self.d,
                self.d_z,
                self.total_tau,
                self.nonlinear_dynamics,
                self.num_layers,
                self.num_hidden,
                self.num_output,
                instantaneous=instantaneous,
            )

        # print("We are setting the Mask here.")
        self.mask = Mask(
            d,
            d_z,
            self.total_tau,
            instantaneous=instantaneous,
            instantaneous_forcing=instantaneous_forcing,
            latent=True,
            fixed=fixed,
            fixed_output_fraction=fixed_output_fraction,
            n_climate=self.n_climate_latents,
        )
        # if self.debug_gt_graph:
        #     if self.instantaneous:
        #         self.mask.fix(self.gt_graph)
        #     else:
        #         self.mask.fix(self.gt_graph[:-1])
        self.instantaneous_forcing = instantaneous_forcing

    def get_adj(self):
        """
        Returns: Matrices of the probabilities from which the masks linking the
        latent variables are sampled
        """
        # return self.mask.get_proba() * self.mask.fixed_mask
        return self.mask.get_proba()  # [:self.tau]#[tau, dz, dz]

    def get_effective_adj(self):
        adj = self.mask.get_proba()

        if self.use_forced_latents and self.map_aerosol_to_climate:
            adj = self.apply_spatial_forcing_mask(adj.unsqueeze(0)).squeeze(0)

        return adj

    def _forcing_dict(self, y_co2=None, y_aerosol=None, y_ch4=None, y_so2=None):
        """Normalize forcing inputs to the canonical four-forcing names."""
        return OrderedDict(
            (
                ("co2", y_co2),
                ("ch4", y_ch4),
                ("aerosol", y_aerosol),
                ("so2", y_so2),
            )
        )

    def _active_forcing_dict(self, forcing_dict):
        active = OrderedDict()
        for name in self.forcing_order:
            forcing = forcing_dict.get(name)
            if self.forcing_latent_dims.get(name, 0) > 0:
                if forcing is None:
                    return None
                active[name] = forcing
        return active

    def _forcing_at_timestep(self, forcing, t):
        return forcing[:, t] if forcing is not None and forcing.dim() == 3 else forcing

    def _transition_forcing_list(self, forcing_slices, i):
        return [forcing_slices[name][:, :, i][:, :, :, None] for name in forcing_slices]

    def _forcing_index_slices(self):
        """
        Return index slices of each forcing block in the full latent vector.

        Full latent order:
        [climate, co2, ch4, aerosol, so2]
        """
        slices = OrderedDict()
        # Optional, we can put start to input param, so support slicing over sub latents
        start = self.n_climate_latents
        for name in self.forcing_order:
            dim = self.forcing_latent_dims.get(name, 0)
            if dim <= 0:
                continue
            slices[name] = slice(start, start + dim)
            start += dim
        return slices

    # Measure the spatial overlap (by grid number) between forcing and climate latent decoder maps.
    # routing[name][i, j] is the normalized alignment weight from forcing latent j to climate latent i.
    # considering z_forcing usually represent larger space than z_climate, dim=1: kind of a single-parent structure from forcings to climate
    # (one forcing z can impact multiple climate z, but can't be impacted by multiple forcing z)
    def forcing_climate_spatial_alignment(self, k=5):
        # with torch.no_grad():
        W_climate = self.autoencoder.get_w_decoder()[0, :, : self.n_climate_latents].clone()

        routing = {}

        for name in ("aerosol", "so2"):
            if self.forcing_latent_dims.get(name, 0) <= 0:
                continue

            W_forcing = self.autoencoder.get_w_forcings(name).clone()
            if W_forcing is None:
                continue

            score = W_climate.T @ W_forcing
            # score: (n_climate, n_forcing)

            k_eff = min(k, score.size(1))
            idx = score.topk(k_eff, dim=1).indices

            hard = torch.zeros_like(score)
            hard.scatter_(1, idx, 1.0)

            routing[name] = hard

        return routing
        # problem with this version: learn small coeffficients around 1/n_forcings
        # W_climate = self.autoencoder.get_w_decoder()[0, :, :self.n_climate_latents]
        # W_climate = torch.relu(W_climate)

        # routing = {}
        # for name in ("aerosol", "so2"):
        #     if self.forcing_latent_dims.get(name, 0) <= 0:
        #         continue

        #     W_forcing = self.autoencoder.get_w_forcings(name)
        #     if W_forcing is None:
        #         continue

        #     W_forcing = torch.relu(W_forcing)

        #     score = W_climate.T @ W_forcing
        #     routing[name] = torch.softmax(score / tau, dim=1) # if dim=0, the total connection from z_forings to z_climate is 1

        # return routing

    # The decoder weight matrix W reveals which forcing and climate latents correspond to similar spatial regions
    def apply_spatial_forcing_mask(self, mask):
        spatial_mask = torch.ones_like(mask)

        routing = self.forcing_climate_spatial_alignment()
        forcing_slices = self._forcing_index_slices()

        n_climate = self.n_climate_latents

        for name in ("aerosol", "so2"):
            if name not in forcing_slices or name not in routing:
                continue

            forcing_slice = forcing_slices[name]
            spatial_mask[:, :, :n_climate, forcing_slice] = routing[name].view(1, 1, n_climate, -1)
        return mask * spatial_mask

    def encode(self, x, y, y_co2=None, y_aerosol=None, y_ch4=None, y_so2=None):
        """
        Encode observations X (history) and Y (target) into latent variables Z.

        Args:
            x: Historical observations, shape (batch, tau, d, d_x).
            y: Target observation, shape (batch, d, d_x).
            y_co2: Optional CO2 forcing, shape (batch, tau+1, d_global, d_y_co2) or (batch, d_y_co2).
            y_aerosol: Optional aerosol forcing, shape (batch, tau+1, d_spatial, d_y_aerosol) or (batch, d_y_aerosol).

        Returns:
            z: Latent variables, shape (batch, tau+1, d, d_z).
            mu: Variational posterior mean for the target timestep, shape (batch, d, d_z).
            std: Variational posterior std for the target timestep, shape (batch, d, d_z).
        """
        b = x.size(0)  # batch size
        z = torch.zeros(b, self.tau + 1, self.d, self.d_z)
        mu = torch.zeros(b, self.d, self.d_z)
        std = torch.zeros(b, self.d, self.d_z)

        # Handle forced latents if enabled
        forcing_dict = self._forcing_dict(y_co2, y_aerosol, y_ch4, y_so2)
        active_forcings = self._active_forcing_dict(forcing_dict) if self.use_forced_latents else None

        if self.use_forced_latents and active_forcings is not None:
            n_climate_latents = self.n_climate_latents

            # y_co2 shape: (batch_size, tau+1, spatial_dim) - NOW SPATIAL like aerosol!
            # y_aerosol shape: (batch_size, tau+1, spatial_dim)
            # We need to process forcings at each timestep separately

            # For SAVAR, d=1, so we only iterate once over i
            for i in range(self.d):
                # Encode climate latents from observations for all timesteps
                for t in range(self.tau):
                    # Extract forcing at timestep t
                    forcing_t = OrderedDict(
                        (name, self._forcing_at_timestep(forcing, t)) for name, forcing in active_forcings.items()
                    )

                    # Encode forcings for this timestep
                    z_forced_t, _, _ = self.autoencoder.encode_forcings(forcing_t)

                    q_mu, q_logvar = self.autoencoder(
                        x[:, t, i], i, encode=True
                    )  # -> this will encode x into climate+forcing latents

                    q_std = torch.exp(0.5 * q_logvar)
                    # Only encode to climate latents
                    z[:, t, i, :n_climate_latents] = q_mu[:, :n_climate_latents] + q_std[
                        :n_climate_latents
                    ] * self.distr_encoder(
                        0, 1, size=(b, n_climate_latents)
                    )  # -> then slice over the first n_climate latents
                    # Fill forced latents with timestep-specific forcings
                    z[:, t, i, n_climate_latents:] = z_forced_t

                # Encode the target timestep (y) using final forcing timestep
                forcing_target = OrderedDict(
                    (name, self._forcing_at_timestep(forcing, -1)) for name, forcing in active_forcings.items()
                )

                z_forced_target, mu_forced, std_forced = self.autoencoder.encode_forcings(forcing_target)

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
                    q_mu, q_logvar = self.autoencoder(x[:, t, i], i, encode=True)

                    q_std = torch.exp(0.5 * q_logvar)
                    z[:, t, i] = q_mu + q_std * self.distr_encoder(0, 1, size=q_mu.size())

                q_mu, q_logvar = self.autoencoder(y[:, i], i, encode=True)

                q_std = torch.exp(0.5 * q_logvar)
                z[:, -1, i] = q_mu + q_std * self.distr_encoder(0, 1, size=q_mu.size())

                mu[:, i] = q_mu
                std[:, i] = q_std

        return z, mu, std

    def transition(self, z, mask):
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

        # TODO: confirm if this can gives less sparsity
        if self.map_aerosol_to_climate:
            # WHAT IF I only add it to the training forward instead of all transition?
            # will this gives less sparsity of the aerosol to climate
            # The mask will be multiplied with a mapping which only allow the the latents interaction that's decodes to the same spatial area
            mask = self.apply_spatial_forcing_mask(mask)

        # here I seperate z_climate and z_forcings
        if self.use_forced_latents:
            z_climate = z[:, :, :, : self.n_climate_latents]  # ([1, 6, 1, 4])
            index_slices = self._forcing_index_slices()
            forcing_slices = OrderedDict((name, z[..., idx]) for name, idx in index_slices.items())
            # forcing_slices["co2"] torch.Size([2, 6, 1, 1])
            # forcing_slices["aerosol"] torch.Size([2, 6, 1, 2])
        else:
            z_climate = z
            forcing_slices = OrderedDict()
        print("forcing_slices")
        for k, v in forcing_slices.items():
            print(k, v.shape)
        for i in range(self.d):
            # Todo, currently only handle z_co2 of d=1 and z_zerosol of d=1, what if there are more forcings? should we name them separately?
            forcing_latents = self._transition_forcing_list(forcing_slices, i)
            if self.transition_param_sharing:
                pz_params = self.transition_model(
                    z_climate[:, :, i][:, :, :, None],
                    mask[
                        :, :, i * self.d_z : (i + 1) * self.d_z
                    ],  # slice over the target dim, but don't make any difference because d=0
                    i,
                    z_forcings=forcing_latents,
                )
            else:
                # Not yet implemented for non-parameter sharing version!
                pz_params = torch.zeros(b, self.d_z, 1)
                for k in range(self.d_z):
                    pz_params[:, k] = self.transition_model(
                        z_climate[:, :, i][:, :, :, None],
                        mask[:, :, i * self.d_z + k],
                        i,
                        k,
                        z_forcings=forcing_latents,
                    )
            # The resulting pz_params's last z_forcings dimensions should stay unchanged after transition
            mu[:, i] = pz_params[:, :, 0]
            std[:, i] = torch.exp(0.5 * self.transition_model.logvar[i])

        # print("This is giving us the pz_mu and pz_std that we use later.")
        return mu, std

    def decode(self, z):
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
            z_for_decode = z[..., : self.n_climate_latents]  # Shape: (batch, d, n_climate)
        else:
            z_for_decode = z

        # only decode from transited climate variables
        for i in range(self.d):
            px_mu, px_logvar = self.autoencoder(z_for_decode[:, i], i, encode=False)

            if px_mu.ndim == mu.ndim:  # In case of linear mixing with one variable, second dimension is too much
                px_mu = px_mu.squeeze()

            mu[:, i] = px_mu
            std[:, i] = torch.exp(0.5 * px_logvar)

        return mu, std

    def forward(
        self,
        x,
        y,
        gt_z,
        iteration,
        xi=None,
        y_co2=None,
        y_aerosol=None,  # BC
        y_ch4=None,
        y_so2=None,
    ):
        if iteration == 1:

            print(f"shape x: {x.shape}, y: {y.shape}")
            if y_co2 is not None and y_aerosol is not None:
                print(f"y_co2: {y_co2.shape}")
                print(f"y_aerosol: {y_aerosol.shape}")
            if y_ch4 is not None and y_so2 is not None:
                print(f"y_ch4: {y_ch4.shape}")
                print(f"y_so2: {y_so2.shape}")
        """Full forward pass: encode, transition, decode, and compute ELBO.

        Args:
            x: Historical observations, shape (batch, tau, d, d_x).
            y: Target observation, shape (batch, d, d_x).
            gt_z: Ground-truth latents (used only when debug_gt_z=True).
            iteration: Current training iteration (unused in forward, passed for API consistency).
            xi: Optional GEV shape parameter override.
            y_co2: Optional CO2 forcing.
            y_aerosol: Optional aerosol forcing, by default is BC
            y_ch4: Optional CH4 forcing.
            y_so2: Optional SO2 forcing.

        Returns:
            Tuple of (elbo, recons, kl, px_mu, forcing_recons_loss_co2,
            forcing_recons_loss_aerosol, encoded_forcing_mu).
        """
        b = x.size(0)  # batch size

        # sample Zs (based on X)

        z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol, y_ch4, y_so2)
        # z (b, tau + 1, d, d_z) q_mu_y and q_std_y (b, d, d_z)

        # Store encoded forcing latent means for supervision loss (if using forced latents)
        encoded_forcing_mu = None
        # get params of the transition model p(z^t | z^{<t})
        mask = self.mask(b)  # [b, tau, d_z, d_z]
        forcing_dict = self._forcing_dict(y_co2, y_aerosol, y_ch4, y_so2)
        active_forcings = self._active_forcing_dict(forcing_dict) if self.use_forced_latents else None
        if self.use_forced_latents:
            n_climate_latents = self.n_climate_latents
        else:
            n_climate_latents = self.d_z
        print("self.n_climate_latents", self.n_climate_latents)
        z_for_transit = z.clone()
        # if not self.instantaneous:
        #     z_for_transit[:, -1, :, :n_climate_latents] = 0 # here mask climate z_t to zero to avoid label leakage

        pz_mu, pz_std = self.transition(z_for_transit, mask)
        # pz_mu ([b, 1, d_z]

        # get params from decoder p(x^t | z^t)
        # we pass only the last z to the decoder, to get xs.

        px_mu, px_std = self.decode(z[:, -1])  # px_mu (b, d, d_x)

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
        # FIX: since the rest latents not involve in forcing reconstruction nor climate prediciton, it should be sliced
        kl = torch.sum(kl_raw[..., :n_climate_latents], dim=[2]).mean()
        # kl = torch.sum(0.5 * (torch.log(pz_std**2) - torch.log(q_std_y**2)) + 0.5 *
        # (q_std_y**2 + (q_mu_y - pz_mu) ** 2) / pz_std**2 - 0.5, dim=[1, 2]).mean()
        assert kl >= 0, f"KL={kl} has to be >= 0"

        elbo = recons - self.coeff_kl * kl

        # Compute forcing reconstruction losses
        forcing_recons_losses = OrderedDict((name, torch.tensor(0.0, device=x.device)) for name in self.forcing_order)
        forcing_recons_losses["gmst_loss"] = 0  # or torch.mean((px_mu.mean(dim=-1) - y.mean(dim=-1)) ** 2)

        if self.use_forced_latents and active_forcings is not None:
            # Extract forcing latents from z (last timestep, first feature dimension, forcing latent indices)
            forcing_arch = getattr(self, "forcing_arch", "baseline")
            if forcing_arch == "baseline":
                if not self._forcing_arch_logged:
                    logger.info("[ForcingArch] Using forcing_arch='baseline' (encoded forced latents)")
                    self._forcing_arch_logged = True
                    # update here: I used the all "previous" steps for reconstruction
                # Shape: (batch, tau, n_forced_latents_total)
                # Decode forcing latents back to forcing space
                z_forced_target = z[:, -1, 0, n_climate_latents:]
                forcing_outputs = self.autoencoder.decode_forcings(z_forced_target)
            elif forcing_arch == "transitioned":
                if not self._forcing_arch_logged:
                    logger.info("[ForcingArch] Using forcing_arch='transitioned' (pz_mu forced latents)")
                    self._forcing_arch_logged = True
                # Use transitioned latents (pz_mu) for forcing reconstruction
                z_forced_target = pz_mu[:, 0, n_climate_latents:]  # Shape: (batch, n_forced_latents_total)
                forcing_outputs = self.autoencoder.decode_forcings(z_forced_target)
            elif forcing_arch == "predefined":
                if not self._forcing_arch_logged:
                    logger.info("[ForcingArch] Using forcing_arch='predefined' (no forcing reconstruction)")
                    self._forcing_arch_logged = True
                # No forcing reconstruction in predefined conditioning mode
                forcing_outputs = {}
            else:
                raise ValueError(f"Unknown forcing_arch='{forcing_arch}'")

            if forcing_arch != "predefined":
                for name, forcing in active_forcings.items():
                    mu_key = f"{name}_mu"

                    if mu_key not in forcing_outputs:
                        continue

                    forcing_target = self._forcing_at_timestep(forcing, -1)
                    if self.forcing_mse:
                        forcing_recons = forcing_outputs[mu_key]

                        forcing_recons_losses[name] = torch.mean((forcing_recons - forcing_target) ** 2)
                    else:
                        logvar_key = f"{name}_logvar"
                        forcing_var = torch.exp(0.5 * forcing_outputs[logvar_key])
                        px_forcing_distr = self.distr_decoder(forcing_outputs[mu_key], forcing_var)
                        forcing_recons_losses[name] = -torch.mean(
                            torch.sum(px_forcing_distr.log_prob(forcing_target), dim=[1])
                        )
                # print("forcing_recons_loss_aerosol and recons",forcing_recons_loss_aerosol, recons)

        return (
            elbo,
            recons,
            kl,
            px_mu,
            forcing_recons_losses["co2"],
            forcing_recons_losses["aerosol"],
            encoded_forcing_mu,
            forcing_recons_losses,  # any additional forcing recons losses goes here
        )

    def predict_pxmu_pxstd(self, x, y, y_co2=None, y_aerosol=None, y_ch4=None, y_so2=None):

        # NOTE: this one was working fine for the CRPS loss because I was not using no_grad...
        # I need to keep the grads if I am going to add to the loss

        b = x.size(0)

        # sample Zs (based on X)
        z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol, y_ch4, y_so2)

        # get params of the transition model p(z^t | z^{<t})
        mask = self.mask(b)

        z_for_transit = z.clone()

        pz_mu, pz_std = self.transition(z_for_transit, mask)

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

        z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol, y_ch4, y_so2)

        mask = self.mask(b)

        z_for_transit = z.clone()

        pz_mu, pz_std = self.transition(z_for_transit, mask)

        # decode
        px_mu, px_std = self.decode(pz_mu)

        return px_mu, y, z, pz_mu, pz_std

    def predict_counterfactual(
        self,
        x,
        y,
        counterfactual_z_index,
        counterfactual_z_value,
        y_co2=None,
        y_aerosol=None,
        y_ch4=None,
        y_so2=None,
    ):

        # Use no grad to speed it up! But I need to keep the grads if I am going to add to the loss.

        """
        This is the prediction function for the model.

        We want to take past time steps and predict the next time step, not to reconstruct the past time steps.
        """

        b = x.size(0)

        z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol, y_ch4, y_so2)

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
        z_for_transit = z.clone()
        # if not self.instantaneous:
        #     z_for_transit[:, -1, :, :n_climate_latents] = 0

        pz_mu, pz_std = self.transition(z_for_transit, mask)

        # decode
        px_mu, px_std = self.decode(pz_mu)

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
            z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol, y_ch4, y_so2)

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)

            z_for_transit = z.clone()
            pz_mu, pz_std = self.transition(z_for_transit, mask)
            # here I am taking the approach of sampling from the Z distributions, and then decoding.
            samples_from_zs = torch.zeros(num_samples, b, self.d, self.d_x)
            z_samples = torch.zeros(num_samples, b, self.d, self.d_z)

            # TODO: Remove this for loop
            for i in range(num_samples):
                z_samples[i] = self.distr_transition(pz_mu, pz_std).sample()
                samples_from_zs[i], some_decoded_samples_std = self.decode(z_samples[i])

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
        self,
        x,
        y,
        num_samples,
        with_zs_logprob: bool = False,
        y_co2=None,
        y_aerosol=None,
        y_ch4=None,
        y_so2=None,
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
            z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol, y_ch4, y_so2)

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)

            forcing_dict = self._forcing_dict(y_co2, y_aerosol, y_ch4, y_so2)
            active_forcings = self._active_forcing_dict(forcing_dict) if self.use_forced_latents else None
            if self.use_forced_latents and active_forcings is not None:
                n_climate_latents = self.n_climate_latents
            else:
                n_climate_latents = self.d_z
            z_for_transit = z.clone()

            # This is very safe operation to ensure no climate from last step can be used
            if not self.instantaneous:
                z_for_transit[:, -1, :, :n_climate_latents] = 0

            pz_mu, pz_std = self.transition(z_for_transit, mask)

            dim = pz_mu.ndim
            new_shape = [num_samples]
            for k in range(dim):
                new_shape.append(1)
            z_samples = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).sample()

            if with_zs_logprob:
                z_samples_logprob = self.distr_transition(pz_mu.repeat(new_shape), pz_std.repeat(new_shape)).log_prob(
                    z_samples
                )

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


if __name__ == "__main__":

    device = "cuda:0"
    var = ["ts"]
    tau = 5
    d_x = 16
    d_co2 = 1
    d = len(var)
    future_time_steps = 1
    num_input = d * tau
    model = LatentTSDCD(
        num_layers=2,
        num_hidden=8,
        num_input=num_input,
        num_output=2,
        num_layers_mixing=2,
        num_hidden_mixing=16,
        position_embedding_dim=10,
        transition_param_sharing=False,
        position_embedding_transition=10,
        coeff_kl=1,
        d=d,
        # Here, everything hardcoded to gaussian because GEV leads to Nan... TBD
        distr_z0="gaussian",
        distr_encoder="gaussian",
        distr_transition="gaussian",
        distr_decoder="gaussian",
        d_x=d_x,
        d_z=7,
        # d_z_global=1,
        tau=tau,
        instantaneous=False,
        instantaneous_forcing=True,
        nonlinear_dynamics=True,
        nonlinear_mixing=True,
        tied_w=False,
        fixed=False,
        d_y_co2=d_co2,
        d_y_aerosol=d_x,
        d_y_ch4=0,
        d_y_so2=0,
        use_forced_latents=True,
        n_forced_latents_co2=1,
        n_forced_latents_aerosol=2,
        n_forced_latents_ch4=0,
        n_forced_latents_so2=0,
        forcing_arch="baseline",
        map_aerosol_to_climate=True,
    )
    # model = model.to(device)
    # If use_forced_latent, d_z= sqrt(d_x) + n_forced_latents_co2, n_forced_latents_aerosol
    batch_size = 2
    adj = model.get_adj()  # [tau, dz, dz]
    x = torch.randn(batch_size, tau, 1, d_x)  # .to(device)
    y = torch.randn(batch_size, future_time_steps, d_x)  # .to(device)
    y_co2 = torch.randn(batch_size, tau + future_time_steps, d_co2)  # .to(device)
    y_aerosol = torch.randn(batch_size, tau + future_time_steps, d_x)  # .to(device)
    y_ch4 = None  # y_co2.clone()+1  # .to(device)
    y_so2 = None  # y_aerosol.clone() +1 # .to(device)
    model.eval()
    torch.manual_seed(0)

    with torch.no_grad():
        y_co2_pert = y_co2.clone()
        y_aerosol_pert = y_aerosol.clone()

        # Make a big perturbation so numerical differences are obvious.
        y_co2_pert[:, -1] += 10.0
        y_aerosol_pert[:, -1] += 10.0

        n_climate = model.d_z - model.n_forced_latents_co2 - model.n_forced_latents_aerosol

        # Use deterministic-ish mask probabilities instead of sampled mask.
        mask = model.get_adj().unsqueeze(0).expand(batch_size, -1, -1, -1)

        z1, mu1, _ = model.encode(x, y, y_co2, y_aerosol)
        z2, mu2, _ = model.encode(x, y, y_co2_pert, y_aerosol_pert)
        print(
            "encode climate z diff:",
            (mu1[..., :n_climate] - mu2[..., :n_climate]).abs().max().item(),
            "|| Expected encode climate z diff: =0",
        )  # expect zero, then confirmed that the climate encoder is not impacted by forcings

        print(
            "encode forcing z diff:",
            (mu1[..., n_climate:] - mu2[..., n_climate:]).abs().max().item(),
            "|| Expected encode forcing z diff: >0",
        )

        pz1, _ = model.transition(z1, mask)
        pz2, _ = model.transition(z2, mask)
        z3 = z2.clone()
        z3[:, -1, :, :n_climate] = 100
        pz3, _ = model.transition(z3, mask)

        print(
            "transition climate pz diff:",
            (pz1[..., :n_climate] - pz2[..., :n_climate]).abs().max().item(),
            "|| Expected climate pz diff: >0",
        )
        # expect >0， then confirm the climate latents are transitted

        print(
            "transition forcing pz diff:",
            (pz1[..., n_climate:] - pz2[..., n_climate:]).abs().max().item(),
            "|| Expected forcing pz diff: =0",
        )  # expect zero, then confirmed that the forcing is **not** impacted by any factors

        print(
            "transition forcing pz diff mask z:", (pz3 - pz2).abs().max().item()
        )  # expect zero, then confirmed that the last step climate is not used

        px1, _ = model.decode(pz1)
        px2, _ = model.decode(pz2)
        print("prediction px diff:", (px1 - px2).abs().max().item(), "|| Expected diff>0")
        # expect >0, the changing forcing impact the final decoded climate
        pz3 = pz2
        pz3[..., n_climate:] = 100
        px3, _ = model.decode(pz3)
        print("prediction px diff mask:", (px3 - px2).abs().max().item(), "|| Expected diff=0")
        # expect zero, the decoder doesn't use any transitted forcings as input

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    optimizer.zero_grad()
    # kl_global = 0
    for i in range(1):
        (
            elbo,
            recons,
            kl,
            px_mu,
            forcing_recons_loss_co2,
            forcing_recons_loss_aerosol,
            encoded_forcing_mu,
            forcing_recons_loss,
        ) = model(x, y, gt_z=None, iteration=i, y_co2=y_co2, y_aerosol=y_aerosol, y_ch4=y_ch4, y_so2=y_so2)
        forcing_recons_loss_so2 = forcing_recons_loss["so2"]
        forcing_recons_loss_ch4 = forcing_recons_loss["ch4"]
        print(
            f"{i}: -elbo {-elbo.item():.4f}, recons {recons.item():.4f}, kl {kl.item():.4f} co2 {forcing_recons_loss_co2.item():.4f} aerosol {forcing_recons_loss_aerosol.item():.4f} so2 {forcing_recons_loss_so2.item():.4f} ch4 {forcing_recons_loss_ch4.item():.4f}"
        )
        loss = (
            -elbo
            + forcing_recons_loss_co2
            + forcing_recons_loss_aerosol
            + forcing_recons_loss["so2"]
            + forcing_recons_loss["ch4"]
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Forward: {px_mu[0]}")
    print(f"Ground truth: {y[0]}")

    with torch.no_grad():
        px_mu, y, z, pz_mu, pz_std = model.predict(x, y, y_co2=y_co2, y_aerosol=y_aerosol, y_ch4=y_ch4, y_so2=y_so2)
        print(f"Prediction: {px_mu[0]}")
        px_mu, y, z, pz_mu, pz_std = model.predict_counterfactual(
            x, y, 1, 0.1, y_co2=y_co2, y_aerosol=y_aerosol, y_ch4=y_ch4, y_so2=y_so2
        )
        print(f"predict_counterfactual: {px_mu[0]}")
        samples_from_xs, samples_from_zs, y = model.predict_sample(
            x, y, 2, y_co2=y_co2, y_aerosol=y_aerosol, y_ch4=y_ch4, y_so2=y_so2
        )
        print(samples_from_xs.shape)
        print(f"predict_sample: {samples_from_xs[0]}")
        samples_from_xs, samples_from_zs, y = model.predict_sample_bayesianfiltering(
            x, y, 2, y_co2=y_co2, y_aerosol=y_aerosol, y_ch4=y_ch4, y_so2=y_so2
        )
        print(f"predict_sample_bayesianfiltering: {samples_from_xs[0]}")
