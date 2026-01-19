# Adapted from the original code for CDSD, Brouillard et al., 2024.

from collections import OrderedDict

import torch
import torch.distributions as distr
import torch.nn as nn


class Mask(nn.Module):
    def __init__(
        self,
        d_x: int,  # number of variables
        d_z: int,  # number of control variables
        drawhard: bool,
    ):
        super().__init__()
        self.d_x = d_x
        self.drawhard = drawhard
        # Here we can just set what we want the output to be.
        self.uniform = distr.uniform.Uniform(0, 1)

        # initialize mask as log(mask_ij) = 1
        self.param = nn.Parameter(torch.ones((d_x + d_z, d_x + d_z)) * 5)

        self.fixed_mask = torch.ones_like(self.param)
        # set diagonal 0 for G_t0
        self.fixed_mask[torch.arange(d_z, d_x + d_z), torch.arange(d_z, d_x + d_z)] = (
            0  # Remove diag for variables except controls
        )
        self.fixed_mask[:, :d_z] = 0
        self.fixed_mask[torch.arange(d_z), torch.arange(d_z)] = 1

    def forward(self, b: int) -> torch.Tensor:
        """
        :param b: batch size
        :param tau: temperature constant for sampling
        """
        adj = gumbel_sigmoid(self.param, self.uniform, b, hard=self.drawhard)
        adj = adj * self.fixed_mask
        return adj

    def get_proba(self) -> torch.Tensor:
        return torch.sigmoid(self.param) * self.fixed_mask


def sample_logistic(shape, uniform):
    u = uniform.sample(shape)
    return torch.log(u) - torch.log(1 - u)


def gumbel_sigmoid(log_alpha, uniform, bs, hard=False):
    shape = tuple([bs] + list(log_alpha.size()))
    logistic_noise = sample_logistic(shape, uniform)
    y_soft = torch.sigmoid((log_alpha + logistic_noise))

    if hard:
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
        self.num_output = num_output

        module_dict = OrderedDict()

        # create model layer by layer
        out_features = num_hidden
        if num_layers == 0:
            out_features = num_output

        module_dict["lin0"] = nn.Linear(num_input, out_features)

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


class TSDCD(nn.Module):
    """Differentiable Causal Discovery for time series with latent variables."""

    def __init__(
        self,
        num_layers: int,
        num_hidden: int,
        position_embedding_dim: int,
        distr_x: str,
        d_x: int,
        d_z: int,  # control variables dimension
        hard_gumbel: bool,
    ):
        super().__init__()

        # nn encoder hyperparameters
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.position_embedding_dim = position_embedding_dim

        self.d_x = d_x
        self.d_z = d_z
        self.total_d = d_x + d_z
        self.hard_gumbel = hard_gumbel

        if distr_x == "gaussian":
            self.distr_x = distr.normal.Normal
        else:
            raise NotImplementedError("This distribution is not implemented yet.")

        # print("We are setting the Mask here.")
        self.mask = Mask(
            d_x,
            d_z,
            drawhard=hard_gumbel,
        )

        self.transition_model = TransitionModelParamSharing(
            self.d_x,
            self.d_z,
            self.num_layers,
            self.num_hidden,
            self.position_embedding_dim,
        )

    def get_adj(self):
        """
        Returns: Matrices of the probabilities from which the masks linking the
        latent variables are sampled
        """
        return self.mask.get_proba()

    def transition(self, x, z, mask):

        b = x.size(0)
        mu = torch.zeros(b, self.total_d)
        std = torch.zeros(b, self.total_d)

        mu = torch.zeros(b, self.total_d)
        for k in range(self.total_d):
            mu[:, k] = self.transition_model(x, z, mask[:, :, k], k).squeeze()
        # px_params = self.transition_model(x, z, mask) # Here this is just a concatenation + multiplication

        # mu = px_params[:, :, 0]

        std = torch.exp(0.5 * self.transition_model.logvar)

        # print("This is giving us the pz_mu and pz_std that we use later.")
        return mu, std

    def forward(self, x, z):

        b = x.size(0)
        mask = self.mask(b)

        px_mu, px_std = self.transition(x.clone(), z.clone(), mask)
        px_distr = self.distr_x(px_mu, px_std)

        recons = torch.mean(torch.sum(px_distr.log_prob(torch.cat((z, x), dim=1)), dim=1))  # log likelihood of the data

        return recons, px_mu, px_std


class TransitionModelParamSharing(nn.Module):
    """Models the transitions between the latent variables Z with neural networks."""

    # Attempt at parameter sharing in the transition model

    def __init__(
        self,
        d_x: int,
        d_z: int,
        num_layers: int,
        num_hidden: int,
        embedding_dim: int = 5,
    ):
        """
        Args:
            d: number of features
            d_z: number of latent variables
            tau: size of the timewindow
            num_layers: number of layers for the neural networks
            num_hidden: number of hidden units
        """
        super().__init__()
        self.d_x = d_x  # number of variables
        self.d_z = d_z
        self.total_d = d_x + d_z

        # initialize NNs
        self.nonlinear_dynamics = num_layers > 0
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        # self.embedding_dim = embedding_dim
        # self.embedding_transition = nn.Embedding(self.total_d, embedding_dim)

        self.logvar = nn.Parameter(torch.ones(1, self.total_d) * -4)
        if self.nonlinear_dynamics:
            print("NON LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(num_layers, num_hidden, self.total_d, 1) for i in range(self.total_d))
            # self.nn = MLP(num_layers, num_hidden, self.total_d + embedding_dim, 1)
        else:
            print("LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(0, 0, self.total_d, 1) for i in range(self.total_d))
            # self.nn = MLP(0, 0, self.total_d + embedding_dim, 1)

    def forward(self, x, z, mask, k):
        """Returns the params of N(z_t | z_{<t}) for a specific feature i and latent variable k NN(G_{tau-1} * z_{t-1},
        ..., G_{tau-k} * z_{t-k})"""

        # j_values = torch.arange(self.total_d, device=x.device).expand(
        #     x.shape[0], -1
        # )  # create a 2D tensor with shape (x.shape[0], self.d_x)
        # embedded_x = self.embedding_transition(j_values)
        all_var = torch.cat((z, x), dim=1)
        x_ = mask * all_var  # .transpose(2, 1)
        # x_ = torch.cat((x_, embedded_x), dim=2)

        param_x = self.nn[k](x_)

        # del embedded_x
        # del masked_x
        del x_

        return param_x
