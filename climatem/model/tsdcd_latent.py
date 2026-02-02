# Adapted from the original code for CDSD, Brouillard et al., 2024.

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
            self.param = (gt_mask > 0) * 10.0
        else:
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
        d: int,
        d_x: int,
        d_z: int,
        tau: int,
        instantaneous: bool,
        nonlinear_mixing: bool,
        nonlinear_dynamics: bool,
        hard_gumbel: bool,
        # no_gt: bool,
        # debug_gt_graph: bool,
        # debug_gt_z: bool,
        # debug_gt_w: bool,
        # gt_graph: torch.tensor = None,
        # gt_w: torch.tensor = None,
        tied_w: bool = False,
        fixed: bool = False,
        fixed_output_fraction: float = 1.0,
        gev_learn_xi: bool = False,
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
        self.tau = tau
        self.instantaneous = instantaneous
        self.nonlinear_mixing = nonlinear_mixing
        self.nonlinear_dynamics = nonlinear_dynamics
        self.hard_gumbel = hard_gumbel
        # self.no_gt = no_gt
        # self.debug_gt_graph = debug_gt_graph
        # self.debug_gt_z = debug_gt_z
        # self.debug_gt_w = debug_gt_w
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
            self.autoencoder = NonLinearAutoEncoderUniqueMLP_noloop(
                d,
                d_x,
                d_z,
                self.num_hidden_mixing,
                self.num_layers_mixing,
                use_gumbel_mask=False,
                tied=tied_w,
                embedding_dim=self.position_embedding_dim,
                gt_w=None,
            )
        else:
            # print('Using linear mixing')
            print("LINEAR MIXING")
            self.autoencoder = LinearAutoEncoder(d, d_x, d_z, tied=tied_w)

        # if debug_gt_w:
        #     self.decoder.w = gt_w

        if self.transition_param_sharing:
            self.transition_model = TransitionModelParamSharing(
                self.d,
                self.d_z,
                self.total_tau,
                self.nonlinear_dynamics,
                self.num_layers,
                self.num_hidden,
                self.num_output,
                self.position_embedding_transition,
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
        # if self.debug_gt_graph:
        #     if self.instantaneous:
        #         self.mask.fix(self.gt_graph)
        #     else:
        #         self.mask.fix(self.gt_graph[:-1])

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

    def transition(self, z, mask):

        b = z.size(0)
        mu = torch.zeros(b, self.d, self.d_z)
        std = torch.zeros(b, self.d, self.d_z)

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

            if self.transition_param_sharing:
                pz_params = self.transition_model(z, mask[:, :, i * self.d_z : (i + 1) * self.d_z], i)
            else:
                pz_params = torch.zeros(b, self.d_z, 1)
                for k in range(self.d_z):
                    pz_params[:, k] = self.transition_model(z, mask[:, :, i * self.d_z + k], i, k)
            mu[:, i] = pz_params[:, :, 0]
            std[:, i] = torch.exp(0.5 * self.transition_model.logvar[i])

        # print("This is giving us the pz_mu and pz_std that we use later.")
        return mu, std

    def decode(self, z):

        mu = torch.zeros(z.size(0), self.d, self.d_x)
        std = torch.zeros(z.size(0), self.d, self.d_x)

        # TODO: Can we remove this for loop
        for i in range(self.d):
            # px_mu, px_logvar = self.encoder_decoder(z[:, i], i, encoder=False)

            px_mu, px_logvar = self.autoencoder(z[:, i], i, encode=False)
            if px_mu.ndim == mu.ndim:  # In case of linear mixing with one variable, second dimension is too much
                # Check that linear autoencoder corresponds to PF when multi varia/bles
                px_mu = px_mu.squeeze()

            mu[:, i] = px_mu
            std[:, i] = torch.exp(0.5 * px_logvar)

        return mu, std

    def forward(self, x, y, gt_z, iteration, xi=None):

        b = x.size(0)

        # sample Zs (based on X)
        z, q_mu_y, q_std_y = self.encode(x, y)
        # if self.debug_gt_z:
        #     z = gt_z

        # get params of the transition model p(z^t | z^{<t})
        mask = self.mask(b)
        if self.instantaneous:
            pz_mu, pz_std = self.transition(z.clone(), mask)
        else:
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask)
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

        elbo = recons - kl

        return elbo, recons, kl, px_mu

    # def predict(self, x, y):
    #    b = x.size(0)

    #    with torch.no_grad():
    #        # sample Zs (based on X)
    #        z, q_mu_y, q_std_y = self.encode(x, y)
    #
    #        # get params of the transition model p(z^t | z^{<t})
    #        mask = self.mask(b)
    #        pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask)
    #        px_mu, px_std = self.decode(pz_mu)
    #    return px_mu, y

    def predict_pxmu_pxstd(self, x, y):

        # NOTE: this one was working fine for the CRPS loss because I was not using no_grad...
        # I need to keep the grads if I am going to add to the loss

        b = x.size(0)

        # sample Zs (based on X)
        z, q_mu_y, q_std_y = self.encode(x, y)

        # get params of the transition model p(z^t | z^{<t})
        mask = self.mask(b)
        if self.instantaneous:
            pz_mu, pz_std = self.transition(z.clone(), mask)
        else:
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask)

        # get params from decoder p(x^t | z^t)
        # we pass only the last z to the decoder, to get xs.
        px_mu, px_std = self.decode(pz_mu)

        return px_mu, px_std

    def predict(self, x, y):

        # Use no grad to speed it up! But I need to keep the grads if I am going to add to the loss.

        """
        This is the prediction function for the model.

        We want to take past time steps and predict the next time step, not to reconstruct the past time steps.
        """
        b = x.size(0)

        # NOTE: we are not using y here. We encode using both x and y,
        # but then we discard the latents from the y encoding.

        z, q_mu_y, q_std_y = self.encode(x, y)

        mask = self.mask(b)

        if self.instantaneous:
            pz_mu, pz_std = self.transition(z.clone(), mask)
        else:
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask)

        # decode
        px_mu, px_std = self.decode(pz_mu)

        return px_mu, y, z, pz_mu, pz_std

    def predict_counterfactual(self, x, y, counterfactual_z_index, counterfactual_z_value):

        # Use no grad to speed it up! But I need to keep the grads if I am going to add to the loss.

        """
        This is the prediction function for the model.

        We want to take past time steps and predict the next time step, not to reconstruct the past time steps.
        """

        b = x.size(0)

        z, q_mu_y, q_std_y = self.encode(x, y)

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
            pz_mu, pz_std = self.transition(z.clone(), mask)
        else:
            pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask)

        # decode
        px_mu, px_std = self.decode(pz_mu)

        return px_mu, y, z, pz_mu, pz_std

    def predict_sample(self, x, y, num_samples):
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

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)

            if self.instantaneous:
                pz_mu, pz_std = self.transition(z.clone(), mask)
            else:
                pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask)

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

    def predict_sample_bayesianfiltering(self, x, y, num_samples, with_zs_logprob: bool = False):
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

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)

            if self.instantaneous:
                pz_mu, pz_std = self.transition(z.clone(), mask)
            else:
                pz_mu, pz_std = self.transition(z[:, :-1].clone(), mask)

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
                z_samples.reshape(z_samples.size(0) * z_samples.size(1), z_samples.size(2), z_samples.size(3))
            )
            samples_from_zs = samples_from_zs.reshape(z_samples.size(0), z_samples.size(1), z_samples.size(2), self.d_x)
            # some_decoded_samples_mu, some_decoded_samples_std = self.decode(z_samples[i])

            # samples_from_zs[i] = some_decoded_samples_mu

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
        self.use_grad_project = True
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
        gt_w=None,
    ):
        super().__init__(d, d_x, d_z, num_hidden, num_layer, use_gumbel_mask, tied, gt_w)
        self.embedding_encoder = nn.Embedding(d_z, embedding_dim)
        self.encoder = MLP(num_layer, num_hidden, d_x + embedding_dim, 1)  # embedding_dim_encoding

        self.decoder = MLP(num_layer, num_hidden, d_z + embedding_dim, 1)
        self.embedding_decoder = nn.Embedding(d_x, embedding_dim)

    def encode(self, x, i):

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
        x_ = torch.cat(
            (mask_ * x.unsqueeze(1), embedded_x), dim=2
        )  # expand dimensions of x for broadcasting - looks good.

        del embedded_x
        del mask_

        mu = self.encoder(x_).squeeze()

        return mu, self.logvar_encoder

    def decode(self, z, i):

        mask = super().get_decode_mask(z.shape[0])
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
            print("NON LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(num_layers, num_hidden, d * d_z * tau, self.num_output) for i in range(d * d_z))
        else:
            print("LINEAR DYNAMICS")
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
            print("NON LINEAR DYNAMICS")
            self.nn = nn.ModuleList(
                MLP(num_layers, num_hidden, d * d_z * tau + embedding_dim, self.num_output) for i in range(d)
            )
        else:
            print("LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(0, 0, d * d_z * tau + embedding_dim, self.num_output) for i in range(d))
        # self.nn = MLP(num_layers, num_hidden, d * k * k, self.num_output)

    def forward(self, z, mask, i):
        """Returns the params of N(z_t | z_{<t}) for a specific feature i and latent variable k NN(G_{tau-1} * z_{t-1},
        ..., G_{tau-k} * z_{t-k})"""

        j_values = torch.arange(self.d_z, device=z.device).expand(
            z.shape[0], -1
        )  # create a 2D tensor with shape (x.shape[0], self.d_z)
        embedded_z = self.embedding_transition(j_values)
        masked_z = (mask * z).transpose(3, 2).reshape((z.shape[0], -1, self.d_z)).transpose(2, 1)
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
