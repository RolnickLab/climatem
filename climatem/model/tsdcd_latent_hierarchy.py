from collections import OrderedDict

import torch
import torch.nn as nn

from .tsdcd_latent import (
    MLP,
    LatentTSDCD,
    LinearAutoEncoder,
    Mask,
    NonLinearAutoEncoderUniqueMLP_noloop,
    TransitionModelParamSharing,
    logger,
)

# For this version, the transition paramter sharing from tsdcd_latent can work, but we may need extra modification to close the impact from z_global to pz_mu's forcing slice


class TransitionModelGlobal(nn.Module):
    """Models the transitions between the latent variables Z2 with neural networks."""

    def __init__(
        self,
        d: int,
        d_z: int,
        d_z_forcing: int,
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
        self.d_z_forcing = d_z_forcing
        self.tau = tau
        output_var = False

        # initialize NNs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        if output_var:
            self.num_output = num_output
        else:
            self.num_output = 1
            # self.logvar = torch.ones(1)  * 0. # nn.Parameter(torch.ones(d) * 0.1)
            # self.logvar = nn.Parameter(torch.ones(d) * -4)
            self.logvar = nn.Parameter(torch.ones(d, d_z) * -4)

        input_dim = d * d_z * tau
        if nonlinear_dynamics:
            print("NON LINEAR DYNAMICS")
            # How global forcing modulate current climate z
            self.nn = nn.ModuleList(MLP(num_layers, num_hidden, input_dim, self.num_output) for i in range(d * d_z))
            # How global forcing modulate current climate z
            self.forcing_modulator = MLP(0, 0, d_z + d_z_forcing, d_z)
        else:
            print("LINEAR DYNAMICS")
            self.nn = nn.ModuleList(MLP(0, 0, input_dim, self.num_output) for i in range(d * d_z))
            self.forcing_modulator = MLP(num_layers, num_hidden, d_z + d_z_forcing, d_z)

    # TODO: alternatively, we can assign different modulators to differnet months (but this can cause mismatch between the real month and its order in the sequence)
    # TODO: add explicit month embedding to forcing modulator, so that the effect can change based on current month

    def forward(self, z_global, i, k, z_forcings=None):
        """Predict next-step global latent after each historical latent state is conditioned on same-time global
        forcings."""
        # z: (b, tau, d_z_global, 1)
        # z_forcings: list of tensors, each  (b, tau, z_f, 1)

        if z_forcings is not None:
            z_forcings = torch.cat(z_forcings, dim=-2)

            x = torch.cat([z_global, z_forcings], dim=-2)  # (b, tau, 2, 1)

            x = x.view(-1, self.d_z + self.d_z_forcing)  # (b*tau, 2)

            delta = self.forcing_modulator(x)  # (b*tau, 1)
            delta = delta.view_as(z_global)  # (b,tau, 1,1)

            z_new = z_global + delta  # (b,tau, 1,1)
        else:
            z_new = z_global

        flat_z = z_new.view(z_global.size(0), -1)
        param_z = self.nn[i * self.d_z + k](flat_z)
        return param_z


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

        # print("The z is of size: ", z.size()) [256, tau, dz, 1])
        batch_size = z.size(0)
        z_forcings = torch.cat(z_forcings, dim=-2)
        n_climate = self.d_z - z_forcings.shape[-2]

        total_z_for_mask = torch.cat([z, z_forcings], dim=-2)  # ([b, tau+1, 7, 1])

        total_z_for_mask = total_z_for_mask.view(mask.size())

        # print("The z is now, after z.view() of size: ", z.size()) [256, tau, dz])

        # print("what is mask * z shape? ", (mask * z).size())

        masked_z = (mask * total_z_for_mask).view(batch_size, -1)
        if z_global is not None:
            flat_z_global = z_global.view(batch_size, -1)

            # global forcings only affect climate targets
            if k < self.d_z - n_climate:
                global_input = flat_z_global
            else:
                global_input = torch.zeros_like(flat_z_global)

            masked_z = torch.cat([masked_z, global_input], dim=1)
        # if z_global is not None:
        #     # 2. Process current z2 (t): flatten without masking
        #     # (b, 1, d, dz2) -> (b, 1*d * dz2)
        #     flat_z2 = z_global.view(batch_size, -1)

        #     # 3. Concatenate along the feature dimension
        #     # Result shape: (b, (tau*d*dz) + (1*d*dz2))
        #     masked_z = torch.cat([masked_z, flat_z2], dim=1)

        # 4. Predict params for N(z1_t | z1_{<t}, z2_t)

        param_z = self.nn[i * self.d_z + k](masked_z)

        # print("What is the shape of param_z?", param_z.size())

        # param_z = self.nn(masked_z)

        return param_z


class HierarchicalLatentTSDCD(LatentTSDCD):
    """
    Differentiable Causal Discovery for time series with latent variables of two levels Extract a global letent from all
    local latents.

    Implements the HierarchicalLatentTSDCD architecture: inherit LatentTSDCD, from an encoder maps observations to local latent
    variables,
    Encoders maps forcings to forcing latents
    A global encoder map local latents to a global latents,

    a learnable causal mask (Gumbel-sigmoid) parameterizes the temporal causal graph between local latents,
    a global transition model predicts future global latents conditioned on past global latents, forced by greenhouse-gas latents,
    and a local transition model predict future local latents conditioned on masked past local latents and current global latents, forced by aerosol latents,

    and a decoder reconstructs observations from local latents.

    New variable name glossary (used throughout forward / loss computation):
        pz2_mu  -- predicted global z mean (latent dynamics / transition output), shape (batch, d, d_z_global) only d_z_global=1 is tested
        pz2_std -- predicted global z std  (latent dynamics / transition output), shape (batch, d, d_z_global)
        qz2_mu  -- variational posterior mean for z_global (alias used in some contexts)

    New loss: loss between global latent and GMST
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
        d_z_global: int,
        tau: int,  # Number of timesteps as input
        instantaneous: bool,
        instantaneous_forcing: bool,
        nonlinear_mixing: bool,
        nonlinear_dynamics: bool,
        nonlinear_global_dynamics: bool,
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
            d_z_global: number of global latent variables, usually set to 1.
        """
        super().__init__(
            num_layers=num_layers,
            num_hidden=num_hidden,
            num_input=num_input,
            num_output=num_output,
            num_layers_mixing=num_layers_mixing,
            num_hidden_mixing=num_hidden_mixing,
            position_embedding_dim=position_embedding_dim,
            transition_param_sharing=transition_param_sharing,
            position_embedding_transition=position_embedding_transition,
            coeff_kl=coeff_kl,
            distr_z0=distr_z0,
            distr_encoder=distr_encoder,
            distr_transition=distr_transition,
            distr_decoder=distr_decoder,
            d=d,  # Number of variables
            d_x=d_x,  # Dimension of observations
            d_z=d_z,  # Dimension of latent space
            tau=tau,  # Number of timesteps as input
            instantaneous=instantaneous,
            instantaneous_forcing=instantaneous_forcing,
            nonlinear_mixing=nonlinear_mixing,
            nonlinear_dynamics=nonlinear_dynamics,
            tied_w=tied_w,
            reduce_encoding_pos_dim=reduce_encoding_pos_dim,
            fixed=fixed,
            fixed_output_fraction=fixed_output_fraction,
            gev_learn_xi=gev_learn_xi,
            use_exogenous=use_exogenous,
            d_y_co2=d_y_co2,
            d_y_aerosol=d_y_aerosol,
            d_y_ch4=d_y_ch4,
            d_y_so2=d_y_so2,
            use_forced_latents=use_forced_latents,
            n_forced_latents_co2=n_forced_latents_co2,
            n_forced_latents_aerosol=n_forced_latents_aerosol,
            n_forced_latents_ch4=n_forced_latents_ch4,
            n_forced_latents_so2=n_forced_latents_so2,
            forcing_arch=forcing_arch,
            map_aerosol_to_climate=map_aerosol_to_climate,
            forcing_mse=forcing_mse,
        )
        self.global_forcing_order = ("co2", "ch4")
        self.local_forcing_order = ("aerosol", "so2")

        self.global_forcing_latent_dims = OrderedDict(
            (forcing, self.forcing_latent_dims[forcing]) for forcing in self.global_forcing_order
        )

        n_forced_global_latents_total = sum(self.global_forcing_latent_dims.values()) if self.use_forced_latents else 0

        self.d_z2 = d_z_global
        if self.nonlinear_mixing:
            logger.info("Non-LINEAR MIXING")
            # Local autoencoder: stay as before, map each variable to their own latent space
            # NEW: Global autoencoder, encode z_global from z_climate
            self.autoencoder_global = NonLinearAutoEncoderUniqueMLP_noloop(
                d,
                self.n_climate_latents,
                self.d_z2,
                self.num_hidden_mixing,
                self.num_layers_mixing,
                tied=tied_w,
                embedding_dim=self.position_embedding_dim,
                reduce_encoding_pos_dim=self.reduce_encoding_pos_dim,
                gt_w=None,
                n_forced_latents_co2=0,
                n_forced_latents_aerosol=0,
                n_forced_latents_ch4=0,
                n_forced_latents_so2=0,
                d_y_co2_spatial=0,  # treat as not using exogenous
                d_y_aerosol_spatial=0,  # treat as not using exogenous
                d_y_ch4_spatial=0,  # treat as not using exogenous
                d_y_so2_spatial=0,  # treat as not using exogenous
                use_forced_latents=False,
                forcing_mse=forcing_mse,
            )
        else:
            # print('Using linear mixing')
            logger.info("LINEAR MIXING")
            self.autoencoder_global = LinearAutoEncoder(
                d,
                self.n_climate_latents,
                self.d_z2,
                tied=tied_w,
                d_y_co2_spatial=0,  # treat as not using exogenous
                d_y_aerosol_spatial=0,  # treat as not using exogenous
                d_y_ch4_spatial=0,  # treat as not using exogenous
                d_y_so2_spatial=0,  # treat as not using exogenous
                use_forced_latents=False,
                n_forced_latents_co2=0,
                n_forced_latents_aerosol=0,
                n_forced_latents_ch4=0,
                n_forced_latents_so2=0,
                forcing_mse=forcing_mse,
            )

        # Global transition model, single latent: no mask (i.e., no sparsity), no parameter-sharing,
        # forward path predicts pz2_mu_t from pz2_mu_<t, each pz2_mu_t is concatenated with z_co2_t
        # so that in the transition, the forcing is transitted together with z2
        self.transition_model_global = TransitionModelGlobal(
            d=self.d,
            d_z=self.d_z2,
            d_z_forcing=n_forced_global_latents_total,
            tau=self.total_tau,
            nonlinear_dynamics=nonlinear_global_dynamics,
            num_layers=self.num_layers,
            num_hidden=self.num_hidden,
            num_output=self.num_output,
        )

        # Local transition model
        if self.transition_param_sharing:
            # Local transition model, transit from z^{<t} and z2^{t}
            # I will keep as before but without the involvement of global forcings (put mask=0)
            self.transition_model = TransitionModelParamSharing(
                self.d,
                self.d_z,
                self.total_tau,
                self.nonlinear_dynamics,
                self.num_layers,
                self.num_hidden,
                self.num_output,
                self.position_embedding_dim,
                d_z_global=self.d_z2,
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
                d_z_global=self.d_z2,
                instantaneous=instantaneous,
            )

        # print("We are setting the Mask here.")
        # The mask is still a full mask for d_z, but global_forcings to climate path closed
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
            n_exclude_global_forcing=n_forced_global_latents_total,
        )
        # TODO: directly force z_global_t to be gsmt_t without the prediction head
        if nonlinear_global_dynamics:
            self.gmst_head = MLP(num_layers, num_hidden, d_z_global, 1)
        else:
            self.gmst_head = MLP(0, 0, d_z_global, 1)

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
        # Extract the global latent solely from climate latents
        z_climate = z[:, :, :, : self.n_climate_latents]  # ([1, 6, 1, 4])

        for i in range(self.d):
            for t in range(self.tau + 1):
                q_mu, q_logvar = self.autoencoder_global(z_climate[:, t, i], i, encode=True)  # q_mu: shape (b, d_z)
                # reparam trick - here we sample from a Gaussian...every time
                q_std = torch.exp(0.5 * q_logvar)
                z2[:, t, i] = q_mu + q_std * self.distr_encoder(0, 1, size=q_mu.size())
            mu[:, i] = q_mu
            std[:, i] = q_std
        return z2, mu, std

    def transition_global(self, z2, z):
        """
        Transition model for z2, fully conencted mask
        Args:
            z (Tensor): Shape (B, T, D_z2), where T = tau corresponds to the
                temporal window [t - tau, ..., t - 1].

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

        # here I seperate z_climate and z_forcings

        index_slices = self._forcing_index_slices()
        global_forcing_slices = OrderedDict(
            (name, z[..., idx]) for name, idx in index_slices.items() if name in self.global_forcing_order
        )
        # global_forcing_slices["co2"]  (b, tau, 1, 1)

        # TODO Can we remove this for loop
        for i in range(self.d):
            z_global_forcing = self._transition_forcing_list(global_forcing_slices, i)
            pz2_params = torch.zeros(b, self.d_z2, 1)
            for k in range(self.d_z2):
                pz2_params[:, k] = self.transition_model_global(
                    z2[:, :, i][:, :, :, None], i, k, z_forcings=z_global_forcing
                )
            mu[:, i] = pz2_params[:, :, 0]
            std[:, i] = torch.exp(0.5 * self.transition_model_global.logvar[i])

        return mu, std

    def transition(self, z, z_global, mask):
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

        if self.map_aerosol_to_climate:
            # The mask will be multiplied with a mapping which only allow the the latents interaction that's decodes to the same spatial area
            mask = self.apply_spatial_forcing_mask(mask)
        # Only climate latents go to the transition model
        # here I seperate z_climate and z_forcings
        z_climate = z[:, :, :, : self.n_climate_latents]  # ([1, 6, 1, 4])
        index_slices = self._forcing_index_slices()
        forcing_slices = OrderedDict((name, z[..., idx]) for name, idx in index_slices.items())

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
                    z_global=z_global,
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
                        z_global=z_global,
                    )
            mu[:, i] = pz_params[:, :, 0]
            std[:, i] = torch.exp(0.5 * self.transition_model.logvar[i])

        # print("This is giving us the pz_mu and pz_std that we use later.")
        return mu, std

    def forward(
        self,
        x,
        y,
        gt_z,
        iteration,
        xi=None,
        y_co2=None,
        y_aerosol=None,
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

        z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol, y_ch4, y_so2)
        z2, q_mu_z2, q_std_z2 = self.encode_global(z)
        # z(b, tau + 1, d, d_z)
        # z2 (b, tau + 1, d, d_z_global)
        encoded_forcing_mu = None

        forcing_dict = self._forcing_dict(y_co2, y_aerosol, y_ch4, y_so2)
        active_forcings = self._active_forcing_dict(forcing_dict) if self.use_forced_latents else None
        #  Here we isolate z_climate from the total z
        # get params of the transition model p(z^t | z^{<t})
        mask = self.mask(b)  # [b, tau, d_z, d_z]
        z_for_transit = z.clone()

        # Transit global
        pz2_mu, pz2_std = self.transition_global(z2=z2[:, :-1].clone(), z=z[:, :-1].clone())
        # pz2_mu  (b, d, dz_global)

        pz_mu, pz_std = self.transition(z_for_transit, z_global=z2[:, -1], mask=mask)

        # get params from decoder p(x^t | z^t)
        # we pass only the last z to the decoder, to get xs.

        px_mu, px_std = self.decode(z[:, -1])  # pz_mu (b,d,d_x)

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
            q_std_z2_safe = q_std_z2.clamp(min=eps)
            pz2_std_safe = pz2_std.clamp(min=eps)

            kl_raw_z2 = (
                0.5 * (torch.log(pz2_std_safe**2) - torch.log(q_std_z2_safe**2))
                + 0.5 * (q_std_z2_safe**2 + (q_mu_z2 - pz2_mu) ** 2) / pz2_std_safe**2
                - 0.5
            )
        else:
            px_distr = self.distr_decoder(px_mu, px_std)
            recons = torch.mean(torch.sum(px_distr.log_prob(y), dim=[1, 2]))  # recons is a scaler

            # compute the KL, the reconstruction and the ELBO
            # kl = distr.kl_divergence(q, p).mean()
            kl_raw = (
                0.5 * (torch.log(pz_std**2) - torch.log(q_std_y**2))
                + 0.5 * (q_std_y**2 + (q_mu_y - pz_mu) ** 2) / pz_std**2
                - 0.5
            )
            kl_raw_z2 = (
                0.5 * (torch.log(pz2_std**2) - torch.log(q_std_z2**2))
                + 0.5 * (q_std_z2**2 + (q_mu_z2 - pz2_mu) ** 2) / pz2_std**2
                - 0.5
            )

        kl_local = torch.sum(kl_raw[..., : self.n_climate_latents], dim=[2]).mean()
        kl_global = torch.sum(kl_raw_z2[..., : self.n_climate_latents], dim=[2]).mean()

        kl = kl_local + kl_global
        # kl = torch.sum(0.5 * (torch.log(pz_std**2) - torch.log(q_std_y**2)) + 0.5 *
        # (q_std_y**2 + (q_mu_y - pz_mu) ** 2) / pz_std**2 - 0.5, dim=[1, 2]).mean()
        assert kl >= 0, f"KL={kl} has to be >= 0"

        elbo = recons - self.coeff_kl * kl

        # The naming can be confusion, but I put gmst loss into the forcing recons loss

        gmst_loss = torch.mean((pz2_mu.mean(dim=-1) - y.mean(dim=-1)) ** 2)

        # Compute forcing reconstruction losses
        forcing_recons_losses = OrderedDict((name, torch.tensor(0.0, device=x.device)) for name in self.forcing_order)
        forcing_recons_losses["gmst_loss"] = gmst_loss
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
                z_forced_target = z[:, -1, 0, self.n_climate_latents :]
                forcing_outputs = self.autoencoder.decode_forcings(z_forced_target)
            elif forcing_arch == "transitioned":
                if not self._forcing_arch_logged:
                    logger.info("[ForcingArch] Using forcing_arch='transitioned' (pz_mu forced latents)")
                    self._forcing_arch_logged = True
                # Use transitioned latents (pz_mu) for forcing reconstruction
                z_forced_target = pz_mu[:, 0, self.n_climate_latents :]  # Shape: (batch, n_forced_latents_total)
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

    def predict_pxmu_pxstd(
        self,
        x,
        y,
        y_co2=None,
        y_aerosol=None,
        y_ch4=None,
        y_so2=None,
    ):

        # NOTE: this one was working fine for the CRPS loss because I was not using no_grad...
        # I need to keep the grads if I am going to add to the loss

        b = x.size(0)

        # sample Zs (based on X)
        z, q_mu_y, q_std_y = self.encode(x, y, y_co2, y_aerosol)
        z2, _, _ = self.encode_global(z)

        # get params of the transition model p(z^t | z^{<t})
        mask = self.mask(b)

        pz2_mu, pz2_std = self.transition_global(z2=z2[:, :-1].clone(), z=z[:, :-1].clone())

        pz_mu, pz_std = self.transition(z.clone(), z_global=pz2_mu, mask=mask)

        # get params from decoder p(x^t | z^t)
        # we pass only the predicted z to the decoder, to get xs.
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

        z2, _, _ = self.encode_global(z)

        mask = self.mask(b)
        pz2_mu, pz2_std = self.transition_global(z2=z2[:, :-1].clone(), z=z[:, :-1].clone())  # (b,1,d_z2)

        pz_mu, pz_std = self.transition(z.clone(), pz2_mu.clone(), mask=mask)

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

        z2, _, _ = self.encode_global(z)

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
        pz2_mu, _ = self.transition_global(z2=z2[:, :-1].clone(), z=z[:, :-1].clone())

        pz_mu, pz_std = self.transition(z.clone(), z_global=pz2_mu, mask=mask)

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

            z2, _, _ = self.encode_global(z)

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)
            pz2_mu, _ = self.transition_global(z2=z2[:, :-1].clone(), z=z[:, :-1].clone())

            pz_mu, pz_std = self.transition(z.clone(), z_global=pz2_mu, mask=mask)

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

            z2, q_mu_z2, q_std_z2 = self.encode_global(z)

            # get params of the transition model p(z^t | z^{<t})
            mask = self.mask(b)

            forcing_dict = self._forcing_dict(y_co2, y_aerosol, y_ch4, y_so2)
            active_forcings = self._active_forcing_dict(forcing_dict) if self.use_forced_latents else None
            if self.use_forced_latents and active_forcings is not None:
                n_climate_latents = self.n_climate_latents
            else:
                n_climate_latents = self.d_z

            z_for_transit = z.clone()
            if not self.instantaneous:
                z_for_transit[:, -1, :, :n_climate_latents] = 0

            pz2_mu, pz2_std = self.transition_global(z2=z2[:, :-1].clone(), z=z_for_transit[:, :-1].clone())
            pz_mu, pz_std = self.transition(z_for_transit, z_global=pz2_mu, mask=mask)

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


if __name__ == "__main__":
    device = "cuda:0"
    var = ["ts"]
    tau = 5
    d_x = 16
    d_co2 = 1
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
        position_embedding_dim=10,
        transition_param_sharing=True,
        position_embedding_transition=10,
        coeff_kl=1,
        d=d,
        # Here, everything hardcoded to gaussian because GEV leads to Nan... TBD
        distr_z0="gaussian",
        distr_encoder="gaussian",
        distr_transition="gaussian",
        distr_decoder="gaussian",
        d_x=d_x,
        d_z=10,
        d_z_global=1,
        tau=tau,
        instantaneous=False,
        instantaneous_forcing=True,
        nonlinear_dynamics=True,
        nonlinear_global_dynamics=False,
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
        map_aerosol_to_climate=False,
        forcing_mse=True,
    )
    # model = model.to(device)
    # If use_forced_latent, d_z= sqrt(d_x) + n_forced_latents_co2, n_forced_latents_aerosol
    batch_size = 2
    adj = model.get_adj()  # [tau, dz, dz]
    x = torch.randn(batch_size, tau, 1, d_x)  # .to(device)
    y = torch.randn(batch_size, future_time_steps, d_x)  # .to(device)
    y_co2 = torch.randn(batch_size, tau + future_time_steps, d_co2)  # .to(device)
    y_aerosol = torch.randn(batch_size, tau + future_time_steps, d_x)  # .to(device)
    y_ch4 = y_co2.clone() + 1  # .to(device)
    y_so2 = y_aerosol.clone() + 1  # .to(device)
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
        mask = model.get_adj() * model.mask.fixed_mask.unsqueeze(0).expand(batch_size, -1, -1, -1)

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

        z_global_1, mu_global_1, _ = model.encode_global(z1)
        z_global_2, mu_global_2, _ = model.encode_global(z2)

        pz_global_1, _ = model.transition_global(z2=z_global_1[:, :-1].clone(), z=z1[:, :-1].clone())
        pz_global_2, _ = model.transition_global(z2=z_global_2[:, :-1].clone(), z=z2[:, :-1].clone())

        pz1, _ = model.transition(z1, mask=mask, z_global=z_global_1[:, -1])
        pz2, _ = model.transition(z2, mask=mask, z_global=z_global_2[:, -1])

        z3 = z2.clone()
        z3[:, -1, :, :n_climate] = 100
        pz3, _ = model.transition(z3, mask=mask, z_global=z_global_2[:, -1])

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
            "transition forcing pz diff mask z:", (pz3 - pz2).abs().max().item(), "|| Expected forcing pz diff: =0"
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

    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # optimizer.zero_grad()
    # # kl_global = 0
    # for i in range(10):
    #     elbo,recons,kl,px_mu,forcing_recons_loss_co2,forcing_recons_loss_aerosol,encoded_forcing_mu, forcing_recons_loss, gmst_loss = model(x, y, gt_z=None, iteration=i,y_co2=y_co2, y_aerosol=y_aerosol,y_ch4=y_ch4, y_so2=y_so2)
    #     forcing_recons_loss_so2 = forcing_recons_loss["so2"]
    #     forcing_recons_loss_ch4 = forcing_recons_loss["ch4"]
    #     print(
    #         f"{i}: -elbo {-elbo.item():.4f}, recons {recons.item():.4f}, kl {kl.item():.4f} co2 {forcing_recons_loss_co2.item():.4f} aerosol {forcing_recons_loss_aerosol.item():.4f} so2 {forcing_recons_loss_so2.item():.4f} ch4 {forcing_recons_loss_ch4.item():.4f} gmst {forcing_recons_loss_ch4.item():.4f}"
    #     )
    #     loss = -elbo+forcing_recons_loss_co2+forcing_recons_loss_aerosol+forcing_recons_loss["so2"]+forcing_recons_loss["ch4"]
    #     loss.backward()
    #     optimizer.step()
    #     optimizer.zero_grad()
    # print(f"Forward: {px_mu[0]}")
    # print(f"Ground truth: {y[0]}")

    # with torch.no_grad():
    #     px_mu, y, z, pz_mu, pz_std = model.predict(x, y,y_co2=y_co2, y_aerosol=y_aerosol,y_ch4=y_ch4, y_so2=y_so2)
    #     print(f"Prediction: {px_mu[0]}")
    #     px_mu, y, z, pz_mu, pz_std = model.predict_counterfactual(x, y, 1, 0.1,y_co2=y_co2, y_aerosol=y_aerosol,y_ch4=y_ch4, y_so2=y_so2)
    #     print(f"predict_counterfactual: {px_mu[0]}")
    #     samples_from_xs, samples_from_zs, y = model.predict_sample(x, y, 2,y_co2=y_co2, y_aerosol=y_aerosol,y_ch4=y_ch4, y_so2=y_so2)
    #     print(samples_from_xs.shape)
    #     print(f"predict_sample: {samples_from_xs[0]}")
    #     samples_from_xs, samples_from_zs, y = model.predict_sample_bayesianfiltering(x, y, 2,y_co2=y_co2, y_aerosol=y_aerosol,y_ch4=y_ch4, y_so2=y_so2)
    #     print(f"predict_sample_bayesianfiltering: {samples_from_xs[0]}")
