"""
This code is inspired by the SAVAR data generation paper and code "A spatiotemporal stochastic climate model for
benchmarking causal discovery methods for teleconnections", Tibau et al.

2022 The main difference with the provided code is the torch/GPU implementation which considerably speeds up the data
generation process
"""

import itertools as it
from copy import deepcopy
from math import pi, sin, sqrt
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from tqdm.auto import tqdm


def dict_to_matrix(links_coeffs, default=0):
    """
    Maps to the coefficient matrix.

    Without time :param links_coeffs: :param default: :return: a matrix coefficient of [j, i, \tau-1] where a link is i
    -> j at \tau
    """
    tau_max = max(abs(lag) for (_, lag), _ in it.chain.from_iterable(links_coeffs.values()))

    n_vars = len(links_coeffs)

    graph = np.ones((n_vars, n_vars, tau_max), dtype=float)
    graph *= default

    for j, values in links_coeffs.items():
        for (i, tau), coeff in values:
            graph[j, i, abs(tau) - 1] = coeff

    return graph


class SAVAR:
    """Main class containing SAVAR model."""

    __slots__ = [
        "links_coeffs",
        "n_vars",
        "time_length",
        "transient",
        "spatial_resolution",
        "tau_max",
        "mode_weights",
        "noise_weights",
        "noise_cov",
        "noise_strength",
        "noise_variance",
        "latent_noise_cov",
        "fast_noise_cov",
        "forcing_dict",
        "season_dict",
        "data_field",
        "noise_data_field",
        "seasonal_data_field",
        "forcing_data_field",
        "linearity",
        "poly_degrees",
        "verbose",
        "model_seed",
        "nnar_model",
    ]

    def __init__(
        self,
        links_coeffs: dict,
        time_length: int,
        mode_weights: np.ndarray,
        transient: int = 200,
        noise_weights: np.ndarray = None,
        noise_strength: float = 1,
        noise_variance: float = 1,
        noise_cov: np.ndarray = None,
        latent_noise_cov: np.ndarray = None,
        fast_cov: np.ndarray = None,
        forcing_dict: dict = None,
        linearity: str = "linear",
        poly_degrees: List[int] = [2],
        season_dict: dict = None,
        data_field: np.ndarray = None,
        noise_data_field: np.ndarray = None,
        seasonal_data_field: np.ndarray = None,
        forcing_data_field: np.ndarray = None,
        verbose: bool = False,
        model_seed: int = None,
    ):

        self.links_coeffs = links_coeffs
        self.time_length = time_length
        self.transient = transient
        self.noise_strength = noise_strength
        self.noise_variance = noise_variance  # TODO: NOT USED.
        self.noise_cov = noise_cov

        self.latent_noise_cov = latent_noise_cov  # D_x
        self.fast_noise_cov = fast_cov  # D_y

        self.mode_weights = mode_weights
        self.noise_weights = noise_weights

        self.forcing_dict = forcing_dict
        self.season_dict = season_dict
        self.linearity = linearity
        self.poly_degrees = poly_degrees

        self.data_field = data_field

        self.verbose = verbose
        self.model_seed = model_seed

        # Computed attributes
        print("Creating attributes")
        self.n_vars = len(links_coeffs)
        self.tau_max = max(abs(lag) for (_, lag), _ in it.chain.from_iterable(self.links_coeffs.values()))
        self.spatial_resolution = deepcopy(self.mode_weights.reshape(self.n_vars, -1).shape[1])
        print("spatial-resolution done")

        if self.noise_weights is None:
            self.noise_weights = deepcopy(self.mode_weights)
        if self.latent_noise_cov is None:
            self.latent_noise_cov = np.eye(self.n_vars)
        if self.fast_noise_cov is None:
            self.fast_noise_cov = np.zeros((self.spatial_resolution, self.spatial_resolution))
        print("copies done")

        # Empty attributes
        self.noise_data_field = noise_data_field
        self.seasonal_data_field = seasonal_data_field
        self.forcing_data_field = forcing_data_field

        if np.random is not None:
            np.random.seed(model_seed)

    def generate_data(self, train_nnar=True) -> None:
        """Generates the data of savar :return:"""
        # Prepare the datafield
        if self.data_field is None:
            if self.verbose:
                print("Creating empty data field")
            # Compute the field
            self.data_field = np.zeros((self.spatial_resolution, self.time_length + self.transient))

        # Add noise first
        if self.noise_data_field is None:
            if self.verbose:
                print("Creating noise data field")
            self._add_noise_field()
        else:
            self.data_field += self.noise_data_field

        # Add seasonality
        if self.season_dict is not None:
            if self.verbose:
                print("Adding seasonality forcing")
            self._add_seasonality_forcing()
        else:
            print("No seasonality")

        # Add external forcing
        if self.forcing_dict is not None:
            if self.verbose:
                print("Adding external forcing")
            initial_data = self.data_field.copy()
            self._add_external_forcing()
            diff = self.data_field - initial_data
            print(f"Max change in data field: {diff.max()}")
            print(f"Mean change in data field: {diff.mean()}")
            print(f"Sample values after forcing applied:\n{diff[:, :5]}")
        else:
            print("No forcing")

            # Compute the data
        if self.linearity == "linear":
            if self.verbose:
                print("Creating linear data")
            self._create_linear()
        elif self.linearity == "polynomial":
            if self.verbose:
                print("Creating polynomial data")
            self._create_polynomial()
        else:
            if self.verbose:
                print("Creating nonlinear data")
            if train_nnar:
                print("Training NNAR model before data generation...")
                self.train_nnar(num_epochs=50, learning_rate=0.001, batch_size=32)
            self._create_nonlinear()

    def generate_cov_noise_matrix(self) -> np.ndarray:
        """
        W in NxL data_field L times T.

        :return:
        """

        W = deepcopy(self.noise_weights).reshape(self.n_vars, -1)
        print(f"noise_weights copied, {W.shape}")
        W_plus = np.linalg.pinv(W)
        print("noise_weights inverted")
        # Can we speed this up? since they are all np.eye
        cov = self.noise_strength * W_plus @ W_plus.transpose()  # + self.fast_noise_cov
        print("cov created inverted")

        return cov

    def _add_noise_field(self):

        if self.noise_cov is None:
            print("Generate covariance matrix")
            self.noise_cov = self.generate_cov_noise_matrix()
            self.noise_cov += 1e-6 * np.eye(self.noise_cov.shape[0])

        # Generate noise from cov
        print("Generate noise_data_field multivariate random")
        mean_torch = torch.Tensor(np.zeros(self.spatial_resolution)).to(device="cuda")
        cov = torch.Tensor(self.noise_cov).to(device="cuda")
        distrib = MultivariateNormal(loc=mean_torch, covariance_matrix=cov)  # . to(device="cuda")
        noise_data_field = distrib.sample(sample_shape=torch.Size([self.time_length + self.transient]))
        self.noise_data_field = noise_data_field.detach().cpu().numpy().transpose()

        # self.noise_data_field = np.random.multivariate_normal(mean=np.zeros(self.spatial_resolution), cov=self.noise_cov,
        #                                                       size=self.time_length + self.transient).transpose()

        self.data_field += self.noise_data_field

    def _add_seasonality_forcing(self):

        # A*sin((2pi/lambda)*x) A = amplitude, lambda = period
        amplitude = self.season_dict["amplitude"]
        period = self.season_dict["period"]
        season_weight = self.season_dict.get("season_weight", None)

        seasonal_trend = np.asarray(
            [amplitude * sin((2 * pi / period) * x) for x in range(self.time_length + self.transient)]
        )

        seasonal_data_field = np.ones_like(self.data_field)
        seasonal_data_field *= seasonal_trend.reshape(1, -1)

        # Apply seasonal weights
        if season_weight is not None:
            season_weight = season_weight.sum(axis=0).reshape(self.spatial_resolution)  # vector dim L
            seasonal_data_field *= season_weight[:, None]  # L times T

        self.seasonal_data_field = seasonal_data_field

        # Add it to the data field.
        self.data_field += seasonal_data_field

    def _add_external_forcing(self):
        """
        Adds external forcing to the data field using PyTorch tensors for GPU acceleration.

        Allows for both linear and nonlinear ramps.
        """
        if self.forcing_dict is None:
            raise TypeError("Forcing dict is empty")

        w_f = deepcopy(self.forcing_dict.get("w_f"))
        f_1 = float(self.forcing_dict.get("f_1", 0))
        f_2 = float(self.forcing_dict.get("f_2", 0))
        f_time_1 = self.forcing_dict.get("f_time_1", 0)
        f_time_2 = self.forcing_dict.get("f_time_2", self.time_length)
        ramp_type = self.forcing_dict.get("ramp_type", "linear")  # Default to linear

        if w_f is None:
            w_f = deepcopy(self.mode_weights)
            w_f = (w_f != 0).astype(int)  # Convert non-zero elements to 1

        print(self.mode_weights.shape)
        # w_f = w_f / (w_f.max() + 1e-8)  # Normalize to range [0,1]

        # Merge last two dims first => shape (d_z, lat*lon)
        temp = w_f.reshape(w_f.shape[0], w_f.shape[1] * w_f.shape[2])
        # sum over dim=0 => shape (lat*lon,)
        w_f_sum = torch.tensor(temp.sum(axis=0), dtype=torch.float32, device="cuda")
        f_time_1 += self.transient
        f_time_2 += self.transient
        time_length = self.time_length + self.transient

        # Generate the forcing trend using torch tensors
        if ramp_type == "linear":
            ramp = torch.linspace(f_1, f_2, f_time_2 - f_time_1, dtype=torch.float32, device="cuda")
        elif ramp_type == "quadratic":
            t = torch.linspace(0, 1, f_time_2 - f_time_1, dtype=torch.float32, device="cuda")
            ramp = f_1 + (f_2 - f_1) * t**2
        elif ramp_type == "exponential":
            t = torch.linspace(0, 1, f_time_2 - f_time_1, dtype=torch.float32, device="cuda")
            ramp = f_1 + (f_2 - f_1) * (torch.exp(t) - 1) / (torch.exp(torch.tensor(1.0)) - 1)
        elif ramp_type == "sigmoid":
            t = torch.linspace(-6, 6, f_time_2 - f_time_1, dtype=torch.float32, device="cuda")
            ramp = f_1 + (f_2 - f_1) * (1 / (1 + torch.exp(-t)))
        elif ramp_type == "sinusoidal":
            t = torch.linspace(0, pi, f_time_2 - f_time_1, dtype=torch.float32, device="cuda")
            ramp = f_1 + (f_2 - f_1) * (0.5 * (1 - torch.cos(t)))
        else:
            raise ValueError(
                "Unsupported ramp type. Choose from 'linear', 'quadratic', 'exponential', 'sigmoid', or 'sinusoidal'."
            )

        # Generate the forcing trend using torch tensors
        trend = torch.cat(
            [
                torch.full((f_time_1,), f_1, dtype=torch.float32, device="cuda"),
                ramp,
                torch.full((time_length - f_time_2,), f_2, dtype=torch.float32, device="cuda"),
            ]
        ).reshape(1, time_length)

        if w_f_sum.dim() == 2:
            w_f_sum = w_f_sum.sum(dim=0, keepdim=True)  # Sum across the correct dimension

        # Compute the forcing field on GPU
        forcing_field = (w_f_sum.reshape(1, -1) * trend.T).T
        self.forcing_data_field = forcing_field.cpu().numpy()

        print(f"Using {ramp_type} ramp: f_1={f_1}, f_2={f_2}, f_time_1={f_time_1}, f_time_2={f_time_2}")

        print(f"Forcing data field mean: {self.forcing_data_field.mean()}")

        print(f"Before addition - Data field mean: {self.data_field.mean()}")

        # data_field_before = self.data_field.copy()

        self.data_field += self.forcing_data_field

        # data_field_after = self.data_field

        print(f"After addition - Data field mean: {self.data_field.mean()}")

        # # Convert tensors to numpy for plotting if necessary
        # if isinstance(w_f_sum, torch.Tensor):
        #     w_f_sum = w_f_sum.cpu().numpy()
        # if isinstance(forcing_field, torch.Tensor):
        #     forcing_field = forcing_field.cpu().numpy()
        # if isinstance(data_field_before, torch.Tensor):
        #     data_field_before = data_field_before.cpu().numpy()
        # if isinstance(data_field_after, torch.Tensor):
        #     data_field_after = data_field_after.cpu().numpy()

        # # Compute mean values over spatial dimensions
        # mean_forcing = forcing_field.mean(axis=0)
        # mean_data_before = data_field_before.mean(axis=0)
        # mean_data_after = data_field_after.mean(axis=0)

        # # Plot 1: Mean Forcing over Time
        # plt.figure(figsize=(10, 4))
        # plt.plot(range(time_length), mean_forcing, label="Mean Forcing", color="blue")
        # plt.axvline(x=f_time_1, linestyle="--", color="gray", label="Start Forcing")
        # plt.axvline(x=f_time_2, linestyle="--", color="gray", label="End Forcing")
        # plt.xlabel("Time Steps")
        # plt.ylabel("Forcing Intensity")
        # plt.title("Evolution of External Forcing Over Time")
        # plt.legend()
        # plt.grid()
        # plt.savefig(f"mean_forcing_over_time_{f_1}_{f_2}_{ramp_type}.png")  # Save to a file
        # plt.close()

        # # Plot 2: Mean Data Before and After Forcing
        # plt.figure(figsize=(10, 4))
        # plt.plot(range(time_length), mean_data_before, label="Data Before Forcing", color="red", linestyle="dashed")
        # plt.plot(range(time_length), mean_data_after, label="Data After Forcing", color="green")
        # plt.axvline(x=f_time_1, linestyle="--", color="gray", label="Start Forcing")
        # plt.axvline(x=f_time_2, linestyle="--", color="gray", label="End Forcing")
        # plt.xlabel("Time Steps")
        # plt.ylabel("Mean Data Value")
        # plt.title("Effect of Forcing on Data Field")
        # plt.legend()
        # plt.grid()
        # plt.savefig(f"mean_data_before_after_forcing_{f_1}_{f_2}_{ramp_type}.png")  # Save to a file
        # plt.close()

    def _create_linear(self):
        """Weights N \times L data_field L \times T."""
        weights = deepcopy(self.mode_weights.reshape(self.n_vars, -1))
        # weights_inv = np.linalg.pinv(weights)
        weights_inv = torch.Tensor(np.linalg.pinv(weights)).to(device="cuda")
        weights = torch.Tensor(weights).to(device="cuda")
        time_len = deepcopy(self.time_length)
        time_len += self.transient
        tau_max = self.tau_max

        # phi = dict_to_matrix(self.links_coeffs)
        phi = torch.Tensor(dict_to_matrix(self.links_coeffs)).to(device="cuda")
        # data_field = deepcopy(self.data_field)
        data_field = torch.Tensor(self.data_field).to(device="cuda")

        print("create_linear")
        for t in tqdm(range(tau_max, time_len)):
            for i in range(tau_max):
                data_field[..., t : t + 1] += weights_inv @ phi[..., i] @ weights @ data_field[..., t - 1 - i : t - i]
                # data_field[..., t:t + 1] += torch.matmul(torch.matmul(torch.matmul(weights_inv, phi[..., i]), weights), data_field[..., t - 1 - i:t - i])

        self.data_field = data_field[..., self.transient :].detach().cpu().numpy()

    def _create_intervened_nextstep(self, input_data, intervened_mode=None, intervention_value=None, intervened_t=None):
        """
        Not tested yet!!! see causal_graph_comparison for a proper function.

        input_data are the tau timesteps that get intervened on
        at mode intervened_mode, with value +intervention_value, at timestep intervened_t

        input_data is here of shape `self.spatial_resolution * self.time_length`.
        This is to keep the savar structure similar to the one of `self.data_field`
        """

        weights = deepcopy(self.mode_weights.reshape(self.n_vars, -1))
        # weights_inv = np.linalg.pinv(weights)
        weights_inv = torch.Tensor(np.linalg.pinv(weights)).to(device="cuda")
        weights = torch.Tensor(weights).to(device="cuda")
        tau = input_data.shape[1]

        # phi = dict_to_matrix(self.links_coeffs)
        phi = torch.Tensor(dict_to_matrix(self.links_coeffs)).to(device="cuda")
        # data_field = deepcopy(self.data_field)
        next_step = torch.zeros(self.spatial_resolution).to(device="cuda")

        change_indices = []

        quadrant_row = intervened_mode // int(sqrt(self.n_vars))
        quadrant_col = intervened_mode % int(sqrt(self.n_vars))

        start_row = quadrant_row * self.spatial_resolution
        start_col = quadrant_col * self.spatial_resolution

        for i in range(self.spatial_resolution):
            for j in range(self.spatial_resolution):
                change_idx = (start_row + i) * int(sqrt(self.n_vars) * self.spatial_resolution) + (start_col + j)
                change_indices.append(change_idx)

        # perform intervention
        input_data[change_indices, intervened_t] += intervention_value

        for i in range(tau):
            next_step += weights_inv @ phi[..., i] @ weights @ input_data[:, -i]

        return next_step

    def train_nnar(self, num_epochs=50, learning_rate=0.001, batch_size=32):
        """
        Method for training a very simple single-layer neural network with sigmoid activation (one neuron).

        We train it here on pairs (past_values, future_value), but this can be adapted as needed.
        """

        # A trivial net:  data_in -> [Linear] -> [Sigmoid] -> data_out
        self.nnar_model = nn.Sequential(nn.Linear(self.spatial_resolution, self.spatial_resolution), nn.Sigmoid()).to(
            "cuda"
        )

        optimizer = torch.optim.Adam(self.nnar_model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        # Create a training dataset from self.data_field: each sample is (X_t, X_{t+1}),
        # (we might later incorporate more lags)

        # collect input-output pairs:
        X = torch.from_numpy(self.data_field[:, :-1].T).float().to("cuda")
        Y = torch.from_numpy(self.data_field[:, 1:].T).float().to("cuda")
        dataset_size = X.shape[0]

        # Simple mini-batch loop
        for epoch in range(num_epochs):
            perm = torch.randperm(dataset_size, device="cuda")
            batch_losses = []

            for i in range(0, dataset_size, batch_size):
                idx = perm[i : i + batch_size]
                x_batch = X[idx]
                y_batch = Y[idx]

                # forward pass
                pred = self.nnar_model(x_batch)
                loss = loss_fn(pred, y_batch)
                batch_losses.append(loss.item())

                # backward + update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {sum(batch_losses)/len(batch_losses):.6f}")

        print("Training of single-layer NNAR model completed.")

    def _create_nonlinear(self):
        """
        Generates nonlinear data by applying a (trained or simple) nonlinearity at each time step. This method uses the
        same logic as _create_linear to step forward in time and adds the nonlinearity (sigmoid) before adding to
        data_field.

        If train_nnar=True was set, we assume self.nnar_model was trained in generate_data().
        Otherwise, we can do a direct inline "torch.sigmoid(...)" approach.
        Can be increased in complexity if needed
        """

        weights = torch.Tensor(np.linalg.pinv(self.mode_weights.reshape(self.n_vars, -1))).to("cuda")
        phi = torch.Tensor(dict_to_matrix(self.links_coeffs)).to("cuda")
        mode_weights_tensor = torch.Tensor(self.mode_weights.reshape(self.n_vars, -1)).to("cuda")
        data_field = torch.Tensor(self.data_field).to("cuda")

        time_len = self.time_length + self.transient
        tau_max = self.tau_max

        print("create_nonlinear (single-layer net + sigmoid)")

        for t in tqdm(range(tau_max, time_len)):
            # Sum up influences from each lag
            nonlinear_contrib = 0.0
            for i in range(tau_max):
                # get linear combination as in _create_linear
                lincombo = weights @ phi[..., i] @ mode_weights_tensor @ data_field[..., (t - 1 - i) : (t - i)]
                # Apply a sigmoid (or feed it through the small neural net if you want more complexity)
                lincombo_nl = torch.sigmoid(lincombo)
                # accumulate
                nonlinear_contrib += lincombo_nl.squeeze(-1)

            # Add the (nonlinear) effect to the data field at time t
            data_field[:, t] += nonlinear_contrib

        self.data_field = data_field[:, self.transient :].detach().cpu().numpy()

    def _create_polynomial(self):
        """Example polynomial autoregression, e.g. x^2 for poly_degree=2."""
        w_np = np.linalg.pinv(self.mode_weights.reshape(self.n_vars, -1))
        phi_np = dict_to_matrix(self.links_coeffs)

        w_torch = torch.Tensor(w_np).to("cuda")
        phi_torch = torch.Tensor(phi_np).to("cuda")
        mw_torch = torch.Tensor(self.mode_weights.reshape(self.n_vars, -1)).to("cuda")
        data_field = torch.Tensor(self.data_field).to("cuda")

        time_len = self.time_length + self.transient
        tau_max = self.tau_max

        print(f"create_polynomial with degrees={self.poly_degrees}")

        for t in tqdm(range(tau_max, time_len)):
            # For each time step, sum over the contributions of all lags
            for i in range(tau_max):
                lincombo = w_torch @ phi_torch[..., i] @ mw_torch @ data_field[..., (t - 1 - i) : (t - i)]

                # For each requested polynomial degree, add its effect
                poly_sum = 0.0
                for deg in self.poly_degrees:
                    poly_sum += lincombo**deg

                data_field[:, t] += poly_sum.squeeze(-1)

        self.data_field = data_field[:, self.transient :].detach().cpu().numpy()
