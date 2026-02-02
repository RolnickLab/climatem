import itertools as it
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib import cm
from tigramite.data_processing import smooth


def check_stability(graph: Union[np.ndarray, dict], lag_first_axis: bool = False, verbose: bool = False):
    """
    Raises an AssertionError if the input graph corresponds to a non-stationary process.

    Parameters
    ----------
    graph: array
        Lagged connectivity matrices. Shape is (n_nodes, n_nodes, max_delay+1)
    lag_first_axis: bool
        Indicates if the lag is in the first axis or in the last
    verbose: bool
        Level of output information
    """

    if isinstance(graph, dict):
        graph = create_graph(graph, return_lag=False)

    # Adapt the Varmodel return to the desired format (lag, N, N) -> (N, N, lag)
    if lag_first_axis:
        graph = np.moveaxis(graph, 0, 2)

    if verbose:
        print("The shape of the graph is", graph.shape)

    # Get the shape from the input graph
    n_nodes, _, period = graph.shape
    # Set the top section as the horizontally stacked matrix of
    # shape (n_nodes, n_nodes * period)
    stability_matrix = scipy.sparse.hstack([scipy.sparse.lil_matrix(graph[:, :, t_slice]) for t_slice in range(period)])
    # Extend an identity matrix of shape
    # (n_nodes * (period - 1), n_nodes * (period - 1)) to shape
    # (n_nodes * (period - 1), n_nodes * period) and stack the top section on
    # top to make the stability matrix of shape
    # (n_nodes * period, n_nodes * period)
    stability_matrix = scipy.sparse.vstack(
        [stability_matrix, scipy.sparse.eye(n_nodes * (period - 1), n_nodes * period)]
    )
    # Check the number of dimensions to see if we can afford to use a dense
    # matrix
    n_eigs = stability_matrix.shape[0]
    if n_eigs <= 25:
        # If it is relatively low in dimensionality, use a dense array
        stability_matrix = stability_matrix.todense()
        eigen_values, _ = scipy.linalg.eig(stability_matrix)
    else:
        # If it is a large dimensionality, convert to a compressed row sorted
        # matrix, as it may be easier for the linear algebra package
        stability_matrix = stability_matrix.tocsr()
        # Get the eigen values of the stability matrix
        eigen_values = scipy.sparse.linalg.eigs(stability_matrix, k=(n_eigs - 2), return_eigenvectors=False)
    # Ensure they all have less than one magnitude

    assert np.all(
        np.abs(eigen_values) < 1.0
    ), "Values given by time lagged connectivity matrix corresponds to a  non-stationary process!"

    if verbose:
        print("The coefficients correspond to an stationary process")


def create_random_mode(
    size: tuple,
    mu: tuple = (0, 0),
    var: tuple = (0.5, 0.5),
    position: tuple = (3, 3, 3, 3),
    plot: bool = False,
    Sigma: np.ndarray = None,
    random: bool = True,
) -> np.ndarray:
    """
    Creates a positive-semidefinite matrix to be used as a covariance matrix of two var
    Then use that covariance to compute a pdf of a bivariate gaussian distribution which
    is used as mode weight. It is random but enfoced to be spred.
    Inspired in:  https://scipython.com/blog/visualizing-the-bivariate-gaussian-distribution/
    and https://stackoverflow.com/questions/619335/a-simple-algorithm-for-generating-positive-semidefinite-matrices

    :param random: Does not create a random, insted uses a ind cov matrix
    :param size
    :param mu tuple with the x and y mean
    :param var used to enforce spread modes. (0, 0) = totally random
    :param position: tuple of the position of the mean
    :param plot:
    """

    # Unpack variables
    size_x, size_y = size
    x_a, x_b, y_a, y_b = position
    mu_x, mu_y = mu
    var_x, var_y = var

    # In case of non invertible
    if Sigma is not None:
        Sigma_o = Sigma.copy()
    else:
        Sigma_o = Sigma

    # Compute the position of the mean
    X = np.linspace(-x_a, x_b, size_x)
    Y = np.linspace(-y_a, y_b, size_y)
    X, Y = np.meshgrid(X, Y)

    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    # Mean vector
    mu = np.array([mu_x, mu_y])

    # Compute almost-random covariance matrix
    if random:
        Sigma = np.random.rand(2, 2)
        Sigma = np.dot(Sigma, Sigma.transpose())  # Make it invertible
        Sigma += +np.array([[var_x, 0], [0, var_y]])
    else:
        if Sigma is None:
            Sigma = np.asarray([[0.5, 0], [0, 0.5]])

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2 * np.pi) ** n * Sigma_det)

    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum("...k,kl,...l->...", pos - mu, Sigma_inv, pos - mu)

    # The actual weight
    Z = np.exp(-fac / 2) / N

    if not np.isfinite(Z).all() or (Z > 0.5).any():
        Z = create_random_mode(size=size, mu=mu, var=var, position=position, plot=False, Sigma=Sigma_o, random=random)

    if plot:
        # Create a surface plot and projected filled contour plot under it.
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)

        # Adjust the limits, ticks and view angle
        ax.set_zlim(-0.15, 0.2)
        ax.set_zticks(np.linspace(0, 0.2, 5))
        ax.view_init(27, -21)
        plt.show()
        plt.close()

    return Z


def create_non_stationarity(
    N_var: int,
    t_sample: int,
    tau: float = 0.5,
    cov_mat: np.ndarray = None,
    sigma: float = 1,
    smoothing_window: int = None,
) -> np.ndarray:
    """
    Returns a (t_sample, N_var) array representing an oscilatory trend created from a N_var-dimensional Ornstein-
    Uhlenbeck process of covariance matrix cov_mat, standard dev : sigma and mean reversal parameter = tau. The
    Ornstein-Uhlenbeck process is smoothed with a Gaussian moving average of windows 2*smoothing_window The mean of the
    O-U process is set to zero inside the function.

    Parameters
    ----------
    cov_mat : array. If it is None, then the identity matrix is used.
    Covariance matrix of the Brownian motion generating the O-U process. Shape is (N_var, N_var).
    N_var: int
        Number of dimension of the O-U process to generate.
    t_sample : int
        Sample size.
    sigma: float (default is 0.3)
        Standard dev of the O-U process.
    tau : float (default is 0.05)
        Mean reversal parameter (how fast the process goes back to its mean).
    smoothing_window : int (default is N_var/10)
        Size of the smoothing windows.

    Returns
    -------
    X_smooth : array. Shape is (t_sample, N_var).
        Smoothed O-U process.
    """
    if cov_mat is None:
        # If it is None, then we use identity
        cov_mat = np.identity(N_var, dtype=float)

    if smoothing_window is None:
        smoothing_window = int(t_sample / 10)  # default value of smoothing windows if not specified

    mu = np.zeros(N_var)  # Mean of the O-U is zero
    dt = 0.001

    sigma_bis = sigma * np.sqrt(2.0 / tau)
    sqrtdt = np.sqrt(dt)

    # Initial value of the process
    X = np.zeros((t_sample, N_var))
    # random initial value of the O-U process around its mean
    X[0, :] = np.random.multivariate_normal(mu, sigma * sigma * cov_mat)

    # generation of the N-dim O-H process from its ODS
    for i in range(t_sample - 1):
        X[i + 1, :] = (
            X[i, :]
            + dt * (-(X[i, :] - mu) / tau)
            + np.random.multivariate_normal(mu, (sigma_bis * sqrtdt) ** 2 * cov_mat)
        )

    # Smoothing using tigramite smoothing function
    try:
        X_smooth = smooth(X, smoothing_window)
    except ValueError:
        raise ValueError(f"Smoothing windows {str(smoothing_window)} is invalid")
    return X_smooth


def create_graph(links_coeffs, return_lag=True):
    """
    :param links_coeffs:
    :param return_lag: if True, return max lag, otherwise returns only np.ndarray
    From the shape of [j, i, tau]
    :return:
    """

    # Define N
    N = len(links_coeffs)
    non_linear = False

    # Detect if it s non-linear link_coeff
    if len(links_coeffs[0][0]) == 3:
        non_linear = True

    # We find the max_lag
    if not non_linear:
        max_lag = max(abs(lag) for (_, lag), _ in it.chain.from_iterable(links_coeffs.values()))
    else:
        max_lag = max(abs(lag) for (_, lag), _, _ in it.chain.from_iterable(links_coeffs.values()))

    # We create an empty graph
    graph = np.zeros((N, N, max_lag + 1))

    # Compute the graph values
    if not non_linear:
        for j, values in links_coeffs.items():
            for (i, tau), coeff in values:
                graph[j, i, abs(tau) - 1] = coeff if tau != 0 else 0
    else:
        for j, values in links_coeffs.items():
            for (i, tau), coeff, _ in values:
                graph[j, i, abs(tau) - 1] = coeff if tau != 0 else 0

    if return_lag:
        return np.asarray(graph), max_lag
    else:
        return np.asarray(graph)


def permute_matrices(
    lat,
    lon,
    modes_inferred,
    modes_gt,
    mat_transition,
    tau,
):
    modes_inferred = modes_inferred.reshape((lat, lon, modes_inferred.shape[-1])).transpose((2, 0, 1))

    idx_gt_flat = np.argmax(modes_gt.reshape(modes_gt.shape[0], -1), axis=1)  # shape: (n_modes,)
    idx_inferred_flat = np.argmax(modes_inferred.reshape(modes_inferred.shape[0], -1), axis=1)  # shape: (n_modes,)

    # Convert flat indices to 2D coordinates (row, col)
    idx_gt = np.array([np.unravel_index(i, (lat, lon)) for i in idx_gt_flat])  # shape: (n_modes, 2)
    idx_inferred = np.array([np.unravel_index(i, (lat, lon)) for i in idx_inferred_flat])  # shape: (n_modes, 2)

    # Compute error matrix using squared Euclidean distance between indices which yields an (n_modes x n_modes) matrix
    permutation_list = ((idx_gt[:, None, :] - idx_inferred[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)

    # Permute
    for k in range(tau):
        mat_transition[k] = mat_transition[k][np.ix_(permutation_list, permutation_list)]

    return mat_transition
