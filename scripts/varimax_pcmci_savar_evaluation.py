"""Baseline evaluation of causal discovery using PCMCI with Varimax-PCA preprocessing.

This script provides a baseline comparison for the ClimatEM causal discovery
model.  It applies Varimax-rotated PCA to extract latent modes from synthetic
SAVAR data and then runs PCMCI (Runge et al., 2019) to infer causal links.
The inferred adjacency matrix is compared against the known ground-truth
structure using precision, recall, F1, and SHD.

The script also loads results from the ClimatEM model (CDSD) for side-by-side
comparison.
"""
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import asarray, diag, dot, eye, sum
from numpy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

parcorr = ParCorr(significance="analytic")


def extract_adjacency_matrix(links_coeffs, N, tau):
    """Extract ground-truth adjacency matrices for each time lag.

    Parameters
    ----------
    links_coeffs : dict
        Dictionary mapping each latent variable index to a list of
        ``((target_var, lag), coefficient)`` tuples describing causal links.
    N : int
        Number of latent variables.
    tau : int
        Maximum time lag to consider.

    Returns
    -------
    adj_matrices : np.ndarray
        Binary adjacency matrices with shape ``(tau, N, N)`` where entry
        ``[t, i, j]`` is 1 if variable *j* causes variable *i* at lag *t+1*.
    """
    # Initialize a 3D array to store adjacency matrices for each time lag (tau x N x N)
    adj_matrices = np.zeros((tau, N, N))

    # Loop through each component and its links
    for key, values in links_coeffs.items():
        for link, coeff in values:
            target_var, lag = link
            time_lag = -lag  # Convert the negative lag to a positive index
            # Only consider lags that are within the specified time window (tau)
            if time_lag <= tau:
                if abs(coeff) > 0.01:
                    adj_matrices[time_lag - 1, key, target_var] = (
                        1  # Fill the adjacency matrix at the appropriate time lag
                    )
                else:
                    adj_matrices[time_lag - 1, key, target_var] = 0

    return adj_matrices


def evaluate_adjacency_matrix(A_inferred, A_ground_truth, threshold):
    """Evaluate precision, recall, F1-score, and SHD between two adjacency matrices.

    Parameters
    ----------
    A_inferred : np.ndarray
        Inferred adjacency matrix (may be real-valued).
    A_ground_truth : np.ndarray
        Ground-truth adjacency matrix (may be real-valued).
    threshold : float
        Threshold for binarising both matrices before comparison.

    Returns
    -------
    precision : float
    recall : float
    f1 : float
    shd : int
        Structural Hamming Distance (false positives + false negatives).
    """
    # Binarize the matrices before comparison
    A_inferred_bin = binarize_matrix(A_inferred, threshold)
    A_ground_truth_bin = binarize_matrix(A_ground_truth, threshold)

    # Flatten the matrices to make comparison easier
    A_inferred_flat = A_inferred_bin.flatten()
    A_ground_truth_flat = A_ground_truth_bin.flatten()

    # Binary classification metrics
    precision = float(precision_score(A_ground_truth_flat, A_inferred_flat))
    recall = float(recall_score(A_ground_truth_flat, A_inferred_flat))
    f1 = float(f1_score(A_ground_truth_flat, A_inferred_flat))

    # Structural Hamming Distance (SHD)
    false_positives = int(np.sum((A_inferred_bin == 1) & (A_ground_truth_bin == 0)))
    false_negatives = int(np.sum((A_inferred_bin == 0) & (A_ground_truth_bin == 1)))
    shd = false_positives + false_negatives

    return precision, recall, f1, shd


def binarize_matrix(A, threshold=0.5):
    """Binarise an adjacency matrix by applying a threshold.

    Parameters
    ----------
    A : np.ndarray
        Real-valued adjacency matrix.
    threshold : float
        Values strictly above this are set to 1; all others to 0.

    Returns
    -------
    np.ndarray
        Integer array with values in {0, 1}.
    """
    return (A > threshold).astype(int)


def varimax(Phi, gamma=1, q=20, tol=1e-6):
    """Compute the Varimax rotation of a factor loading matrix.

    Implements the standard Varimax criterion (Kaiser, 1958) via SVD-based
    iterative optimisation.

    Parameters
    ----------
    Phi : np.ndarray
        Factor loading matrix of shape ``(p, k)`` where *p* is the number of
        observed variables and *k* is the number of factors.
    gamma : float, optional
        Rotation parameter. ``gamma=1`` gives standard Varimax.
    q : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance (ratio of successive singular-value sums).

    Returns
    -------
    rotated : np.ndarray
        Rotated loading matrix ``Phi @ R``.
    R : np.ndarray
        Orthogonal rotation matrix.
    """
    p, k = Phi.shape
    R = eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u, s, vh = svd(dot(Phi.T, asarray(Lambda) ** 3 - (gamma / p) * dot(Lambda, diag(diag(dot(Lambda.T, Lambda))))))
        R = dot(u, vh)
        d = sum(s)
        if d / d_old < tol:
            break
    return dot(Phi, R), R


if __name__ == "__main__":

    # load your existing JSON config
    config_path = Path("configs/single_param_file_savar.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    exp   = cfg["exp_params"]
    data  = cfg["data_params"]
    savar = cfg["savar_params"]

    # pull out exactly the bits you used to hard-code
    tau        = exp["tau"]
    n_modes    = exp["d_z"]               # latent dim = number of modes
    comp_size  = savar["comp_size"]
    time_len   = savar["time_len"]
    is_forced  = savar["is_forced"]
    seasonality = savar["seasonality"]
    overlap    = savar["overlap"]
    difficulty = savar["difficulty"]
    lat = lon = int(np.sqrt(n_modes)) * comp_size
    noise_val = savar["noise_val"]

    var_names = []
    for k in range(n_modes):
        var_names.append(rf"$X^{k}$")

    # NOTE: Adjust these paths to match your local environment.
    savar_folder = "/home/ka/ka_iti/ka_qa4548/my_projects/climatem/workspace/pfs7wor9/ka_qa4548-data/SAVAR_DATA_TEST"
    # Load gt mode weights
    savar_fname = f"modes_{n_modes}_tl_{time_len}_isforced_{is_forced}_difficulty_{difficulty}_noisestrength_{noise_val}_seasonality_{seasonality}_overlap_{overlap}"
    # Get the gt mode weights
    modes_gt = np.load(savar_folder + f"/{savar_fname}_mode_weights.npy")

    params_file = savar_folder + f"/{savar_fname}_parameters.npy"
    params = np.load(params_file, allow_pickle=True).item()
    links_coeffs = params["links_coeffs"]

    adj_gt = extract_adjacency_matrix(links_coeffs, n_modes, tau)
    n_gt_connections = (np.array(adj_gt) > 0).sum()

    # NOTE: Adjust these paths to match your local environment.
    cdsd_adj_inferred_path = Path("/home/ka/ka_iti/ka_qa4548/my_projects/climatem/workspace/pfs7wor9/ka_qa4548-results/SAVAR_DATA_TEST/var_savar_scenarios_piControl_nonlinear_False_tau_5_z_9_lr_0.001_bs_256_spreg_0_ormuinit_100000.0_spmuinit_0.1_spthres_0.05_fixed_False_num_ensembles_1_instantaneous_False_crpscoef_1_spcoef_0_tempspcoef_0_overlap_0.3_forcing_True/plots/graphs.npy")
    cdsd_modes_inferred_path = Path("/home/ka/ka_iti/ka_qa4548/my_projects/climatem/workspace/pfs7wor9/ka_qa4548-results/SAVAR_DATA_TEST/var_savar_scenarios_piControl_nonlinear_False_tau_5_z_9_lr_0.001_bs_256_spreg_0_ormuinit_100000.0_spmuinit_0.1_spthres_0.05_fixed_False_num_ensembles_1_instantaneous_False_crpscoef_1_spcoef_0_tempspcoef_0_overlap_0.3_forcing_True/plots/w_decoder.npy")
    modes_inferred = np.load(cdsd_modes_inferred_path)
    adj_w = np.load(cdsd_adj_inferred_path)

    # Find the permutation
    modes_inferred = modes_inferred.reshape((lat, lon, modes_inferred.shape[-1])).transpose((2, 0, 1))

    # Get the flat index of the maximum for each mode
    idx_gt_flat = np.argmax(modes_gt.reshape(modes_gt.shape[0], -1), axis=1)          # shape: (n_modes,)
    idx_inferred_flat = np.argmax(modes_inferred.reshape(modes_inferred.shape[0], -1), axis=1)  # shape: (n_modes,)

    # Convert flat indices to 2D coordinates (row, col)
    idx_gt = np.array([np.unravel_index(i, (lat, lon)) for i in idx_gt_flat])         # shape: (n_modes, 2)
    idx_inferred = np.array([np.unravel_index(i, (lat, lon)) for i in idx_inferred_flat])  # shape: (n_modes, 2)

    # Compute error matrix using squared Euclidean distance between indices which yields an (n_modes x n_modes) matrix
    permutation_list = ((idx_gt[:, None, :] - idx_inferred[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    logger.info("permutation_list: %s", permutation_list)

    # Permute
    for k in range(tau):
        adj_w[k] = adj_w[k][np.ix_(permutation_list, permutation_list)]

    logger.info("PERMUTED THE MATRICES")

    precision, recall, f1, shd = evaluate_adjacency_matrix(adj_w, adj_gt, 0.9)

    logger.info("difficulty %s results:", difficulty)
    logger.info("Precision: %s, Recall: %s, F1 Score: %s, SHD: %s", precision, recall, f1, shd)
