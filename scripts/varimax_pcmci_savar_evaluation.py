import json
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

parcorr = ParCorr(significance="analytic")


def extract_adjacency_matrix(links_coeffs, N, tau):
    """
    Extract the ground truth adjacency matrices for each time lag from the links_coeffs.

    Args:
        links_coeffs (dict): The dictionary of causal links between latent variables.
        N (int): The number of latent variables.
        tau (int): The maximum time lag.

    Returns:
        adj_matrices (np.ndarray): The ground truth adjacency matrices (tau x N x N),
                                where each matrix corresponds to a different time lag.
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
    """Evaluates the precision, recall, F1-score, and Structural Hamming Distance (SHD) between the inferred and ground
    truth adjacency matrices."""
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


def extract_adjacency_matrix(links_coeffs, N, tau):
    """
    Extract the ground truth adjacency matrices for each time lag from the links_coeffs.

    Args:
        links_coeffs (dict): The dictionary of causal links between latent variables.
        N (int): The number of latent variables.
        tau (int): The maximum time lag.

    Returns:
        adj_matrices (np.ndarray): The ground truth adjacency matrices (tau x N x N),
                                where each matrix corresponds to a different time lag.
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


def binarize_matrix(A, threshold=0.5):
    """Binarizes the adjacency matrix by applying a threshold."""
    return (A > threshold).astype(int)


def varimax(Phi, gamma=1, q=20, tol=1e-6):
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

    savar_folder = Path("/home/ka/ka_iti/ka_qa4548/my_projects/climatem/workspace/pfs7wor9/ka_qa4548-data/SAVAR_DATA_TEST")
    # Load gt mode weights
    savar_fname = f"modes_{n_modes}_tl_{time_len}_isforced_{is_forced}_difficulty_{difficulty}_noisestrength_{noise_val}_seasonality_{seasonality}_overlap_{overlap}"
    # Get the gt mode weights
    modes_gt = np.load(savar_folder / f"{savar_fname}_mode_weights.npy")

    #savar_data = np.load(savar_folder / savar_fname)
    params_file = savar_folder / f"{savar_fname}_parameters.npy"
    params = np.load(params_file, allow_pickle=True).item()
    links_coeffs = params["links_coeffs"]

    # modes_gt = np.load(savar_folder / f"{savar_fname[:-4]}_mode_weights.npy")
    # modes_gt -= modes_gt.mean()
    # modes_gt /= modes_gt.std()

    adj_gt = extract_adjacency_matrix(links_coeffs, n_modes, tau)
    n_gt_connections = (np.array(adj_gt) > 0).sum()

    # load CDSD results (already permuted / aligned)
    cdsd_adj_inferred_path = Path("/home/ka/ka_iti/ka_qa4548/my_projects/climatem/workspace/pfs7wor9/ka_qa4548-results/SAVAR_DATA_TEST/var_savar_scenarios_piControl_nonlinear_False_tau_5_z_9_lr_0.001_bs_256_spreg_0_ormuinit_100000.0_spmuinit_0.1_spthres_0.05_fixed_False_num_ensembles_1_instantaneous_False_crpscoef_1_spcoef_0_tempspcoef_0_overlap_0.3_forcing_True/plots/graphs.npy")
    cdsd_modes_inferred_path = Path("/home/ka/ka_iti/ka_qa4548/my_projects/climatem/workspace/pfs7wor9/ka_qa4548-results/SAVAR_DATA_TEST/var_savar_scenarios_piControl_nonlinear_False_tau_5_z_9_lr_0.001_bs_256_spreg_0_ormuinit_100000.0_spmuinit_0.1_spthres_0.05_fixed_False_num_ensembles_1_instantaneous_False_crpscoef_1_spcoef_0_tempspcoef_0_overlap_0.3_forcing_True/plots/w_decoder.npy")
    modes_inferred = np.load(cdsd_modes_inferred_path)
    adj_w = np.load(cdsd_adj_inferred_path)

    ############################

    # # Fit PCA + varimax
    # pca_model = PCA(n_modes).fit(savar_data.T)
    # latent_data = pca_model.transform(savar_data.T)
    # varimaxpcs, varimax_rotation = varimax(latent_data)

    # # To recover which mode is which and permute accordingly when evaluating
    # inverse_varimax = dot(latent_data, np.linalg.pinv(varimax_rotation))
    # reverted_data = pca_model.inverse_transform(inverse_varimax)

    # dataframe = pp.DataFrame(varimaxpcs, datatime={0: np.arange(len(varimaxpcs))}, var_names=var_names)
    # # Run PCMCI
    # pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)

    # results = pcmci.run_pcmci(tau_min=1, tau_max=5, pc_alpha=None, alpha_level=0.001)

    # Permute accordingly before evaluating learned graph.
    # individual_modes = np.zeros((n_modes, time_len, lat, lon))
    # for k in range(n_modes):
    #     latent_data_bis = np.zeros(latent_data.shape)
    #     latent_data_bis[:, k] = latent_data[:, k]
    #     inverse_varimax = dot(latent_data_bis, np.linalg.pinv(varimax_rotation))
    #     reverted_data = pca_model.inverse_transform(inverse_varimax)
    #     individual_modes[k] = reverted_data.reshape((-1, lat, lon))
    # individual_modes = individual_modes.std(1)
    # individual_modes -= individual_modes.mean()
    # individual_modes /= individual_modes.std()

    # permutation_list = ((modes_gt[:, None] - individual_modes[None]) ** 2).sum((2, 3)).argmin(1)

    # # Get adjacency matrix from PCMCI graph
    # graph = results["graph"]
    # graph[
    #     results["val_matrix"]
    #     < np.abs(results["val_matrix"].flatten()[results["val_matrix"].flatten().argsort()[::-1][n_gt_connections - 1]])
    # ] = ""

    # adj_matrix_inferred = np.zeros((tau, n_modes, n_modes))
    # for k in range(n_modes):
    #     graph_k = graph[k]
    #     for j in range(n_modes):
    #         adj_matrix_inferred[:, k, j] = graph_k[j][1:] == "-->"

    # for k in range(tau):
    #     adj_matrix_inferred[k] = adj_matrix_inferred[k][np.ix_(permutation_list, permutation_list)]
    # adj_matrix_inferred = adj_matrix_inferred.transpose((0, 2, 1))

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
    print("permutation_list:", permutation_list)

    # Permute 
    for k in range(tau):
        adj_w[k] = adj_w[k][np.ix_(permutation_list, permutation_list)]

    print("PERMUTED THE MATRICES")

    precision, recall, f1, shd = evaluate_adjacency_matrix(adj_w, adj_gt, 0.9)

    print(f"difficuly {difficulty} results:")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, SHD: {shd}")
