"""
Evaluation utilities for comparing learned causal graphs against ground truth.

Provides functions to compute standard causal-discovery metrics -- Structural Hamming Distance (SHD), precision, recall,
and F1 -- between inferred and ground-truth adjacency matrices.  Also includes helpers for permuting matrices to align
latent orderings, plotting adjacency heatmaps, and extracting human-readable latent equations from adjacency structures.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def get_permutation_list(mat_adj_w, modes_gt, lat, lon):  # , remove_n_latents=0
    """
    Find the permutation that best aligns inferred modes to ground-truth modes.

    Alignment is based on the 2-D grid location of the maximum value of each
    spatial mode.  The permutation minimises the sum of squared Euclidean
    distances between the peak locations of ground-truth and inferred modes.

    Parameters
    ----------
    mat_adj_w : np.ndarray
        Inferred mode weight matrix, shape ``(lat*lon, n_modes)``.
    modes_gt : np.ndarray
        Ground-truth mode weights, shape ``(n_modes, lat, lon)``.
    lat, lon : int
        Spatial grid dimensions.

    Returns
    -------
    np.ndarray
        Integer array of length ``n_modes`` mapping each ground-truth mode
        index to the best-matching inferred mode index.
    """
    mat_adj_w = mat_adj_w.reshape((lat, lon, mat_adj_w.shape[1])).transpose((2, 0, 1))

    idx_gt = np.where(modes_gt == modes_gt.max((1, 2))[:, None, None])
    idx_inferred = np.array(np.where(mat_adj_w == mat_adj_w.max((1, 2))[:, None, None]))

    idx_gt = np.array(idx_gt)[1:]
    idx_inferred = idx_inferred[1:]

    # if remove_n_latents==0:
    return ((idx_gt[:, :, None] - idx_inferred[:, None, :]) ** 2).sum(0).argmin(1)  # .tolist()
    # else:
    #     return ((idx_gt[:, :, None] - idx_inferred[:, None, :])**2).sum(0).argmin(0).tolist()


def get_permutation_list_hardcoded_100(mat_adj_w, modes_gt, lat, lon):  # , remove_n_latents=0
    """
    Compute mode permutation for exactly 100 modes on a 10x10 spatial grid.

    This is a specialised (and possibly dead-code) variant of
    ``get_permutation_list`` that assumes 100 modes laid out on a 10x10 grid
    of 10x10 spatial patches.  Each mode is matched by finding the maximum
    activation within its expected patch.

    Parameters
    ----------
    mat_adj_w : np.ndarray
        Inferred mode weight matrix, shape ``(lat*lon, 100)``.
    modes_gt : np.ndarray
        Ground-truth mode weights, shape ``(100, lat, lon)``.
    lat, lon : int
        Spatial grid dimensions (expected to be 100 each).

    Returns
    -------
    list of int
        Permutation mapping ground-truth mode indices to inferred mode indices.
    """
    mat_adj_w = mat_adj_w.reshape((lat, lon, mat_adj_w.shape[1])).transpose((2, 0, 1))

    permutation_list = []
    for k in range(100):
        permutation_list.append(
            mat_adj_w[:, (k // 10) * 10 : (k // 10) * 10 + 10, (k % 10) * 10 : (k % 10) * 10 + 10].max((1, 2)).argmax()
        )
    return permutation_list


def load_adjacency_matrix(csv_file):
    """
    Loads the adjacency matrix from a CSV file, skipping the first row.

    Parameters:
        csv_file (str): The path to the CSV file containing the adjacency matrix.

    Returns:
        np.ndarray: The adjacency matrix as a NumPy array.
    """
    # Load the CSV file into a Pandas DataFrame, skipping the first row
    df = pd.read_csv(csv_file, header=None, skiprows=1)
    adjacency_matrix = df.values
    return np.array(adjacency_matrix)


def permute_matrix(matrix, permutation):
    """
    Permutes the rows and columns of the matrix based on the given permutation.

    Parameters:
        matrix (np.ndarray): The adjacency matrix to be permuted.
        permutation (list): The list containing the new order of indices.

    Returns:
        np.ndarray: The permuted adjacency matrix.
    """
    # Convert permutation list to a NumPy array
    permuted_matrix = matrix[np.ix_(permutation, permutation)]
    return permuted_matrix


def load_and_permute_all_matrices(modes_inferred, modes_gt, adj_w, adj_gt, lat, lon, tau):
    """
    Loads and permutes multiple adjacency matrices, one for each time lag.

    Parameters:
        csv_files (list): List of CSV file paths for each time lag.
        permutation (list): List containing the new order of indices.

    Returns:
        np.ndarray: A 3D NumPy array containing all permuted adjacency matrices
                    where the shape is (number_of_time_lags, n, n).
    """
    # Find the permutation
    modes_inferred = modes_inferred.reshape((lat, lon, modes_inferred.shape[-1])).transpose((2, 0, 1))

    # Get the flat index of the maximum for each mode
    idx_gt_flat = np.argmax(modes_gt.reshape(modes_gt.shape[0], -1), axis=1)  # shape: (n_modes,)
    idx_inferred_flat = np.argmax(modes_inferred.reshape(modes_inferred.shape[0], -1), axis=1)  # shape: (n_modes,)

    # Convert flat indices to 2D coordinates (row, col)
    idx_gt = np.array([np.unravel_index(i, (lat, lon)) for i in idx_gt_flat])  # shape: (n_modes, 2)
    idx_inferred = np.array([np.unravel_index(i, (lat, lon)) for i in idx_inferred_flat])  # shape: (n_modes, 2)

    # Compute error matrix using squared Euclidean distance between indices which yields an (n_modes x n_modes) matrix
    permutation_list = ((idx_gt[:, None, :] - idx_inferred[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    logger.info("permutation_list: %s", permutation_list)

    # Permute
    for k in range(tau):
        adj_w[k] = adj_w[k][np.ix_(permutation_list, permutation_list)]

    logger.info("PERMUTED THE MATRICES")

    return adj_w


def binarize_matrix(A, threshold=0.5):
    """
    Binarise an adjacency matrix by applying a threshold.

    Parameters
    ----------
    A : np.ndarray
        Real-valued adjacency matrix.
    threshold : float
        Values strictly above this become 1; all others become 0.

    Returns
    -------
    np.ndarray
        Integer array with values in {0, 1}.
    """
    return (A > threshold).astype(int)


def plot_adjacency_matrix(
    mat1: np.ndarray,
    mat2: np.ndarray,
    mat3: np.ndarray,
    path: str,
    name: str,
    no_gt: bool = False,
    iteration: int = 0,
    plot_through_time: bool = True,
    plot_last_row_col: bool = True,
):
    """
    Plot the adjacency matrices learned and compare them to the ground truth, the first dimension of the matrix should
    be the time (tau).

    Args:
        mat1: learned adjacency matrices
        mat2: ground-truth adjacency matrices
        mat3: original learned (unpermuted) adjacency matrices
        path: path where to save the plot
        name: name of the plot
        no_gt: if True, does not use the ground-truth graph
        iteration: iteration number for saving plot name
        plot_through_time: if True, saves the plot with iteration number
        plot_last_row_col: if True, plots the last row and column, otherwise skips them
    """
    tau = mat1.shape[0]  # Get the number of time steps
    subfig_names = ["Learned", "Ground Truth", "Original Learned (Unpermuted)"]

    fig = plt.figure(constrained_layout=True)
    fig.suptitle("Adjacency matrices:")

    if no_gt:
        nrows = 1
    else:
        nrows = 2

    # Determine the range for rows and columns
    if plot_last_row_col:
        row_col_slice = slice(None)  # Plot all rows and columns
    else:
        row_col_slice = slice(0, -2)  # Skip the last row and column

    if tau == 1:
        axes = fig.subplots(nrows=nrows, ncols=1)
        for row in range(nrows):
            if no_gt:
                ax = axes
            else:
                ax = axes[row]
            if row == 0:
                sns.heatmap(mat1[0][row_col_slice, row_col_slice], ax=ax, cbar=False, vmin=-1, vmax=1, cmap="Blues")
            elif row == 1:
                sns.heatmap(mat2[0], ax=ax, cbar=False, vmin=-1, vmax=1, cmap="Blues")
            elif row == 2:
                sns.heatmap(mat3[0][row_col_slice, row_col_slice], ax=ax, cbar=False, vmin=-1, vmax=1, cmap="Blues")
    else:
        subfigs = fig.subfigures(nrows=nrows, ncols=1)
        for row in range(nrows):
            if nrows == 1:
                subfig = subfigs
            else:
                subfig = subfigs[row]
            subfig.suptitle(f"{subfig_names[row]}")

            axes = subfig.subplots(nrows=1, ncols=tau)
            for i in range(tau):
                axes[i].set_title(f"t - {i+1}")
                if row == 0:
                    sns.heatmap(
                        mat1[i][row_col_slice, row_col_slice], ax=axes[i], cbar=False, vmin=-1, vmax=1, cmap="Blues"
                    )
                elif row == 1:
                    sns.heatmap(mat2[i], ax=axes[i], cbar=False, vmin=-1, vmax=1, cmap="Blues")
                elif row == 2:
                    sns.heatmap(
                        mat3[i][row_col_slice, row_col_slice], ax=axes[i], cbar=False, vmin=-1, vmax=1, cmap="Blues"
                    )

    if plot_through_time:
        fname = f"{name}_{iteration}.png"
    else:
        fname = f"{name}.png"

    plt.savefig(path / fname)
    plt.close()


def plot_adjacency_with_forcing_labels(
    mat_inferred: np.ndarray,
    mat_gt: np.ndarray,
    forcing_indices: dict,
    path: str,
    name: str = "adjacency_with_labels",
    threshold: float = 0.5,
    tau_idx: int = 0,
):
    """
    Plot adjacency matrices with labeled axes showing climate modes, CO2, and aerosol indices.

    Args:
        mat_inferred: Inferred adjacency matrices (tau x N x N)
        mat_gt: Ground truth adjacency matrices (tau x N x N)
        forcing_indices: Dict with 'co2', 'aerosol' index lists and 'n_total'
        path: Path where to save the plot
        name: Name of the plot file
        threshold: Binarization threshold
        tau_idx: Which time lag to plot (0-indexed)
    """
    co2_indices = forcing_indices.get("co2", [])
    aerosol_indices = forcing_indices.get("aerosol", [])
    n_total = forcing_indices.get("n_total", mat_inferred.shape[1])
    n_climate = n_total - len(co2_indices) - len(aerosol_indices)

    # Create labels for each index
    labels = []
    for i in range(n_total):
        if i < n_climate:
            labels.append(f"M{i}")
        elif i in co2_indices:
            labels.append("CO2")
        elif i in aerosol_indices:
            aero_idx = aerosol_indices.index(i)
            labels.append(f"A{aero_idx}")

    # Binarize matrices
    mat_inferred_bin = binarize_matrix(mat_inferred[tau_idx], threshold)
    mat_gt_bin = binarize_matrix(mat_gt[tau_idx], threshold)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot inferred adjacency
    axes[0].imshow(mat_inferred_bin, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    axes[0].set_title(f"Inferred Adjacency (t-{tau_idx+1})", fontsize=12)
    axes[0].set_xticks(range(n_total))
    axes[0].set_yticks(range(n_total))
    axes[0].set_xticklabels(labels, fontsize=8)
    axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_xlabel("Source (cause)")
    axes[0].set_ylabel("Target (effect)")

    # Plot ground truth adjacency
    im2 = axes[1].imshow(mat_gt_bin, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    axes[1].set_title(f"Ground Truth Adjacency (t-{tau_idx+1})", fontsize=12)
    axes[1].set_xticks(range(n_total))
    axes[1].set_yticks(range(n_total))
    axes[1].set_xticklabels(labels, fontsize=8)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_xlabel("Source (cause)")
    axes[1].set_ylabel("Target (effect)")

    # Add separating lines between climate modes and forcing latents
    for ax in axes:
        # Line between climate modes and CO2
        ax.axhline(y=n_climate - 0.5, color="red", linewidth=1.5, linestyle="--")
        ax.axvline(x=n_climate - 0.5, color="red", linewidth=1.5, linestyle="--")
        # Line between CO2 and aerosols (if both exist)
        if co2_indices and aerosol_indices:
            co2_end = max(co2_indices) + 0.5
            ax.axhline(y=co2_end, color="orange", linewidth=1, linestyle=":")
            ax.axvline(x=co2_end, color="orange", linewidth=1, linestyle=":")

    # Add colorbar
    fig.colorbar(im2, ax=axes, shrink=0.6, label="Edge present")

    # Add legend for regions
    legend_text = (
        f"M0-M{n_climate-1}: Climate Modes | CO2: CO2 Forcing | A0-A{len(aerosol_indices)-1}: Aerosol Forcings"
    )
    fig.text(0.5, 0.02, legend_text, ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(Path(path) / f"{name}.png", dpi=150)
    plt.close()

    logger.info(f"Saved labeled adjacency plot to {Path(path) / name}.png")


def plot_adjacency_all_lags_with_labels(
    mat_inferred: np.ndarray,
    mat_gt: np.ndarray,
    forcing_indices: dict,
    path: str,
    name: str = "adjacency_all_lags",
    threshold: float = 0.5,
):
    """
    Plot adjacency matrices for all time lags with labeled axes.

    Args:
        mat_inferred: Inferred adjacency matrices (tau x N x N)
        mat_gt: Ground truth adjacency matrices (tau x N x N)
        forcing_indices: Dict with 'co2', 'aerosol' index lists and 'n_total'
        path: Path where to save the plot
        name: Name of the plot file
        threshold: Binarization threshold
    """
    tau = mat_inferred.shape[0]
    co2_indices = forcing_indices.get("co2", [])
    aerosol_indices = forcing_indices.get("aerosol", [])
    n_total = forcing_indices.get("n_total", mat_inferred.shape[1])
    n_climate = n_total - len(co2_indices) - len(aerosol_indices)

    # Create labels
    labels = []
    for i in range(n_total):
        if i < n_climate:
            labels.append(f"M{i}")
        elif i in co2_indices:
            labels.append("CO2")
        elif i in aerosol_indices:
            aero_idx = aerosol_indices.index(i)
            labels.append(f"A{aero_idx}")

    # Create figure with 2 rows (inferred, gt) x tau columns
    fig, axes = plt.subplots(2, tau, figsize=(4 * tau, 8))

    if tau == 1:
        axes = axes.reshape(2, 1)

    for t in range(tau):
        mat_inf_bin = binarize_matrix(mat_inferred[t], threshold)
        mat_gt_bin = binarize_matrix(mat_gt[t], threshold)

        # Inferred
        axes[0, t].imshow(mat_inf_bin, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        axes[0, t].set_title(f"Inferred t-{t+1}", fontsize=10)
        if t == 0:
            axes[0, t].set_ylabel("Target")
            axes[0, t].set_yticks(range(n_total))
            axes[0, t].set_yticklabels(labels, fontsize=7)
        else:
            axes[0, t].set_yticks([])

        # Ground truth
        axes[1, t].imshow(mat_gt_bin, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        axes[1, t].set_title(f"GT t-{t+1}", fontsize=10)
        axes[1, t].set_xlabel("Source")
        axes[1, t].set_xticks(range(n_total))
        axes[1, t].set_xticklabels(labels, fontsize=7, rotation=45)
        if t == 0:
            axes[1, t].set_ylabel("Target")
            axes[1, t].set_yticks(range(n_total))
            axes[1, t].set_yticklabels(labels, fontsize=7)
        else:
            axes[1, t].set_yticks([])

        # Add separator lines
        for row in range(2):
            axes[row, t].axhline(y=n_climate - 0.5, color="red", linewidth=1, linestyle="--")
            axes[row, t].axvline(x=n_climate - 0.5, color="red", linewidth=1, linestyle="--")

    plt.suptitle("Adjacency Matrices by Time Lag (Red line separates climate modes from forcings)", fontsize=12)
    plt.tight_layout()
    plt.savefig(Path(path) / f"{name}.png", dpi=150)
    plt.close()

    logger.info(f"Saved multi-lag adjacency plot to {Path(path) / name}.png")


def evaluate_adjacency_matrix(A_inferred, A_ground_truth, threshold):
    """
    Evaluate precision, recall, F1, and SHD between inferred and ground-truth graphs.

    Both matrices are binarised with the given *threshold* before comparison.

    Parameters
    ----------
    A_inferred : np.ndarray
        Inferred adjacency matrix (possibly real-valued).
    A_ground_truth : np.ndarray
        Ground-truth adjacency matrix.
    threshold : float
        Values strictly above this become 1; all others become 0.

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
    logger.info(f"N inferred links: {A_inferred_bin.sum()}")
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


def evaluate_adjacency_by_link_type(A_inferred, A_ground_truth, threshold, forcing_indices):
    """
    Evaluate adjacency matrix metrics separately for different link types.

    Args:
        A_inferred: Inferred adjacency matrices (tau x N x N)
        A_ground_truth: Ground truth adjacency matrices (tau x N x N)
        threshold: Binarization threshold
        forcing_indices: Dict with 'co2', 'aerosol' index lists and 'n_total'

    Returns:
        Dict with metrics for each link type:
        - 'overall': Overall metrics
        - 'climate_to_climate': Climate mode ↔ Climate mode
        - 'co2_to_climate': CO2 → Climate mode
        - 'aerosol_to_climate': Aerosol → Climate mode
        - 'forcing_autoreg': Forcing autoregressive (CO2→CO2, aerosol→aerosol)
    """
    A_inferred_bin = binarize_matrix(A_inferred, threshold)
    A_ground_truth_bin = binarize_matrix(A_ground_truth, threshold)

    co2_indices = set(forcing_indices.get("co2", []))
    aerosol_indices = set(forcing_indices.get("aerosol", []))
    n_total = forcing_indices.get("n_total", A_inferred.shape[1])
    n_climate = n_total - len(co2_indices) - len(aerosol_indices)

    results = {}

    # Overall metrics
    results["overall"] = _compute_metrics(A_inferred_bin.flatten(), A_ground_truth_bin.flatten())

    # Climate ↔ Climate (indices 0 to n_climate-1)
    climate_mask = np.zeros_like(A_inferred_bin, dtype=bool)
    climate_mask[:, :n_climate, :n_climate] = True
    results["climate_to_climate"] = _compute_metrics(A_inferred_bin[climate_mask], A_ground_truth_bin[climate_mask])

    # CO2 → Climate (column indices in co2_indices, row indices in climate)
    co2_to_climate_mask = np.zeros_like(A_inferred_bin, dtype=bool)
    for co2_idx in co2_indices:
        co2_to_climate_mask[:, :n_climate, co2_idx] = True
    if co2_to_climate_mask.any():
        results["co2_to_climate"] = _compute_metrics(
            A_inferred_bin[co2_to_climate_mask], A_ground_truth_bin[co2_to_climate_mask]
        )
    else:
        results["co2_to_climate"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "shd": 0, "n_gt_links": 0}

    # Aerosol → Climate (column indices in aerosol_indices, row indices in climate)
    aerosol_to_climate_mask = np.zeros_like(A_inferred_bin, dtype=bool)
    for aerosol_idx in aerosol_indices:
        aerosol_to_climate_mask[:, :n_climate, aerosol_idx] = True
    if aerosol_to_climate_mask.any():
        results["aerosol_to_climate"] = _compute_metrics(
            A_inferred_bin[aerosol_to_climate_mask], A_ground_truth_bin[aerosol_to_climate_mask]
        )
    else:
        results["aerosol_to_climate"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "shd": 0, "n_gt_links": 0}

    # Forcing autoregressive (CO2→CO2, aerosol→aerosol diagonal)
    forcing_autoreg_mask = np.zeros_like(A_inferred_bin, dtype=bool)
    for idx in co2_indices | aerosol_indices:
        forcing_autoreg_mask[:, idx, idx] = True
    if forcing_autoreg_mask.any():
        results["forcing_autoreg"] = _compute_metrics(
            A_inferred_bin[forcing_autoreg_mask], A_ground_truth_bin[forcing_autoreg_mask]
        )
    else:
        results["forcing_autoreg"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "shd": 0, "n_gt_links": 0}

    return results


def _compute_metrics(inferred_flat, gt_flat):
    """
    Compute precision, recall, F1, and SHD from flattened binary arrays.

    Parameters
    ----------
    inferred_flat : np.ndarray
        Flattened binary inferred adjacency values.
    gt_flat : np.ndarray
        Flattened binary ground-truth adjacency values.

    Returns
    -------
    dict
        Dictionary with keys ``'precision'``, ``'recall'``, ``'f1'``,
        ``'shd'``, and ``'n_gt_links'``.
    """
    if len(inferred_flat) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "shd": 0, "n_gt_links": 0}

    n_gt_links = int(gt_flat.sum())
    n_inferred_links = int(inferred_flat.sum())

    if n_gt_links == 0 and n_inferred_links == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "shd": 0, "n_gt_links": 0}
    elif n_gt_links == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "shd": n_inferred_links, "n_gt_links": 0}
    elif n_inferred_links == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "shd": n_gt_links, "n_gt_links": n_gt_links}

    precision = float(precision_score(gt_flat, inferred_flat, zero_division=0))
    recall = float(recall_score(gt_flat, inferred_flat, zero_division=0))
    f1 = float(f1_score(gt_flat, inferred_flat, zero_division=0))

    false_positives = int(np.sum((inferred_flat == 1) & (gt_flat == 0)))
    false_negatives = int(np.sum((inferred_flat == 0) & (gt_flat == 1)))
    shd = false_positives + false_negatives

    return {"precision": precision, "recall": recall, "f1": f1, "shd": shd, "n_gt_links": n_gt_links}


def print_evaluation_by_link_type(results):
    """
    Log evaluation metrics grouped by causal link type.

    Parameters
    ----------
    results : dict
        Dictionary returned by ``evaluate_adjacency_by_link_type``, mapping
        link-type names (e.g. ``'climate_to_climate'``) to metric dicts.
    """
    logger.info("\n%s", "=" * 70)
    logger.info("EVALUATION BY LINK TYPE")
    logger.info("=" * 70)

    for link_type, metrics in results.items():
        logger.info("\n%s:", link_type.upper().replace("_", " "))
        logger.info("  GT links: %s", metrics["n_gt_links"])
        logger.info("  Precision: %.3f", metrics["precision"])
        logger.info("  Recall:    %.3f", metrics["recall"])
        logger.info("  F1:        %.3f", metrics["f1"])
        logger.info("  SHD:       %s", metrics["shd"])

    logger.info("\n%s", "=" * 70)


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
    for target, values in links_coeffs.items():
        for link, coeff in values:
            source, lag = link
            time_lag = -lag  # Convert the negative lag to a positive index
            # Only consider lags that are within the specified time window (tau)
            if time_lag <= tau:
                if abs(coeff) > 0.01:
                    adj_matrices[time_lag - 1, target, source] = (
                        1  # Fill the adjacency matrix at the appropriate time lag
                    )
                else:
                    adj_matrices[time_lag - 1, target, source] = 0

    return adj_matrices


def extract_latent_equations(links_coeffs):
    """
    Convert a ``links_coeffs`` dictionary into human-readable latent equations.

    Parameters
    ----------
    links_coeffs : dict
        Mapping from latent variable index to a list of
        ``((linked_var, lag), coefficient)`` tuples.

    Returns
    -------
    dict
        Mapping from latent variable index to a string equation, e.g.
        ``"L0(t) = 0.5 * L1(t - 1) + 0.3 * L0(t - 2)"``.
    """
    equations = {}

    for latent_var, links in links_coeffs.items():
        equation_terms = []
        for (linked_var, lag), coeff in links:
            term = f"{coeff} * L{linked_var}(t{f' - {abs(lag)}' if lag != 0 else ''})"
            equation_terms.append(term)

        equation = " + ".join(equation_terms)
        equations[latent_var] = f"L{latent_var}(t) = {equation}"

    return equations


def extract_equations_from_adjacency(adj_matrices):
    """
    Derive human-readable latent equations from a stack of adjacency matrices.

    Parameters
    ----------
    adj_matrices : np.ndarray
        Adjacency matrices with shape ``(num_lags, num_latents, num_latents)``.
        Non-zero entries indicate a causal link.

    Returns
    -------
    dict
        Mapping from latent variable index to a string equation built from
        the non-zero entries of the adjacency matrices across all lags.
    """
    num_lags, num_latents, _ = adj_matrices.shape  # 5 lags, 16 latents

    equations = {}
    for latent_var in range(num_latents):
        equation_terms = []
        for lag in range(num_lags):
            adj_matrix_at_lag = adj_matrices[lag]  # Get the adjacency matrix for the current lag
            for linked_var in range(num_latents):
                coeff = adj_matrix_at_lag[latent_var, linked_var]
                if coeff != 0:  # Only include non-zero coefficients
                    term = f"{coeff} * L{linked_var}(t - {lag+1})"
                    equation_terms.append(term)

        # Join the terms to create the equation
        if equation_terms:
            equation = " + ".join(equation_terms)
            equations[latent_var] = f"L{latent_var}(t) = {equation}"
        else:
            equations[latent_var] = f"L{latent_var}(t) = 0"  # No dependencies found

    return equations


def main(csv_file, permutation):
    """
    Main function to load, permute, and return the adjacency matrix.

    Parameters:
        csv_file (str): The path to the CSV file containing the adjacency matrix.
        permutation (list): The list containing the new order of indices.

    Returns:
        np.ndarray: The permuted adjacency matrix.
    """
    # Load the adjacency matrix
    adjacency_matrix = load_adjacency_matrix(csv_file)

    # Permute the adjacency matrix
    permuted_matrix = permute_matrix(adjacency_matrix, permutation)

    return permuted_matrix


def save_equations_to_json(equations, filename):
    """
    Serialise a dictionary of latent equations to a JSON file.

    Parameters
    ----------
    equations : dict
        Mapping from latent variable index to equation string.
    filename : str or Path
        Destination file path.
    """
    with open(filename, "w") as json_file:
        json.dump(equations, json_file, indent=4)
    logger.info(f"Equations saved to {filename}")


# Example usage:
# NOTE: The paths below are hardcoded to a specific user/cluster environment.
#       Adjust savar_path, results_path, and config_path to match your setup.
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    threshold = 0.5

    # load your existing JSON config
    config_path = Path("configs/single_param_file_savar.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    exp = cfg["exp_params"]
    data = cfg["data_params"]
    savar = cfg["savar_params"]

    # pull out exactly the bits you used to hard-code
    tau = exp["tau"]
    n_modes = exp["d_z"]  # latent dim = number of modes
    comp_size = savar["comp_size"]
    time_len = savar["time_len"]
    is_forced = savar["is_forced"]
    seasonality = savar["seasonality"]
    overlap = savar["overlap"]
    difficulty = savar["difficulty"]
    lat = lon = int(np.sqrt(n_modes)) * comp_size
    noise_val = savar["noise_val"]

    home_path = str(Path.home())
    savar_path = "/my_projects/climatem/workspace/pfs7wor9/ka_qa4548-data/SAVAR_DATA_TEST"
    results_path = Path(
        "my_projects/climatem/workspace/pfs7wor9/ka_qa4548-results/SAVAR_DATA_TEST/var_savar_scenarios_piControl_nonlinear_False_tau_5_z_9_lr_0.001_bs_256_spreg_0_ormuinit_100000.0_spmuinit_0.1_spthres_0.05_fixed_False_num_ensembles_1_instantaneous_False_crpscoef_1_spcoef_0_tempspcoef_0_overlap_0.3_forcing_True"
    )

    # Load ground truthh modes
    savar_folder = home_path + savar_path
    savar_fname = f"modes_{n_modes}_tl_{time_len}_isforced_{is_forced}_difficulty_{difficulty}_noisestrength_{noise_val}_seasonality_{seasonality}_overlap_{overlap}"
    # modes_gt_path = savar_folder / Path(f"/{savar_fname}_mode_weights.npy")
    modes_gt = np.load(f"{savar_folder}/{savar_fname}_mode_weights.npy")

    result_folder = home_path / results_path
    # load CDSD results
    cdsd_adj_inferred_path = result_folder / Path("plots/graphs.npy")
    cdsd_modes_inferred_path = result_folder / Path("plots/w_decoder.npy")
    modes_inferred = np.load(cdsd_modes_inferred_path)
    adjacency_inferred = np.load(cdsd_adj_inferred_path)

    # if n_modes == 100:
    #     # With lots of modes some modes are equal and the other function breaks. This function works for the specifics params of the 100 modes dataset.
    #     permutation_list = get_permutation_list(mat_adj_w, modes_gt, lat, lon)
    # else:
    #     permutation_list = get_permutation_list(mat_adj_w, modes_gt, lat, lon)
    permuted_matrices = np.array(
        load_and_permute_all_matrices(modes_inferred, modes_gt, adjacency_inferred, adjacency_inferred, lat, lon, tau)
    )

    # Load parameters from npy file
    params_file = f"{savar_folder}/{savar_fname}_parameters.npy"
    params = np.load(params_file, allow_pickle=True).item()
    links_coeffs = params["links_coeffs"]

    gt_adj_list = extract_adjacency_matrix(links_coeffs, n_modes, tau)

    plot_adjacency_matrix(
        mat1=binarize_matrix(permuted_matrices, threshold),
        mat2=gt_adj_list,
        mat3=gt_adj_list,
        path=result_folder,
        name=f"permuted_adjacency_thr_{threshold}",
        no_gt=False,
        iteration=20000,
        plot_through_time=True,
    )

    save_equations_to_json(extract_latent_equations(links_coeffs), result_folder / "gt_eq")
    save_equations_to_json(
        extract_equations_from_adjacency(binarize_matrix(permuted_matrices, threshold)),
        result_folder / f"thr_{threshold}_results_eq",
    )

    precision, recall, f1, shd = evaluate_adjacency_matrix(permuted_matrices, gt_adj_list, threshold)
    logger.info(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, SHD: {shd}")
    results = {"precision": precision, "recall": recall, "f1_score": f1, "shd": shd}
    # Save results as a JSON file
    json_filename = result_folder / f"thr_{threshold}_evaluation_results.json"
    with open(json_filename, "w") as json_file:
        json.dump(results, json_file)
