import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path


def get_permutation_list(mat_adj_w, modes_gt, lat, lon): #, remove_n_latents=0

    mat_adj_w = mat_adj_w.reshape((lat, lon, mat_adj_w.shape[1])).transpose((2, 0, 1))

    idx_gt = np.where(modes_gt == modes_gt.max((1, 2))[:, None, None])
    idx_inferred = np.array(np.where(mat_adj_w== mat_adj_w.max((1, 2))[:, None, None]))

    idx_gt = np.array(idx_gt)[1:]
    idx_inferred = idx_inferred[1:]

    # if remove_n_latents==0:
    return ((idx_gt[:, :, None] - idx_inferred[:, None, :])**2).sum(0).argmin(1) #.tolist()
    # else:
    #     return ((idx_gt[:, :, None] - idx_inferred[:, None, :])**2).sum(0).argmin(0).tolist()

def get_permutation_list_hardcoded_100(mat_adj_w, modes_gt, lat, lon): #, remove_n_latents=0

    mat_adj_w = mat_adj_w.reshape((lat, lon, mat_adj_w.shape[1])).transpose((2, 0, 1))

    permutation_list = []
    for k in range(100):
        permutation_list.append(mat_adj_w[:, (k//10)*10:(k//10)*10+10, (k%10)*10:(k%10)*10+10].max((1, 2)).argmax())
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

def load_and_permute_all_matrices(csv_files, permutation, remove_modes = []):
    """
    Loads and permutes multiple adjacency matrices, one for each time lag.
    
    Parameters:
        csv_files (list): List of CSV file paths for each time lag.
        permutation (list): List containing the new order of indices.
    
    Returns:
        np.ndarray: A 3D NumPy array containing all permuted adjacency matrices
                    where the shape is (number_of_time_lags, n, n).
    """
    permuted_matrices = []
    
    for csv_file in csv_files:
        # Load the adjacency matrix
        adjacency_matrix = load_adjacency_matrix(csv_file)

        if len(remove_modes):
            adjacency_matrix = np.delete(adjacency_matrix, remove_modes, 0)
            adjacency_matrix = np.delete(adjacency_matrix, remove_modes, 1) 
        
        # Permute the adjacency matrix
        permuted_matrix = permute_matrix(adjacency_matrix, permutation)
        
        # Append the permuted matrix to the list
        permuted_matrices.append(permuted_matrix)
    
    # Convert the list of permuted matrices to a NumPy array
    return np.array(permuted_matrices)

def binarize_matrix(A, threshold=0.5):
    """
    Binarizes the adjacency matrix by applying a threshold.
    """
    return (A > threshold).astype(int)

def plot_adjacency_matrix(mat1: np.ndarray, mat2: np.ndarray, mat3: np.ndarray, 
                          path: str, name: str, no_gt: bool = False,
                          iteration: int = 0, plot_through_time: bool = True,
                          plot_last_row_col: bool = True):
    """ 
    Plot the adjacency matrices learned and compare them to the ground truth,
    the first dimension of the matrix should be the time (tau).
    
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
                sns.heatmap(mat1[0][row_col_slice, row_col_slice], ax=ax, cbar=False, vmin=-1, vmax=1,
                            cmap="Blues")
            elif row == 1:
                sns.heatmap(mat2[0], ax=ax, cbar=False, vmin=-1, vmax=1,
                            cmap="Blues")
            elif row == 2:
                sns.heatmap(mat3[0][row_col_slice, row_col_slice], ax=ax, cbar=False, vmin=-1, vmax=1,
                            cmap="Blues")
    else:
        subfigs = fig.subfigures(nrows=nrows, ncols=1)
        for row in range(nrows):
            if nrows == 1:
                subfig = subfigs
            else:
                subfig = subfigs[row]
            subfig.suptitle(f'{subfig_names[row]}')

            axes = subfig.subplots(nrows=1, ncols=tau)
            for i in range(tau):
                axes[i].set_title(f"t - {i+1}")
                if row == 0:
                    sns.heatmap(mat1[i][row_col_slice, row_col_slice], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues")
                elif row == 1:
                    sns.heatmap(mat2[i], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues")
                elif row == 2:
                    sns.heatmap(mat3[i][row_col_slice, row_col_slice], ax=axes[i], cbar=False, vmin=-1, vmax=1,
                                cmap="Blues")

    if plot_through_time:
        fname = f"{name}_{iteration}.png"
    else:
        fname = f"{name}.png"

    plt.savefig(path / fname)
    plt.close()


def evaluate_adjacency_matrix(A_inferred, A_ground_truth, threshold):
    """
    Evaluates the precision, recall, F1-score, and Structural Hamming Distance (SHD)
    between the inferred and ground truth adjacency matrices.
    """
    # Binarize the matrices before comparison
    A_inferred_bin = binarize_matrix(A_inferred, threshold)
    print(f"N inferred links: {A_inferred_bin.sum()}")
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
                    adj_matrices[time_lag - 1, key, target_var] = 1  # Fill the adjacency matrix at the appropriate time lag
                else:
                    adj_matrices[time_lag - 1, key, target_var] = 0


    return adj_matrices

def extract_latent_equations(links_coeffs):
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
    with open(filename, 'w') as json_file:
        json.dump(equations, json_file, indent=4)
    print(f"Equations saved to {filename}")

# Example usage:
if __name__ == "__main__":

    # Set parameters here
    home_path = Path('$HOME')
    tau = 5
    threshold = 0.75
    n_modes_gt = 25
    difficulty = "easy"
    iteration = 2999
    comp_size = 25

    n_modes = n_modes_gt
    lat = lon = int(np.sqrt(n_modes))*comp_size
    folder_results = home_path / f'predictions_{n_modes_gt}_{difficulty}'
    savar_folder = home_path / Path('savar_data')
    savar_fname = f"m_{n_modes_gt}_{difficulty}_savar_name"
    run_name = f"model_results_folder"
    results_path = folder_results / f"{savar_fname}_{run_name}"
    csv_files = [results_path / f'adjacency_transition_time_{i}_iteration_{iteration}.csv' for i in np.arange(5, 0, -1)]

    # Get the permuted adjacency matrices for all time lags
    modes_gt = np.load(savar_folder / f'{savar_fname}_mode_weights.npy')
    mat_adj_w = np.load(results_path / f'adj_w_iteration_{iteration}.npy')[0]

    if n_modes == 100:
        # With lots of modes some modes are equal and the other function breaks. This function works for the specifics params of the 100 modes dataset. 
        permutation_list = get_permutation_list_hardcoded_100(mat_adj_w, modes_gt, lat, lon)
    else:
        permutation_list = get_permutation_list(mat_adj_w, modes_gt, lat, lon)
    permuted_matrices = np.array(load_and_permute_all_matrices(csv_files, permutation_list))


    # Load parameters from npy file
    params_file = savar_folder / f'{savar_fname}_parameters.npy'
    params = np.load(params_file, allow_pickle=True).item()
    links_coeffs = params["links_coeffs"]

    gt_adj_list = extract_adjacency_matrix(links_coeffs, n_modes_gt, tau)

    plot_adjacency_matrix(mat1=binarize_matrix(permuted_matrices, threshold),
                                   mat2=gt_adj_list,
                                   mat3=gt_adj_list,
                                   path=results_path,
                                   name=f'permuted_adjacency_thr_{threshold}',
                                   no_gt=False,
                                   iteration=iteration,
                                   plot_through_time=True)
        
    save_equations_to_json(extract_latent_equations(links_coeffs), results_path / 'gt_eq')
    save_equations_to_json(extract_equations_from_adjacency(binarize_matrix(permuted_matrices, threshold)), results_path / f"thr_{threshold}_results_eq")


    precision, recall, f1, shd = evaluate_adjacency_matrix(permuted_matrices, gt_adj_list, threshold)
    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}, SHD: {shd}')
    results = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "shd": shd
    }
    # Save results as a JSON file
    json_filename = results_path / f"thr_{threshold}_evaluation_results.json"
    with open(json_filename, 'w') as json_file:
        json.dump(results, json_file)
