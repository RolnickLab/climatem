import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
import torch 
from tqdm.auto import tqdm

from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd

from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from sklearn.decomposition import PCA


### Here set SAVAR paths and load data ####
n_gt_connections = 6
difficulty = "med_easy"
noisestrength = 1

tau = 5
n_modes = 4
time_len = 10_000
lat = lon = 50
var_names = []
for k in range(n_modes):
    var_names.append(fr'$X^{k}$')
savar_folder = Path('/home/mila/j/julien.boussard/causal_model/savar/input')
savar_fname = f"m_{n_modes}_tl_10000_isforced_False_difficulty_{difficulty}_noisestrength_{noisestrength}_nooverlap_noseasonality.npy"
savar_data = np.load(savar_folder / savar_fname)
# link_coeffs_fname = f"m_{n_modes}_tl_10000_isforced_False_difficulty_{difficulty}_noisestrength_{noisestrength}_nooverlap_noseasonality_links_coeffs.csv"
# links_coeff = pd.read_csv(savar_folder / link_coeffs_fname)
params_file = savar_folder / f'{savar_fname[:-4]}_parameters.npy'
params = np.load(params_file, allow_pickle=True).item()
links_coeffs = params["links_coeffs"]
modes_gt = np.load(savar_folder / f'{savar_fname[:-4]}_mode_weights.npy')
modes_gt -= modes_gt.mean()
modes_gt /= modes_gt.std()
json_results_fname = savar_folder / f'varimax_pcmci_{savar_fname[:-4]}_results.json'

############################

parcorr = ParCorr(significance='analytic')

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


def evaluate_adjacency_matrix(A_inferred, A_ground_truth, threshold):
    """
    Evaluates the precision, recall, F1-score, and Structural Hamming Distance (SHD)
    between the inferred and ground truth adjacency matrices.
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

def binarize_matrix(A, threshold=0.5):
    """
    Binarizes the adjacency matrix by applying a threshold.
    """
    return (A > threshold).astype(int)

def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R), R

# if __name__ == "__main__":

pca_model = PCA(4).fit(savar_data.T)
latent_data = pca_model.transform(savar_data.T)
varimaxpcs, varimax_rotation = varimax(latent_data)

# To recover which mode is which and permute accordingly
inverse_varimax = dot(latent_data, np.linalg.pinv(varimax_rotation))
reverted_data = pca_model.inverse_transform(inverse_varimax)

dataframe = pp.DataFrame(varimaxpcs, 
                         datatime = {0:np.arange(len(varimaxpcs))}, 
                         var_names=var_names)

pcmci = PCMCI(
    dataframe=dataframe, 
    cond_ind_test=parcorr,
    verbosity=1)

results = pcmci.run_pcmci(
    tau_min=1, tau_max=5, pc_alpha=None, alpha_level=0.001
)

# Find graph + permute + save results + compare 

# To recover which mode is which 
# inverse_varimax = dot(latent_data, np.linalg.pinv(varimax_rotation))
# reverted_data = pca_model.inverse_transform(inverse_varimax)

individual_modes = np.zeros((n_modes, time_len, lat, lon))
for k in range(n_modes):
    latent_data_bis = np.zeros(latent_data.shape)
    latent_data_bis[:, k] = latent_data[:, k]
    inverse_varimax = dot(latent_data_bis, np.linalg.pinv(varimax_rotation))
    reverted_data = pca_model.inverse_transform(inverse_varimax)
    individual_modes[k] = reverted_data.reshape((-1, lat, lon))
individual_modes = individual_modes.std(1)
individual_modes -= individual_modes.mean()
individual_modes /= individual_modes.std()

permutation_list = ((modes_gt[:, None] - individual_modes[None])**2).sum((2,3)).argmin(1)

graph = results['graph']
graph[results['val_matrix']<np.abs(results['val_matrix'].flatten()[results['val_matrix'].flatten().argsort()[::-1][n_gt_connections-1]])] = ''
adj_matrix_inferred = np.zeros((tau, n_modes, n_modes))

for k in range(n_modes):
    graph_k = graph[k]
    for j in range(n_modes):
        adj_matrix_inferred[:, k, j] = graph_k[j][1:] == '-->'

for k in range(tau):
    adj_matrix_inferred[k] = adj_matrix_inferred[k][np.ix_(permutation_list, permutation_list)]        
adj_matrix_inferred = adj_matrix_inferred.transpose((0, 2, 1))

gt_adj_list = extract_adjacency_matrix(links_coeffs, n_modes, tau)
precision, recall, f1, shd = evaluate_adjacency_matrix(adj_matrix_inferred, gt_adj_list, 0.9)

print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}, SHD: {shd}')
results = {
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "shd": shd
}
# Save results as a JSON file
with open(json_results_fname, 'w') as json_file:
    json.dump(results, json_file)
