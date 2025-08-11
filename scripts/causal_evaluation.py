import climatem.synthetic_data.graph_evaluation as ge
from climatem.utils import parse_args, update_config_withparse
import climatem.config
import climatem
import climatem.config as cfg
import os
import json
import platform
import glob
import numpy as np

# TODO check imports!!!

# rsync with local

# steps:
# save config file with data and access params from there
# add output file & error file to data folder as well
# TODO can't do because error & output is saved in scratch (can we access that from the code? I think so)
# TODO maybe just write a script to move the files to the data folder after? idk
# done not sure where to save config file, also there /is/ a params json saved but it doesn't save all the params used

# TODO need to create new folder so data is not overwritten when you make a new run

# done Access etiher json or npy
#     modes_4_tl_10000_isforced_False_difficulty_easy_noisestrength_0.2_seasonality_False_overlap_False_parameters
# done extract params
#     params = np.load(params_file, allow_pickle=True).item()
#     links_coeffs = params["links_coeffs"
# create adjacency matrics from links coeffs
# permute links coefs
# import f1, precision, recall, harmad distance, etc. & evaluate

# TODO need to save tau to paramerters.csv file

# Example usage:
if __name__ == "__main__":

    args = parse_args()

    # Set parameters here
    # TODO either move all of this to a json file for json_path or find a way to extract this information from the params

    home_path = os.getenv("HOME")
    print("home path: ", home_path)

    # determine if running on cluster or local machine
    if platform.system() == "Linux":
        # on Mila cluster
        scratch_path = os.getenv("SCRATCH")
        print("scratch path: ", scratch_path)

    # get user's scratch directory on Mila cluster:
    else:
        # on Mac or PC?
        # TODO this is hardcoded for Christina's local machine, need to fix
        scratch_path = os.path.join(home_path, "dev", "mila", "scratch")
        print("local scratch path: ", scratch_path)

    # TODO need to find config path in the right data directory
    # TODO need to get correct data directory
    # how to find correct savar data dir?
    data_path = os.path.join(scratch_path, "data", "SAVAR_DATA_TEST")
    print("savar folder: ", data_path)
    # TODO how do i figure out which config to use?
    config_path = os.path.join(data_path, args.config_path)

    print("config path: ", config_path)

    # name from parameters file

    with open(config_path, "r") as f:
        params = json.load(f)

    # this is duplicated from main_picabu, probably should be its own function
    experiment_params = cfg.expParams(**params["exp_params"])
    data_params = cfg.dataParams(**params["data_params"])
    gt_params = cfg.gtParams(**params["gt_params"])
    train_params = cfg.trainParams(**params["train_params"])
    model_params = cfg.modelParams(**params["model_params"])
    optim_params = cfg.optimParams(**params["optim_params"])
    plot_params = cfg.plotParams(**params["plot_params"])
    savar_params = cfg.savarParams(**params["savar_params"])

    # relevant params
    tau = experiment_params.tau
    n_modes = experiment_params.d_z
    difficulty = savar_params.difficulty
    iteration = 2999
    comp_size = savar_params.comp_size
    tl = savar_params.time_len
    isforced = savar_params.is_forced
    noisestrength = savar_params.noise_val
    seasonality = savar_params.seasonality
    overlap = savar_params.overlap

    # folder to save results
    # TODO where should I shave the results?
    # results_path = os.path.join(scratch_path, "results", "SAVAR_DATA_TEST", f"eval_{n_modes}_{difficulty}")
    # os.makedirs(results_path, exist_ok=True)

    # Load parameters from npy file <-- get ground truth graph
    # TODO need to get correct params graph with correct
    params_file = glob.glob(
        f"{data_path}/modes_{n_modes}_tl_{tl}_isforced_{isforced}_diff_{difficulty}_noisestrength_{noisestrength}_seasonality_{seasonality}_overlap_{overlap}_parameters.npy"
    )[0]

    params = np.load(params_file, allow_pickle=True).item()
    gt_graph = params["links_coeffs"]

    print("Ground truth graph: ", gt_graph)

    # TODO next step create adjacency matrix out of gt graph
    # self.gt_adj = np.array(extract_adjacency_matrix(links_coeffs, self.n_per_col**2, tau))

    # TODO permute adjacency matrix --> maybe use the permute function from graph_evaluation.py

    # TODO import f1, precision, recall, harmad distance, etc. & evaluate

    # remember that integers are np integers
    # data structure: dictionary
    # {0: [((0, np.int64(-2)), 0.5)], 1: [((1, np.int64(-3)), 0.38)], 2: [((2, np.int64(-2)), 0.36)], 3: [((3, np.int64(-4)), 0.38)]}

    # run up until here

    # # main process execution
    # gt_adj_list = extract_adjacency_matrix(links_coeffs, n_modes_gt, tau)
    #
    #
    # # plot
    # plot_adjacency_matrix(
    #     mat1=binarize_matrix(permuted_matrices, threshold),
    #     mat2=gt_adj_list,
    #     mat3=gt_adj_list,
    #     path=results_path,
    #     name=f"permuted_adjacency_thr_{threshold}",
    #     no_gt=False,
    #     iteration=iteration,
    #     plot_through_time=True,
    # )
    #
    # # save to json
    # save_equations_to_json(extract_latent_equations(links_coeffs), results_path / "gt_eq")
    # save_equations_to_json(
    #     extract_equations_from_adjacency(binarize_matrix(permuted_matrices, threshold)),
    #     results_path / f"thr_{threshold}_results_eq",
    # )
    #
    # precision, recall, f1, shd = evaluate_adjacency_matrix(permuted_matrices, gt_adj_list, threshold)
    # print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, SHD: {shd}")
    # results = {"precision": precision, "recall": recall, "f1_score": f1, "shd": shd}
    # # Save results as a JSON file
    # json_filename = results_path / f"thr_{threshold}_evaluation_results.json"
    # with open(json_filename, "w") as json_file:
    #     json.dump(results, json_file)
