from typing import List

from climatem.constants import SEQ_LEN_MAPPING


class expParams:
    def __init__(
        self,
        exp_path,  # Path to where the output will be saved i.e. model runs, plots
        _target_: str = "climatem.data_loader.climate_datamodule.ClimateDataModule",
        latent: bool = True,  # Are you using latent variables or not (if not, learn causal variables between all observations)
        d_z: int = 90,  # Latent dimension
        d_x: int = 6250,  # Observation dimension
        lon: int = 144,  # Longitude
        lat: int = 96,  # Latitude
        tau: int = 5,  # Number of timesteps
        random_seed: int = 1,
        gpu: bool = True,  # Running code on GPU?
        num_workers: int = 0,
        pin_memory: bool = False,
        verbose: bool = True,
    ):
        self.exp_path = exp_path
        self._target_ = _target_
        self.latent = latent
        self.d_z = d_z
        self.d_x = d_x
        self.lon = lon
        self.lat = lat
        self.tau = tau
        self.random_seed = random_seed
        self.gpu = gpu
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose


class dataParams:
    def __init__(
        self,
        data_dir,  # The processed (normalized, deseasonalized, numpy...) data will be stored here
        climateset_data,  # The raw data is found here (typically .grib or .nc files)
        reload_climate_set_data,  # If True, will reload the numpy data directly from data_dir
        icosahedral_coordinates_path,  # Path to coordinates
        train_historical_years,  # If "historical" in train_scenarios use these years to train
        test_years,  # use these years to test
        train_years,  # use these years to train
        train_scenarios,  # training scenarios i.e. piControl, ssp245 ...
        test_scenarios,  # test scenarios
        train_models,  # train_models i.e. Nor-ESM
        #  test_models, TODO: enable training and testing on two different models
        in_var_ids,  # input variables i.e. ts, pr, gases. If "savar" uses synthetic data
        out_var_ids,  # output variables i.e. ts, pr, gases
        num_ensembles: int = 1,  # number of ensembles
        num_levels: int = 1,
        temp_res: str = "mon",  # temporal resolution. Only "mon" is accepted for now
        batch_size: int = 256,  # batch size for loading the data
        eval_batch_size: int = 256,  # batch size for loading the evaluation data
        global_normalization: bool = True,  # normalize the data?
        seasonality_removal: bool = False,  # deseasonalize the data?
        channels_last: bool = False,  # last dimension of data is the channel
        ishdf5: bool = False,  # numpy vs hdf5. for now only numpy is supported. Redundant with next param
        data_format: str = "numpy",  # numpy vs hdf5. for now only numpy is supported
        seq_to_seq: bool = True,  # predicting a sequence from a sequence?
        train_val_interval_length: int = 11,
        load_train_into_mem: bool = True,
        load_test_into_mem: bool = True,
        num_months_aggregated: List[int] = [
            1
        ],  # Aggregate num_months_aggregated months i.e. if you want yearly temporal resolution set this param to [12]
    ):
        self.data_dir = data_dir
        self.climateset_data = climateset_data
        self.reload_climate_set_data = reload_climate_set_data
        self.icosahedral_coordinates_path = icosahedral_coordinates_path
        self.train_historical_years = train_historical_years
        self.test_years = test_years
        self.train_years = train_years
        self.train_scenarios = train_scenarios
        self.test_scenarios = test_scenarios
        self.train_models = train_models
        # self.test_models = test_models
        self.in_var_ids = in_var_ids
        self.out_var_ids = out_var_ids
        self.num_ensembles = num_ensembles
        self.num_levels = num_levels
        try:
            self.seq_len = SEQ_LEN_MAPPING[temp_res]
        except ValueError:
            print(f"Only monthly resolution is implemented for now, you entered resolution {temp_res}")
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.global_normalization = global_normalization
        self.seasonality_removal = seasonality_removal
        self.channels_last = channels_last
        self.ishdf5 = ishdf5
        self.data_format = data_format
        self.seq_to_seq = seq_to_seq
        self.train_val_interval_length = train_val_interval_length
        self.load_train_into_mem = load_train_into_mem
        self.load_test_into_mem = load_test_into_mem
        self.num_months_aggregated = num_months_aggregated


# This class is only for debugging and for setting some params to the true aprams when training picabu
class gtParams:
    def __init__(
        self,
        no_gt: bool = True,  # do we have GT to compare? If synthetic data, will be True and overwritten
        debug_gt_z: bool = False,  # below params help debugging the code when we have ground truth
        debug_gt_w: bool = False,
        debug_gt_graph: bool = False,
    ):
        self.no_gt = no_gt
        self.debug_gt_z = debug_gt_z
        self.debug_gt_w = debug_gt_w
        self.debug_gt_graph = debug_gt_graph


class trainParams:
    def __init__(
        self,
        ratio_train: float = 0.9,
        lr: float = 0.001,
        lr_scheduler_epochs: List[int] = [10000, 20000],
        lr_scheduler_gamma: float = 1,  # multiply lr by this value at iterations specified in lr_scheduler_epochs
        max_iteration: int = 100000,  # maximum trainign iteration
        patience: int = 5000,  # Only learn mapping from obs to latents for patience iteration
        patience_post_thresh: int = 50,  # NOT SURE: if mapping converges before patience, and for patience_post_thresh it's stable, then optimize everything
        valid_freq: int = 5,  # get validation metrics every valid_freq iteration
    ):
        self.ratio_train = ratio_train
        self.ratio_valid = 1 - self.ratio_train
        self.lr = lr
        self.lr_scheduler_epochs = lr_scheduler_epochs
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.max_iteration = max_iteration
        self.patience = patience
        self.patience_post_thresh = patience_post_thresh
        self.valid_freq = valid_freq


class modelParams:
    def __init__(
        self,
        instantaneous: bool = False,  # Allow instantaneous connections?
        no_w_constraint: bool = False,  # If True, no single parent assumption i.e. no causal graph
        tied_w: bool = False,  # NOT SURE, to clarify
        nonlinear_mixing: bool = True,  # If False, latent dynamics are linear
        num_hidden_mixing: int = 16,  # MLP params for latent dynamics if non-linear
        num_layers_mixing: int = 2,
        nonlinear_dynamics: bool = True,
        num_hidden: int = 8,  # MLP params for mapping from obs to latents. If 0, then linear. SHould add a flag as `nonlinear_mixing`
        num_layers: int = 2,
        num_output: int = 2,  # NOT SURE
        fixed: bool = False,  # Do we fix the causal graph? Should be in gt_params maybe
        fixed_output_fraction=None,  # NOT SURE, Remove this?
        tau_neigh: int = 0,  # NOT SURE
        hard_gumbel: bool = False,  # NOT SURE
    ):
        self.instantaneous = instantaneous
        self.no_w_constraint = no_w_constraint
        self.tied_w = tied_w
        self.nonlinear_mixing = nonlinear_mixing
        self.nonlinear_dynamics = nonlinear_dynamics
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_output = num_output
        self.num_hidden_mixing = num_hidden_mixing
        self.num_layers_mixing = num_layers_mixing
        self.fixed = fixed
        self.fixed_output_fraction = fixed_output_fraction
        self.tau_neigh = tau_neigh
        self.hard_gumbel = hard_gumbel


class optimParams:
    def __init__(
        self,
        optimizer: str = "rmsprop",
        use_sparsity_constraint: bool = True,  # If False, use sparsity penalty
        crps_coeff: float = 1,  # Loss penalty coefficient for CRPS
        spectral_coeff: float = 20,  # for spatial spectrum
        temporal_spectral_coeff: float = 2000,  # for temporal spectrum
        coeff_kl: float = 1,  # for KL div
        reg_coeff: float = 0.01,  # for sparsity penalty if penalty
        reg_coeff_connect: float = 0,  # for cluster connectivity penalty if we want to enforce it
        fraction_highest_wavenumbers: float = None,
        fraction_lowest_wavenumbers: float = None,
        schedule_reg: int = 0,  # when we start adding penalties to the loss
        schedule_ortho: int = 0,  # when we start adding ortho constraint to the loss
        schedule_sparsity: int = 0,  # when we start adding sparsity constraint to the loss
        ortho_mu_init: float = 10_000,  # Initial orthogonality constraint coeff
        ortho_mu_mult_factor: float = 1.2,  # Multiply coeff by mult_factor every ortho_min_iter_convergence
        ortho_omega_gamma: float = 0.01,  # Not sure, related to ALM
        ortho_omega_mu: float = 0.9,  # Not sure, related to ALM
        ortho_h_threshold: float = 0.01,  # orthogonality threshold i.e. achieved when below this threshold
        ortho_min_iter_convergence: float = 1_000,  # orthogonality threshold i.e. achieved when below above threshold for at least ortho_min_iter_convergence
        sparsity_mu_init: float = 0.1,  # Below same aprams for sparsity  and acyclicity constraint
        sparsity_mu_mult_factor: float = 1.2,
        sparsity_omega_gamma: float = 0.01,
        sparsity_omega_mu: float = 0.95,
        sparsity_h_threshold: float = 0.0001,
        sparsity_min_iter_convergence: float = 1_000,
        sparsity_upper_threshold: float = 0.5,
        acyclic_mu_init: float = 1,
        acyclic_mu_mult_factor: float = 2,
        acyclic_omega_gamma: float = 0.01,
        acyclic_omega_mu: float = 0.9,
        acyclic_h_threshold: float = 1e-8,
        acyclic_min_iter_convergence: float = 1_000,
        mu_acyclic_init: float = 0,
        h_acyclic_threshold: float = 0,
    ):
        self.optimizer = optimizer
        self.use_sparsity_constraint = use_sparsity_constraint
        self.crps_coeff = crps_coeff
        self.spectral_coeff = spectral_coeff
        self.temporal_spectral_coeff = temporal_spectral_coeff
        self.coeff_kl = coeff_kl
        self.reg_coeff = reg_coeff
        self.reg_coeff_connect = reg_coeff_connect

        self.fraction_highest_wavenumbers = fraction_highest_wavenumbers
        self.fraction_lowest_wavenumbers = fraction_lowest_wavenumbers

        self.schedule_reg = schedule_reg
        self.schedule_ortho = schedule_ortho
        self.schedule_sparsity = schedule_sparsity

        self.ortho_mu_init = ortho_mu_init
        self.ortho_mu_mult_factor = ortho_mu_mult_factor
        self.ortho_omega_gamma = ortho_omega_gamma
        self.ortho_omega_mu = ortho_omega_mu
        self.ortho_h_threshold = ortho_h_threshold
        self.ortho_min_iter_convergence = ortho_min_iter_convergence

        self.sparsity_mu_init = sparsity_mu_init
        self.sparsity_mu_mult_factor = sparsity_mu_mult_factor
        self.sparsity_omega_gamma = sparsity_omega_gamma
        self.sparsity_omega_mu = sparsity_omega_mu
        self.sparsity_h_threshold = sparsity_h_threshold
        self.sparsity_min_iter_convergence = sparsity_min_iter_convergence
        self.sparsity_upper_threshold = sparsity_upper_threshold

        self.acyclic_mu_init = acyclic_mu_init
        self.acyclic_mu_mult_factor = acyclic_mu_mult_factor
        self.acyclic_omega_gamma = acyclic_omega_gamma
        self.acyclic_omega_mu = acyclic_omega_mu
        self.acyclic_h_threshold = acyclic_h_threshold
        self.acyclic_min_iter_convergence = acyclic_min_iter_convergence
        self.mu_acyclic_init = mu_acyclic_init
        self.h_acyclic_threshold = h_acyclic_threshold


class plotParams:
    def __init__(
        self, plot_freq: int = 500, plot_through_time: bool = True, print_freq: int = 500, savar: bool = False
    ):
        self.plot_freq = plot_freq
        self.plot_through_time = plot_through_time
        self.print_freq = print_freq
        self.savar = savar


class savarParams:
    # Params for generating synthetic data
    def __init__(
        self,
        time_len: int = 10_000,  # Time length of the data
        comp_size: int = 10,  # Each component size
        noise_val: float = 0.2,  # Noise variance relative to signal
        n_per_col: int = 2,  # square grid, equivalent of lat/lon
        difficulty: str = "easy",  # easy, med_easy, med_hard, hard: difficulty of the graph
        seasonality: bool = False,  # Seasonality in synthetic data
        overlap: bool = False,  # Modes overlap
        is_forced: bool = False,  # Forcings in synthetic data
        plot_original_data: bool = True,
    ):
        self.time_len = time_len
        self.comp_size = comp_size
        self.noise_val = noise_val
        self.n_per_col = n_per_col
        self.difficulty = difficulty
        self.seasonality = seasonality
        self.overlap = overlap
        self.is_forced = is_forced
        self.plot_original_data = plot_original_data


class rolloutParams:
    # Params for generating synthetic data
    def __init__(
        self,
        final_30_years_of_ssps: bool = True,  # Do prediction on the last years?
        batch_size: int = 10,  # number of initial conditions to look at the rollout on
        num_particles: int = 50,  # number of particles to propagate at each step
        num_particles_per_particle: int = 10,  # num particles to sample for each particle and compute fft
        num_timesteps: int = 1200,  # Time length of the prediction
        score: str = "log_bayesian",  # log_bayesian should be used
        tempering: bool = True,  # tempering the variance when sampling allows to propagate uncertainty
        sample_trajectories: bool = False,  # sample each trajectory separately
        batch_memory: bool = True,
    ):
        self.num_timesteps = num_timesteps
        self.final_30_years_of_ssps = final_30_years_of_ssps
        self.score = score
        self.tempering = tempering
        self.batch_size = batch_size
        self.num_particles = num_particles
        self.num_particles_per_particle = num_particles_per_particle
        self.sample_trajectories = sample_trajectories
        self.batch_memory = batch_memory
