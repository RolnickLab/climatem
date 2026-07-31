from typing import List

from climatem.constants import SEQ_LEN_MAPPING


class expParams:
    """Experiment setup: paths, dimensions, random seed, and hardware config."""

    def __init__(
        self,
        exp_path,  # Path to where the output will be saved i.e. model runs, plots
        _target_: str = "climatem.data_loader.climate_datamodule.ClimateDataModule",
        latent: bool = True,  # Are you using latent variables or not (if not, learn causal variables between all observations)
        d_z: int = 90,  # Latent dimension
        d_z_global: int = 0,  # Higher-level latent dimension
        d_x: int = 6250,  # Observation dimension
        lon: int = 144,  # Longitude
        lat: int = 96,  # Latitude
        tau: int = 5,  # Number of timesteps
        future_timesteps: int = 1,  # Number of future timesteps to include in the future
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
        self.d_z_global = d_z_global
        self.d_x = d_x
        self.lon = lon
        self.lat = lat
        self.tau = tau
        self.future_timesteps = future_timesteps
        self.random_seed = random_seed
        self.gpu = gpu
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose


class dataParams:
    """Data loading: paths, scenarios, variables, batch size, and preprocessing options."""

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
        map_to_healpix: bool = False,
        global_normalization: bool = True,  # normalize the data?
        seasonality_removal: bool = False,  # deseasonalize the data?
        channels_last: bool = False,  # last dimension of data is the channel
        ishdf5: bool = False,  # numpy vs hdf5. for now only numpy is supported. Redundant with next param
        data_format: str = "numpy",  # numpy vs hdf5. for now only numpy is supported
        forcing_conditioning: str = "raw",  # how to condition on forcings: raw | template | mode | region (SAVAR)
        seq_to_seq: bool = True,  # predicting a sequence from a sequence?
        train_val_interval_length: int = 11,
        load_train_into_mem: bool = True,
        load_test_into_mem: bool = True,
        num_months_aggregated: List[int] = [
            1
        ],  # Aggregate num_months_aggregated months i.e. if you want yearly temporal resolution set this param to [12]
        **kwargs,  # accept any new keys in the parameter configs (for reloading the returned extra parameters)
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
        self.temp_res = temp_res
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.map_to_healpix = map_to_healpix
        self.global_normalization = global_normalization
        self.seasonality_removal = seasonality_removal
        self.channels_last = channels_last
        self.ishdf5 = ishdf5
        self.data_format = data_format
        self.forcing_conditioning = forcing_conditioning
        self.seq_to_seq = seq_to_seq
        self.train_val_interval_length = train_val_interval_length
        self.load_train_into_mem = load_train_into_mem
        self.load_test_into_mem = load_test_into_mem
        self.num_months_aggregated = num_months_aggregated


# # This class is only for debugging and for setting some params to the true aprams when training picabu
# class gtParams:
#     def __init__(
#         self,
#         no_gt: bool = True,  # do we have GT to compare? If synthetic data, will be True and overwritten
#         debug_gt_z: bool = False,  # below params help debugging the code when we have ground truth
#         debug_gt_w: bool = False,
#         debug_gt_graph: bool = False,
#     ):
#         self.no_gt = no_gt
#         self.debug_gt_z = debug_gt_z
#         self.debug_gt_w = debug_gt_w
#         self.debug_gt_graph = debug_gt_graph


class trainParams:
    """Training loop: learning rate, iterations, patience for phase transition, and validation frequency."""

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
        # here valid_freq is critical for updating the parameters of the ALM method as they get updated every valid_freq
        **kwargs,
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
    """Model architecture: latent dynamics type, MLP sizes, embedding, and causal mask options."""

    def __init__(
        self,
        instantaneous: bool = False,  # Allow instantaneous connections?
        instantaneous_forcing: bool = False,  # NEW: allow instantaneous  connection from forcing to climate
        no_w_constraint: bool = False,  # If True, no single parent assumption i.e. no causal graph
        tied_w: bool = False,  # NOT SURE, to clarify
        nonlinear_mixing: bool = True,  # If False, latent dynamics are linear
        num_hidden_mixing: int = 16,  # MLP params for latent dynamics if non-linear
        num_layers_mixing: int = 2,
        nonlinear_dynamics: bool = True,
        nonlinear_global_dynamics: bool = True,
        num_hidden: int = 8,  # MLP params for mapping from obs to latents. If 0, then linear. SHould add a flag as `nonlinear_mixing`
        num_layers: int = 2,
        num_output: int = 2,  # NOT SURE
        position_embedding_dim: int = 100,  # Dimension of positional embedding
        reduce_encoding_pos_dim: bool = False,  # Reduce encoder positional embedding dimension by x10
        tau_neigh: int = 0,  # Legacy neighborhood radius used in older configs
        hard_gumbel: bool = False,  # Legacy mask sampling flag used in analysis scripts
        transition_param_sharing: bool = True,
        position_embedding_transition: int = 100,
        fixed: bool = False,  # Do we fix the causal graph? Should be in gt_params maybe
        fixed_output_fraction=None,  # This is used if we fix the mask, and want to get a fix number of 0 and 1
        constraint_func: str = "trace",  # This is used for the constraint - trace is the correct one here
        use_exogenous: bool = True,  # NEW: Enable conditioning on exogenous forcings (CO2 + aerosols)
        d_y_co2: int = 1,  # NEW: Dimension of CO2 forcing (typically 1 for global, or spatial_dim for local)
        d_y_aerosol: int = 900,  # NEW: Dimension of aerosol forcing (typically spatial_dim for local effects),
        d_y_ch4: int = 0,  # NEW: Dimension of CH4 forcing (typically 1 for global, or spatial_dim for local)
        d_y_so2: int = 0,  # NEW: Dimension of SO2 forcing (typically spatial_dim for local effects)
        use_forced_latents: bool = True,  # NEW: Map forcings directly to dedicated latent dimensions
        n_forced_latents_co2: int = 1,  # NEW: Number of latent dimensions for CO2 forcing
        n_forced_latents_aerosol: int = 2,  # NEW: Number of latent dimensions for aerosol forcing
        n_forced_latents_ch4: int = 0,  # NEW: Number of latent dimensions for CO2 forcing
        n_forced_latents_so2: int = 0,  # NEW: Number of latent dimensions for aerosol forcing
        forcing_arch: str = "baseline",  # NEW: baseline | transitioned | predefined
        map_aerosol_to_climate: bool = False,  # Added: if only allow global impacts from aerosol to climate
        forcing_mse: bool = False,
        # first check if z_climate and z_aerosol are mapped to the same locations. If so, then allow transition from z_aerosol to z_climate
    ):
        self.instantaneous = instantaneous
        self.instantaneous_forcing = instantaneous_forcing
        self.no_w_constraint = no_w_constraint
        self.tied_w = tied_w
        self.nonlinear_mixing = nonlinear_mixing
        self.nonlinear_dynamics = nonlinear_dynamics
        self.nonlinear_global_dynamics = nonlinear_global_dynamics
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.num_output = num_output
        self.num_hidden_mixing = num_hidden_mixing
        self.num_layers_mixing = num_layers_mixing
        self.position_embedding_dim = position_embedding_dim
        self.reduce_encoding_pos_dim = reduce_encoding_pos_dim
        self.tau_neigh = tau_neigh
        self.hard_gumbel = hard_gumbel
        self.transition_param_sharing = transition_param_sharing
        self.position_embedding_transition = position_embedding_transition
        self.fixed = fixed
        self.fixed_output_fraction = fixed_output_fraction
        self.constraint_func = constraint_func
        self.use_exogenous = use_exogenous
        self.d_y_co2 = d_y_co2
        self.d_y_aerosol = d_y_aerosol
        self.d_y_ch4 = d_y_ch4
        self.d_y_so2 = d_y_so2
        self.use_forced_latents = use_forced_latents
        self.n_forced_latents_co2 = n_forced_latents_co2
        self.n_forced_latents_aerosol = n_forced_latents_aerosol
        self.n_forced_latents_ch4 = n_forced_latents_ch4
        self.n_forced_latents_so2 = n_forced_latents_so2
        self.forcing_arch = forcing_arch
        self.map_aerosol_to_climate = map_aerosol_to_climate
        self.forcing_mse = forcing_mse


class optimParams:
    """Optimization: loss coefficients, ALM penalty parameters, and constraint schedules."""

    def __init__(
        self,
        optimizer: str = "rmsprop",
        use_sparsity_constraint: bool = True,  # If False, use sparsity penalty
        binarize_transition: bool = True,  # If True, start adding the variance term of the matrix instead of the sparsity once constraint has been achieved
        crps_coeff: float = 1,  # Loss penalty coefficient for CRPS
        spectral_coeff: float = 20,  # for spatial spectrum
        temporal_spectral_coeff: float = 2000,  # for temporal spectrum
        coeff_kl: float = 1,  # for KL div
        loss_decay_future_timesteps: float = 1,  # if we predict more than 1 timestep, the loss ecay will reduce the weight of far away timesteps in the loss
        reg_coeff: float = 0.01,  # for sparsity penalty if penalty
        reg_coeff_connect: float = 0,  # for cluster connectivity penalty if we want to enforce it
        fraction_highest_wavenumbers: float = None,
        fraction_lowest_wavenumbers: float = None,
        take_log_spectra: bool = True,
        scheduler_spectra: List[
            int
        ] = None,  # the spectra term coefficient in the loss will be linearly increased from 0 to 1 if this is not None, ex: [0, 30_000, 50_000]
        schedule_reg: int = 0,  # when we start adding penalties to the loss
        schedule_ortho: int = 0,  # when we start adding ortho constraint to the loss
        schedule_sparsity: int = 0,  # when we start adding sparsity constraint to the loss
        ortho_mu_init: float = 10_000,  # Initial orthogonality constraint coeff
        ortho_mu_mult_factor: float = 1.2,  # Multiply coeff by mult_factor every ortho_min_iter_convergence
        ortho_omega_gamma: float = 0.01,  # Not sure, related to ALM
        ortho_omega_mu: float = 0.9,  # Not sure, related to ALM
        ortho_h_threshold: float = 0.01,  # orthogonality threshold i.e. achieved when below this threshold
        ortho_min_iter_convergence: float = 1_000,  # orthogonality threshold i.e. achieved when below above threshold for at least ortho_min_iter_convergence
        ortho_spatial_mu_init: float = 10_000,  # Initial orthogonality constraint coeff for spatial forcings
        ortho_spatial_mu_mult_factor: float = 1.2,  # Multiply coeff by mult_factor every ortho_min_iter_convergence for spatial forcings
        ortho_spatial_omega_gamma: float = 0.01,  # Not sure, related to ALM
        ortho_spatial_omega_mu: float = 0.9,  # Not sure, related to ALM
        ortho_spatial_h_threshold: float = 0.1,  # orthogonality threshold i.e. achieved when below this threshold for spatial forcings
        ortho_spatial_min_iter_convergence: float = 1_000,  # orthogonality threshold i.e. achieved when below above threshold for at least ortho_min_iter_convergence
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
        forcing_co2_coeff: float = 10.0,  # Weight for CO2 forcing reconstruction loss
        forcing_aerosol_coeff: float = 10.0,  # Weight for aerosol (BC) forcing reconstruction loss
        forcing_ch4_coeff: float = 10.0,  # Weight for CH4 forcing reconstruction loss
        forcing_so2_coeff: float = 10.0,  # Weight for SO2 forcing reconstruction loss
        gmst_coeff: float = 0,  # Weight for GMST loss
        forcing_latent_supervision_coeff: float = 10.0,  # Weight for direct forcing latent supervision loss
        decoder_utilization_coeff: float = 0.1,  # Penalty coefficient for underutilized forcing latent decoder weights
        min_forcing_decoder_norm: float = 1.5,  # Target minimum L2 norm for forcing latent decoder weights
        udpate_ALM_using_valid: bool = True,  # If False use training loss convergence if True uses valid loss convergence
        udpate_ALM_using_nll: bool = True,  # If False use augmented loss convergence if True uses NLL convergence
        update_ALM_spatial: bool = False,
    ):

        self.optimizer = optimizer
        self.use_sparsity_constraint = use_sparsity_constraint
        self.binarize_transition = binarize_transition
        self.crps_coeff = crps_coeff
        self.spectral_coeff = spectral_coeff
        self.temporal_spectral_coeff = temporal_spectral_coeff
        self.loss_decay_future_timesteps = loss_decay_future_timesteps
        self.coeff_kl = coeff_kl
        self.reg_coeff = reg_coeff
        self.reg_coeff_connect = reg_coeff_connect

        self.fraction_highest_wavenumbers = fraction_highest_wavenumbers
        self.fraction_lowest_wavenumbers = fraction_lowest_wavenumbers
        self.take_log_spectra = take_log_spectra
        self.scheduler_spectra = scheduler_spectra

        self.schedule_reg = schedule_reg
        self.schedule_ortho = schedule_ortho
        self.schedule_sparsity = schedule_sparsity

        self.ortho_mu_init = ortho_mu_init
        self.ortho_mu_mult_factor = ortho_mu_mult_factor
        self.ortho_omega_gamma = ortho_omega_gamma
        self.ortho_omega_mu = ortho_omega_mu
        self.ortho_h_threshold = ortho_h_threshold
        self.ortho_min_iter_convergence = ortho_min_iter_convergence

        self.ortho_spatial_mu_init = ortho_spatial_mu_init
        self.ortho_spatial_mu_mult_factor = ortho_spatial_mu_mult_factor
        self.ortho_spatial_omega_gamma = ortho_spatial_omega_gamma
        self.ortho_spatial_omega_mu = ortho_spatial_omega_mu
        self.ortho_spatial_h_threshold = ortho_spatial_h_threshold
        self.ortho_spatial_min_iter_convergence = ortho_spatial_min_iter_convergence

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

        self.forcing_co2_coeff = forcing_co2_coeff
        self.forcing_aerosol_coeff = forcing_aerosol_coeff
        self.forcing_ch4_coeff = forcing_ch4_coeff
        self.forcing_so2_coeff = forcing_so2_coeff
        self.gmst_coeff = gmst_coeff
        self.forcing_latent_supervision_coeff = forcing_latent_supervision_coeff
        self.decoder_utilization_coeff = decoder_utilization_coeff
        self.min_forcing_decoder_norm = min_forcing_decoder_norm

        self.udpate_ALM_using_valid = udpate_ALM_using_valid
        self.udpate_ALM_using_nll = udpate_ALM_using_nll
        self.update_ALM_spatial = update_ALM_spatial


class plotParams:
    """Plotting frequency and toggle options for training diagnostics."""

    def __init__(
        self, plot_freq: int = 500, plot_through_time: bool = True, print_freq: int = 500, savar: bool = False
    ):
        self.plot_freq = plot_freq
        self.plot_through_time = plot_through_time
        self.print_freq = print_freq
        self.savar = savar


class savarParams:
    """
    Configuration for SAVAR synthetic data generation.

    Controls all aspects of the Seasonal Vector Auto-Regressive data generator:
    spatial grid, temporal length, causal graph structure, seasonality, external
    forcing (CO2 + aerosol), noise characteristics, and background state.
    See ``climatem/synthetic_data/savar.py`` for the generator implementation.
    """

    def __init__(
        self,
        # Basic data generation parameters
        time_len: int = 10_000,  # Total number of timesteps to generate (longer = more data for training)
        comp_size: int = 10,  # Size of each spatial component/mode
        noise_val: float = 0.02,  # Noise strength relative to signal (higher = noisier data)
        n_per_col: int = 2,  # Number of grid points per row/column in square spatial grid (total spatial size = n_per_col^2 * comp_size)
        # Causal graph structure
        difficulty: str = "easy",  # Complexity of causal graph: "easy" (sparse), "med_easy", "med_hard", "hard" (dense/complex)
        # Seasonality parameters
        seasonality: bool = False,  # Whether to add seasonal variations (e.g., annual cycles like climate data)
        periods: List[float] = [
            365,
            182.5,
            60,
        ],  # Seasonal periods in days (e.g., annual=365, semi-annual=182.5, bi-monthly=60)
        amplitudes: List[float] = [0.06, 0.02, 0.01],  # Amplitude of each seasonal component (matched to periods list)
        phases: List[float] = [
            0.0,
            0.7853981634,
            1.5707963268,
        ],  # Phase shifts for seasonality in radians (0, π/4, π/2)
        yearly_jitter_amp: float = 0.05,  # Year-to-year random variation in seasonal amplitude (adds realism)
        yearly_jitter_phase: float = 0.10,  # Year-to-year random variation in seasonal phase (adds realism)
        # Spatial structure
        overlap: float = 0,  # Whether spatial modes can overlap between 0 and 1 (True = modes share spatial regions)
        # External forcing parameters
        is_forced: bool = True,  # Whether to include external forcings like CO2 and aerosols (mimics climate change)
        f_1: int = 0,  # Initial forcing value at start of ramp (baseline level). NOTE: used as float downstream
        f_2: int = 1,  # Final forcing value at end of ramp (target level). NOTE: used as float downstream
        f_time_1: int = 4000,  # Timestep when forcing ramp begins (relative to start after transient)
        f_time_2: int = 8000,  # Timestep when forcing ramp ends and forcing becomes constant at f_2
        ramp_type: str = "linear",  # Temporal evolution of forcing: "linear", "quadratic", "exponential", "sigmoid", "sinusoidal"
        # Dynamics type
        linearity: str = "linear",  # Type of dynamics: "linear" (VAR model), "polynomial", or "nonlinear" (neural net)
        poly_degrees: List[int] = [
            2
        ],  # Polynomial degrees to use if linearity="polynomial" (e.g., [2] for quadratic, [2,3] for quad+cubic)
        # Visualization
        plot_original_data: bool = True,  # Whether to generate plots during data generation
        # Separate forcing fields (more realistic than single forcing)
        use_separate_forcings: bool = True,  # Use distinct CO2 and aerosol forcing fields with different dynamics
        forcing_amplification: float = 1.2,  # Overall scaling factor for forcing magnitudes
        # Aerosol forcing parameters
        aerosol_scale: float = 0.02,  # Strength of aerosol forcing (typically negative for cooling effect, positive here for magnitude)
        aerosol_spatial_contrast: float = 1.05,  # Regional variability of aerosol effects (>1 increases heterogeneity across space)
        aerosol_ramp_up_time: int = 2000,  # When aerosol forcing starts increasing (default: 20% of time_len)
        aerosol_peak_time: int = 5000,  # When aerosol forcing reaches maximum (default: 50% of time_len)
        aerosol_decline_time: int = 8000,  # When aerosol forcing finishes declining to baseline (default: 80% of time_len)
        aerosol_timing_stagger: float = 0.3,  # Fraction of timeline to stagger aerosol latents (creates distinct temporal patterns per latent)
        # Forcing causal structure parameters
        n_co2_latents: int = 1,  # Number of latent variables representing CO2 forcing in causal graph (typically 1 for global)
        n_aerosol_latents: int = 2,  # Number of latent variables representing aerosol forcing (multiple for regional effects)
        co2_effect_strength: float = 0.25,  # Causal coefficient strength for CO2 → climate mode links (larger = stronger influence)
        aerosol_effect_strength: float = 0.20,  # Causal coefficient strength for aerosol → climate mode links (larger = stronger influence)
        # Noise temporal correlation (AR(1) / Ornstein-Uhlenbeck)
        noise_ar1_rho: float = 0.95,  # AR(1) persistence parameter ρ (0=white noise, 0.95=realistic red noise). Can also be "decay" for mode-dependent ρₖ = exp(-k/K)
        noise_ar1: bool = True,  # Use AR(1) (red) noise instead of white noise for realistic temporal correlations
        # Background state parameters
        enable_background: bool = False,  # Whether to add low-frequency background state (slow climate mean state drift)
        background_strength: float = 0.3,  # Strength relative to mode std (if < 1 and mode="relative") or absolute magnitude
        background_strength_mode: str = "relative",  # "relative" to mode std or "absolute"
        background_smoothness: float = 0.15,  # Controls spatial frequency (higher = smoother spatial patterns)
        background_timescale_rho: float = 0.995,  # AR(1) persistence (higher = slower temporal evolution, 0.995 ≈ 200 step timescale)
        background_n_modes: int = 3,  # Number of low-frequency Fourier components for spatial smoothness
        use_correct_hyperparams: bool = True,  # Override some of the model params to match those of savar data if true
    ):
        self.time_len = time_len
        self.comp_size = comp_size
        self.noise_val = noise_val
        self.n_per_col = n_per_col
        self.difficulty = difficulty
        self.seasonality = seasonality
        self.periods = periods
        self.amplitudes = amplitudes
        self.phases = phases
        self.yearly_jitter_amp = yearly_jitter_amp
        self.yearly_jitter_phase = yearly_jitter_phase
        self.overlap = overlap
        self.is_forced = is_forced
        self.f_1 = f_1
        self.f_2 = f_2
        self.f_time_1 = f_time_1
        self.f_time_2 = f_time_2
        self.ramp_type = ramp_type
        self.linearity = linearity
        self.poly_degrees = poly_degrees
        self.plot_original_data = plot_original_data
        self.use_separate_forcings = use_separate_forcings
        self.forcing_amplification = forcing_amplification
        self.aerosol_scale = aerosol_scale
        self.aerosol_spatial_contrast = aerosol_spatial_contrast
        self.aerosol_ramp_up_time = aerosol_ramp_up_time
        self.aerosol_peak_time = aerosol_peak_time
        self.aerosol_decline_time = aerosol_decline_time
        self.aerosol_timing_stagger = aerosol_timing_stagger
        # Forcing causal structure
        self.n_co2_latents = n_co2_latents
        self.n_aerosol_latents = n_aerosol_latents
        self.co2_effect_strength = co2_effect_strength
        self.aerosol_effect_strength = aerosol_effect_strength
        # Noise temporal correlation
        self.noise_ar1_rho = noise_ar1_rho
        self.noise_ar1 = noise_ar1
        # Background state parameters
        self.enable_background = enable_background
        self.background_strength = background_strength
        self.background_strength_mode = background_strength_mode
        self.background_smoothness = background_smoothness
        self.background_timescale_rho = background_timescale_rho
        self.background_n_modes = background_n_modes
        self.use_correct_hyperparams = use_correct_hyperparams


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
