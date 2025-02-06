from typing import List, Optional, Union

class expParams:
    def __init__(self, 
                 exp_path,
                 _target_,
                 latent: bool = True,
                 d_z: int = 90, 
                 d_x: int = 6250,
                 lon: int = 144, 
                 lat: int = 96,    
                 tau: int = 5,
                 random_seed: int = 1,
                 seed: int = 11, 
                 gpu: bool = True,
                 float: bool = False,
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 verbose: bool = True
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
        self.seed = seed
        self.gpu = gpu
        self.float = float
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.verbose = verbose

class dataParams:
    def __init__(self, 
                 data_dir,
                 icosahedral_coordinates_path,
                 train_historical_years, 
                 test_years,
                 train_years,
                 train_scenarios,
                 test_scenarios,
                 train_models,
                 in_var_ids,
                 out_var_ids,
                 num_ensembles: int = 1, 
                 num_levels: int = 1, 
                 seq_len: int = 12,
                 batch_size: int = 256,    
                 eval_batch_size: int = 256,
                 val_split: float = 0.1,
                 seasonality_removal: bool = False,
                 channels_last: bool = False,
                 ishdf5: bool = False,
                 data_format: str = "numpy",
                 seq_to_seq: bool = True,
                 train_val_interval_length: int = 11, 
                 load_train_into_mem: bool = True,
                 load_test_into_mem: bool = True,
                 num_months_aggregated: List[int] = [1],
                ):
        self.data_dir = data_dir
        self.icosahedral_coordinates_path = icosahedral_coordinates_path
        self.train_historical_years = train_historical_years
        self.test_years = test_years
        self.train_years = train_years
        self.train_scenarios = train_scenarios
        self.test_scenarios = test_scenarios
        self.train_models = train_models
        self.in_var_ids = in_var_ids
        self.out_var_ids = out_var_ids
        self.num_ensembles = num_ensembles
        self.num_levels = num_levels
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.val_split = val_split
        self.seasonality_removal = seasonality_removal
        self.channels_last = channels_last
        self.ishdf5 = ishdf5
        self.data_format = data_format
        self.seq_to_seq = seq_to_seq
        self.train_val_interval_length = train_val_interval_length
        self.load_train_into_mem = load_train_into_mem
        self.load_test_into_mem = load_test_into_mem
        self.num_months_aggregated = num_months_aggregated

class gtParams:
    def __init__(self, 
                 no_gt: bool = True,
                 debug_gt_z: bool = False,
                 debug_gt_w: bool = False,
                 debug_gt_graph: bool = False
                ):
        self.no_gt = no_gt
        self.debug_gt_z = debug_gt_z
        self.debug_gt_w = debug_gt_w
        self.debug_gt_graph = debug_gt_graph

class trainParams:
    def __init__(self, 
                 ratio_train: float = 0.9, 
                 batch_size: int = 6000, 
                 lr: float = 0.001,    
                 lr_scheduler_epochs: List[int] = [10000, 20000],
                 lr_scheduler_gamma: float = 1,    
                 max_iteration: int = 100000,
                 patience: int = 5000,
                 patience_post_thresh: int = 50,
                 valid_freq: int = 5
                ):
        self.ratio_train = ratio_train
        self.ratio_valid = 1-self.ratio_train
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_epochs = lr_scheduler_epochs
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.max_iteration = max_iteration
        self.patience = patience
        self.patience_post_thresh = patience_post_thresh
        self.valid_freq = valid_freq

class modelParams:
    def __init__(self, 
                 instantaneous: bool = False,
                 no_w_constraint: bool = False,
                 tied_w: bool = False,
                 nonlinear_mixing: bool = True,
                 num_hidden: int = 8, 
                 num_layers: int = 2,
                 num_output: int = 2, 
                 num_hidden_mixing: int = 16,    
                 num_layers_mixing: int = 2,
                 fixed: bool = False,
                 fixed_output_fraction = None, # Remove this? 
                 tau_neigh: int = 0,
                 hard_gumbel: bool = False
                ):
        self.instantaneous = instantaneous
        self.no_w_constraint = no_w_constraint
        self.tied_w = tied_w
        self.nonlinear_mixing = nonlinear_mixing
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
    def __init__(self, 
                 optimizer: str = "rmsprop",
                 crps_coeff: float = 1, 
                 spectral_coeff: float = 20, 
                 temporal_spectral_coeff: float = 2000,
                 coeff_kl: float = 1,   
                 reg_coeff: float = 0.01,   
                 reg_coeff_connect: float = 0,   

                 schedule_reg: int = 0,
                 schedule_ortho: int = 0,
                 schedule_sparsity: int = 0,

                 ortho_mu_init: float = 10_000,
                 ortho_mu_mult_factor: float = 1.2,
                 ortho_omega_gamma: float = 0.01,
                 ortho_omega_mu: float = 0.9,
                 ortho_h_threshold: float = 0.01,
                 ortho_min_iter_convergence: float = 1_000,

                 sparsity_mu_init: float = 0.1,
                 sparsity_mu_mult_factor: float = 1.2,
                 sparsity_omega_gamma: float = 0.01,
                 sparsity_omega_mu: float = 0.95,
                 sparsity_h_threshold: float = 0.0001,
                 sparsity_min_iter_convergence: float = 1_000,
                 sparsity_upper_threshold: float = 0.5,
                
                 # Is the below params needed? Only when instantaneous connections? 
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
        self.crps_coeff = crps_coeff
        self.spectral_coeff = spectral_coeff
        self.temporal_spectral_coeff = temporal_spectral_coeff
        self.coeff_kl = coeff_kl
        self.reg_coeff = reg_coeff
        self.reg_coeff_connect = reg_coeff_connect

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
    def __init__(self, 
                 plot_freq: int = 500,
                 plot_through_time: bool = True,
                 print_freq: int = 500
                ):
        self.plot_freq = plot_freq
        self.plot_through_time = plot_through_time
        self.print_freq = print_freq

