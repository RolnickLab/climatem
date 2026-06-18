import numpy as np
import torch
from tqdm import trange
"""
bayesian filter with multi-variate version (default using the ground truth forcings)
the spectral is computed using only the climate variable (ts only)
"""
def calculate_fft_mean_std_across_all_noresm(datamodule, accelerator):

    # Start again at the beginning of the dataloader.
    train_dataloader = iter(datamodule.train_dataloader(accelerator))

    # iterate through the data and append all the y values together
    y_all = []
    for i in range(len(train_dataloader)):
        batch = next(train_dataloader)
        if isinstance(batch, dict):
            # New format with forcings
            y_whole_dataloader = batch["y"]
        else:
            # Legacy format (tuple)
            _, y_whole_dataloader = batch
        y_all.append(y_whole_dataloader[:, 0, 0].unsqueeze(1)) # only take the first variable and unsqueeze to keep the variable dim
    y_all = torch.cat(y_all, dim=0)
    y_all = torch.nan_to_num(y_all)

    # make sure we reset the dataloader
    train_dataloader = iter(datamodule.train_dataloader(accelerator))

    y_true_fft_data = torch.abs(torch.fft.rfft(y_all[:, :, :], dim=2))

    # calculate the mean and std of the fft of the true data across all the data
    y_true_fft_mean = y_true_fft_data.mean(dim=0)
    y_true_fft_std = y_true_fft_data.std(dim=0)

    return y_true_fft_mean, y_true_fft_std



def logscore_the_samples_for_spatial_spectra_bayesian(
    y_true_fft_mean,
    y_true_fft_std,
    y_pred_samples,
    coords: np.ndarray,
    sigma: float = 1.0,
    num_particles: int = 100,
    batch_size: int = 64,
    distribution_spatial_spectra: str = "laplace",
    tempering: bool = False,
):
    """
    Calculate the spatial spectra of the true values and the predicted values, and then calculate a score between them.
    This is a measure of how well the model is predicting the spatial spectra of the true values.

    Args:
        true_values: torch.Tensor, observed values in a batch
        y_pred: torch.Tensor, a selection of predicted values
        num_particles: int, the number of samples that have been taken from the model
    """

    fft_pred = torch.abs(torch.fft.rfft(y_pred_samples[:, :, :], dim=3))

    # extend fft_true so it is the same value but extended to the same shape as fft_pred
    fft_true = y_true_fft_mean.repeat(num_particles, batch_size, 1, 1)
    fft_true_std = y_true_fft_std.repeat(num_particles, batch_size, 1, 1)

    if fft_pred.dim() == fft_true.dim() + 1:
#         print("I am flattening the preds here.")
        fft_pred = torch.flatten(fft_pred, start_dim=0, end_dim=1)

    assert fft_true.shape == fft_pred.shape
    assert fft_true_std.shape == fft_pred.shape # ([500, 8, 5, 1537])
    if distribution_spatial_spectra == "laplace":
        spatial_spectra_score = torch.abs((fft_pred - fft_true) / (fft_true_std))
    elif distribution_spatial_spectra == "gaussian":
        spatial_spectra_score = ((fft_pred - fft_true) ** 2) / (2 * fft_true_std**2)

#     print("Spatial spectra score shape before summing:", spatial_spectra_score.shape)

    spatial_spectra_score = -torch.sum(spatial_spectra_score, dim=(2, 3))
    if tempering: 
#         spatial_spectra_score /= y_true_fft_mean.shape[1]
        # print(f"shape of FFT mean is {y_true_fft_mean.shape} and dim 1 is {y_true_fft_mean.shape[1]}")
        spatial_spectra_score /= np.sqrt(y_true_fft_mean.shape[1])

#     print("The spatial spectra score shape should be (num_particles, num_batch_size):", spatial_spectra_score.shape)
    # score = ...
    return spatial_spectra_score


def particle_filter_weighting_bayesian(
    model,
    x,
    y,
    y_true_fft_mean,
    y_true_fft_std,
    coordinates,
    num_particles: int = 100,
    num_particles_per_particle: int = 10,
    timesteps: int = 120,
    score: str = "variance",
    save_dir: str = None,
    save_name: str = None,
    batch_size: int = 16,
    tempering: bool = False,
    sample_trajectories: bool = False,
    batch_memory: bool = False,
    all_y_ssp= None
):
    """
    Implement a particle filter to make a set of autoregressive predictions, where each created sample is evaluated by
    some score, and we do a particle filter to select only best samples to continue the autoregressive rollout.

    We need to pass the directory to save stuff to, and the stem of the filenames...
    if batch_memory: will loop over initial conditions (batch_size)
    else: no loop, faster but much higher memory

    TODO: REMOVE FOR LOOP OVER BATCH - torch/model can deal with the additional row?
    TODO: Code is quite confusing because here x is latent and z is reconstruction + y is fixed obs corresponding to FFT
    """

    print("Initial number of particles:", num_particles)
    num_variables = x.shape[2]
    for t in trange(timesteps):

        # Prediction
        # make all the new predictions, taking samples from the latents

        if t == 0:
            print("This is the first timestep, so I am going to generate samples from the initial latents.")
            if score == "log_bayesian":
                print(f"x shape {x.shape}") # (8, 5, d, 3072)
                print(f"y shape {y.shape}") #(8, d, 3072)

                if not batch_memory:
                    unused_samples_from_xs, samples_from_zs, y, logscore_samples_fromzs, pz2_mu = (
                        model.predict_sample_bayesianfiltering(
                            x, y, num_particles * num_particles_per_particle, with_zs_logprob=True,
                        )
                    )
                    torch.cuda.empty_cache()
                    logscore_samples_fromzs = torch.sum(logscore_samples_fromzs, -1).squeeze(2)
                    if tempering: 
                        logscore_samples_fromzs /= np.sqrt(model.d_z)
                else:
                    batch_size = x.shape[0]
                    samples_from_zs = []
                    logscore_samples_fromzs = []
                    pz2_mu = []
                    forcings = []
                    for k in range(batch_size):
                        print(f"processing {k}")
                        unused_samples_from_xs, samples_from_zs_batch_, unused_y, logscore_samples_fromzs_batch, pz2_mu_batch = (
                            model.predict_sample_bayesianfiltering(
                                x[k][None], y[k][None], num_particles * num_particles_per_particle, with_zs_logprob=True,
                            )
                        )
                        samples_from_zs_batch = samples_from_zs_batch_[:,:,:1] #only take the first variable
                        if num_variables > 1:
                            forcings_batch = samples_from_zs_batch_[:,:,1:]
                        # print("samples_from_zs_batch",samples_from_zs_batch.shape) #[500, 1, 5, 3072]
                        # print("logscore_samples_fromzs_batch",logscore_samples_fromzs_batch.shape) #[500, 1, 5, 60]
                        # we only care about the 1st variable's score
                        logscore_samples_fromzs_batch = logscore_samples_fromzs_batch[:,:,:1]  #only take the first variable's scpre
                        # samples_from_zs_batch ([500, 1, 1, 3072]) pz2_mu_batch torch.Size([1, 1, 1])
                        torch.cuda.empty_cache()
                        logscore_samples_fromzs_batch = torch.sum(logscore_samples_fromzs_batch, -1).squeeze(2)
                        if tempering: 
                            logscore_samples_fromzs_batch /= np.sqrt(model.d_z)
                        samples_from_zs.append(samples_from_zs_batch)
                        if num_variables > 1:
                            forcings.append(forcings_batch)
                        logscore_samples_fromzs.append(logscore_samples_fromzs_batch)
                        pz2_mu.append(pz2_mu_batch)
                    samples_from_zs = torch.cat(samples_from_zs, dim=1) #(500,8,1,3072)
                    if num_variables > 1:
                        forcings = torch.cat(forcings, dim=1)
                    logscore_samples_fromzs = torch.cat(logscore_samples_fromzs, dim=-1)[None]
                    pz2_mu= torch.cat(pz2_mu, dim=0)[None] #(1,8,1,1)

#                 print(f"unused_samples_from_xs shape {unused_samples_from_xs.shape}")
#                 print(f"samples_from_zs shape {samples_from_zs.shape}")
            else:
                unused_samples_from_xs, samples_from_zs, y = model.predict_sample_bayesianfiltering(
                    x, y, num_particles * num_particles_per_particle, with_zs_logprob=False,
                )

        else:
            # print("Not the first timestep, so generating samples using initial particles.")
            # px_mu, y, z, pz_mu, pz_std = model.predict(x, y, num_particles)
            # note, here I think x is no. of samples - dimensional
            # REMOVE THIS FOR LOOP IF POSSIBLE
#             for i in trange(num_particles):
#                 print(f"Generating mean sample for particle {i}")
                # px_mu, y, z, pz_mu, pz_std = model.predict(x[:, i, :, :], y[i, :, :])

                # New code
                # Here for each particle at time t predict num_particles_per_particle at time t+1
            
            assert x.ndim == 5
            assert y.ndim == 3
            
            # print(f"x.shape {x.shape}")
            # print(f"y.shape {y.shape}")

            x_reshaped = x.reshape((-1, x.shape[2], x.shape[3], x.shape[4]))
            y_reshaped = y.repeat(x.shape[0], 1, 1, 1).reshape((-1, y.shape[1], y.shape[2]))
            # print(f"x_reshaped.shape {x_reshaped.shape}")
            # print(f"y_reshaped.shape {y_reshaped.shape}")
            if score == "log_bayesian":
                if not batch_memory:
                    unused_samples_from_xs, samples_from_zs, y_reshaped, logscore_samples_fromzs, pz2_mu = (
                        model.predict_sample_bayesianfiltering(
                            x_reshaped, y_reshaped, num_particles_per_particle, with_zs_logprob=True,
                        )
                    ) # finds n_particles_per_particle * n_particles, here, for each k in n_particles the corresponding n_particles_per_particle are in [k, k+n_particles, ..., k+n_particles_per_particle*n_particles]
                    torch.cuda.empty_cache()
                    logscore_samples_fromzs = torch.sum(logscore_samples_fromzs, -1).squeeze()
                    if tempering: 
                        logscore_samples_fromzs /= np.sqrt(model.d_z)
                    logscore_samples_fromzs = logscore_samples_fromzs.reshape((logscore_samples_fromzs.shape[0], x.shape[0], x.shape[1]))
                else:
                    samples_from_zs = []
                    logscore_samples_fromzs = []
                    pz2_mu=[]
                    forcings=[]
                    for k in range(batch_size):
                        if all_y_ssp is not None:
                            input_y =all_y_ssp[t,k].repeat(x.shape[0], 1, 1)
                        else:
                            input_y = y[k].repeat(x.shape[0], 1, 1)
                        unused_samples_from_xs, samples_from_zs_batch_, unused_y_reshaped, logscore_samples_fromzs_batch, pz2_mu_batch = (
                            model.predict_sample_bayesianfiltering(
                                x[:, k], input_y, num_particles_per_particle, with_zs_logprob=True,
                            )
                        ) # finds n_particles_per_particle * n_particles, here, for each k in n_particles the corresponding n_particles_per_particle are in [k, k+n_particles, ..., k+n_particles_per_particle*n_particles]
                        # samples_from_zs_batch [10, 50, 3072]-> will select the best one particle out of 10
                        samples_from_zs_batch = samples_from_zs_batch_[:,:,:1] # only take the 1st variable
                        if num_variables > 1:
                            forcings_batch = samples_from_zs_batch_[:,:,1:]
                        logscore_samples_fromzs_batch = logscore_samples_fromzs_batch[:,:,:1]
                        # pz2_mu_batch [50, 1, 1]
                    
                        torch.cuda.empty_cache()
                        logscore_samples_fromzs_batch = torch.sum(logscore_samples_fromzs_batch, -1).squeeze()
                        if tempering: 
                            logscore_samples_fromzs_batch /= np.sqrt(model.d_z)
                        logscore_samples_fromzs_batch = logscore_samples_fromzs_batch.reshape((logscore_samples_fromzs_batch.shape[0], x.shape[0]))
                        # print(f"logscore_samples_fromzs_batch shape {logscore_samples_fromzs_batch.shape}")
                        # print(f"should be npp*np")
                        logscore_samples_fromzs.append(logscore_samples_fromzs_batch[:, None])
                        samples_from_zs.append(samples_from_zs_batch[:, :, None])
                        if num_variables > 1:
                            forcings.append(forcings_batch[:, :, None])
                        pz2_mu.append(pz2_mu_batch[:, None])
                    samples_from_zs = torch.cat(samples_from_zs, dim=2) # shape (10, 50, 8, 1, 3072)
                    if num_variables > 1:
                        forcings= torch.cat(forcings, dim=2)
                    # print("should be npp*np*bs*1*6250")
                    # samples_from_zs = samples_from_zs.reshape((-1, samples_from_zs.shape[2], samples_from_zs.shape[3], samples_from_zs.shape[4]))
                    logscore_samples_fromzs = torch.cat(logscore_samples_fromzs, dim=-1)
                    logscore_samples_fromzs = logscore_samples_fromzs.reshape((-1, x.shape[0], x.shape[1]))
                    pz2_mu = torch.cat(pz2_mu, dim=1)
                    # print(f"samples_from_zs shape {samples_from_zs.shape}")
                    # print(f"should be 85, 3, 1, 6250")
                    # print(f"logscore_samples_fromzs shape {logscore_samples_fromzs.shape}")
                    # print(f"should be npp, np, bs")

            else:
                samples_from_zs, y, unused_z, unused_pz_mu, unused_pz_std, pz2_mu = model.predict(x_reshaped, y_reshaped)
            
            if not batch_memory: 
                samples_from_zs = samples_from_zs.reshape((samples_from_zs.shape[0], x.shape[0], x.shape[1], samples_from_zs.shape[2], samples_from_zs.shape[3]))

# if i == 0:
#                 samples_from_zs = next_sample_from_zs.unsqueeze(0)
#                 logscore_samples_fromzs = next_logscore_samples_fromzs.unsqueeze(0)
#             else:
#                 samples_from_zs = torch.cat([samples_from_zs, next_sample_from_zs.unsqueeze(0)], dim=0)
#                 logscore_samples_fromzs = torch.cat(
#                     [logscore_samples_fromzs, next_logscore_samples_fromzs.unsqueeze(0)], dim=0
#                 )
# #             next_sample_from_zs.reshape()

        # then calculate the score of each of the samples
        # Update the weights, where we want the weights to increase as the score improves

        if score == "spatial_spectra":
            new_weights = score_the_samples_for_spatial_spectra(
                y,
                samples_from_zs,
                coords=coordinates,
                num_particles=num_particles * num_particles_per_particle,
                mid_latitudes=True,
            )
        elif score == "log_bayesian":
            # This is [K, L, batch size]
            # print(f"y shape {y.shape}")
            if t > 0:
                # In correct dimension?? should be
                logscore_samples_fromzs = torch.flatten(logscore_samples_fromzs, start_dim=0, end_dim=1)
                samples_from_zs = torch.flatten(samples_from_zs, start_dim=0, end_dim=1)
                if num_variables > 1:
                    forcings = torch.flatten(forcings, start_dim=0, end_dim=1)
#             else:
#                 samples_from_zs = samples_from_zs.unsqueeze(2)
            # print(f"samples_from_zs shape {samples_from_zs.shape}")
            # print(f"y_true_fft_mean shape {y_true_fft_mean.shape}")
            # This is [K*L, batch size, 1, 6250]--> Is this expected?
            # Then fft_true shape after repeating: torch.Size([K*L, batch size, 1, 3126]
            scores_spatial_spectra = logscore_the_samples_for_spatial_spectra_bayesian(
                y_true_fft_mean,
                y_true_fft_std,
                samples_from_zs,
                coords=coordinates,
                num_particles=num_particles * num_particles_per_particle,
                batch_size=batch_size,
                tempering=tempering,
            )
#             print(f"spatial_spectra shape {scores_spatial_spectra.shape}")#num_samples, bs
            # This is [K*L, batch size]
            new_weights = logscore_samples_fromzs + scores_spatial_spectra
            if new_weights.ndim == 3:
                new_weights = new_weights[0]
            # print(f"new_weights.shape {new_weights.shape}")
        #             new_weights = torch.exp(new_weights) # Here we might be able to sample directly from the log probabilities in torch to avoid taking the exp
        else:
            raise ValueError("Score must be either variance or spatial_spectra")

#         print("New log weights are higher is better if log_bayesian otherwise lower...")
#         print("Shape of new weights:", new_weights.shape)

#         #         print('Minimum of the new weights, along the first dimension:', torch.min(new_weights, dim=0))
#         #         print('Maximum of the new weights, along the 0th dimension:', torch.max(new_weights, dim=0))
#         print("What is the shape of the min calculated above:", torch.min(new_weights, dim=0).values.shape)

        # normalise the weights along the first dimension

        # NO NEED FOR THIS IF sample_trajectories (and no need for tempering)
        # TODO below this!!
        max_weight = torch.max(new_weights, dim=0)
        if score != "log_bayesian":
            min_weight = torch.min(new_weights, dim=0)
        else:
            # might get overflows here - might need to clip...for torch.exp
            new_weights = torch.exp(new_weights - max_weight.values)
        new_weights = new_weights / torch.sum(new_weights, dim=0)
        # clip the new_weights to avoid numerical instability
        new_weights = torch.clamp(new_weights, min=1e-8, max=1.0)
        new_weights = new_weights / torch.sum(new_weights, dim=0)

        # REMOVE THIS FOR LOOP IF POSSIBLE
#         print(f"For loop batch resample batch_size {batch_size}")
        if not sample_trajectories:
            resampled_indices = torch.multinomial(new_weights.T, num_particles, replacement=True).T
        else:
            # Here, every num_particles_per_particle we should sample one i.e. we track each trajectory
            resampled_indices = torch.zeros([num_particles, batch_size], dtype = torch.long)
            for k in range(num_particles):
                idx_trajectory = torch.arange(k, k+(num_particles_per_particle)*num_particles, num_particles)
                resampled_indices[k] = idx_trajectory[new_weights[idx_trajectory].argmax(0)]

        # for i in range(batch_size):

        #     resampled_indices = torch.multinomial(new_weights[:, i], num_particles, replacement=True)
        #     # append these resampled indices to n array so we get an output of shape (5, batch_size)
        #     if i == 0:
        #         resampled_indices_array = resampled_indices.unsqueeze(1)
        #     else:
        #         resampled_indices_array = torch.cat([resampled_indices_array, resampled_indices.unsqueeze(1)], dim=1)

        # Use list comprehension to collect resampled indices for each column
        # resampled_indices_array2 = torch.stack([torch.multinomial(new_weights[:, i], num_particles, replacement=True) for i in range(256)], dim=1)

        # assert that the two resampled indices are the same
        # assert torch.all(resampled_indices_array == resampled_indices_array2)
        # samples_from_zs [500, 8, 1, 3072] forcings[500, 8, 4, 3072]
        if num_variables > 1:
            samples_from_zs_full = torch.cat([samples_from_zs, forcings], dim=2)
        else:
            samples_from_zs_full  = samples_from_zs
        selected_samples = samples_from_zs_full[resampled_indices, torch.arange(batch_size)]
        if t == 0:
            pz2_mu = pz2_mu.repeat(num_particles, 1, 1, 1)
        # selected_samples_global = pz2_mu[resampled_indices, torch.arange(batch_size)]
        np.save(save_dir / f"{save_name}_{t}.npy", selected_samples.detach().cpu().numpy())
        if getattr(model, "d_z_global", None) is not None:
            np.save(save_dir / f"{save_name}_{t}_global.npy", pz2_mu.detach().cpu().numpy())
#         print("Saved the selected samples with name:", f"{save_name}_{_}.npy")

        if t == 0:
            x = x.repeat(num_particles, 1, 1, 1, 1)
#             print("Shape of x after repeating, in the first timestep:", x.shape)

        x = x[:, :, 1:, :, :]

        # now we just need to unsqueeze the selected samples, so that we can concatenate them to x
        selected_samples = selected_samples.unsqueeze(2)
        # pz2_mu will not be reused for rollout
        

#         print("What is the shape of x, just before we concatenate?", x.shape)

        # then we need to append the selected samples to x, along the right axis
        # Here shouldn't it be the unused samples fromxs???
        x = torch.cat([x, selected_samples], dim=2)

        # then we are going back to the top of the loop

    return selected_samples

