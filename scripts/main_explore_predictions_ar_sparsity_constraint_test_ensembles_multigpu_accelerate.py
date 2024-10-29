# Here we have a quick main where we are testing data loading with different ensemble members and ideally with different climate models.

import argparse
import time
import sys
import warnings
import os
import json
import torch
import numpy as np
from metrics import mcc_latent, shd, precision_recall, edge_errors, w_mae
from model.tsdcd import TSDCD
from model.tsdcd_latent_explore import LatentTSDCD

# from data_loader import DataLoader

# first replace this with the modified data loader
#from climate_data_loader import CausalClimateDataModule
from climate_data_loader_test_ensembles_multigpu import CausalClimateDataModule

from train import Training

# NOTE: here I am working with the use of the constrained graph sparsity:

# this does work to some extent...
#from train_latent_constrain_graph import TrainingLatent

from train_latent_constrain_graph_multigpu import TrainingLatent

from accelerate import Accelerator

accelerator = Accelerator()


class Bunch:
    """
    A class that has one variable for each entry of a dictionary.
    """

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def to_dict(self):
        return self.__dict__

    # def fancy_print(self, prefix=''):
    #     str_list = []
    #     for key, val in self.__dict__.items():
    #         str_list.append(prefix + f"{key} = {val}")
    #     return '\n'.join(str_list)


def main(hp):
    """
    :param hp: object containing hyperparameter values
    """
    t0 = time.time()

    # Control as much randomness as possible
    torch.manual_seed(hp.random_seed)
    np.random.seed(hp.random_seed)

    # Use GPU
    # TODO: Make everything Double instead of FLoat on GPU
    if hp.gpu:
        if hp.float:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        if hp.float:
            torch.set_default_tensor_type("torch.FloatTensor")
        else:
            torch.set_default_tensor_type("torch.FloatTensor")

    # Create folder
    # args.exp_path = os.path.join(args.exp_path, f"exp{args.exp_id}")
    # if not os.path.exists(args.exp_path):
    #     os.makedirs(args.exp_path)

    # generate data and split train/test
    if hp.gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    config_fname = hp.config_exp_path
    with open(config_fname) as f:
        data_params = json.load(f)

    if hp.ishdf5:
        print("IS HDF5")
        return
    else:
        datamodule = CausalClimateDataModule(**data_params) #...
        datamodule.setup()

    train_dataloader = iter(datamodule.train_dataloader())
     # val_dataloader = iter(datamodule.val_dataloader())

    x, y = next(train_dataloader)
    # data_loader = DataLoader(ratio_train=hp.ratio_train,
    #                          ratio_valid=hp.ratio_valid,
    #                          data_path=hp.data_path,
    #                          data_format=hp.data_format,
    #                          latent=hp.latent,
    #                          no_gt=hp.no_gt,
    #                          debug_gt_w=hp.debug_gt_w,
    #                          instantaneous=hp.instantaneous,
    #                          tau=hp.tau, device=device)

    # initialize model
    d = x.shape[2]
    print("d:", d)

    if hp.instantaneous:
        print('Using instantaneous connections')
        num_input = d * (hp.tau + 1) * (hp.tau_neigh * 2 + 1)
    else:
        num_input = d * hp.tau * (hp.tau_neigh * 2 + 1)

    if not hp.latent:
        model = TSDCD(model_type="fixed",
                      num_layers=hp.num_layers,
                      num_hidden=hp.num_hidden,
                      num_input=num_input,
                      num_output=2,
                      d=d,
                      tau=hp.tau,
                      tau_neigh=hp.tau_neigh,
                      instantaneous=hp.instantaneous,
                      hard_gumbel=hp.hard_gumbel)
    else:
        model = LatentTSDCD(num_layers=hp.num_layers,
                            num_hidden=hp.num_hidden,
                            num_input=num_input,
                            num_output=2,
                            num_layers_mixing=hp.num_layers_mixing,
                            num_hidden_mixing=hp.num_hidden_mixing,
                            coeff_kl=hp.coeff_kl,
                            d=d,
                            distr_z0="gaussian",
                            distr_encoder="gaussian",
                            distr_transition="gaussian",
                            distr_decoder="gaussian",
                            d_x=hp.d_x,
                            d_z=hp.d_z,
                            tau=hp.tau,
                            instantaneous=hp.instantaneous,
                            nonlinear_mixing=hp.nonlinear_mixing,
                            hard_gumbel=hp.hard_gumbel,
                            no_gt=hp.no_gt,
                            debug_gt_graph=hp.debug_gt_graph,
                            debug_gt_z=hp.debug_gt_z,
                            debug_gt_w=hp.debug_gt_w,
                            # gt_w=data_loader.gt_w,
                            # gt_graph=data_loader.gt_graph,
                            tied_w=hp.tied_w,
                            # NOTE: seb adding fixed to try to test when we have a fixed graph
                            # also 
                            fixed=hp.fixed,
                            fixed_output_fraction=hp.fixed_output_fraction)

    #print("Trying to move model to GPU...")
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    #print("Where is my model?", next(model.parameters()).device)
    
    
    # print what device the model is on at this point
    #print("Where is my model?", next(model.parameters()).device)

    # thinking about accelerator here...
    #device = accelerator.device
    #model.to(device)
    
    name = f"var_{datamodule.hparams.in_var_ids}_scenarios_{datamodule.hparams.train_scenarios[0]}_tau_{hp.tau}_z_{hp.d_z}_lr_{hp.lr}_spreg_{hp.reg_coeff}_ormuinit_{hp.ortho_mu_init}_spmuinit_{hp.sparsity_mu_init}_spthres_{hp.sparsity_upper_threshold}_fixed_{hp.fixed}_num_ensembles_{datamodule.hparams.num_ensembles}_instantaneous_{hp.instantaneous}_crpscoef_{hp.crps_coeff}_spcoef_{hp.spectral_coeff}"
    hp.exp_path = hp.exp_path + name

    # create path to exp and save hyperparameters
    save_path = os.path.join(hp.exp_path, "train")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(os.path.join(hp.exp_path, "params.json"), "w") as file:
        json.dump(vars(hp), file, indent=4)

    # # load the best metrics
    # with open(os.path.join(hp.data_path, "best_metrics.json"), 'r') as f:
    #     best_metrics = json.load(f)
    best_metrics = {'recons': 0, 'kl': 0, 'mcc': 0, 'elbo': 0}

    # train
    if not hp.latent:
        trainer = Training(model, datamodule, hp)
    else:
        trainer = TrainingLatent(model, datamodule, hp, best_metrics, d)
    
    # where is the model at this point?
    print("Where is my model?", next(trainer.model.parameters()).device)
    
    valid_loss = trainer.train_with_QPM()

    # save final results, (MSE)
    metrics = {"shd": 0., "precision": 0., "recall": 0., "train_mse": 0., "val_mse": 0., "mcc": 0.}
    # if we have the GT, also compute (SHD, Pr, Re, MCC)
    if not hp.no_gt:
        # Here can remove this ---
        if hp.instantaneous:
            gt_graph = trainer.gt_dag
        else:
            gt_graph = trainer.gt_dag[:-1]  # remove the graph G_t

        learned_graph = trainer.model.get_adj().detach().numpy().reshape(gt_graph.shape[0], gt_graph.shape[1], -1)

        score, cc_program_perm, assignments, z, z_hat, _ = mcc_latent(trainer.model, trainer.data)
        permutation = np.zeros((gt_graph.shape[1], gt_graph.shape[1]))
        permutation[np.arange(gt_graph.shape[1]), assignments[1]] = 1
        gt_graph = permutation.T @ gt_graph @ permutation

        metrics['mcc'] = score
        metrics['w_mse'] = w_mae(trainer.model.autoencoder.get_w_decoder().detach().numpy()[:, :, assignments[1]], datamodule.gt_w)
        metrics['shd'] = shd(learned_graph, gt_graph)
        metrics['precision'], metrics['recall'] = precision_recall(learned_graph, gt_graph)
        errors = edge_errors(learned_graph, gt_graph)
        metrics['tp'] = errors['tp']
        metrics['fp'] = errors['fp']
        metrics['tn'] = errors['tn']
        metrics['fn'] = errors['fn']
        metrics['n_edge_gt_graph'] = np.sum(gt_graph)
        metrics['n_edge_learned_graph'] = np.sum(learned_graph)
        metrics['execution_time'] = time.time() - t0

        for key, val in valid_loss.items():
            metrics[key] = val
    
    # assert that trainer.model is in eval mode
    if trainer.model.training:
        print("Model is in train mode")
    else:
        print("Model is in eval mode") 
    
    #train_mse, train_smape, train_y, train_y_pred, train_x_original = autoregress_prediction_original(trainer, datamodule, False, hp)
    #val_mse, val_smape, val_y, val_y_pred, val_x_original = autoregress_prediction_original(trainer, datamodule, True, hp)

    # NOTE: just dummies here for now
    train_mse, train_smape, val_mse, val_smape = 10.0, 10.0, 10.0, 10.0

    # save the metrics
    metrics['train_mse']= train_mse
    metrics['train_smape']= train_smape
    metrics['val_mse'] = val_mse
    metrics['val_smape'] = val_smape

    # save the metrics
    with open(os.path.join(hp.exp_path, "results.json"), "w") as file:
        json.dump(metrics, file, indent=4)

    # finally, save the model
    torch.save(trainer.model.state_dict(), os.path.join(hp.exp_path, "model.pth"))

# NOTE:(seb) rewrite so that we make the iters again!
def autoregress_prediction_original(trainer, datamodule, valid, hp, timesteps=60):
    """
    Calculate the MSE and SMAPE between X_{t+1} and X_hat_{t+1}
    But also do autoregressive lead out for set number of months
    """
    
    print("What mode is the trainer in autoregress predictions? Is it in training mode?")
    print(trainer.model.training)

    # change the model to eval mode
    trainer.model.eval()
    # assert it is in eval
    if trainer.model.training:
        print("Now I tried to eval, and model is in train mode")
    else:
        print("Now I tried to eval, and model is in eval mode")

    if not valid:
        print("in training data mode")

        # Make the iterator again, since otherwise we have iterated through it already...
        train_dataloader = iter(datamodule.train_dataloader())
        x, y = next(train_dataloader)
        
        # NOTE:(seb) what we had before, 29/5/24:
        #x, y = next(trainer.data_loader_train)
        
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
        y = y[:, 0]
        z = None

        # NOTE:(seb) one point here is that maybe the model resets?
        # ensure these are correct
        y_pred, y, z, pz_mu, pz_std = trainer.model.predict(x, y)
                
        # make a copy of y_pred, which is a tensor
        x_original = x.clone().detach()
        y_original = y.clone().detach()
        y_original_pred = y_pred.clone().detach()
        
        # save these original values, x_original, y_orginal, y_original_pred
        np.save(os.path.join(hp.exp_path, "train_x_ar_0.npy"), x_original.detach().cpu().numpy())
        np.save(os.path.join(hp.exp_path, "train_y_ar_0.npy"), y_original.detach().cpu().numpy())
        np.save(os.path.join(hp.exp_path, "train_y_pred_ar_0.npy"), y_original_pred.detach().cpu().numpy())
        #np.save(os.path.join(hp.exp_path, "train_encoded_z_ar_0.npy"), z.detach().cpu().numpy())
        #np.save(os.path.join(hp.exp_path, "train_pz_mu_ar_0.npy"), pz_mu.detach().cpu().numpy())
        #np.save(os.path.join(hp.exp_path, "train_pz_std_ar_0.npy"), pz_std.detach().cpu().numpy())

        # Now doing the autoregressive rollout...
        for i in range(1, timesteps):
            
            # assert that x_original and x are the same
            if i == 1:
                assert torch.allclose(x_original, x)

            # remove the first timestep, so now we have (tau - 1) timesteps, 
            # then append the prediction
            x = x[:, 1:, :, :]
            x = torch.cat([x, y_pred.unsqueeze(1)], dim=1)

            # then predict the next timestep
            # y at this point is pointless!!!
            y_pred, y, z, pz_mu, pz_std = trainer.model.predict(x, y)

            print(f"**********THE AUTOREGRESSIVE ITERATION THAT I AM SAVING IS {i}**********")
            np.save(os.path.join(hp.exp_path, f"train_x_ar_{i}.npy"), x.detach().cpu().numpy())
            np.save(os.path.join(hp.exp_path, f"train_y_ar_{i}.npy"), y.detach().cpu().numpy())
            np.save(os.path.join(hp.exp_path, f"train_y_pred_ar_{i}.npy"), y_pred.detach().cpu().numpy())
            #np.save(os.path.join(hp.exp_path, f"train_encoded_z_ar_{i}.npy"), z.detach().cpu().numpy())
            #np.save(os.path.join(hp.exp_path, f"train_pz_mu_ar_{i}.npy"), pz_mu.detach().cpu().numpy())
            #np.save(os.path.join(hp.exp_path, f"train_pz_std_ar_{i}.npy"), pz_std.detach().cpu().numpy())
        
        print(f"did autoregressive for training, {timesteps} timesteps")

    else:
        # bs = np.min([trainer.data.n_valid, 1000])
        # Make the iterator again   
        val_dataloader = iter(datamodule.val_dataloader())
        x, y = next(val_dataloader)        

        # old, using existing dataloader
        #x, y = next(trainer.data_loader_val)
        
        y = torch.nan_to_num(y)
        x = torch.nan_to_num(x)
        y = y[:, 0]
        z = None
        # swap
        y_pred, y, z, pz_mu, pz_std = trainer.model.predict(x, y)
        
        # make a copy of y_pred, which is a tensor
        x_original = x.clone().detach()
        y_original = y.clone().detach()
        y_original_pred = y_pred.clone().detach()

        # saving these
        np.save(os.path.join(hp.exp_path, "val_x_ar_0.npy"), x_original.detach().cpu().numpy())
        np.save(os.path.join(hp.exp_path, "val_y_ar_0.npy"), y_original.detach().cpu().numpy())
        np.save(os.path.join(hp.exp_path, "val_y_pred_ar_0.npy"), y_original_pred.detach().cpu().numpy())
        #np.save(os.path.join(hp.exp_path, "val_encoded_z_ar_0.npy"), z.detach().cpu().numpy())
        #np.save(os.path.join(hp.exp_path, "val_pz_mu_ar_0.npy"), pz_mu.detach().cpu().numpy())
        #np.save(os.path.join(hp.exp_path, "val_pz_std_ar_0.npy"), pz_std.detach().cpu().numpy())

        for i in range(1, timesteps):
            
            # remove the first timestep, so now we have (tau - 1) timesteps 
            x = x[:, 1:, :, :]
            x = torch.cat([x, y_pred.unsqueeze(1)], dim=1)

            # then predict the next timestep
            # y at this point is not being updated!!!
            y_pred, y, z, pz_mu, pz_std = trainer.model.predict(x, y)
            
            np.save(os.path.join(hp.exp_path, f"val_x_ar_{i}.npy"), x.detach().cpu().numpy())
            np.save(os.path.join(hp.exp_path, f"val_y_ar_{i}.npy"), y.detach().cpu().numpy())
            np.save(os.path.join(hp.exp_path, f"val_y_pred_ar_{i}.npy"), y_pred.detach().cpu().numpy())
            #np.save(os.path.join(hp.exp_path, f"val_encoded_z_ar_{i}.npy"), z.detach().cpu().numpy())
            #np.save(os.path.join(hp.exp_path, f"val_pz_mu_ar_{i}.npy"), pz_mu.detach().cpu().numpy())
            #np.save(os.path.join(hp.exp_path, f"val_pz_std_ar_{i}.npy"), pz_std.detach().cpu().numpy())
        
        print(f"did autoregressive for validation, {timesteps} timesteps") 

        # not finished, probably need to add some metrics here.
        # what are the shapes here?
    
    with torch.no_grad():
        
        # NOTE: printing a selection of pointless metrics...
        
        # I guess there are different MAEs that we can calculate here.
        # The first element [:, 0, :, :] has the poorest persistence, whereas the last element [:, -1, :, :] has the best persistence - this is where I want to add the prediction
        mae1 = torch.mean(torch.abs(y_original - y_original_pred))
        print('SOME KIND OF MAE...this is just the mean of all the elements!:', mae1)
        
        mae1_persis_0th = torch.mean(torch.abs(y_original - x_original[:, 0, :, :]))
        mae1_persis_1st = torch.mean(torch.abs(y_original - x_original[:, 1, :, :]))
        mae1_persis_2nd = torch.mean(torch.abs(y_original - x_original[:, 2, :, :]))
        mae1_persis_3rd = torch.mean(torch.abs(y_original - x_original[:, 3, :, :]))
        mae1_persis_4th = torch.mean(torch.abs(y_original - x_original[:, 4, :, :]))

        mae_x_0th_to_last = torch.mean(torch.abs(x_original[:, 4, :, :] - x_original[:, 0, :, :]))
        mae_x_1st_to_last = torch.mean(torch.abs(x_original[:, 4, :, :] - x_original[:, 1, :, :]))
        mae_x_2nd_to_last = torch.mean(torch.abs(x_original[:, 4, :, :] - x_original[:, 2, :, :]))
        mae_x_3rd_to_last = torch.mean(torch.abs(x_original[:, 4, :, :] - x_original[:, 3, :, :]))

        print('MAE1 PERSISTENCE 0th:', mae1_persis_0th)
        print('MAE1 PERSISTENCE 1st:', mae1_persis_1st)
        print('MAE1 PERSISTENCE 2nd:', mae1_persis_2nd)
        print('MAE1 PERSISTENCE 3rd:', mae1_persis_3rd)
        print('MAE1 PERSISTENCE 4th:', mae1_persis_4th)

        print('MAE X 0th to last:', mae_x_0th_to_last)
        print('MAE X 1st to last:', mae_x_1st_to_last)
        print('MAE X 2nd to last:', mae_x_2nd_to_last)
        print('MAE X 3rd to last:', mae_x_3rd_to_last)


        print('CALCULATING MSE AND SMAPE')  
        # why is this half here...and actually why sum and mean     
        mse = torch.mean(torch.sum(0.5 * (y_original - y_original_pred) ** 2, dim=2))
        print("MSE:", mse)
        print("MSE shape:", mse.shape)

        mse2 = torch.mean((y_original - y_original_pred) ** 2)
        print("MSE2:", mse2)
        print("MSE2 shape:", mse2.shape)

        smape = torch.mean(torch.sum(2 * (y_original - y_original_pred).abs() / (y_original.abs() + y_original_pred.abs()), dim = 2))
        print("SMAPE:", smape)
        print()

    return mse.item(), smape.item(), y_original, y_original_pred, x_original



def prediction_original(trainer, valid):
    """
    Calculate the MSE and SMAPE between X_{t+1} and X_hat_{t+1}, using orignal code from causal_model_original
    """
    if not valid:
        
        #NOTE: SH, this is the original code here from causal_model_original accessed 06/05/2024, but I don't think the trainer.data.sample still works...
        #bs = np.min([trainer.data.n_train, 1000])
        #x, y, _ = trainer.data.sample(bs, valid=False)
        #y, y_pred = trainer.model.predict(x, y)

        x, y = next(trainer.data_loader_train)
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
        y = y[:, 0]
        z = None
        y, y_pred = trainer.model.predict(x, y)

    else:
        #bs = np.min([trainer.data.n_valid, 1000])
        #x, y, _ = trainer.data.sample(bs, valid=True)
        #y, y_pred = trainer.model.predict(x, y)

        x, y = next(trainer.data_loader_val)
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
        y = y[:, 0]
        z = None
        y, y_pred = trainer.model.predict(x, y)

    with torch.no_grad():
        mse = torch.mean(torch.sum(0.5 * (y - y_pred) ** 2, dim=2))
        smape = torch.mean(torch.sum(2 * (y - y_pred).abs() / (y.abs() + y_pred.abs()), dim = 2))
    
    return mse.item(), smape.item(), y, y_pred, x



def prediction(trainer, valid):
    """
    Calculate the MSE and SMAPE between X_{t+1} and X_hat_{t+1}
    """
    
    print("***********CALLING PREDICTION***********")
    
    if not valid:
        print("in training data mode")
        
        # might try this batch size thing actually!
        # bs = np.min([trainer.datamodule.n_train, 1000])
        #x, y = trainer.data.get_batch(bs, "train")

        x, y = next(trainer.data_loader_train)
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
        y = y[:, 0]
        z = None
        nll, recons, kl, y_pred = trainer.get_nll(x, y, z)
    else:
        print("in validation mode")
        # bs = np.min([trainer.data.n_valid, 1000])
        x, y = next(trainer.data_loader_val)
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
        y = y[:, 0]
        z = None
        nll, recons, kl, y_pred = trainer.get_nll(x, y, z)
    
    #print('HERE ARE THE SHAPES OF Y_PRED AND Y')
    #print()
    #print("y_pred shape:", y_pred.shape)
    #print("y shape:", y.shape)
    #print()

    # what are the shapes here?
    with torch.no_grad():
        
        # I guess there are different MAEs that we can calculate here.
        mae1 = torch.mean(torch.abs(y - y_pred))
        print('SOME KIND OF MAE...this is just the mean of all the elements!:', mae1)
        mae2 = torch.mean(torch.abs(y - y_pred), dim=2)
        print('SOME KIND OF MAE with dim...this is very long:', mae2.shape)
        #print("1st thing of MAE:", mae[0])
        

        print('CALCULATING MSE AND SMAPE')  
        # why is this half here...and actually why sum and mean     
        mse = torch.mean(torch.sum(0.5 * (y - y_pred) ** 2, dim=2))
        print("MSE:", mse)
        print("MSE shape:", mse.shape)

        mse2 = torch.mean((y - y_pred) ** 2)
        print("MSE2:", mse2)
        print("MSE2 shape:", mse2.shape)

        smape = torch.mean(torch.sum(2 * (y - y_pred).abs() / (y.abs() + y_pred.abs()), dim = 2))
        print("SMAPE:", smape)
        print()
    return mse.item(), smape.item(), y, y_pred


def autoregress_prediction(trainer, valid, hp, timesteps=60):
    """
    Calculate the MSE and SMAPE between X_{t+1} and X_hat_{t+1}
    But also do autoregressive lead out for set number of months
    """
    
    print("***********CALLING AUTOREGRESS PREDICTION***********")
    
    if not valid:
        print("in training data mode")

        x, y = next(trainer.data_loader_train)
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
        y = y[:, 0]
        z = None
        nll, recons, kl, y_pred = trainer.get_nll(x, y, z)
        # make a copy of y_pred, which is a tensor
        y_pred_t = y_pred.clone().detach()
        x_original = x.clone().detach()

        print("trying autoregressive in training")
        for i in range(1, timesteps):
            x = torch.cat([x, y_pred.unsqueeze(1)], dim=1)
            x = x[:, 1:, :, :]
            nll, recons, kl, y_pred = trainer.get_nll(x, y, z)
            np.save(os.path.join(hp.exp_path, f"train_y_pred_ar_{i}.npy"), y_pred.detach().cpu().numpy())
        print("did autoregressive for training, 60 timesteps")

    else:
        print("in validation mode")
        # bs = np.min([trainer.data.n_valid, 1000])
        x, y = next(trainer.data_loader_val)
        print("x shape:", x.shape)
        print("y shape:", y.shape)
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
        y = y[:, 0]
        z = None
        nll, recons, kl, y_pred = trainer.get_nll(x, y, z)
        # make a copy of y_pred, which is a tensor
        y_pred_t = y_pred.clone().detach()
        x_original = x.clone().detach()

        for i in range(1, timesteps):
            x = torch.cat([x, y_pred.unsqueeze(1)], dim=1)
            x = x[:, 1:, :, :]
            nll, recons, kl, y_pred = trainer.get_nll(x, y, z)
            np.save(os.path.join(hp.exp_path, f"val_y_pred_ar_{i}.npy"), y_pred.detach().cpu().numpy())
    
    #print('HERE ARE THE SHAPES OF Y_PRED AND Y')
    #print()
    #print("y_pred shape:", y_pred.shape)
    #print("y shape:", y.shape)
    #print()

    # what are the shapes here?
    with torch.no_grad():
        
        # I guess there are different MAEs that we can calculate here.
        # The first element [:, 0, :, :] has the poorest persistence, whereas the last element [:, -1, :, :] has the best persistence - this is where I want to add the prediction
        mae1 = torch.mean(torch.abs(y - y_pred_t))
        print('SOME KIND OF MAE...this is just the mean of all the elements!:', mae1)
        mae1_persis_0th = torch.mean(torch.abs(y - x_original[:, 0, :, :]))
        mae1_persis_1st = torch.mean(torch.abs(y - x_original[:, 1, :, :]))
        mae1_persis_4th = torch.mean(torch.abs(y - x_original[:, 3, :, :]))
        mae1_persis_5th = torch.mean(torch.abs(y - x_original[:, -1, :, :]))
        print('MAE1 PERSISTENCE 0th:', mae1_persis_0th)
        print('MAE1 PERSISTENCE 1st:', mae1_persis_1st)
        print('MAE1 PERSISTENCE 4th:', mae1_persis_4th)
        print('MAE1 PERSISTENCE 5th:', mae1_persis_5th)
        
        
        mae2 = torch.mean(torch.abs(y - y_pred_t), dim=2)
        print('SOME KIND OF MAE with dim...this is very long:', mae2.shape)
        #print("1st thing of MAE:", mae[0])

        mae2_persis_0th = torch.mean(torch.mean(torch.abs(y - x_original[:, 0, :, :]), dim=2), dim=0)
        mae2_persis_1st = torch.mean(torch.mean(torch.abs(y - x_original[:, 1, :, :]), dim=2), dim=0)
        mae2_persis_5th = torch.mean(torch.mean(torch.abs(y - x_original[:, 4, :, :]), dim=2), dim=0)
        
        print('MAE2 PERSISTENCE 0th shape:', mae2_persis_0th.shape)
        print('MAE2 PERSISTENCE 0th:', mae2_persis_0th)
        print('MAE2 PERSISTENCE 1st:', mae2_persis_1st)
        print('MAE2 PERSISTENCE 5th:', mae2_persis_5th)



        print('CALCULATING MSE AND SMAPE')  
        # why is this half here...and actually why sum and mean     
        mse = torch.mean(torch.sum(0.5 * (y - y_pred_t) ** 2, dim=2))
        print("MSE:", mse)
        print("MSE shape:", mse.shape)

        mse2 = torch.mean((y - y_pred_t) ** 2)
        print("MSE2:", mse2)
        print("MSE2 shape:", mse2.shape)

        smape = torch.mean(torch.sum(2 * (y - y_pred_t).abs() / (y.abs() + y_pred_t.abs()), dim = 2))
        print("SMAPE:", smape)
        print()


    return mse.item(), smape.item(), y, y_pred_t, x_original

def ar_prediction(trainer, valid, hp):
    """
    Predict X_hat_{t+1} and then append this to the input to predict X_hat_{t+2}
    Also calculate the MSE and SMAPE between X_{t+n} and X_hat_{t+n}
    """

    #new_iter = trainer.data_loader_train

    # I have this problem where I want to go over the data again here...
    # So I need to make the iterator again...

    if not valid:
        #bs = np.min([trainer.data.n_train, 1000])
        x, y = next(trainer.data_loader_train)
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
        y = y[:, 0]
        z = None
        nll, recons, kl, y_pred = trainer.get_nll(x, y, z)

        # drop the first timestep from x
        # x = x[:, 1:, :, :]
        # append the prediction to the input
        
        print("trying autoregressive in training")
        for i in range(1, 60):
            x = torch.cat([x, y_pred.unsqueeze(1)], dim=1)
            x = x[:, 1:, :, :]
            nll, recons, kl, y_pred = trainer.get_nll(x, y, z)
            np.save(os.path.join(hp.exp_path, f"train_y_pred_ar_{i}.npy"), y_pred.detach().cpu().numpy())
        print("did autoregressive for training, 60 timesteps")

    else:
        # bs = np.min([trainer.data.n_valid, 1000])
        x, y = next(trainer.data_loader_val)
        x = torch.nan_to_num(x)
        y = torch.nan_to_num(y)
        y = y[:, 0]
        z = None
        nll, recons, kl, y_pred = trainer.get_nll(x, y, z)

        for i in range(1, 60):
            x = torch.cat([x, y_pred.unsqueeze(1)], dim=1)
            x = x[:, 1:, :, :]
            nll, recons, kl, y_pred = trainer.get_nll(x, y, z)
            np.save(os.path.join(hp.exp_path, f"val_y_pred_ar_{i}.npy"), y_pred.detach().cpu().numpy())

    # ideally I want to have something here where we connect the predictions to the truth...need to look at dataloader
    with torch.no_grad():

        mae = torch.mean(torch.abs(y - y_pred), dim=2)
        mse = torch.mean(torch.sum(0.5 * (y - y_pred) ** 2, dim=2))
        smape = torch.mean(torch.sum(2 * (y - y_pred).abs() / (y.abs() + y_pred.abs()), dim=2))

    return mse.item(), smape.item(), y, y_pred  


# NOTE: SH - I am going to write a function to extract the predictions here too

def assert_args(args):
    """
    Raise errors or warnings if some args should not take some combination of
    values.
    """
    # raise errors if some args should not take some combination of values
    if args.no_gt and (args.debug_gt_graph or args.debug_gt_z or args.debug_gt_w):
        raise ValueError("Since no_gt==True, all other args should not use ground-truth values")

    if args.latent and (args.d_z is None or args.d_x is None or args.d_z <= 0 or args.d_x <= 0):
        raise ValueError("When using latent model, you need to define d_z and d_x with integer values greater than 0")

    if args.ratio_valid == 0:
        args.ratio_valid = 1 - args.ratio_train
    if args.ratio_train + args.ratio_valid > 1:
        raise ValueError("The sum of the ratio for training and validation set is higher than 1")

    # string input with limited possible values
    supported_dataformat = ["numpy", "hdf5"]
    if args.data_format not in supported_dataformat:
        raise ValueError(f"This file format ({args.data_format}) is not \
                         supported. Supported types are: {supported_dataformat}")
    supported_optimizer = ["sgd", "rmsprop"]
    if args.optimizer not in supported_optimizer:
        raise ValueError(f"This optimizer type ({args.optimizer}) is not \
                         supported. Supported types are: {supported_optimizer}")

    # warnings, strange choice of args combination
    if not args.latent and args.debug_gt_z:
        warnings.warn("Are you sure you want to use gt_z even if you don't have latents")
    if args.latent and (args.d_z > args.d_x):
        warnings.warn("Are you sure you want to have a higher dimension for d_z than d_x")

    return args


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Causal models for climate data")
    # for the default values, check default_params.json

    parser.add_argument("--exp-path", type=str, default="../../causal_climate_exp/",
                        help="Path to experiments")
    parser.add_argument("--config-exp-path", type=str, default="../emulator/configs/datamodule/climate.json",
                        help="Path to a json file with specifics of experiments")
    parser.add_argument("--config-path", type=str, default="default_params.json",
                        help="Path to a json file with values for all hyperparameters")
    parser.add_argument("--use-data-config", action="store_true",
                        help="If true, overwrite some parameters to fit \
                        parameters that have been used to generate data")
    parser.add_argument("--exp-id", type=int,
                        help="ID specific to the experiment")

    # For synthetic datasets, can use the ground-truth values to do ablation studies
    parser.add_argument("--debug-gt-z", action="store_true",
                        help="If true, use the ground truth value of Z (use only to debug)")
    parser.add_argument("--debug-gt-w", action="store_true",
                        help="If true, use the ground truth value of W (use only to debug)")
    parser.add_argument("--debug-gt-graph", action="store_true",
                        help="If true, use the ground truth graph (use only to debug)")
    parser.add_argument("--no-w-constraint", action="store_true",
                        help="If True, does not apply constraint on W (non-negativity and ortho)")

    # Dataset properties
    parser.add_argument("--data-path", type=str, help="Path to the dataset")
    parser.add_argument("--data-format", type=str, help="numpy|hdf5")
    parser.add_argument("--no-gt", action="store_true",
                        help="If True, does not use any ground-truth for plotting and metrics")
    
    # dataset transformation
    parser.add_argument("--seasonality-removal", action="store_true", help="Deseasonalise the data")

    # specific to model with latent variables
    parser.add_argument("--latent", action="store_true", help="Use the model that assumes latent variables")
    parser.add_argument("--tied-w", action="store_true", help="Use the same matrix W, as the decoder, for the encoder")
    parser.add_argument("--nonlinear-mixing", action="store_true", help="The encoder/decoder use NN")
    parser.add_argument("--coeff-kl", type=float, help="coefficient that is multiplied to the KL term ")
    parser.add_argument("--d-z", type=int, help="if latent, d_z is the number of cluster z")
    parser.add_argument("--d-x", type=int, help="if latent, d_x is the number of gridcells")

    # NOTE: seb setting the transition matrix as fixed in different ways as an ablation study
    # and adding a fixed_output parameter
    parser.add_argument("--fixed", action="store_true", help="Keep transition matrix fixed as all ones")
    parser.add_argument("--fixed-output-fraction", type=float, help="Fraction of 1s and 0s in fixed output matrix, ")
    

    parser.add_argument("--instantaneous", action="store_true", help="Use instantaneous connections")
    parser.add_argument("--tau", type=int, help="Number of past timesteps to consider")
    parser.add_argument("--tau-neigh", type=int, help="Radius of neighbor cells to consider")
    parser.add_argument("--ratio-train", type=int, help="Proportion of the data used for the training set")
    parser.add_argument("--ratio-valid", type=int, help="Proportion of the data used for the validation set")
    parser.add_argument("--batch-size", type=int, help="Number of samples per minibatch")

    # Model hyperparameters: architecture
    parser.add_argument("--num-hidden", type=int, help="Number of hidden units")
    parser.add_argument("--num-layers", type=int, help="Number of hidden layers")
    parser.add_argument("--num-output", type=int, help="Number of output units")

    parser.add_argument("--num-hidden-mixing", type=int, help="Number of hidden \
                        units for the encoder/decoder learning the mixing function")
    parser.add_argument("--num-layers-mixing", type=int, help="Number of hidden \
                        layers for the encoder/decoder learning the mixing function")

    # Model hyperparameters: optimization
    parser.add_argument("--optimizer", type=str, help="sgd|rmsprop")
    parser.add_argument("--reg-coeff", type=float, help="Coefficient for the sparsity regularisation term")
    parser.add_argument("--reg-coeff-connect", type=float, help="Coefficient for the connectivity regularisation term")
    parser.add_argument("--lr", type=float, help="Initial learning rate")
    parser.add_argument("--lr-scheduler-epochs", type=lambda x: list(map(int, x.split(","))), help="Number of iterations to decrease lr by lr-scheduler-gamma")
    parser.add_argument("--lr-scheduler-gamma", type=int, help="Value by which to multiply LR at each iteration lr-schedule, 1 does not decrease LR, 0.1 does by 10")
    parser.add_argument("--random-seed", type=int, help="Random seed for torch and numpy")
    parser.add_argument("--schedule-reg", type=int,
                        help="Start applying the sparsity regularization only after X number of steps")
    parser.add_argument("--schedule-ortho", type=int,
                        help="Start applying the orthogonality constraint only after X number of steps")
    parser.add_argument("--schedule-sparsity", type=int,
                        help="Start applying the sparsity constraint only after X number of steps")
    parser.add_argument("--hard-gumbel", action="store_true",
                        help="If true, use the hard version when sampling the masks")

    # ALM/QPM options
    # orthogonality constraint
    parser.add_argument("--ortho-mu-init", type=float,
                        help="initial value of mu for the constraint")
    parser.add_argument("--ortho-mu-mult-factor", type=float,
                        help="Multiply mu by this amount when constraint not sufficiently decreasing")
    parser.add_argument("--ortho-omega-gamma", type=float,
                        help="Precision to declare convergence of subproblems")
    parser.add_argument("--ortho-omega-mu", type=float,
                        help="After subproblem solved, h should have reduced by this ratio")
    parser.add_argument("--ortho-h-threshold", type=float,
                        help="Can stop if h smaller than h-threshold")
    parser.add_argument("--ortho-min-iter-convergence", type=int,
                        help="Minimal number of iteration before checking if has converged")
    
    # NOTE:(seb) adding same for the sparsity constraint
    # sparsity constraint
    parser.add_argument("--sparsity-mu-init", type=float,
                        help="initial value of mu for the constraint")
    parser.add_argument("--sparsity-mu-mult-factor", type=float,
                        help="Multiply mu by this amount when constraint not sufficiently decreasing")
    parser.add_argument("--sparsity-omega-gamma", type=float,
                        help="Precision to declare convergence of subproblems")
    parser.add_argument("--sparsity-omega-mu", type=float,
                        help="After subproblem solved, h should have reduced by this ratio")
    parser.add_argument("--sparsity-h-threshold", type=float,
                        help="Can stop if h smaller than h-threshold")
    parser.add_argument("--sparsity-min-iter-convergence", type=int,
                        help="Minimal number of iteration before checking if has converged")
    
    # NOTE:(seb) adding an argument for upper threshold of the sparsity constraint
    parser.add_argument("--sparsity-upper-threshold", type=float, help="Upper threshold for the sparsity constraint")

    # acyclicity constraint
    parser.add_argument("--acyclic-mu-init", type=float,
                        help="initial value of mu for the constraint")
    parser.add_argument("--acyclic-mu-mult-factor", type=float,
                        help="Multiply mu by this amount when constraint not sufficiently decreasing")
    parser.add_argument("--acyclic-omega-gamma", type=float,
                        help="Precision to declare convergence of subproblems")
    parser.add_argument("--acyclic-omega-mu", type=float,
                        help="After subproblem solved, h should have reduced by this ratio")
    parser.add_argument("--acyclic-h-threshold", type=float,
                        help="Can stop if h smaller than h-threshold")
    parser.add_argument("--acyclic-min-iter-convergence", type=int,
                        help="Minimal number of iteration before checking if has converged")

    parser.add_argument("--mu-acyclic-init", type=float,
                        help="initial value of mu for the acyclicity constraint")
    parser.add_argument("--h-acyclic-threshold", type=float,
                        help="Can stop if h smaller than h-threshold")

    parser.add_argument("--max-iteration", type=int,
                        help="Maximal number of iteration before stopping")
    parser.add_argument("--patience", type=int,
                        help="Patience used after the acyclicity constraint is respected")
    parser.add_argument("--patience-post-thresh", type=int,
                        help="Patience used after the thresholding of the adjacency matrix")
    
    # adding loss coefficients
    parser.add_argument("--crps-coeff", type=float,
                        help="Coefficient for the CRPS term of the loss")
    parser.add_argument("--spectral-coeff", type=float,
                        help="Coefficient for the spectral term of the loss")

    # logging
    parser.add_argument("--valid-freq", type=int, help="Frequency of evaluating the loss on the validation set")
    parser.add_argument("--plot-freq", type=int, help="Plotting frequency")
    parser.add_argument("--plot-through-time", action="store_true", help="If true, save each plot in a \
                        different file with a name depending on the iteration")
    parser.add_argument("--print-freq", type=int, help="Printing frequency")

    # device and numerical precision
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    parser.add_argument("--float", action="store_true", help="Use Float precision")

    parser.add_argument("--ishdf5", action="store_true", help="Use GPU")
    args = parser.parse_args()

    # if a json file with params is given,
    # update params accordingly
    if args.config_path != "":
        print(f"using config file: {args.config_path}")
        default_params = vars(args)
        with open(args.config_path, 'r') as f:
            params = json.load(f)

        for key, val in params.items():
            if default_params[key] is None or not default_params[key]:
                default_params[key] = val
        args = Bunch(**default_params)

    # use some parameters from the data generating process';;
    if args.use_data_config != "":
        with open(args.config_exp_path, 'r') as f:
            params = json.load(f)
        args.d_x = params['d_x']
        if 'latent' in params:
            args.latent = params['latent']
            if args.latent:
                args.d_z = params['d_z']
        if 'tau' in params:
            args.tau = params['tau']
        if 'neighborhood' in params:
            args.tau_neigh = params['neighborhood']

    # args.nonlinear_mixing = True
    args.latent = True
    print(args.no_gt)

    args = assert_args(args)

    main(args)
