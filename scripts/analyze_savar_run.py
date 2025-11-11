#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from climatem.config import (
    dataParams,
    expParams,
    gtParams,
    modelParams,
    optimParams,
    plotParams,
    savarParams,
    trainParams,
)
from climatem.data_loader.causal_datamodule import CausalClimateDataModule
from climatem.model.tsdcd_latent import LatentTSDCD
from climatem.utils import load_config


def _expand_path(path_str: str, fallback: Path) -> str:
    expanded = Path(os.path.expandvars(path_str))
    if expanded.exists():
        return str(expanded)
    return str(fallback)


def _load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found")
    return load_config(config_path)


def _instantiate_params(cfg: Dict, project_root: Path):
    exp_cfg = dict(cfg["exp_params"])
    exp_cfg["exp_path"] = str(project_root / "ws" / "SAVAR_RESULTS")
    experiment_params = expParams(**exp_cfg)

    data_cfg = dict(cfg["data_params"])
    data_cfg.pop("seq_len", None)
    data_cfg["data_dir"] = _expand_path(data_cfg["data_dir"], project_root / "ws" / "SAVAR_DATA")
    data_cfg["climateset_data"] = _expand_path(
        data_cfg.get("climateset_data", data_cfg["data_dir"]),
        project_root / "ws" / "SAVAR_DATA",
    )
    data_cfg["icosahedral_coordinates_path"] = _expand_path(
        data_cfg["icosahedral_coordinates_path"],
        project_root / "mappings" / "vertex_lonlat_mapping.npy",
    )
    data_params_obj = dataParams(**data_cfg)

    gt_params_obj = gtParams(**cfg["gt_params"])

    train_cfg = dict(cfg["train_params"])
    train_cfg.pop("ratio_valid", None)
    train_params_obj = trainParams(**train_cfg)

    model_params_obj = modelParams(**cfg["model_params"])
    optim_params_obj = optimParams(**cfg["optim_params"])

    plot_cfg = cfg.get("plot_params", {})
    plot_params_obj = plotParams(**plot_cfg) if plot_cfg else plotParams()

    savar_cfg = cfg.get("savar_params", {})
    savar_params_obj = savarParams(**savar_cfg) if savar_cfg else savarParams()

    return (
        experiment_params,
        data_params_obj,
        gt_params_obj,
        train_params_obj,
        model_params_obj,
        optim_params_obj,
        plot_params_obj,
        savar_params_obj,
    )


def _encode_means(model: LatentTSDCD, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    b, tau, d, _ = x.shape
    device = x.device
    history = torch.zeros(b, tau, d, model.d_z, device=device)
    future = torch.zeros(b, d, model.d_z, device=device)

    for i in range(d):
        for t in range(tau):
            q_mu, _ = model.autoencoder(x[:, t, i], i, encode=True)
            history[:, t, i] = q_mu
        q_mu_y, _ = model.autoencoder(y[:, i], i, encode=True)
        future[:, i] = q_mu_y

    return history, future


def _summarize_prediction(pred: torch.Tensor, truth: torch.Tensor) -> Dict[str, float]:
    pred_flat = pred.reshape(pred.shape[0], -1)
    truth_flat = truth.reshape(truth.shape[0], -1)

    diff = pred_flat - truth_flat
    rmse = torch.sqrt(torch.mean(diff**2)).item()
    mae = torch.mean(torch.abs(diff)).item()

    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    truth_centered = truth_flat - truth_flat.mean(dim=1, keepdim=True)
    corr = F.cosine_similarity(pred_centered, truth_centered, dim=1).mean().item()

    pred_std = pred_flat.std(dim=1, unbiased=False).mean().item()
    truth_std = truth_flat.std(dim=1, unbiased=False).mean().item()
    amp_ratio = pred_std / truth_std if truth_std > 0 else float("nan")

    sign_agreement = ((pred_flat.sign() == truth_flat.sign()) | (truth_flat == 0)).float().mean().item()

    return {
        "rmse": rmse,
        "mae": mae,
        "corr": corr,
        "amp_ratio": amp_ratio,
        "sign_agreement": sign_agreement,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose latent transition behaviour on SAVAR runs.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Experiment directory (contains training_results/model.pth).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path("configs/single_param_file_savar.json"),
        help="Config file used for training.",
    )
    parser.add_argument("--rollout-steps", type=int, default=6, help="Number of autoregressive steps to compute.")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force GPU execution. By default, the script runs on CPU to avoid latent sampling device mismatches.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    cfg = _load_config(args.config_path.resolve())

    (
        experiment_params,
        data_params_obj,
        gt_params_obj,
        train_params_obj,
        model_params_obj,
        optim_params_obj,
        plot_params_obj,
        savar_params_obj,
    ) = _instantiate_params(cfg, project_root)

    force_gpu = args.gpu
    gpu_available = torch.cuda.is_available()
    use_gpu = force_gpu and gpu_available and experiment_params.gpu

    if force_gpu and not gpu_available:
        print("--gpu requested but CUDA is not available; falling back to CPU.")

    if not use_gpu and experiment_params.gpu:
        print("Running diagnostics on CPU to avoid device mismatches in latent sampling.")

    experiment_params.gpu = use_gpu
    device = torch.device("cuda" if use_gpu else "cpu")

    if not use_gpu:
        torch.set_default_tensor_type("torch.FloatTensor")

    torch.set_grad_enabled(False)

    datamodule = CausalClimateDataModule(
        tau=experiment_params.tau,
        future_timesteps=experiment_params.future_timesteps,
        num_months_aggregated=data_params_obj.num_months_aggregated,
        train_val_interval_length=data_params_obj.train_val_interval_length,
        in_var_ids=data_params_obj.in_var_ids,
        out_var_ids=data_params_obj.out_var_ids,
        train_years=data_params_obj.train_years,
        train_historical_years=data_params_obj.train_historical_years,
        test_years=data_params_obj.test_years,
        val_split=1 - train_params_obj.ratio_train,
        seq_to_seq=data_params_obj.seq_to_seq,
        channels_last=data_params_obj.channels_last,
        train_scenarios=data_params_obj.train_scenarios,
        test_scenarios=data_params_obj.test_scenarios,
        train_models=data_params_obj.train_models,
        batch_size=data_params_obj.batch_size,
        eval_batch_size=data_params_obj.eval_batch_size,
        num_workers=experiment_params.num_workers,
        pin_memory=experiment_params.pin_memory,
        load_train_into_mem=data_params_obj.load_train_into_mem,
        load_test_into_mem=data_params_obj.load_test_into_mem,
        verbose=experiment_params.verbose,
        seed=experiment_params.random_seed,
        seq_len=data_params_obj.seq_len,
        data_dir=data_params_obj.climateset_data,
        output_save_dir=data_params_obj.data_dir,
        num_ensembles=data_params_obj.num_ensembles,
        lon=experiment_params.lon,
        lat=experiment_params.lat,
        num_levels=data_params_obj.num_levels,
        global_normalization=data_params_obj.global_normalization,
        seasonality_removal=data_params_obj.seasonality_removal,
        reload_climate_set_data=data_params_obj.reload_climate_set_data,
        icosahedral_coordinates_path=data_params_obj.icosahedral_coordinates_path,
        time_len=savar_params_obj.time_len,
        comp_size=savar_params_obj.comp_size,
        noise_val=savar_params_obj.noise_val,
        n_per_col=savar_params_obj.n_per_col,
        difficulty=savar_params_obj.difficulty,
        seasonality=savar_params_obj.seasonality,
        overlap=savar_params_obj.overlap,
        is_forced=savar_params_obj.is_forced,
        f_1=savar_params_obj.f_1,
        f_2=savar_params_obj.f_2,
        f_time_1=savar_params_obj.f_time_1,
        f_time_2=savar_params_obj.f_time_2,
        ramp_type=savar_params_obj.ramp_type,
        linearity=savar_params_obj.linearity,
        poly_degrees=savar_params_obj.poly_degrees,
        plot_original_data=savar_params_obj.plot_original_data,
    )
    datamodule.setup()

    d = len(data_params_obj.in_var_ids)
    num_input = d * experiment_params.tau * (model_params_obj.tau_neigh * 2 + 1)

    model = LatentTSDCD(
        num_layers=model_params_obj.num_layers,
        num_hidden=model_params_obj.num_hidden,
        num_input=num_input,
        num_output=model_params_obj.num_output,
        num_layers_mixing=model_params_obj.num_layers_mixing,
        num_hidden_mixing=model_params_obj.num_hidden_mixing,
        position_embedding_dim=model_params_obj.position_embedding_dim,
        reduce_encoding_pos_dim=model_params_obj.reduce_encoding_pos_dim,
        coeff_kl=optim_params_obj.coeff_kl,
        d=d,
        distr_z0="gaussian",
        distr_encoder="gaussian",
        distr_transition="gaussian",
        distr_decoder="gaussian",
        d_x=experiment_params.d_x,
        d_z=experiment_params.d_z,
        tau=experiment_params.tau,
        instantaneous=model_params_obj.instantaneous,
        nonlinear_dynamics=model_params_obj.nonlinear_dynamics,
        nonlinear_mixing=model_params_obj.nonlinear_mixing,
        hard_gumbel=model_params_obj.hard_gumbel,
        no_gt=gt_params_obj.no_gt,
        debug_gt_graph=gt_params_obj.debug_gt_graph,
        debug_gt_z=gt_params_obj.debug_gt_z,
        debug_gt_w=gt_params_obj.debug_gt_w,
        tied_w=model_params_obj.tied_w,
        fixed=model_params_obj.fixed,
        fixed_output_fraction=model_params_obj.fixed_output_fraction,
    ).to(device)
    model.eval()

    model_path = run_dir / "training_results" / "model.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint {model_path} not found")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    val_batch = next(iter(datamodule.val_dataloader()))
    x_val, y_val = val_batch
    x_val = torch.nan_to_num(x_val).to(device)
    y_val = torch.nan_to_num(y_val[:, 0]).to(device)

    px_mu_tf, _, _, pz_mu_tf, _ = model.predict(x_val, y_val, teacher_forcing=True)
    _, q_mu_future = _encode_means(model, x_val, y_val)
    latent_mae_tf = torch.mean(torch.abs(pz_mu_tf - q_mu_future)).item()
    latent_norm_tf = pz_mu_tf.norm(dim=-1).mean().item()
    latent_norm_target = q_mu_future.norm(dim=-1).mean().item()
    tf_summary = _summarize_prediction(px_mu_tf.cpu(), y_val.cpu())

    px_mu_free, _, _, pz_mu_free, _ = model.predict(x_val, teacher_forcing=False)
    free_summary = _summarize_prediction(px_mu_free.cpu(), y_val.cpu())
    latent_mae_free = torch.mean(torch.abs(pz_mu_free - q_mu_future)).item()
    latent_norm_free = pz_mu_free.norm(dim=-1).mean().item()

    rollout_preds = []
    rollout_latents = []
    history = x_val.clone()
    for step in range(args.rollout_steps):
        px_mu_step, _, _, pz_mu_step, _ = model.predict(history, teacher_forcing=False)
        rollout_preds.append(px_mu_step.cpu())
        rollout_latents.append(pz_mu_step.cpu())
        history = torch.cat([history[:, 1:], px_mu_step.unsqueeze(1)], dim=1)

    rollout_stats = []
    for idx, (pred_step, latent_step) in enumerate(zip(rollout_preds, rollout_latents), start=1):
        rollout_stats.append(
            {
                "step": idx,
                "pred_mean": pred_step.mean().item(),
                "pred_std": pred_step.std(unbiased=False).item(),
                "latent_mean_abs": latent_step.abs().mean().item(),
                "latent_norm": latent_step.norm(dim=-1).mean().item(),
            }
        )

    print("\nTeacher-forcing check:")
    print(f"  latent MAE (pred vs encoded): {latent_mae_tf:.4f}")
    print(f"  latent norm predicted / target: {latent_norm_tf:.4f} / {latent_norm_target:.4f}")
    print(
        f"  reconstruction -> RMSE: {tf_summary['rmse']:.4f}, "
        f"MAE: {tf_summary['mae']:.4f}, Corr: {tf_summary['corr']:.4f}, "
        f"Amplitude ratio: {tf_summary['amp_ratio']:.4f}, Sign agreement: {tf_summary['sign_agreement']:.4f}"
    )

    print("\nFree-run one step check:")
    print(f"  latent MAE (pred vs encoded): {latent_mae_free:.4f}")
    print(f"  latent norm predicted: {latent_norm_free:.4f}")
    print(
        f"  prediction -> RMSE: {free_summary['rmse']:.4f}, "
        f"MAE: {free_summary['mae']:.4f}, Corr: {free_summary['corr']:.4f}, "
        f"Amplitude ratio: {free_summary['amp_ratio']:.4f}, Sign agreement: {free_summary['sign_agreement']:.4f}"
    )

    print("\nAutoregressive rollout (no teacher forcing):")
    for stats in rollout_stats:
        print(
            f"  step {stats['step']:>2}: pred_mean={stats['pred_mean']:+.4f}, "
            f"pred_std={stats['pred_std']:.4f}, latent_mean_abs={stats['latent_mean_abs']:.4f}, "
            f"latent_norm={stats['latent_norm']:.4f}"
        )


if __name__ == "__main__":
    main()
