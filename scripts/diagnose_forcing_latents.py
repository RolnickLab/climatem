#!/usr/bin/env python3
"""
Diagnostic script to check forcing latent encoder health.

Run this on a trained model checkpoint to diagnose why forcing latents aren't learning.
"""

import sys
import torch
import numpy as np
from pathlib import Path

def diagnose_forcing_encoders(checkpoint_path):
    """Load checkpoint and analyze forcing encoder parameters."""

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    print("\n" + "="*80)
    print("FORCING ENCODER PARAMETER ANALYSIS")
    print("="*80)

    # Check if forcing encoders exist
    has_co2_encoder = any('co2_forcing_encoder' in k for k in state_dict.keys())
    has_aerosol_encoder = any('aerosol_forcing_encoder' in k for k in state_dict.keys())

    if not has_co2_encoder and not has_aerosol_encoder:
        print("\n❌ NO FORCING ENCODERS FOUND!")
        print("Model does not have use_forced_latents=True or not trained with forcing.")
        return

    print(f"\n✓ CO2 Encoder: {'Present' if has_co2_encoder else 'Missing'}")
    print(f"✓ Aerosol Encoder: {'Present' if has_aerosol_encoder else 'Missing'}")

    # Analyze encoder weights
    print("\n" + "-"*80)
    print("ENCODER WEIGHT MAGNITUDES")
    print("-"*80)

    encoder_stats = {}

    # CO2 encoder
    if has_co2_encoder:
        co2_keys = [k for k in state_dict.keys() if 'co2_forcing_encoder_mu' in k and 'weight' in k]
        if co2_keys:
            co2_weight = state_dict[co2_keys[0]]
            encoder_stats['co2_encoder'] = {
                'key': co2_keys[0],
                'shape': co2_weight.shape,
                'mean': co2_weight.mean().item(),
                'std': co2_weight.std().item(),
                'max': co2_weight.max().item(),
                'norm': co2_weight.norm().item()
            }
            print(f"\nCO2 Encoder Weight:")
            print(f"  Shape: {co2_weight.shape}")
            print(f"  Mean: {co2_weight.mean().item():.6f}")
            print(f"  Std: {co2_weight.std().item():.6f}")
            print(f"  L2 Norm: {co2_weight.norm().item():.6f}")

    # Aerosol encoder
    if has_aerosol_encoder:
        aerosol_keys = [k for k in state_dict.keys() if 'aerosol_forcing_encoder_mu' in k and 'weight' in k]
        if aerosol_keys:
            aerosol_weight = state_dict[aerosol_keys[0]]
            encoder_stats['aerosol_encoder'] = {
                'key': aerosol_keys[0],
                'shape': aerosol_weight.shape,
                'mean': aerosol_weight.mean().item(),
                'std': aerosol_weight.std().item(),
                'max': aerosol_weight.max().item(),
                'norm': aerosol_weight.norm().item()
            }
            print(f"\nAerosol Encoder Weight:")
            print(f"  Shape: {aerosol_weight.shape}")
            print(f"  Mean: {aerosol_weight.mean().item():.6f}")
            print(f"  Std: {aerosol_weight.std().item():.6f}")
            print(f"  L2 Norm: {aerosol_weight.norm().item():.6f}")

    # Climate encoder for comparison
    climate_keys = [k for k in state_dict.keys() if 'mu_encoder' in k and 'weight' in k and 'forcing' not in k]
    if climate_keys:
        climate_weight = state_dict[climate_keys[0]]
        encoder_stats['climate_encoder'] = {
            'key': climate_keys[0],
            'shape': climate_weight.shape,
            'mean': climate_weight.mean().item(),
            'std': climate_weight.std().item(),
            'max': climate_weight.max().item(),
            'norm': climate_weight.norm().item()
        }
        print(f"\nClimate Encoder Weight (for comparison):")
        print(f"  Shape: {climate_weight.shape}")
        print(f"  Mean: {climate_weight.mean().item():.6f}")
        print(f"  Std: {climate_weight.std().item():.6f}")
        print(f"  L2 Norm: {climate_weight.norm().item():.6f}")

    # Analyze encoder logvars
    print("\n" + "-"*80)
    print("ENCODER LOGVAR VALUES")
    print("-"*80)

    # CO2 logvar
    co2_logvar_keys = [k for k in state_dict.keys() if 'co2_forcing_encoder_logvar' in k]
    if co2_logvar_keys:
        co2_logvar = state_dict[co2_logvar_keys[0]]
        print(f"\nCO2 Encoder Logvar:")
        print(f"  Shape: {co2_logvar.shape}")
        print(f"  Values: {co2_logvar.numpy()}")
        print(f"  Std: {torch.exp(0.5 * co2_logvar).numpy()}")

    # Aerosol logvar
    aerosol_logvar_keys = [k for k in state_dict.keys() if 'aerosol_forcing_encoder_logvar' in k]
    if aerosol_logvar_keys:
        aerosol_logvar = state_dict[aerosol_logvar_keys[0]]
        print(f"\nAerosol Encoder Logvar:")
        print(f"  Shape: {aerosol_logvar.shape}")
        print(f"  Values: {aerosol_logvar.numpy()}")
        print(f"  Std: {torch.exp(0.5 * aerosol_logvar).numpy()}")

    # Climate logvar
    climate_logvar_keys = [k for k in state_dict.keys() if k.endswith('logvar_encoder') and 'forcing' not in k]
    if climate_logvar_keys:
        climate_logvar = state_dict[climate_logvar_keys[0]]
        print(f"\nClimate Encoder Logvar (for comparison):")
        print(f"  Shape: {climate_logvar.shape}")
        print(f"  Mean: {climate_logvar.mean().item():.6f}")
        print(f"  Range: [{climate_logvar.min().item():.6f}, {climate_logvar.max().item():.6f}]")

    # Analyze decoder weights
    print("\n" + "-"*80)
    print("DECODER WEIGHT ANALYSIS (by latent)")
    print("-"*80)

    decoder_keys = [k for k in state_dict.keys() if 'decoder' in k and 'weight' in k.lower() and 'w_adj' not in k]
    if decoder_keys:
        # Try to find decoder output layer
        decoder_weight_key = [k for k in decoder_keys if 'layers' in k or 'output' in k]
        if decoder_weight_key:
            print(f"\nFound decoder weights: {decoder_weight_key[0]}")

    # Try to load w_adj directly from checkpoint if available
    w_adj_keys = [k for k in state_dict.keys() if 'w_adj' in k]
    if w_adj_keys:
        print(f"\nFound w_adj in checkpoint: {w_adj_keys}")
        for key in w_adj_keys:
            w_adj = state_dict[key]
            print(f"\n  {key}:")
            print(f"    Shape: {w_adj.shape}")
            if w_adj.dim() >= 2:
                # Compute L2 norm per latent
                norms = torch.norm(w_adj, dim=tuple(range(w_adj.dim()-1)))
                print(f"    Per-latent L2 norms: {norms.numpy()}")

                # Check for dead latents
                threshold = norms.mean() * 0.01  # 1% of mean
                dead_latents = (norms < threshold).nonzero(as_tuple=True)[0]
                if len(dead_latents) > 0:
                    print(f"    ⚠️  DEAD LATENTS (norm < {threshold:.6f}): {dead_latents.numpy()}")

    # Check for optimizer state
    print("\n" + "-"*80)
    print("OPTIMIZER STATE")
    print("-"*80)

    if 'optimizer_state_dict' in checkpoint:
        opt_state = checkpoint['optimizer_state_dict']
        print("\n✓ Optimizer state found in checkpoint")

        # Check if forcing encoder parameters have optimizer state
        if 'state' in opt_state:
            param_groups = opt_state.get('param_groups', [])
            print(f"  Number of parameter groups: {len(param_groups)}")

            for i, group in enumerate(param_groups):
                print(f"    Group {i}: lr={group.get('lr', 'N/A')}, {len(group.get('params', []))} parameters")
    else:
        print("\n❌ No optimizer state in checkpoint")

    # Summary and diagnosis
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)

    issues_found = []

    # Check encoder weight norms
    if 'co2_encoder' in encoder_stats and 'climate_encoder' in encoder_stats:
        ratio = encoder_stats['co2_encoder']['norm'] / encoder_stats['climate_encoder']['norm']
        if ratio < 0.1:
            issues_found.append(f"CO2 encoder weights are {ratio:.2%} of climate encoder (too small)")
        print(f"\n✓ CO2/Climate encoder weight ratio: {ratio:.2%}")

    if 'aerosol_encoder' in encoder_stats and 'climate_encoder' in encoder_stats:
        ratio = encoder_stats['aerosol_encoder']['norm'] / encoder_stats['climate_encoder']['norm']
        if ratio < 0.1:
            issues_found.append(f"Aerosol encoder weights are {ratio:.2%} of climate encoder (too small)")
        print(f"✓ Aerosol/Climate encoder weight ratio: {ratio:.2%}")

    # Check logvar values
    if co2_logvar_keys:
        co2_std = torch.exp(0.5 * state_dict[co2_logvar_keys[0]])
        if (co2_std < 0.01).any():
            issues_found.append("CO2 encoder has very small variance (std < 0.01)")
        print(f"✓ CO2 latent std range: [{co2_std.min().item():.4f}, {co2_std.max().item():.4f}]")

    if aerosol_logvar_keys:
        aerosol_std = torch.exp(0.5 * state_dict[aerosol_logvar_keys[0]])
        if (aerosol_std < 0.01).any():
            issues_found.append("Aerosol encoder has very small variance (std < 0.01)")
        print(f"✓ Aerosol latent std range: [{aerosol_std.min().item():.4f}, {aerosol_std.max().item():.4f}]")

    if issues_found:
        print("\n⚠️  ISSUES DETECTED:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ No obvious parameter issues detected")
        print("  Problem may be in training dynamics or architecture")

    print("\n" + "="*80)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_forcing_latents.py <checkpoint_path>")
        print("\nExample:")
        print("  python diagnose_forcing_latents.py /path/to/results/model_40000.pt")
        sys.exit(1)

    checkpoint_path = Path(sys.argv[1])
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    diagnose_forcing_encoders(checkpoint_path)
