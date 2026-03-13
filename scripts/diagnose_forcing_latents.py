#!/usr/bin/env python3
"""
Diagnostic script to check forcing latent encoder health.

Run this on a trained model checkpoint to diagnose why forcing latents aren't learning.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

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

    # List all forcing-related keys for debugging
    forcing_keys = [k for k in state_dict.keys() if 'forcing' in k.lower()]
    if forcing_keys:
        print(f"\nFound {len(forcing_keys)} forcing-related parameters:")
        for key in sorted(forcing_keys)[:20]:  # Show first 20
            print(f"  - {key}")
        if len(forcing_keys) > 20:
            print(f"  ... and {len(forcing_keys) - 20} more")

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
            print(f"\nCO2 Encoder ALL layers:")
            for key in sorted(co2_keys):
                w = state_dict[key]
                print(f"  {key}: shape={w.shape}, norm={w.norm().item():.4f}")

            # Use first layer for stats
            co2_weight = state_dict[co2_keys[0]]
            encoder_stats['co2_encoder'] = {
                'key': co2_keys[0],
                'shape': co2_weight.shape,
                'mean': co2_weight.mean().item(),
                'std': co2_weight.std().item(),
                'max': co2_weight.max().item(),
                'norm': co2_weight.norm().item()
            }

            # Check final layer output dimension
            final_layer_keys = [k for k in co2_keys if 'lin2' in k]
            if final_layer_keys:
                final_w = state_dict[final_layer_keys[0]]
                n_output = final_w.shape[0]
                print(f"\n  Final layer outputs {n_output} dimensions (expected 1 for n_forced_latents_co2)")
                if n_output != 1:
                    print(f"  ⚠️ ENCODER BUG: CO2 encoder outputs {n_output} instead of 1 latent!")

    # Aerosol encoder
    if has_aerosol_encoder:
        aerosol_keys = [k for k in state_dict.keys() if 'aerosol_forcing_encoder_mu' in k and 'weight' in k]
        if aerosol_keys:
            print(f"\nAerosol Encoder ALL layers:")
            for key in sorted(aerosol_keys):
                w = state_dict[key]
                print(f"  {key}: shape={w.shape}, norm={w.norm().item():.4f}")

            # Use first layer for stats
            aerosol_weight = state_dict[aerosol_keys[0]]
            encoder_stats['aerosol_encoder'] = {
                'key': aerosol_keys[0],
                'shape': aerosol_weight.shape,
                'mean': aerosol_weight.mean().item(),
                'std': aerosol_weight.std().item(),
                'max': aerosol_weight.max().item(),
                'norm': aerosol_weight.norm().item()
            }

            # Check final layer output dimension
            final_layer_keys = [k for k in aerosol_keys if 'lin2' in k]
            if final_layer_keys:
                final_w = state_dict[final_layer_keys[0]]
                n_output = final_w.shape[0]
                print(f"\n  ⚠️ CRITICAL: Final layer outputs {n_output} dimensions (expected 4 for n_forced_latents_aerosol)")
                if n_output != 4:
                    print(f"  ⚠️ ENCODER BUG: Aerosol encoder outputs {n_output} instead of 4 latents!")

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

    # CRITICAL: Check encoder correlation (are CO2 and aerosol encoders learning the same thing?)
    print("\n" + "-"*80)
    print("ENCODER CORRELATION ANALYSIS (Redundancy Check)")
    print("-"*80)

    encoder_correlation = None
    if has_co2_encoder and has_aerosol_encoder:
        co2_keys = [k for k in state_dict.keys() if 'co2_forcing_encoder_mu' in k and 'weight' in k]
        aerosol_keys = [k for k in state_dict.keys() if 'aerosol_forcing_encoder_mu' in k and 'weight' in k]

        if co2_keys and aerosol_keys:
            co2_w = state_dict[co2_keys[0]]
            aerosol_w = state_dict[aerosol_keys[0]]

            # Flatten weights to compute correlation
            co2_flat = co2_w.flatten().numpy()

            # For aerosol, we may have multiple output dimensions (n_aerosol_latents)
            # Check each aerosol latent encoder separately
            if aerosol_w.dim() == 2:
                # Shape: (input_dim, n_aerosol_latents)
                n_aerosol_latents = aerosol_w.shape[1]
                print(f"\nComparing CO2 encoder to {n_aerosol_latents} aerosol latent encoders:")

                correlations = []
                for i in range(n_aerosol_latents):
                    aerosol_flat = aerosol_w[:, i].flatten().numpy()

                    # Ensure same length for correlation
                    min_len = min(len(co2_flat), len(aerosol_flat))
                    if min_len > 1:
                        corr, p_value = pearsonr(co2_flat[:min_len], aerosol_flat[:min_len])
                        correlations.append(corr)
                        print(f"  CO2 ↔ Aerosol[{i}]: r={corr:.4f} (p={p_value:.4e})")

                encoder_correlation = np.mean(correlations)
                print(f"\n  Mean correlation: {encoder_correlation:.4f}")

                if encoder_correlation > 0.8:
                    print(f"  ⚠️  HIGH CORRELATION (>0.8): Encoders may have collapsed to similar solutions!")
                elif encoder_correlation > 0.5:
                    print(f"  ⚠️  MODERATE CORRELATION (>0.5): Some redundancy detected")
                else:
                    print(f"  ✓ Low correlation: Encoders appear to learn distinct patterns")
            else:
                # Fallback for other shapes
                aerosol_flat = aerosol_w.flatten().numpy()
                min_len = min(len(co2_flat), len(aerosol_flat))
                if min_len > 1:
                    corr, p_value = pearsonr(co2_flat[:min_len], aerosol_flat[:min_len])
                    encoder_correlation = corr
                    print(f"\n  CO2 ↔ Aerosol encoder correlation: r={corr:.4f} (p={p_value:.4e})")

                    if abs(corr) > 0.8:
                        print(f"  ⚠️  HIGH CORRELATION: Encoders collapsed!")
                    elif abs(corr) > 0.5:
                        print(f"  ⚠️  MODERATE CORRELATION: Some redundancy")
                    else:
                        print(f"  ✓ Low correlation: Distinct patterns")

    # Check bias terms
    co2_bias_keys = [k for k in state_dict.keys() if 'co2_forcing_encoder_mu' in k and 'bias' in k]
    aerosol_bias_keys = [k for k in state_dict.keys() if 'aerosol_forcing_encoder_mu' in k and 'bias' in k]

    if co2_bias_keys and aerosol_bias_keys:
        co2_bias = state_dict[co2_bias_keys[0]]
        aerosol_bias = state_dict[aerosol_bias_keys[0]]

        print(f"\n  Bias comparison:")
        print(f"    CO2 bias:     {co2_bias.numpy()}")
        print(f"    Aerosol bias: {aerosol_bias.numpy()}")

        # Check if aerosol biases are just scaled/shifted versions of CO2
        if co2_bias.numel() == 1 and aerosol_bias.numel() > 1:
            bias_std = aerosol_bias.std()
            if bias_std < 0.01:
                print(f"    ⚠️  All aerosol biases are nearly identical (std={bias_std:.4f})")

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
    decoder_norms = None
    if w_adj_keys:
        print(f"\nFound w_adj in checkpoint: {w_adj_keys}")
        for key in w_adj_keys:
            w_adj = state_dict[key]
            print(f"\n  {key}:")
            print(f"    Shape: {w_adj.shape}")
            if w_adj.dim() >= 2:
                # Compute L2 norm per latent (column norms for decoder utilization)
                # w_adj shape is typically (d_x, d_z) or (d, d_x, d_z)
                if w_adj.dim() == 3:
                    # Average over feature dimension d
                    norms = torch.norm(w_adj, dim=(0, 1))  # Norm over (d, d_x)
                else:
                    norms = torch.norm(w_adj, dim=0)  # Norm over d_x (rows)

                decoder_norms = norms
                print(f"    Per-latent decoder L2 norms: {norms.numpy()}")

                # Check for dead latents
                threshold = norms.mean() * 0.01  # 1% of mean
                dead_latents = (norms < threshold).nonzero(as_tuple=True)[0]
                if len(dead_latents) > 0:
                    print(f"    ⚠️  DEAD LATENTS (norm < {threshold:.6f}): {dead_latents.numpy()}")

                # Check if forcing latents have significantly lower norms
                # Assuming latents are ordered: [climate_latents, co2_latents, aerosol_latents]
                # This needs to be inferred from config or specified
                if len(norms) == 9:  # Example: 4 climate + 1 CO2 + 4 aerosol
                    climate_norms = norms[:4]
                    co2_norms = norms[4:5]
                    aerosol_norms = norms[5:9]

                    print(f"\n    Decoder utilization by latent type:")
                    print(f"      Climate latents (0-3): mean={climate_norms.mean():.4f}, std={climate_norms.std():.4f}")
                    print(f"      CO2 latent (4):         {co2_norms[0]:.4f}")
                    print(f"      Aerosol latents (5-8): mean={aerosol_norms.mean():.4f}, std={aerosol_norms.std():.4f}")

                    if co2_norms.mean() < climate_norms.mean() * 0.2:
                        print(f"      ⚠️  CO2 decoder usage is {(co2_norms.mean()/climate_norms.mean()):.1%} of climate")
                    if aerosol_norms.mean() < climate_norms.mean() * 0.2:
                        print(f"      ⚠️  Aerosol decoder usage is {(aerosol_norms.mean()/climate_norms.mean()):.1%} of climate")

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
    recommendations = []

    # Check encoder weight norms
    if 'co2_encoder' in encoder_stats and 'climate_encoder' in encoder_stats:
        ratio = encoder_stats['co2_encoder']['norm'] / encoder_stats['climate_encoder']['norm']
        if ratio < 0.1:
            issues_found.append(f"CO2 encoder weights are {ratio:.2%} of climate encoder (too small)")
            recommendations.append("Increase forcing_latent_supervision_coeff or check gradient flow")
        print(f"\n✓ CO2/Climate encoder weight ratio: {ratio:.2%}")

    if 'aerosol_encoder' in encoder_stats and 'climate_encoder' in encoder_stats:
        ratio = encoder_stats['aerosol_encoder']['norm'] / encoder_stats['climate_encoder']['norm']
        if ratio < 0.1:
            issues_found.append(f"Aerosol encoder weights are {ratio:.2%} of climate encoder (too small)")
            recommendations.append("Increase forcing_latent_supervision_coeff or aerosol_effect_strength")
        print(f"✓ Aerosol/Climate encoder weight ratio: {ratio:.2%}")

    # Check encoder correlation
    if encoder_correlation is not None:
        if encoder_correlation > 0.8:
            issues_found.append(f"CO2 and aerosol encoders highly correlated (r={encoder_correlation:.3f})")
            recommendations.append("CRITICAL: Encoders collapsed! Add diversity loss or increase aerosol_timing_stagger")
        elif encoder_correlation > 0.5:
            issues_found.append(f"CO2 and aerosol encoders moderately correlated (r={encoder_correlation:.3f})")
            recommendations.append("Increase temporal diversity (aerosol_timing_stagger) or spatial contrast")
        print(f"✓ CO2↔Aerosol encoder correlation: {encoder_correlation:.3f}")

    # Check decoder utilization
    if decoder_norms is not None and len(decoder_norms) == 9:
        climate_norms = decoder_norms[:4]
        forcing_norms = decoder_norms[4:]
        ratio = forcing_norms.mean() / climate_norms.mean()
        if ratio < 0.3:
            issues_found.append(f"Forcing latents underutilized by decoder ({ratio:.1%} vs climate)")
            recommendations.append("Increase decoder_utilization_coeff or min_forcing_decoder_norm")
        print(f"✓ Forcing/Climate decoder usage ratio: {ratio:.2%}")

    # Check logvar values
    if co2_logvar_keys:
        co2_std = torch.exp(0.5 * state_dict[co2_logvar_keys[0]])
        if (co2_std < 0.01).any():
            issues_found.append("CO2 encoder has very small variance (std < 0.01)")
            recommendations.append("Check if CO2 encoder is receiving gradients during training")
        print(f"✓ CO2 latent std range: [{co2_std.min().item():.4f}, {co2_std.max().item():.4f}]")

    if aerosol_logvar_keys:
        aerosol_std = torch.exp(0.5 * state_dict[aerosol_logvar_keys[0]])
        if (aerosol_std < 0.01).any():
            issues_found.append("Aerosol encoder has very small variance (std < 0.01)")
            recommendations.append("Check if aerosol encoder is receiving gradients during training")
        print(f"✓ Aerosol latent std range: [{aerosol_std.min().item():.4f}, {aerosol_std.max().item():.4f}]")

    # Print summary
    if issues_found:
        print("\n" + "="*80)
        print("⚠️  ISSUES DETECTED:")
        print("="*80)
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")

        if recommendations:
            print("\n" + "="*80)
            print("💡 RECOMMENDATIONS:")
            print("="*80)
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
    else:
        print("\n✓ No obvious parameter issues detected")
        print("  If forcing attribution is still poor, check:")
        print("    1. Training loss curves (are forcing losses decreasing?)")
        print("    2. Ground truth forcing data quality")
        print("    3. Whether forcing effects are strong enough in synthetic data")

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
