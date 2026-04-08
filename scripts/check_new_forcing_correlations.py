#!/usr/bin/env python3
"""
Check forcing latent correlations in the NEW SAVAR dataset with orthogonal spatial templates.
"""

import numpy as np
from scipy.stats import pearsonr

# NEW data directory with updated parameters
data_dir = "/hkfs/work/workspace_haic/scratch/qa4548-climate_ws/SAVAR_DATA_TEST/m_4_tl_10000_ifd_True_dif_easy_ns_0.25_ses_False_ol_False_f1_0_f2_5.0_ft1_2500_ft2_7500_rmp_sinusoidal_lin_linear_pds_[2]_asp_2.0_asc_1.5_art_2500_apt_5000_adt_7500"

co2_latent = np.load(f"{data_dir}/co2_latent_trajectory.npy")
aerosol_latent = np.load(f"{data_dir}/aerosol_latent_trajectory.npy")

print(f"CO2 latent: {co2_latent.shape}")
print(f"Aerosol latent: {aerosol_latent.shape}")

# Aerosol is (4, time) so transpose to (time, 4)
if aerosol_latent.shape[0] == 4:
    aerosol_latent = aerosol_latent.T
    print(f"Transposed aerosol to: {aerosol_latent.shape}")

print("\n" + "="*80)
print("GROUND TRUTH FORCING LATENT CORRELATIONS (NEW DATA)")
print("="*80)

# CO2 vs each of the 4 aerosol latents
print("\nCO2 ↔ Aerosol latents (temporal correlation):")
for i in range(4):
    corr, p = pearsonr(co2_latent, aerosol_latent[:, i])
    print(f"  CO2 ↔ Aerosol[{i}]: r={corr:.4f}, p={p:.4e}")

# CO2 vs mean of aerosols
aerosol_mean = aerosol_latent.mean(axis=1)
corr_mean, p_mean = pearsonr(co2_latent, aerosol_mean)
print(f"\n  CO2 ↔ Mean(Aerosol): r={corr_mean:.4f}, p={p_mean:.4e}")

if abs(corr_mean) > 0.5:
    print("  ⚠️  MODERATE CORRELATION: Limited aerosol independence")
elif abs(corr_mean) > 0.3:
    print("  ⚠️  WEAK CORRELATION: Some CO2-aerosol coupling remains")
else:
    print("  ✓ Low correlation: Good aerosol independence")

# Inter-aerosol correlations
print("\n" + "="*80)
print("INTER-AEROSOL CORRELATIONS (should be diverse)")
print("="*80)
inter_aerosol_corrs = []
for i in range(4):
    for j in range(i+1, 4):
        corr, _ = pearsonr(aerosol_latent[:, i], aerosol_latent[:, j])
        inter_aerosol_corrs.append(abs(corr))
        print(f"  Aerosol[{i}] ↔ Aerosol[{j}]: r={corr:.4f}")

mean_inter_corr = np.mean(inter_aerosol_corrs)
print(f"\n  Mean |correlation|: {mean_inter_corr:.4f}")

if mean_inter_corr > 0.5:
    print("  ⚠️  HIGH: Aerosol latents still too correlated")
elif mean_inter_corr > 0.3:
    print("  ⚠️  MODERATE: Some redundancy remains")
else:
    print("  ✓ Low correlation: Good aerosol diversity")

# Signal strength
print("\n" + "="*80)
print("FORCING SIGNAL STRENGTH")
print("="*80)
print(f"CO2 std: {co2_latent.std():.4f}")
print(f"Aerosol stds: {[f'{aerosol_latent[:, i].std():.4f}' for i in range(4)]}")
print(f"Mean aerosol std: {aerosol_latent.std(axis=0).mean():.4f}")

aerosol_to_co2_ratio = aerosol_latent.std(axis=0).mean() / co2_latent.std()
print(f"\nAerosol/CO2 signal ratio: {aerosol_to_co2_ratio:.4f}")

if aerosol_to_co2_ratio < 0.2:
    print("  ⚠️  Aerosol signal still much weaker than CO2")
elif aerosol_to_co2_ratio < 0.5:
    print("  ⚠️  Aerosol signal somewhat weak compared to CO2")
else:
    print("  ✓ Aerosol signal strength comparable to CO2")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

issues = []
if abs(corr_mean) > 0.4:
    issues.append(f"CO2-Aerosol correlation still moderate (r={corr_mean:.3f})")
if mean_inter_corr > 0.4:
    issues.append(f"Inter-aerosol correlation high ({mean_inter_corr:.3f})")
if aerosol_to_co2_ratio < 0.3:
    issues.append(f"Aerosol signal weak ({aerosol_to_co2_ratio:.1%} of CO2)")

if issues:
    print("\n⚠️  Issues detected:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    print("\nOrthogonal spatial templates may need further tuning.")
else:
    print("\n✓ Ground truth forcing data looks good!")
    print("  - Low CO2-aerosol correlation")
    print("  - Low inter-aerosol correlation")
    print("  - Adequate signal strength")
    print("\n→ Ready to train model on this dataset")
