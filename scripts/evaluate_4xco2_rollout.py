"""Thesis a3 test: compare particle-filter rollouts under abrupt-4xCO2 forcing with the held-out NorESM2-LM run.

Reads the per-step trajectories that ``scripts/rollout_bf.py`` writes
(``rollouts/<setting>_<scenario>/trajectory_iteration_<t>.npy`` and ``gt_all_y_<scenario>.npz``) for one or more
trained runs, converts the normalised anomalies back to kelvin (the loader z-scores piControl-climatology anomalies
with the historical target mean/std stored in ``*_train+val_statistics.npy``), and reports global-mean warming per
decade for prediction and truth.

Usage
-----
python scripts/evaluate_4xco2_rollout.py --stats <...train+val_statistics.npy> --out <dir> \
    baseline=<run folder> A=<run folder> B2=<run folder>

Each ``name=<run folder>`` points at a training-run folder; the rollout sub-folder is found automatically
(first match of ``rollouts/*_<scenario>``), or give the rollout folder itself. ``--scenario ssp370`` evaluates an
in-range scenario rollout the same way (b1 check).
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MONTHS_PER_YEAR = 12


def find_rollout_dir(run_folder: Path, scenario: str) -> Path:
    if run_folder.name.endswith(scenario):
        return run_folder
    hits = sorted(glob.glob(str(run_folder / "rollouts" / f"*_{scenario}")))
    if not hits:
        raise FileNotFoundError(f"no rollouts/*_{scenario} under {run_folder}")
    if len(hits) > 1:
        print(f"[warn] {len(hits)} rollout folders under {run_folder}, using the newest")
        hits.sort(key=lambda p: Path(p).stat().st_mtime)
    return Path(hits[-1])


def load_trajectories(rollout_dir: Path):
    """Return (T, n_particles, npix) predictions in normalised units."""
    files = sorted(glob.glob(str(rollout_dir / "trajectory_iteration_*.npy")))
    files = [f for f in files if not f.endswith("_global.npy")]
    files.sort(key=lambda f: int(Path(f).stem.split("_")[-1]))
    if not files:
        raise FileNotFoundError(f"no trajectory_iteration_*.npy in {rollout_dir}")
    out = []
    for f in files:
        a = np.load(f)  # (n_particles, 1, 1, npix) in Shan's layout
        out.append(a.reshape(a.shape[0], -1, a.shape[-1]).mean(axis=1))  # (n_particles, npix)
    return np.stack(out, axis=0)


def load_truth(rollout_dir: Path, scenario: str):
    data = np.load(rollout_dir / f"gt_all_y_{scenario}.npz")
    gt = data["all_y"]  # (T, 1, 1, npix)
    return gt.reshape(gt.shape[0], -1, gt.shape[-1]).mean(axis=1)  # (T, npix)


def to_kelvin(x, mean, std):
    """Normalised anomaly -> kelvin anomaly relative to the piControl climatology."""
    return x * std + mean


def gmst(x):
    """HEALPix pixels are equal-area, so the global mean is the plain pixel mean."""
    return x.mean(axis=-1)


def annual(x):
    n = (len(x) // MONTHS_PER_YEAR) * MONTHS_PER_YEAR
    return x[:n].reshape(-1, MONTHS_PER_YEAR).mean(axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="name=<run folder or rollout folder>")
    ap.add_argument("--stats", required=True, help="*_train+val_statistics.npy written by the loader")
    ap.add_argument("--out", required=True, help="output folder for the csv/json/png")
    ap.add_argument("--windows", default="1-10,11-20,41-50,77-86", help="year windows for the table")
    ap.add_argument("--scenario", default="abrupt-4xCO2", help="rollout scenario folder suffix, e.g. ssp370")
    args = ap.parse_args()

    stats = np.load(args.stats, allow_pickle=True).item()
    t_mean = float(np.squeeze(stats["target"]["mean"]))
    t_std = float(np.squeeze(stats["target"]["std"]))
    print(f"target normalisation: mean {t_mean:.5f} std {t_std:.5f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    windows = [tuple(int(v) for v in w.split("-")) for w in args.windows.split(",")]

    results = {}
    truth_annual = None
    fig, ax = plt.subplots(figsize=(10, 5))
    for spec in args.runs:
        name, folder = spec.split("=", 1)
        rdir = find_rollout_dir(Path(folder), args.scenario)
        pred = load_trajectories(rdir)  # (T, P, npix)
        truth = load_truth(rdir, args.scenario)  # (T, npix)
        T = min(len(pred), len(truth))
        pred, truth = pred[:T], truth[:T]
        g_pred = gmst(to_kelvin(pred, t_mean, t_std))  # (T, P)
        g_true = gmst(to_kelvin(truth, t_mean, t_std))  # (T,)
        a_pred = annual(g_pred.mean(axis=1))
        a_sd = annual(g_pred.std(axis=1))
        a_true = annual(g_true)
        if truth_annual is None:
            truth_annual = a_true
        row = {"rollout_dir": str(rdir), "months": int(T), "years": int(len(a_pred))}
        for lo, hi in windows:
            sl = slice(lo - 1, hi)
            if hi > len(a_pred):
                continue
            row[f"pred_K_y{lo}-{hi}"] = float(a_pred[sl].mean())
            row[f"true_K_y{lo}-{hi}"] = float(a_true[sl].mean())
            row[f"ratio_y{lo}-{hi}"] = float(a_pred[sl].mean() / a_true[sl].mean())
        row["rmse_annual_K"] = float(np.sqrt(np.mean((a_pred - a_true) ** 2)))
        results[name] = row
        yrs = np.arange(1, len(a_pred) + 1)
        ax.plot(yrs, a_pred, label=f"{name} (particle mean)")
        ax.fill_between(yrs, a_pred - a_sd, a_pred + a_sd, alpha=0.15)
        np.savez(
            out / f"gmst_{name}.npz", pred_monthly=g_pred, true_monthly=g_true, pred_annual=a_pred, true_annual=a_true
        )

    if truth_annual is not None:
        ax.plot(
            np.arange(1, len(truth_annual) + 1),
            truth_annual,
            color="black",
            lw=2,
            label=f"NorESM2-LM {args.scenario} (held out)",
        )
    ax.set_xlabel(f"year of the {args.scenario} rollout")
    ax.set_ylabel("global-mean surface temperature anomaly vs piControl climatology [K]")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"gmst_{args.scenario}_rollouts.png", dpi=130)

    with open(out / "summary.json", "w") as f:
        json.dump(results, f, indent=1)
    keys = [k for k in next(iter(results.values())) if k.startswith(("pred_K", "true_K", "ratio", "rmse"))]
    with open(out / "summary.csv", "w") as f:
        f.write("run," + ",".join(keys) + "\n")
        for name, row in results.items():
            f.write(name + "," + ",".join(f"{row[k]:.3f}" for k in keys) + "\n")
    print(open(out / "summary.csv").read())
    print(f"written to {out}")


if __name__ == "__main__":
    main()
