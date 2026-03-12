import os
import sys
import glob
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Helpers to parse parameters from file path
# ----------------------------
import re

def extract_nx(file_path: str) -> int:
    p = file_path.replace("\\", "/")
    m = re.search(r"/nx_(\d+)_long_\d+/", p)
    if not m:
        raise ValueError(f"Cannot parse nx from path: {file_path}")
    return int(m.group(1))



def extract_tol_krylov(file_path: str) -> float:
    # path contains ".../rel_tol_krylov_1e-12/..."
    return float(file_path.split("rel_tol_krylov_")[1].split("/")[0])


def extract_tol_esti(file_path: str) -> float:
    # NOTE: your folder name looks like "rel_tol_esti_1e-06"
    # If your folder is actually "tol_esti_1e-06", change the split key below.
    return float(file_path.split("rel_tol_esti_")[1].split("/")[0])


def extract_FE(file_path: str) -> int:
    # path contains ".../FE_2/..."
    return int(file_path.split("FE_")[1].split("/")[0])


def get_value(df: pd.DataFrame, name: str):
    """Safely get value by Name from the CSV; return None if missing."""
    s = df.loc[df["Name"] == name, "Value"]
    if s.empty:
        return None
    try:
        return float(s.values[0])
    except Exception:
        return None


def mean_std(vals):
    """Return (mean, std) where std is sample std (ddof=1) if len>1 else 0."""
    vals = np.array(vals, dtype=float)
    m = float(vals.mean())
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return m, sd


# ----------------------------
# Main
# ----------------------------
def main():
    # --- paths ---
    verification_dir = Path("verification")

    if not verification_dir.exists():
        print(f"[ERROR] Cannot find folder: {verification_dir.resolve()}")
        sys.exit(1)

    yaml_path = verification_dir / "gen_verif_files.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] Cannot find config YAML: {yaml_path.resolve()}")
        sys.exit(1)

    # --- read config ---
    with yaml_path.open("r") as f:
        configs_gen = yaml.load(f, Loader=yaml.SafeLoader)

    if configs_gen["types"]["couple"] != "decouple" or configs_gen["types"]["healthy"] != "healthy":
        print("it is not decouple and healthy")
        sys.exit(0)

    folder_path = (
        verification_dir
        / f"{configs_gen['types']['couple']}_{configs_gen['types']['healthy']}_{configs_gen['types']['property']}"
        / "nx"
    )

    # --- selection filters ---
    selected_tol_krylov_values = [1e-3, 1e-6, 1e-12]
    selected_tol_esti_values = [1e-6]
    selected_FE_values = [2]

    # --- metrics (y) ---
    y1 = "L2_norm"
    y2 = "elapsed_krylov"
    y3 = "iter_krylov"

    # Labels for plots
    y1_label = r"$L_2$ Norm (Pa)"
    y2_label = r"Computation time in Perfusion Model (s)"
    y3_label = r"Iteration in Perfusion Model"

    # --- find all Error_values*.csv ---
    # includes: Error_values.csv, Error_values_1.csv, Error_values_2.csv, ...
    csv_pattern = str(folder_path / "**" / "Error_values*.csv")
    csv_files = glob.glob(csv_pattern, recursive=True)

    if not csv_files:
        print(f"[WARN] No CSV found with pattern: {csv_pattern}")
        sys.exit(0)

    # data_by_tol_krylov[tol]['per_nx'][nx] = lists of replicate values
    data_by_tol_krylov = {tol: {"per_nx": {}} for tol in selected_tol_krylov_values}

    # --- load all csvs & collect replicate values ---
    for file_path in csv_files:
        # Normalize to forward slashes for split logic (important on Windows)
        file_path_norm = file_path.replace("\\", "/")

        try:
            nx = extract_nx(file_path_norm)
            tol_krylov = extract_tol_krylov(file_path_norm)
            tol_esti = extract_tol_esti(file_path_norm)
            FE = extract_FE(file_path_norm)
        except Exception as e:
            print(f"[SKIP] cannot parse path: {file_path} | reason: {e}")
            continue

        if (tol_krylov not in selected_tol_krylov_values) or (tol_esti not in selected_tol_esti_values) or (FE not in selected_FE_values):
            continue

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"[SKIP] cannot read CSV: {file_path} | reason: {e}")
            continue

        v1 = get_value(df, y1)
        v2 = get_value(df, y2)
        v3 = get_value(df, y3)

        if v1 is None or v2 is None or v3 is None:
            print(f"[SKIP] missing metric in {file_path}")
            continue

        per_nx = data_by_tol_krylov[tol_krylov]["per_nx"]
        per_nx.setdefault(nx, {"flow": [], "time": [], "iter": []})

        per_nx[nx]["flow"].append(v1)
        per_nx[nx]["time"].append(v2)
        per_nx[nx]["iter"].append(v3)

    # Optional: print replicate counts
    for tol, d in data_by_tol_krylov.items():
        per_nx = d["per_nx"]
        if not per_nx:
            continue
        print(f"\n=== tol_krylov = {tol:g} ===")
        for nx in sorted(per_nx.keys()):
            print(f"nx={nx:>4} | #files={len(per_nx[nx]['time'])}")

    # ----------------------------
    # Plot 1: L2_norm vs nx (mean ± std)
    # ----------------------------
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    for tol_krylov, d in data_by_tol_krylov.items():
        per_nx = d["per_nx"]
        if len(per_nx) < 2:
            continue

        nxs = np.array(sorted(per_nx.keys()), dtype=float)
        l2_mean = []
        l2_std = []

        for nx in nxs:
            m, sd = mean_std(per_nx[int(nx)]["flow"])
            l2_mean.append(m)
            l2_std.append(sd)

        l2_mean = np.array(l2_mean)
        l2_std = np.array(l2_std)

        # slope in log-log (use mean)
        if np.all(l2_mean > 0):
            slope, _ = np.polyfit(np.log10(nxs), np.log10(l2_mean), 1)
            print(f"L2 slope (tol_krylov={tol_krylov:g}): {slope:.4f}")

        exp = int(np.log10(tol_krylov))
        ax1.errorbar(
            nxs,
            l2_mean,
            yerr=l2_std,
            fmt="o--",
            capsize=3,
            label=rf"$tol_{{\mathrm{{krylov}}}} = 10^{{{exp}}}$",
        )

    ax1.set_xlabel(r"$n_x$", fontsize=20)
    ax1.set_ylabel(y1_label, fontsize=18)
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.grid(True, which="both")
    ax1.legend(fontsize=14)

    plt.tight_layout()
    out1 = f"nx_L2_{configs_gen['types']['property']}.png"
    plt.savefig(out1, dpi=300)

    # ----------------------------
    # Plot 2: elapsed_krylov vs nx (mean ± std)
    # ----------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for tol_krylov, d in data_by_tol_krylov.items():
        per_nx = d["per_nx"]
        if len(per_nx) < 2:
            continue

        nxs = np.array(sorted(per_nx.keys()), dtype=float)
        t_mean = []
        t_std = []

        for nx in nxs:
            m, sd = mean_std(per_nx[int(nx)]["time"])
            t_mean.append(m)
            t_std.append(sd)

        t_mean = np.array(t_mean)
        t_std = np.array(t_std)

        # slope in log-log (use mean)
        if np.all(t_mean > 0):
            slope, _ = np.polyfit(np.log10(nxs), np.log10(t_mean), 1)
            print(f"Time slope (tol_krylov={tol_krylov:g}): {slope:.4f}")

        exp = int(np.log10(tol_krylov))
        ax2.errorbar(
            nxs,
            t_mean,
            yerr=t_std,
            fmt="s-",
            capsize=3,
            label=rf"$tol_{{\mathrm{{krylov}}}} = 10^{{{exp}}}$",
            markersize=1,
        )

    ax2.set_xlabel(r"$n_x$", fontsize=20)
    ax2.set_ylabel(y2_label, fontsize=18)
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.grid(True, which="both")
    ax2.legend(fontsize=14)

    plt.tight_layout()
    out2 = f"nx_time_{configs_gen['types']['property']}.png"
    plt.savefig(out2, dpi=300)
    plt.show()
    print(f"\nSaved figures:\n- {out1}\n- {out2}")


if __name__ == "__main__":
    main()
