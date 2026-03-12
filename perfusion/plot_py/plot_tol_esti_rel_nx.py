import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import yaml
import sys
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter


def extract_nx(file_path):
    return int(file_path.split('nx_')[1].split('_')[0]) 

def extract_tol_krylov(file_path):
    return float(file_path.split('rel_tol_krylov_')[1].split('/')[0])

def extract_tol_esti(file_path):
    return float(file_path.split('tol_esti_')[1].split('/')[0])

def extract_FE(file_path):
    return float(file_path.split('FE_')[1].split('/')[0])

def get_rel_error(file_path, param):
    data = pd.read_csv(file_path)
    return data.loc[
        data["Name"] == param, "Value"
    ].values[0]

# Change directory and load the config
os.chdir('./verification')
with open('gen_verif_files.yaml', "r") as configfile:
    configs_gen = yaml.load(configfile, yaml.SafeLoader)

if configs_gen['types']['couple'] != 'decouple' or configs_gen['types']['healthy'] != 'healthy':
    print("It is not decouple and healthy")
    sys.exit()

folder_path = f"./{configs_gen['types']['couple']}_{configs_gen['types']['healthy']}_{configs_gen['types']['property']}/nx/"

# Selected values for plotting
selected_tol_krylov_values = [1e-12]
selected_tol_esti_values = [1e-1, 1e-2, 1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]
selected_FE_values = [2]
selected_nx = [16,64,144]
# selected_nx = [16]
AnalytCouplingPointsResistance = 8912694532.42066

# Find all csv files that match
csv_files = glob.glob(os.path.join(folder_path, '**/Error_values.csv'), recursive=True)

# store data
data_by_nx = {nx: {"tol": [], "err": []} for nx in selected_nx}
y1 = "CouplingPointsResistance"
y1_label = r'$\epsilon_R$'

y1 = "iter_esti_1"
y1_label = "Number of Iterations"

# y1 = "elapsed_esti_1"
# y1_label = "Computation Time (s)"

# -------- read data --------
for fp in csv_files:
    try:
        nx = extract_nx(fp)
        tol_esti = extract_tol_esti(fp)
        tol_krylov = extract_tol_krylov(fp)
        err = get_rel_error(fp, y1)

    except Exception:
        continue

    # if nx in selected_nx:
    #     if nx == 16:
    #         AnalytCouplingPointsResistance = 9212382768.32
    #     elif nx == 64:
    #         AnalytCouplingPointsResistance = 8935044435.78
    #     elif nx == 144:
    #         AnalytCouplingPointsResistance = 8917237761.34

    if nx in selected_nx and tol_krylov in selected_tol_krylov_values and tol_esti in selected_tol_esti_values:
        data_by_nx[nx]["tol"].append(tol_esti)
        if y1 == "CouplingPointsResistance":
            data_by_nx[nx]["err"].append(-(AnalytCouplingPointsResistance - err)/AnalytCouplingPointsResistance)
            # data_by_nx[nx]["err"].append(err)
        else:
            data_by_nx[nx]["err"].append(err)

# -------- plot --------
plt.figure(figsize=(8, 6))
print("data_by_nx", data_by_nx)
for nx, data in data_by_nx.items():
    if not data["tol"]:
        continue

    # sort by tol_esti
    idx = np.argsort(data["tol"])
    tol_sorted = np.array(data["tol"])[idx]
    err_sorted = np.array(data["err"])[idx]
    #semilogx
    plt.loglog(
        tol_sorted, err_sorted,
        marker='o',
        label=rf'$n_x = {nx}$'
    )
ax = plt.gca()

# Major ticks formatting (show as regular numbers, no scientific notation)
ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

# Minor ticks formatting (show as regular numbers, no scientific notation)
# ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
plt.tick_params(axis='both', labelsize=16)
plt.yticks([10, 20, 30, 40, 60])

plt.xlabel(r'$\mathrm{tol}_{\mathrm{R}}$', fontsize=18)
plt.ylabel(y1_label, fontsize=18)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
# plt.savefig("tol_esti_vs_resist_error.png", dpi=300)
plt.savefig("../figure/tol_R_vs_" + y1 + ".png", dpi=300)
plt.show()