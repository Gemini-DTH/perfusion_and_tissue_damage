import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import yaml
import sys


def extract_nx(file_path):
    return int(file_path.split('nx_')[1].split('/')[0]) 

def extract_cpld_conv_crit(file_path):
    return float(file_path.split('cpld_conv_crit_')[1].split('/')[0])

def get_rel_error(file_path, param):
    data = pd.read_csv(file_path)
    return data.loc[
        data["Name"] == param, "Value"
    ].values[0]


## add for verification
os.chdir('./verification')
with open('gen_verif_files.yaml', "r") as configfile:
        configs_gen = yaml.load(configfile, yaml.SafeLoader)

if configs_gen['types']['couple'] != 'couple' or configs_gen['types']['healthy'] != 'unhealthy':
    print("it is not couple and unhealthy")
    sys.exit()

folder_path = f"./{configs_gen['types']['couple']}_{configs_gen['types']['healthy']}_{configs_gen['types']['property']}/nx/"

selected_cpld_conv_crit = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18,1e-19]
selected_nx = [16,64,144]
csv_files = glob.glob(os.path.join(folder_path, '**/Error_values.csv'), recursive=True)

data_by_nx = {nx: {"tol": [], "err": []} for nx in selected_nx}

# y1 = "L2_norm"
# y1_label = r'$L_2$ Norm (Pa)'
# type1 = "L2"

# y1 = "total_elapsed"
# y1_label = r'Computation Time (s)'
# type1 = "time"
y1 = "total_iter"
y1_label = r'Number of Iterations'
type1 = "iter"
for fp in csv_files:
    try:
        nx = extract_nx(fp)
        cpld_conv_crit = extract_cpld_conv_crit(fp)
        err = get_rel_error(fp, y1)
    except Exception:
        continue

    if nx in selected_nx and cpld_conv_crit in selected_cpld_conv_crit:
        data_by_nx[nx]["tol"].append(cpld_conv_crit)
        data_by_nx[nx]["err"].append(err)

plt.figure(figsize=(8, 6))
print("data_by_nx", data_by_nx)
for nx, data in data_by_nx.items():
    if not data["tol"]:
        continue

    # sort by tol_esti
    idx = np.argsort(data["tol"])
    tol_sorted = np.array(data["tol"])[idx]
    err_sorted = np.array(data["err"])[idx]

    plt.loglog(
        tol_sorted, err_sorted,
        marker='o',
        label=rf'$n_x = {nx}$'
    )
plt.tick_params(axis='both', labelsize=16)
plt.xlabel(r'$\mathrm{tol}_{\mathrm{couple}}$', fontsize=20)
plt.ylabel(y1_label, fontsize = 20)
plt.grid(True)
plt.legend(fontsize=14)
plt.tight_layout()
# plt.savefig("tol_esti_vs_resist_error.png", dpi=300)
plt.savefig(f"tol_{type1}_unhealthy.png", dpi=300)
plt.show()
