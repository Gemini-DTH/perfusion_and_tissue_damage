import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import yaml
import sys
from matplotlib.ticker import MaxNLocator


def extract_nx(file_path):
    return int(file_path.split('nx_')[1].split('_')[0]) 

def extract_tol_krylov(file_path):
    return float(file_path.split('rel_tol_krylov_')[1].split('/')[0])

def extract_tol_esti(file_path):
    return float(file_path.split('tol_esti_')[1].split('/')[0])

def extract_FE(file_path):
    return float(file_path.split('FE_')[1].split('/')[0])


## add for verification
os.chdir('./verification')
with open('gen_verif_files.yaml', "r") as configfile:
        configs_gen = yaml.load(configfile, yaml.SafeLoader)

if configs_gen['types']['couple'] != 'decouple' or configs_gen['types']['healthy'] != 'healthy':
    print("it is not decouple and healthy")
    sys.exit()

folder_path = f"./{configs_gen['types']['couple']}_{configs_gen['types']['healthy']}_{configs_gen['types']['property']}/nx/"


selected_tol_krylov_values = [1e-3, 1e-6, 1e-12] 
selected_tol_esti_values = [1e-6]
selected_FE_values = [2] 
csv_files = glob.glob(os.path.join(folder_path, '**/Error_values.csv'), recursive=True)

data_by_tol_krylov = {tol_krylov: {'nx': [], 'flow': [], 'computation_time': [], 'iter_krylov': []} for tol_krylov in selected_tol_krylov_values}

y1 = "L2_norm"
y2 = "elapsed_krylov"
y3 = "iter_krylov"
y1_label = r'$L_2$ Norm'
y2_label = r'$\epsilon_F$'
y1_unit = ' (Pa)'
y2_unit = ''

# y1 = "L2_norm"
# y2 = "FlowRelErr3d"
# y1_label = r'$L_2$ Norm'
# y2_label = "FlowRelErr3d"
# y1_unit = ' '
# y2_unit = ' '

for file_path in csv_files:
    # Read the data from the CSV file
    data = pd.read_csv(file_path)
    
    # Extract nx and tol_krylov values
    nx = extract_nx(file_path)
    tol_krylov = extract_tol_krylov(file_path)
    tol_esti = extract_tol_esti(file_path)
    FE = extract_FE(file_path)

    # Only proceed if the tol_krylov is in the selected list
    if tol_krylov in selected_tol_krylov_values and tol_esti in selected_tol_esti_values and FE in selected_FE_values:
        flow_rel_err = data[data["Name"] == y1]["Value"].values[0]
        time_value = data[data["Name"] == y2]["Value"].values[0]
        iter_value = data[data["Name"] == y3]["Value"].values[0]
        
        data_by_tol_krylov[tol_krylov]['nx'].append(nx)
        data_by_tol_krylov[tol_krylov]['flow'].append(flow_rel_err)
        data_by_tol_krylov[tol_krylov]['computation_time'].append(time_value)
        data_by_tol_krylov[tol_krylov]['iter_krylov'].append(iter_value)


import numpy as np
import matplotlib.pyplot as plt

fig1, ax1 = plt.subplots(figsize=(8, 6))

for tol_krylov, data in data_by_tol_krylov.items():
    if len(data['nx']) < 2:
        continue

    # sort by nx
    idx = np.argsort(data['nx'])
    nx_sorted = np.array(data['nx'])[idx]
    l2_sorted = np.array(data['flow'])[idx]

    # slope (log-log)
    slope, _ = np.polyfit(np.log10(nx_sorted), np.log10(l2_sorted), 1)
    print(f"L2 slope (tol_perfusion={tol_krylov:g}): {slope:.4f}")

    exp = int(np.log10(tol_krylov))
    ax1.loglog(
        nx_sorted, l2_sorted,
        marker='o', linestyle='--',
        label=rf'$tol_{{\mathrm{{perfusion}}}} = 10^{{{exp}}}$'
    )

# ax1.set_xlabel(r'$n_x$', fontsize=20)
# ax1.set_ylabel(r'$L_2$ norm (Pa)', fontsize=20)
# ax1.set_xscale('log', base=2)
# ax1.tick_params(axis='both', which='major', labelsize=18)
# ax1.grid(True)
# ax1.legend(fontsize=16)

# plt.tight_layout()
# plt.savefig(f"nx_L2_{configs_gen['types']['property']}.png", dpi=300)
# plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 6))

for tol_krylov, data in data_by_tol_krylov.items():
    if len(data['nx']) < 2:
        continue

    # sort by nx
    idx = np.argsort(data['nx'])
    nx_sorted = np.array(data['nx'])[idx]
    time_sorted = np.array(data['computation_time'])[idx]

    # slope (scaling)
    slope, _ = np.polyfit(np.log10(nx_sorted), np.log10(time_sorted), 1)
    print(f"Time slope (tol_krylov={tol_krylov:g}): {slope:.4f}")

    exp = int(np.log10(tol_krylov))
    ax2.loglog(
        nx_sorted, time_sorted,
        marker='s', linestyle='-',
        label=rf'$tol_{{\mathrm{{perfusion}}}} = 10^{{{exp}}}$'
    )

ax2.set_xlabel(r'$n_x$', fontsize=20)
ax2.set_ylabel(r'Computation time in Perfusion Model(s)', fontsize=18)
ax2.set_xscale('log', base=2)
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.grid(True)
ax2.legend(fontsize=16)

plt.tight_layout()
plt.savefig(f"nx_time_{configs_gen['types']['property']}.png", dpi=300)
plt.show()

# fig3, ax3 = plt.subplots(figsize=(8, 6))

# for tol_krylov, data in data_by_tol_krylov.items():
#     if len(data['nx']) < 2:
#         continue

#     # sort by nx
#     idx = np.argsort(data['nx'])
#     nx_sorted = np.array(data['nx'])[idx]
#     iter_sorted = np.array(data['iter_krylov'])[idx]
#     print("nx_sorted", nx_sorted)
#     print("iter_sorted", iter_sorted)

#     # slope (scaling)
#     slope, _ = np.polyfit(np.log10(nx_sorted), np.log10(iter_sorted), 1)
#     print(f"Iter slope (tol_krylov={tol_krylov:g}): {slope:.4f}")

#     exp = int(np.log10(tol_krylov))
#     ax3.loglog(
#         nx_sorted, iter_sorted,
#         marker='s', linestyle='-',
#         label=rf'$tol_{{\mathrm{{perfusion}}}} = 10^{{{exp}}}$'
#     )

# ax3.set_xlabel(r'$n_x$', fontsize=20)
# ax3.set_ylabel(r'Iteration in Perfusion Model', fontsize=20)
# ax3.set_xscale('log', base=2)
# ax3.tick_params(axis='both', which='major', labelsize=18)
# ax3.grid(True)
# ax3.legend(fontsize=16)

# plt.tight_layout()
# plt.savefig(f"nx_iter_{configs_gen['types']['property']}.png", dpi=300)
# plt.show()
