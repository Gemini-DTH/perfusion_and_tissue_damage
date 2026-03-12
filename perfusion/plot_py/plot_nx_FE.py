import pandas as pd
import numpy as np
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
    return float(file_path.split('rel_tol_esti_')[1].split('/')[0])

def extract_FE(file_path):
    return float(file_path.split('FE_')[1].split('/')[0])

def get_L2_norm(file_path):
    data = pd.read_csv(file_path)
    return data.loc[data["Name"] == "L2_norm", "Value"].values[0]

    
def sort_by_nx(nx_list, FE_list):
    if not nx_list or not FE_list:
        return [], []
    pairs = sorted(zip(nx_list, FE_list), key=lambda t: t[0])
    nx_sorted = [p[0] for p in pairs]
    FE_sorted = [p[1] for p in pairs]
    return nx_sorted, FE_sorted

def compute_slope(nx, y):
    logx = np.log(nx)
    logy = np.log(y)
    p, _ = np.polyfit(logx, logy, 1)
    return p


## add for verification
os.chdir('./verification')
with open('gen_verif_files.yaml', "r") as configfile:
        configs_gen = yaml.load(configfile, yaml.SafeLoader)

if configs_gen['types']['couple'] != 'decouple' or configs_gen['types']['healthy'] != 'healthy':
    print("it is not decouple and healthy")
    sys.exit()

folder_path = f"./{configs_gen['types']['couple']}_{configs_gen['types']['healthy']}_{configs_gen['types']['property']}/nx/"


selected_tol_krylov_values = [1e-12] 
selected_tol_esti_values = [1e-6]
selected_FE_values = [1,2,3] 
csv_files = glob.glob(os.path.join(folder_path, '**/Error_values.csv'), recursive=True)

# Lists to store extracted data
nx_values = []
L2_norm_FE1 = []
L2_norm_FE2 = []
L2_norm_FE3 = []


# nx v.s. L2
# nx v.s. FlowRelErr3d
# nx v.s. 
# y1 = "FlowRelErr3d"
# y2 = "elapsed_krylov"
y1 = "L2_norm"
y2 = "elapsed_krylov"
y1_label = r'$L_2$ Norm'
y2_label = r'$\epsilon_F$'
y1_unit = ' (Pa)'
y2_unit = ''

# store per FE
nx1, y1 = [], []
nx2, y2 = [], []
nx3, y3 = [], []

# Loop through each file and extract nx, FE, and L2 norm values
for file_path in csv_files:
    nx = extract_nx(file_path)
    fe = extract_FE(file_path)
    l2 = get_L2_norm(file_path)
    tol_krylov = extract_tol_krylov(file_path)
    tol_esti = extract_tol_esti(file_path)
    print("tol_krylov", tol_krylov)
    print("tol_esti", tol_esti)
    print("nx", nx)
    print("fe", fe)

    if tol_krylov in selected_tol_krylov_values and tol_esti in selected_tol_esti_values:
        
        if fe == 1:
            nx1.append(nx); y1.append(l2)
        elif fe == 2:
            nx2.append(nx); y2.append(l2)
        elif fe == 3:
            nx3.append(nx); y3.append(l2)

# sort each line by nx
nx1, y1 = sort_by_nx(nx1, y1)
nx2, y2 = sort_by_nx(nx2, y2)
nx3, y3 = sort_by_nx(nx3, y3)
# plot
fig, ax = plt.subplots(figsize=(10, 6))
if nx1:
    ax.loglog(nx1, y1, label="linear", marker="o")
    slope1 = compute_slope(nx1, y1)
    print(f"FE1 slope = {slope1:.2f}")

if nx2:
    ax.loglog(nx2, y2, label="quadratic", marker="x")
    slope2 = compute_slope(nx2, y2)
    print(f"FE2 slope = {slope2:.2f}")

if nx3:
    ax.loglog(nx3, y3, label="cubic", marker="s")
    slope3 = compute_slope(nx3, y3)
    print(f"FE3 slope = {slope3:.2f}")


ax.tick_params(axis='x', which='major', labelsize=20)  
ax.tick_params(axis='y', which='major', labelsize=20)  
ax.set_xscale('log', base=2)
ax.set_xlabel("$n_x$", fontsize = 20)
ax.set_ylabel(r'$L_2$ Norm (Pa)', fontsize = 20)
ax.legend(fontsize=16)
ax.grid(True)
plt.savefig("nx_FE.png", dpi=300, bbox_inches="tight")
plt.show()