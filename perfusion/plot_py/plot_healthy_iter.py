import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import yaml
import sys


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

selected_tol_krylov_values = [1e-12] 
selected_tol_esti_values = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12]
selected_FE_values = [2] 
selected_nx = [32] 
csv_files = glob.glob(os.path.join(folder_path, '**/Error_values.csv'), recursive=True)

# Lists to store the extracted data
nx_values = []
flow_values = []
tol_esti_values = []
computation_time = []

# y1 = "iter_esti_1"
y1 = "elapsed_esti_1"
y1_label = r'$L_2$ Norm'
y2_label = 'Computation Time'
y1_unit = ''
y2_unit = ' (s)'

cpld_conv_crit_list = []
iter_list = []

# Loop through the csv files and extract data based on selected tol_krylov values
for file_path in csv_files:
    # Read the data from the CSV file
    data = pd.read_csv(file_path)
    
    # Extract nx and tol_krylov values
    nx = extract_nx(file_path)
    tol_krylov = extract_tol_krylov(file_path)
    tol_esti = extract_tol_esti(file_path)
    FE = extract_FE(file_path)
    # Only proceed if the tol_krylov is in the selected list
    if nx in selected_nx and tol_krylov in selected_tol_krylov_values \
    and tol_esti in selected_tol_esti_values and FE in selected_FE_values:
        iteration = data[data["Name"] == y1]["Value"].values[0]
        cpld_conv_crit_list.append(tol_esti)
        iter_list.append(iteration)
# Sort the lists by cpld_conv_crit_list (small to big)
sorted_pairs = sorted(zip(cpld_conv_crit_list, iter_list))

# Unzip the sorted pairs back into separate lists
sorted_cpld_conv_crit_list, sorted_iter_list = zip(*sorted_pairs)

# Convert back to lists (since zip returns a tuple)
sorted_cpld_conv_crit_list = list(sorted_cpld_conv_crit_list)
sorted_iter_list = list(sorted_iter_list)

# print("iter_list", iter_list)
# print("cpld_conv_crit_list", cpld_conv_crit_list)
import numpy as np
import matplotlib.pyplot as plt

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(8, 6))
plt.loglog(sorted_cpld_conv_crit_list, sorted_iter_list, marker='o', linestyle='-', color='b')
slope_time, intercept_time = np.polyfit(sorted_cpld_conv_crit_list, sorted_iter_list, 1 )

print(f"tol_krylov = {selected_tol_krylov_values[0]:g}: "
        f" slope_time = {slope_time:.4f}")
# Add labels and title
plt.xlabel(r'$tol_{esti}$', fontsize = 20)
plt.ylabel(y2_label + y2_unit, fontsize = 20)
# plt.ylabel('computational time', fontsize = 12)
plt.tick_params(axis='both', which='major', labelsize=14)


# plt.savefig("../figure/healthy_tol_iter.png", dpi=300, bbox_inches='tight')
plt.savefig("../figure/healthy_tol_time.png", dpi=300, bbox_inches='tight')
