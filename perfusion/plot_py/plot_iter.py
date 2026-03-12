import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import yaml
import sys


def extract_nx(file_path):
    return int(file_path.split('nx_')[1].split('/')[0]) 

def extract_cpld_conv_crit(file_path):
    return float(file_path.split('cpld_conv_crit_')[1].split('/')[0])

## add for verification
os.chdir('./verification')
with open('gen_verif_files.yaml', "r") as configfile:
        configs_gen = yaml.load(configfile, yaml.SafeLoader)

if configs_gen['types']['couple'] != 'couple' or configs_gen['types']['healthy'] != 'unhealthy':
    print("it is not couple and unhealthy")
    sys.exit()

folder_path = f"./{configs_gen['types']['couple']}_{configs_gen['types']['healthy']}_{configs_gen['types']['property']}/nx/"

selected_cpld_conv_crit = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15,1e-16, 1e-17, 1e-18, 1e-19, 1e-20]
selected_nx = [32] 
csv_files = glob.glob(os.path.join(folder_path, '**/Error_values.csv'), recursive=True)

# Lists to store the extracted data
nx_values = []
flow_values = []
tol_esti_values = []
computation_time = []

y1 = "total_iter"
y1 = "L2_norm"
y2 = "total_elapsed"
y1 = "total_elapsed"
y1_label = "total_iter"
y1_label = "total_elapsed"
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
    cpld_conv_crit = extract_cpld_conv_crit(file_path)
    
    # Only proceed if the tol_krylov is in the selected list
    if cpld_conv_crit in selected_cpld_conv_crit and nx in selected_nx:
        print("nx", nx)
        iteration = data[data["Name"] == y1]["Value"].values[0]
        print("cpld_conv_crit", cpld_conv_crit)
        print("iteration", iteration)
        cpld_conv_crit_list.append(cpld_conv_crit)
        print("cpld_conv_crit_list", cpld_conv_crit_list)
        iter_list.append(iteration)
# Sort the lists by cpld_conv_crit_list (small to big)
sorted_pairs = sorted(zip(cpld_conv_crit_list, iter_list))

# Unzip the sorted pairs back into separate lists
sorted_cpld_conv_crit_list, sorted_iter_list = zip(*sorted_pairs)

# Convert back to lists (since zip returns a tuple)
sorted_cpld_conv_crit_list = list(sorted_cpld_conv_crit_list)
sorted_iter_list = list(sorted_iter_list)
import numpy as np
import matplotlib.pyplot as plt

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(8, 6))
plt.loglog(sorted_cpld_conv_crit_list, sorted_iter_list, marker='o', linestyle='-', color='b')

# Add labels and title
plt.xlabel('cpld_conv_crit')
# plt.ylabel('Iterations')
plt.ylabel('time')
plt.title(f"{configs_gen['types']['couple']}_{configs_gen['types']['healthy']}_{configs_gen['types']['property']}_time")

# plt.savefig("../figure/unhealthy_tol_iter.png", dpi=300, bbox_inches='tight')
plt.savefig("../figure/unhealthy_tol_time.png", dpi=300, bbox_inches='tight')

