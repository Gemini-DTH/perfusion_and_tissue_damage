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



selected_tol_krylov_values = [1e-6] 
selected_tol_esti_values = [1e-6]
selected_FE_values = [2] 
csv_files = glob.glob(os.path.join(folder_path, '**/Error_values.csv'), recursive=True)

data_by_tol_esti = {tol_esti: {'nx': [], 'flow': [], 'computation_time': []} for tol_esti in selected_tol_esti_values}

# Lists to store the extracted data
nx_values = []
flow_values = []
tol_esti_values = []
computation_time = []

# nx v.s. L2
# nx v.s. FlowRelErr3d
# nx v.s. 
# y1 = "FlowRelErr3d"
# y2 = "elapsed_krylov"
y1 = "FlowRelErr3d"
y1_label = r'$\epsilon_Q$'
y1_unit = ''

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
    if tol_krylov in selected_tol_krylov_values and tol_esti in selected_tol_esti_values and FE in selected_FE_values:
        flow_rel_err = data[data["Name"] == y1]["Value"].values[0]  
        data_by_tol_esti[tol_esti]['nx'].append(nx)
        print("nx", nx)
        data_by_tol_esti[tol_esti]['flow'].append(flow_rel_err)
        print("flow_rel_err", flow_rel_err)
        

# Sort the data by nx_values (ascending order)
sorted_indices = sorted(range(len(nx_values)), key=lambda i: nx_values[i])

# Reorder nx_values, flow_values, and tol_krylov_values based on sorted_indices
nx_values_sorted = [nx_values[i] for i in sorted_indices]
flow_values_sorted = [flow_values[i] for i in sorted_indices]
tol_esti_values_sorted = [tol_esti_values[i] for i in sorted_indices]
import numpy as np
import matplotlib.pyplot as plt

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(8, 6))

# Define line styles for each tol_krylov value (you can add more styles if needed)
line_styles = ['-', '--', ':', '-.']
line_style_map = {tol_esti: line_styles[i % len(line_styles)] for i, tol_esti in enumerate(data_by_tol_esti)}


# Loop over each tol_krylov value and fit a log-log model (log(y) = a log(x) + b)
for tol_esti, data in data_by_tol_esti.items():
    # Sort the data by nx_values (ascending order)
    sorted_indices = sorted(range(len(data['nx'])), key=lambda i: data['nx'][i])

    # Reorder data based on sorted_indices
    nx_values_sorted = np.array([data['nx'][i] for i in sorted_indices])
    flow_values_sorted = np.array([data['flow'][i] for i in sorted_indices])

    exp = int(np.log10(tol_esti))
    # Plot the data points (dots) on log-log scale
    ax1.loglog(nx_values_sorted, flow_values_sorted, 
               marker='o', linestyle='--',
               label=rf'$tol_{{\mathrm{{krylov}}}} = 10^{{{exp}}}$ ' + y1_label)

# Set labels for the first y-axis (FlowRelErr3d)
ax1.set_xlabel(r'$n_x$', fontsize = 20)
# ax1.set_ylabel(r'$\epsilon_F$')
ax1.set_ylabel(y1_label + y1_unit, color='b', fontsize = 20)
ax1.tick_params(axis='y', labelcolor='b')
ax1.tick_params(axis='x', which='major', labelsize=14)  
ax1.tick_params(axis='y', which='major', labelsize=14)  


# Loop over each tol_krylov value and fit a log-log model for computation time vs nx
for tol_esti, data in data_by_tol_esti.items():
    # Sort the data by nx_values (ascending order)
    sorted_indices = sorted(range(len(data['nx'])), key=lambda i: data['nx'][i])

    # Reorder data based on sorted_indices
    nx_values_sorted = np.array([data['nx'][i] for i in sorted_indices])

    logx_flow = np.log10(nx_values_sorted)
    print("logx_flow", logx_flow)
    logy_flow = np.log10(flow_values_sorted)
    # y = a x + b  ⇒  斜率 a 就是 log–log 圖上的 slope
    slope_flow, intercept_flow = np.polyfit(logx_flow, logy_flow, 1)
   

    print(f"tol_esti = {tol_esti:g}: "
          f"slope_flow = {slope_flow:.4f}")


    exp = int(np.log10(tol_esti))
    # Plot the data points (dots) on log-log scale
  
ax1.set_xscale('log', base=2)

lines1, labels1 = ax1.get_legend_handles_labels()


plt.tight_layout()
plt.subplots_adjust(right=0.75)

plt.savefig("nx_flow_error.png", dpi=300, bbox_inches='tight')
plt.show()
