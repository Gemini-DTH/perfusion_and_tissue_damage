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

def extract_FE(file_path):
    return float(file_path.split('FE_')[1].split('/')[0])

## add for verification
os.chdir('./verification')
with open('gen_verif_files.yaml', "r") as configfile:
        configs_gen = yaml.load(configfile, yaml.SafeLoader)

if configs_gen['types']['couple'] != 'couple' or configs_gen['types']['healthy'] != 'unhealthy':
    print("it is not couple and healthy")
    sys.exit()

folder_path = f"./{configs_gen['types']['couple']}_{configs_gen['types']['healthy']}_{configs_gen['types']['property']}/nx/"



selected_cpld_conv_crit = [1e-4,1e-6]
selected_FE_values = [2] 
selected_nx_values = [16,32,64,128,144,160]
csv_files = glob.glob(os.path.join(folder_path, '**/Error_values.csv'), recursive=True)

data_by_tol_krylov = {tol_krylov: {'nx': [], 'flow': [], 'computation_time': []} for tol_krylov in selected_cpld_conv_crit}

# Lists to store the extracted data
nx_values = []
flow_values = []
tol_krylov_values = []
computation_time = []

y1 = "L2_norm"
y2 = "elapsed"
y1_label = r'$L_2$ Norm'
y2_label = r'time'
y1_unit = ' (Pa)'
y2_unit = ''

# y1 = "L2_norm"
# y2 = "FlowRelErr3d"
# y1_label = r'$L_2$ Norm'
# y2_label = "FlowRelErr3d"
# y1_unit = ' '
# y2_unit = ' '
# Loop through the csv files and extract data based on selected tol_krylov values
for file_path in csv_files:
    # Read the data from the CSV file
    data = pd.read_csv(file_path)
    
    # Extract nx and tol_krylov values
    nx = extract_nx(file_path)
    cpld_conv_crit = extract_cpld_conv_crit(file_path)
    # FE = extract_FE(file_path)

    
    # Only proceed if the tol_krylov is in the selected list
    if cpld_conv_crit in selected_cpld_conv_crit and nx in selected_nx_values:
        flow_rel_err = data[data["Name"] == y1]["Value"].values[0]
        time_value = data[data["Name"] == y2]["Value"].values[0] 
        
        data_by_tol_krylov[cpld_conv_crit]['nx'].append(nx)
        print("nx", nx)
        data_by_tol_krylov[cpld_conv_crit]['flow'].append(flow_rel_err)
        print("flow_rel_err", flow_rel_err)
        data_by_tol_krylov[cpld_conv_crit]['computation_time'].append(time_value)
        print("time_value", time_value)

# Sort the data by nx_values (ascending order)
sorted_indices = sorted(range(len(nx_values)), key=lambda i: nx_values[i])

# Reorder nx_values, flow_values, and tol_krylov_values based on sorted_indices
nx_values_sorted = [nx_values[i] for i in sorted_indices]
flow_values_sorted = [flow_values[i] for i in sorted_indices]
tol_krylov_values_sorted = [tol_krylov_values[i] for i in sorted_indices]
computation_time_sorted = [computation_time[i] for i in sorted_indices]
import numpy as np
import matplotlib.pyplot as plt

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(8, 6))

# Define line styles for each tol_krylov value (you can add more styles if needed)
line_styles = ['-', '--', ':', '-.']
line_style_map = {tol_krylov: line_styles[i % len(line_styles)] for i, tol_krylov in enumerate(data_by_tol_krylov)}


# Loop over each tol_krylov value and fit a log-log model (log(y) = a log(x) + b)
for tol_krylov, data in data_by_tol_krylov.items():
    # Sort the data by nx_values (ascending order)
    sorted_indices = sorted(range(len(data['nx'])), key=lambda i: data['nx'][i])

    # Reorder data based on sorted_indices
    nx_values_sorted = np.array([data['nx'][i] for i in sorted_indices])
    flow_values_sorted = np.array([data['flow'][i] for i in sorted_indices])

    exp = int(np.log10(tol_krylov))
    # Plot the data points (dots) on log-log scale
    ax1.loglog(nx_values_sorted, flow_values_sorted, 
               marker='o', linestyle='--',
               label=rf'$tol_{{\mathrm{{krylov}}}} = 10^{{{exp}}}$ ' + y1_label)

# Set labels for the first y-axis (FlowRelErr3d)
ax1.set_xlabel(r'$n_x$')
# ax1.set_ylabel(r'$\epsilon_F$')
ax1.set_ylabel(y1_label + y1_unit)
ax1.tick_params(axis='y')

# Create a second y-axis for Computation Time
ax2 = ax1.twinx()

# Loop over each tol_krylov value and fit a log-log model for computation time vs nx
for tol_krylov, data in data_by_tol_krylov.items():
    # Sort the data by nx_values (ascending order)
    sorted_indices = sorted(range(len(data['nx'])), key=lambda i: data['nx'][i])

    # Reorder data based on sorted_indices
    nx_values_sorted = np.array([data['nx'][i] for i in sorted_indices])
    computation_time_sorted = np.array([data['computation_time'][i] for i in sorted_indices])
    print("nx_values_sorted", nx_values_sorted)
    logx_flow = np.log10(nx_values_sorted)
    print("logx_flow", logx_flow)
    logy_flow = np.log10(flow_values_sorted)
    print("logy_flow", logy_flow)
    logx_time = np.log10(nx_values_sorted)
    print("logx_time", logx_time)
    logy_time = np.log10(computation_time_sorted)
    print("logy_time", logy_time)

    # y = a x + b  ⇒  斜率 a 就是 log–log 圖上的 slope
    slope_flow, intercept_flow = np.polyfit(logx_flow, logy_flow, 1)
    slope_time, intercept_time = np.polyfit(logx_time, logy_time, 1)

    # 印出到 terminal
    print(f"tol_krylov = {tol_krylov:g}: "
          f"slope_flow = {slope_flow:.4f}, slope_time = {slope_time:.4f}")


    exp = int(np.log10(tol_krylov))
    # Plot the data points (dots) on log-log scale
    ax2.loglog(nx_values_sorted, computation_time_sorted, 
               marker='s', linestyle='-', 
               label=rf'$tol_{{\mathrm{{krylov}}}} = 10^{{{exp}}}$ ' + y2_label)

# Set labels for the second y-axis (Computation Time)
ax2.set_ylabel(y2_label + y2_unit)
ax2.tick_params(axis='y')

ax1.set_xscale('log', base=2)
ax2.set_xscale('log', base=2)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc='center left',         
    bbox_to_anchor=(1.2, 0.5), 
    borderaxespad=0.
)

# 為右邊 legend 留空間
plt.tight_layout()
plt.subplots_adjust(right=0.75)

plt.savefig("nx_error.png", dpi=300, bbox_inches='tight')
plt.show()
