import pandas as pd
import matplotlib.pyplot as plt
import os
import glob


def extract_nx(file_path):
    return int(file_path.split('nx_')[1].split('_')[0]) 
    # return int(file_path.split('nx_')[1].split('/')[0]) 

def extract_tol_esti(file_path):
    return float(file_path.split('rel_tol_krylov_')[1].split('/')[0])
    # return float(file_path.split('cpld_conv_crit_')[1].split('/')[0])


# folder_path = "verification/couple_unhealthy_homo/nx/"
folder_path = "verification/decouple_healthy_homo/nx/"
# selected_tol_esti_values = [1e-02,1e-03,1e-06, 1e-10]  # Replace with the desired tolerance estimates
selected_tol_esti_values = [1e-08,1e-06,1e-12,1e-20, 1e-4] 
selected_tol_esti_values = [1e-3, 1e-6, 1e-12] 
# selected_tol_esti_values = [1e-06,1e-12, 1e-20]  
# Use glob to find all the Error_values.csv files in subdirectories
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
y1 = "L2_norm"
y2 = "elapsed_krylov"
y1_label = r'$L_2$ Norm'
y2_label = 'Time'
y1_unit = ' (Pa)'
y2_unit = ' (s)'
# Loop through the csv files and extract data based on selected tol_esti values
for file_path in csv_files:
    # Read the data from the CSV file
    data = pd.read_csv(file_path)
    
    # Extract nx and tol_esti values
    nx = extract_nx(file_path)
    tol_esti = extract_tol_esti(file_path)
    
    # Only proceed if the tol_esti is in the selected list
    if tol_esti in selected_tol_esti_values:
        flow_rel_err = data[data["Name"] == y1]["Value"].values[0]
        time_value = data[data["Name"] == y2]["Value"].values[0]
        
        data_by_tol_esti[tol_esti]['nx'].append(nx)
        data_by_tol_esti[tol_esti]['flow'].append(flow_rel_err)
        data_by_tol_esti[tol_esti]['computation_time'].append(time_value)

# Sort the data by nx_values (ascending order)
sorted_indices = sorted(range(len(nx_values)), key=lambda i: nx_values[i])

# Reorder nx_values, flow_values, and tol_esti_values based on sorted_indices
nx_values_sorted = [nx_values[i] for i in sorted_indices]
flow_values_sorted = [flow_values[i] for i in sorted_indices]
tol_esti_values_sorted = [tol_esti_values[i] for i in sorted_indices]
computation_time_sorted = [computation_time[i] for i in sorted_indices]
import numpy as np
import matplotlib.pyplot as plt

# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(8, 6))

# Define line styles for each tol_esti value (you can add more styles if needed)
line_styles = ['-', '--', ':', '-.']
line_style_map = {tol_esti: line_styles[i % len(line_styles)] for i, tol_esti in enumerate(data_by_tol_esti)}


# Loop over each tol_esti value and fit a log-log model (log(y) = a log(x) + b)
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
               label=rf'$tol_{{\mathrm{{esti}}}} = 10^{{{exp}}}$ ' + y1_label)

# Set labels for the first y-axis (FlowRelErr3d)
ax1.set_xlabel(r'$n_x$')
# ax1.set_ylabel(r'$\epsilon_F$')
ax1.set_ylabel(y1_label + y1_unit, color='b')
ax1.tick_params(axis='y', labelcolor='b')

# Create a second y-axis for Computation Time
ax2 = ax1.twinx()

# Loop over each tol_esti value and fit a log-log model for computation time vs nx
for tol_esti, data in data_by_tol_esti.items():
    # Sort the data by nx_values (ascending order)
    sorted_indices = sorted(range(len(data['nx'])), key=lambda i: data['nx'][i])

    # Reorder data based on sorted_indices
    nx_values_sorted = np.array([data['nx'][i] for i in sorted_indices])
    computation_time_sorted = np.array([data['computation_time'][i] for i in sorted_indices])

    exp = int(np.log10(tol_esti))
    # Plot the data points (dots) on log-log scale
    ax2.loglog(nx_values_sorted, computation_time_sorted, 
               marker='s', linestyle='-', 
               label=rf'$tol_{{\mathrm{{esti}}}} = 10^{{{exp}}}$ ' + y2_label)

# Set labels for the second y-axis (Computation Time)
ax2.set_ylabel(y2_label + y2_unit, color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

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

plt.savefig("nx_error_block.png", dpi=300, bbox_inches='tight')
plt.show()
