import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the CSV file
file_path = r"./verification/decouple_healthy_heter_fit/nx/nx_32_long_5/rel_tol_esti_1e-06/rel_tol_krylov_1e-06/cpld_crit_1e-06/FE_2/rel_p.csv"
file_path = r"./verification/couple_unhealthy_heter_fit/nx/nx_32/cpld_conv_crit_1e-06/rel_p.csv"
data = pd.read_csv(file_path)

# Step 2: Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(data['x0 (mm)'].to_numpy(), data['relative error pressure'].abs().to_numpy(), marker='o', linestyle='-', color='b')

# Labeling the axes
plt.xlabel('x (mm)', fontsize = 20)
plt.ylabel('Relative Error of Pressure', fontsize = 20)
plt.tick_params(axis='x', which='major', labelsize=14)  
plt.tick_params(axis='y', which='major', labelsize=14)  

ax = plt.gca()
ax.yaxis.get_offset_text().set_fontsize(14)

# Display the plot
plt.grid(True)

# Save the plot as an image (e.g., in PNG format)
output_image_path = 'relative_error.png'  # Set the path where you want to save the image
plt.savefig(output_image_path, format='png', dpi=300, bbox_inches='tight')

plt.show()
