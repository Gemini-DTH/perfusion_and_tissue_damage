import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Read CSV
path = "./verification/decouple_healthy_homo/nx_32_short_5/rel_tol_esti_0.0001/rel_tol_krylov_1e-06/cpld_crit_1e-06/"
filename = path + "grad(p1)_K1_vel.csv"
df = pd.read_csv(filename)

print("df", df)
df['DOF Coordinates'] = df['DOF Coordinates'].apply(lambda coord: coord.replace(' ', ','))

df['x'] = df['DOF Coordinates'].apply(lambda coord: eval(coord)[0])  # Extract the x coordinate

velocity = df['velocity (mm/s)']

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(df['x'], velocity, marker='o', linestyle='-', color='b', label='Velocity')

ax.set_xlabel('x Coordinate (mm)', fontsize=12)
ax.set_ylabel('Velocity (mm/s)', fontsize=12)
ax.set_title('Velocity vs x Coordinate', fontsize=14)

plt.legend()
ax.grid(True)
plt.savefig("./decouple_healthy_homo/velocity_vs_x_plot.png")
