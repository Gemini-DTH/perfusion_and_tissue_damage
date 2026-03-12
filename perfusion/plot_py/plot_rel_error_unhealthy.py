import pandas as pd
import matplotlib.pyplot as plt
import re
from matplotlib.ticker import ScalarFormatter, MaxNLocator

file_paths = [
    r"./verification/couple_unhealthy_heter_fit/nx/nx_32/cpld_conv_crit_1e-12/rel_p.csv",
    # r"./verification/couple_unhealthy_heter_fit/nx/nx_32/cpld_conv_crit_1e-06/rel_p.csv"
]

plt.figure(figsize=(10, 6))

for path in file_paths:
    data = pd.read_csv(path)
    print("data['relative error pressure']", data['relative error pressure'])
    # 🔍 extract value after cpld_crit_
    match = re.search(r'cpld_crit_([^/]+)', path)
    label = f"cpld_crit = {match.group(1)}" if match else "unknown"

    plt.plot(
        data['x0 (mm)'].to_numpy(),
        data['relative error pressure'].abs().to_numpy(),
        marker='o',
        linestyle='-',
        label=None
    )

plt.xlabel('x (mm)', fontsize=20)
plt.ylabel('Relative Error of Pressure', fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=14)

ax = plt.gca()

# Apply ScalarFormatter for y-axis with power limits for scientific notation
yfmt = ScalarFormatter()
yfmt.set_powerlimits((-3, 3))  # Adjust the limits for scientific notation

ax.yaxis.set_major_formatter(yfmt)  # Set the y-axis formatter to the custom ScalarFormatter

# Set integer ticks for y-axis using MaxNLocator
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.grid(True)
plt.legend(fontsize=14)
plt.savefig('relative_error_stroke.png', dpi=300, bbox_inches='tight')
plt.show()
