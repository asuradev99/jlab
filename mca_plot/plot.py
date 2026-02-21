import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Set LaTeX font 
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load data and plot grid
data = np.loadtxt('uranium.spe', skiprows=12, max_rows=2047)  # Load lines 13 to 2060
bins = np.linspace(0, 3.0, 2048)  # 2048 bin boundaries
plt.grid(True, alpha=0.3)

# Manually create the histogram with a filled step plot
plt.fill_between(bins[:-1], data, step='post', color='dodgerblue', alpha=0.9, label='Data')
plt.xlabel('Energy (MeV)')
plt.ylabel('Bin Count')
plt.title(' Natural Uranium Sample Energy Spectrum')

#Claude Code helped me to format the y axis with one digit and add annotation for fun
class OneDigitScalarFormatter(ScalarFormatter):
    def _set_format(self):
        self.format = "%.1g"

formatter = OneDigitScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))
plt.gca().yaxis.set_major_formatter(formatter)

plt.text(0.5, 0.95, r'Acquisition time: $110 \times 10^3$ seconds',
         transform=plt.gca().transAxes, ha='center', va='top', fontsize=8)
# End of Claude Code addition


plt.savefig("uranium_bar_plot.pdf")
plt.savefig("uranium_bar_plot.png", dpi=300)  
