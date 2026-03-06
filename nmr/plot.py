import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('purcell_nmr.csv', header=None)

x = data[0]
y = data[1]

# Create the plot
plt.figure()
plt.loglog(x, y)
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Log-Log Plot of purcell_nmr.csv')
plt.grid(True)
plt.show()

