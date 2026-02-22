import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.special import factorial 
from scipy import stats
import pandas as pd

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util


plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure(figsize=(10,6), dpi=300)

# Draw samples from a poissonian distribution
# use pandas to open pendulumData (1).csv
df = pd.read_csv("pendulumData (1).csv")
#print columns
print(df.columns)


# plt.plot(samples)

# By putting the number of bins to be the maximum of our data, we generate a bin for every integer
freqs, bins = np.histogram(samples, bins=np.max(samples), range=(0,np.max(samples)))
bin_centers = 0.5*(bins[1:]+bins[:-1]) # Adding the left edge and right edge together and then dividing by 2
sig = np.sqrt(freqs)


# plt.scatter(bin_centers,freqs)
bin_centers, freqs, sig = util.delete_zeros(bin_centers, freqs,sig) # Take out the zeros from our data
popt, pcov = opt.curve_fit(f=util.gaussian, xdata=bin_centers, ydata=freqs, sigma=sig, p0=[5,200, 2], absolute_sigma=True) # Fit function
uncert = np.sqrt(np.diag(pcov)) # Extract uncertainty from fit

print(popt)
print(uncert)

x = np.linspace(start=0,stop=np.max(samples),num=100,endpoint=True) # Define x for smoother plotting of our poisson fit function

# Plot chi2
chi2 = util.chisq(util.gaussian,popt,bin_centers,freqs,sig)
print(chi2)
dof = len(bin_centers) - 2
chi2_probs = 1 - stats.chi2.cdf(chi2,dof) # Get the probability by subtracting 1 from the cumulative distribution function

# add text annotations to the plot with the chi2 value, dof, and chi2 probability

plt.text(0.5, 0.95, r'$\chi^2 = {:.2f}$, DOF = {}, $p = {:.2f}$'.format(chi2, dof, chi2_probs),
         transform=plt.gca().transAxes, ha='center', va='top', fontsize=8)

#plot vertical lines at the mean and one standard deviation away from the mean
mean = popt[0]
stddev = popt[2]
plt.fill_betweenx(y=[0, 50], x1=mean - stddev, x2=mean + stddev, color='grey', alpha=0.1, label='±1 Std Dev')

# use a step plot and "bins" as the bin edges to plot the gaussian fit 
# (Previous code remains unchanged)
# plt.step(bin_centers, freqs, where='mid', label='Data', color='blue') # Step plot using bin centers
plt.fill_between(bin_centers, freqs, step='mid', color='lightblue', alpha=0.7) # Fill area underneath the step plot

plt.plot(x, util.gaussian(x, *popt)) # Plot fit function

plt.errorbar(bin_centers, freqs, yerr=sig, linewidth=1, ls='none', fmt='', capsize=3, color='lightblue') # Plot actual data
plt.axvline(mean, color='black', linestyle='dashed', linewidth=1, label='Mean')


plt.xlabel('Bin Number')
plt.ylim(bottom=0)  # Set the lower limit of the y-axis to 0
plt.ylabel('Bin Count')
plt.title('Test Curve Fit')
plt.legend()  # Add legend to explain colors
plt.savefig('example_plot.png', dpi=300)
