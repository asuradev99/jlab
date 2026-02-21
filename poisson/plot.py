import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

fname = "FileC017.txt"
df = pd.read_csv(fname,skiprows=7,delim_whitespace=True,
                     names = ['Event', 'Time', 'Date', 'TimeStamp', 'ADC1', 'ADC2', 'SiPM', 'Temp', 'Pressure', 'DeadTime', 'Coincident', 'ID'])
event = df["Event"].array
timestamp = df["TimeStamp"].array

t1 = 0
t2 = 1000
print("First time stamp is ", timestamp[0])
print("Last time stamp is ",timestamp[-1])
plt.figure(figsize = (4,4))

# counts, e1,e2, xxx = plt.hist2d(event[0:10000],timestamp[0:10000],bins=(100,100),cmin=0.1)
# plt.ylabel("Timestamp [ms]")
# plt.xlabel("Event number")
# plt.show()
#make histogram with 100 10s bins

plt.figure(figsize = (4,4))
counts, e2, xxx = plt.hist(timestamp/1000,range=(0,1000),bins=100)
counts2, e3, xxx = plt.hist(timestamp/1000,range=(1000,2000),bins=100)

# plt.xlabel("time [s]")
# plt.ylabel("No of events")
# plt.show()

# Compute the running average from 0 up to the jth interval based on the counts variable
counts_running_avg = np.cumsum(counts) / np.arange(1, len(counts) + 1)
counts_running_avg_2 = np.cumsum(counts2) / np.arange(1, len(counts2) + 1)

# Plot the running average
plt.figure(figsize = (4,4))
plt.plot(e2[:-1], counts_running_avg, label='Running Average from t=0-1000s (10 bins)', color='red')
plt.plot(e3[:-1]-1000, counts_running_avg_2, label='Running Average from t=1000-2000s (10 bins)', color='blue')

# add error bars to the running average plot where each error bar is the standard deviation of the counts in the corresponding bin
bin_stderr = np.sqrt(counts_running_avg[-1])/np.sqrt(np.arange(1, len(counts_running_avg)+1))  # Standard error for Poisson distribution
bin_stderr_2 = np.sqrt(counts_running_avg_2[-1])/np.sqrt(np.arange(1, len(counts_running_avg_2)+1))  # Standard error for Poisson distribution
# plot the error bars
plt.errorbar(e2[:-1], counts_running_avg, yerr=bin_stderr, fmt='o', color='red', label='Standard Deviation')
plt.errorbar(e3[:-1] - 1000, counts_running_avg_2, yerr=bin_stderr_2, fmt='o', color='blue', label='Standard Deviation')

plt.xlabel("time [s]")
plt.ylabel("Cumulative Avg No of events")
plt.legend()
plt.show()

#find the p value for the null hypothesis that the two count rates are the same using the z test
mean1 = counts_running_avg[-1]
mean2 = counts_running_avg_2[-1]
std1 = np.sqrt(mean1 / len(counts_running_avg))
std2 = np.sqrt(mean2 / len(counts_running_avg_2))
z = (mean1 - mean2) / np.sqrt(std1**2 + std2**2)
p_value = 2 * (1 - norm.cdf(abs(z)))  # Two-tailed test
print("Z-score:", z)
print("P-value:", p_value)



