import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


pendulum_data = pd.read_csv('./pendulumData.csv', delimiter=',')
cutoff = 30
# need explanation 
def iqr(data):
    q75, q25 = np.percentile(data, [75 ,25])
    iqr = q75 - q25
    return iqr

iqr_val = iqr(pendulum_data['g (m/s2)'])

median  = pendulum_data['g (m/s2)'].median()

pendulum_data = pendulum_data[np.abs(pendulum_data['g (m/s2)'] - median) <= iqr_val*cutoff]

sigma = np.std(pendulum_data['g (m/s2)'])
print(sigma)


# label the x axis g (m/s2)
plt.xlabel(r'g (m/$s^2$)')

# label the y axis Counts
plt.ylabel('Counts')

# plot histogram
# plot error bars with sqrt of counts from the column 

#plot a bar plot where the height of each bar represents the counts per bin width
plt.hist(pendulum_data[r'g (m/s2)'], bins=30, edgecolor='black')

#fit the data with a gaussian based on the mean and standard deviation of the data
mean = np.mean(pendulum_data['g (m/s2)'])
stdev = np.std(pendulum_data['g (m/s2)'])

#plot a  rectangle showing verticle width of one  standard deviation at the mean
plt.axvspan(mean - stdev, mean + stdev, color='yellow', alpha=0.5, label='1 Std Dev')

# plot a verticle line where the mean is
plt.axvline(mean, color='black', linestyle='dashed', linewidth=1, label='Mean')

# plot a gaussian with the mean and standard deviation of the data
def gaussian(x, amp, mean, stddev):
    return amp * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))
x = np.linspace(mean - 4*stdev, mean + 4*stdev, 100)
y = gaussian(x, amp=max(np.histogram(pendulum_data['g (m/s2)'], bins=30)[0]), mean=mean, stddev=stdev)
plt.plot(x, y, color='red', label='Gaussian Fit')




plt.savefig("histogram.pdf")
