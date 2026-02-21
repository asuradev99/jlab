import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.special import factorial 
from scipy import stats
import math

def dgaussian(x, mu, std):     return 1 / (np.sqrt(2*np.pi) * std) * np.exp( -(x-mu)**2 / (2*std**2) )

def gaussian(x, mu, A, std):     return A / (np.sqrt(2*np.pi) * std) * np.exp( -(x-mu)**2 / (2*std**2) )


def two_gaussian(x, mu1, A1, std1, mu2, A2, std2):     
    return A1 / std1 * np.exp( -(x-mu1)**2 / (2*std1**2) ) + A2/ std2 * np.exp( -(x-mu2)**2 / (2*std2**2) )

def poisson(x,lam,a):
    '''
    Inputs:
    a = normalization constant
    x = corresponding number of events, should be an array
    lam = expected value, which is equal to the variance for poisson distribution
    Outputs: probability of x number of events occuring, scaled by constant a
    '''
    return a*lam**x*np.exp(-lam)/factorial(x)

def delete_zeros(bins,counts,err):
    '''
    Inputs:
    bins = the frequency bin centers
    counts = the frequency data
    err = the error data
    Output:
    new_bins = bin centers, but if the frequency of a bin center is zero the bin is removed
    new_counts = frequency data, but if "                    "
    new_err = error data, but if "                       "
    '''
    zeros = np.where(counts==0) # Find the indices where the frequency data is zero
    mask = np.ones(len(counts),dtype=bool) # create a mask of True values
    mask[zeros[0]] = False # Turn the zero parts of the mask False
    new_counts = counts[mask] # Recreate the bin data without the False parts
    new_bins = bins[mask] # Recreate the frequency data without the False parts
    new_err = err[mask] # Recrete the "      "
    return new_bins, new_counts, new_err

def chisq(func,popt,x,y,sig):
    '''
    Inputs:
    func = function to generate expected value
    x = x data
    y = y data
    sig = sigma data
    Outputs:
    chi2 = chi-squared value
    '''
    expected_vals = func(x, *popt) # Again, better off using *popt
    return np.sum((y-expected_vals)**2/sig**2)

def fit_gaussian(counts, bins=50, p0=None):
    '''
    Fit a Gaussian to a histogram of counts.
    Inputs:
    counts       = 1D array of raw data values
    bins         = number of histogram bins (default 50)
    p0           = initial parameter guess [mu, A, std]; inferred from data if None
    Outputs (tuple):
    popt         = fitted parameters [mu, A=1, std] (amplitude forced to 1 for normalised PDF)
    uncert       = 1-sigma uncertainties on popt from covariance matrix
    bin_centers  = centres of non-zero bins used in fit
    freqs        = normalised frequency (probability density) for each bin_center
    sig          = Poissonian uncertainty on each freqs value
    bin_edges    = full array of bin edges from np.histogram
    normalization= bin_width × N, used to convert between density and counts
    chi2         = chi-squared of the fit (computed before amplitude is forced to 1)
    dof          = degrees of freedom (len(bin_centers) - 2)
    chi2_prob    = p-value of the chi-squared
    '''
    if p0 is None:
        p0 = [np.mean(counts), 1, np.std(counts)]
    freqs, bin_edges = np.histogram(counts, bins=bins)
    normalization = (bin_edges[1] - bin_edges[0]) * len(counts)
    sig = np.sqrt(freqs) / normalization
    freqs = freqs / normalization
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    mask = freqs > 0
    bin_centers, freqs, sig = bin_centers[mask], freqs[mask], sig[mask]
    popt, pcov = opt.curve_fit(f=gaussian, xdata=bin_centers, ydata=freqs, sigma=sig, p0=p0, absolute_sigma=True)
    uncert = np.sqrt(np.diag(pcov))
    chi2 = chisq(gaussian, popt, bin_centers, freqs, sig)
    dof = len(bin_centers) - 2
    chi2_prob = 1 - stats.chi2.cdf(chi2, dof)
    popt[1] = 1.0  # force amplitude to 1 so the curve is a normalised PDF
    return popt, uncert, bin_centers, freqs, sig, bin_edges, normalization, chi2, dof, chi2_prob
