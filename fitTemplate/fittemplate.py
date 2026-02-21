'''
Contributors: JLAB Staff, Tuan Nguyen
(Credit to Jim Kiger, William Mcgehee, and Xuangcheng Shao
    for writing the original matlab script)
Last Updated: 2019-Jun-15

This python script serves as a code template for performing
curve fitting using the methods outlined in Bevington and 
Robinson

The code serves two purposes:
    1) To show a workflow starting from importaning some 
        example data, manipulating it into arrays, fitting the
        data to a chosen functional form, and then making plots.
    2) To show how tools such as gradient descent and
       parabolic expansion methods are implemented
    3) To teach good vector arithmetic techniques in 
       numpy/matplotlib, such as broadcasting and matrix
       operations

Students should try to read and understand the code.
Therefore, it is suggested students add print() statements
to see exactly what happens under the hood

DO NOT USE THIS SCRIPT AS A BLACK BOX!!!
'''

### First, import some widely used python libraries
import numpy as np
from scipy.stats import chi2, t
from scipy.special import gammaln
import matplotlib.pyplot as plt
### import all our custom curve-fitting functions from the other files you downloaded in this example
from fitlin import *
from gradsearch import *
from levmar import *
### == setup matplotlib to display latex correctly ==
### These commands make changes to your matplotlibrc settings, which are 
### overall paramterers that customize the details of how matplotlib behaves.
### It includes things like default font sizes, typefaces, color schemes, etc.
### You have a matplotlibrc file somewhere in your computer that contains
### these settings. It is loaded when you import matplotlib above. The commands
### below don't change that file, but they change the rc values within this
### python session. So if you run this code once -- changing the rc setting --
### then comment out the below lines and run again, the setting is still 
### changed. However, if you close out of the python session and restart, your
### rc settings go back to the defaults from matplotlibrc. That can be 
### confusing.
plt.rc('text', usetex=True) #default is False. True uses LaTeX for all text.
plt.rc('font', family='sans-serif')


######## STEP 1: Load the data from a file ########

### there are three sample datasets:
###     bev61.txt  - a linear dataset from Bevington, 
###         in which the uncertainties are in the third column
###     bev81.txt  - a poissonian dataset from Bevington, in which the data are counts, 
###         so the uncertainties are assumed poisson, and you calculate them yourself.
###     gauss3.txt - a gaussian dataset, again with no uncertainties supplied in the file,
###         so, we'll have to create an array for errors, assuming, say, a constant error.

# bev61 = np.loadtxt('bev61.txt')
# x = bev61[:,0]
# y = bev61[:,1]
# sig = bev61[:,2]

# bev81 = np.loadtxt('bev81.txt')
# x = bev81[:,0]
# y = bev81[:,1]
# sig = np.sqrt(y)

gauss3 = np.loadtxt('gauss3.dat')
x = gauss3[:,0]
y = gauss3[:,1]
sig = np.ones(x.size) * np.sqrt(6.25)

### In all of these examples, the (x,y) data we imported is the data we want to fit
### to some functional form. In future experiments, the data you import may be raw
### data that needs some manipulation before it can be compared to theory via fitting.
### For example: the data may be a set of events, but perhaps the function from theory
### that you are comparing to is the probability distribution of events, so you need
### to histogram the imported data into new (x,y) variables before doing the fit.





######## STEP 2: Perform Fitting ########


    
### there are three fitting routines included.
### choose either to use the linear fit or the nonlinear Levenerg-Marquardt
### fit or the nonlinear gradient search.
### The included fitting routines follow the presentation in the textbook
### by Bevington, so you should be able to understand everything about their
### output. However, these routines are not optimized numerical algorithms.
### A more expert approach would perform fits using scipy.optimize.leastsq() .

### use the linear fitting algorithm? (you probably want nonlinear)
linear = False
### If non linear, use the marquardt fitting algorithm? (you generally should)
marq = True

    #### LINEAR FIT ####

if linear:
    popt, perr, chisq, yfit = fitlin(x, y, sig)



    #### NONLINEAR FIT ####

### some useful nonlinear functions
### You'll just use one of these as your fit function, or define a new one yourself.

def sin(x, A, C, w, phi):        return A * np.sin(w*x - phi) + C
def gaussian(x, A, mu, std):     return A / std * np.exp( -(x-mu)**2 / (2*std**2) )
def lorentzian(x, A, mu, gamma): return A * gamma / ( (x-mu)**2 + gamma**2 )
def poisson(x, A, lam):          return A * np.exp( x*np.log(lam) - lam - gammaln(x+1))
def NIST(x, A1, k1, A2, mu2, std2, A3, mu3, std3):  # gauss3 fit function
    return A1 * np.exp(-k1*x) + \
           A2 * np.exp( -(x-mu2)**2 / (2*std2**2) ) + \
           A3 * np.exp( -(x-mu3)**2 / (2*std3**2) )
def bevFunc(x, A0, A1, A2, t1, t2):                 # bev81 fit function
    return A0 + A1 * np.exp(-x/t1) + A2 * np.exp(-x/t2)

if not linear:


    ### require initial parameters
    # p0 = np.array([10, 900, 80, 27, 225])               # bev81 initial parameters
    p0 = np.array([94, .01, 90, 113, 20, 74, 140, 20])  # gauss3 initial parameters

    if marq:
        # popt, perr, pcov, chisq, yfit = levmar(x, y, sig, bevFunc, p0)
        popt, perr, pcov, chisq, yfit = levmar(x, y, sig, NIST, p0)
    else:
        # popt, perr, chisq, yfit = gradsearch(x, y, sig, bevFunc, p0)
        popt, perr, chisq, yfit = gradsearch(x, y, sig, NIST, p0)







######## PART 3: Report Statistics ########

    #### Calculate Chi-Squared Statistics ####

### to print a formatted list of floating numbers, the general syntax is
### *['{0:'width'.'decimals'd/f/e}'.format(val) for val in list]
###     0 asks str.format(val) to fill {0} with the first val
###     'width' is how many characters to you want to print
###     'decimals' is how many numbers to print after the decimal point
###     use d to print as an integer
###     use f to print as a normal float
###     use e to print in scientific notation

print('\nUncertainties:', *['{0:9.1e}'.format(err) for err in perr])
dof = x.size - popt.size    # degrees of freedom
print('Degrees of Freedom = {0:5d}'.format(dof))
rchisq = chisq / dof        # reduced chi-squared
prob = 100 * (1 - chi2.cdf(chisq, dof))
print('Reduced Chi-Squared = {0:6.2f}\nProbability = {1:6.1f} %'.format(rchisq,prob))
if not linear and marq:
    print('\nElements of the error matrix (Marquardt method only)\n')
    for i in range(popt.size):
        print(*['{0:9.1e}'.format(err) for err in pcov[:,i]])



    #### Calculate Residual Statistics ####

residuals = y - yfit
rss = np.sum(residuals**2)
rstd = np.std(residuals)
print('\nResidual Sum of Squares = {0:8.3f}'.format(rss))
print('Residual Standard Deviation = {0:8.3f}\n'.format(rstd))



    #### Print Fit Parameters with Errors ####

### to make a matrix out of several vectors, the general syntax is
### np.stack([vec1,vec2,...], axis=n)
###     axis is the orientation you want the vectors to be in
###     axis=0 means the vectors will be row vectors
###     axis=1 means the vectors will be column vectors

print("Summary of best fit parameters and uncertainties")
print(np.stack([popt,perr], axis=1))



    #### Calculate Confidence Levels ####

level = 0.68                # confidence level
width = t.ppf(level,dof) * perr
lower = popt - width
upper = popt + width
FitResults = np.stack([popt,lower,upper], axis=1)







######## PART 4: Plot Figures ########

    #### Plot Fit to Data ####

### to choose which subplot you want to draw to, the general syntax is
### plt.subplot('numRows', 'numCols', 'subplotIndex')
###     'numRows' is the number of rows you want
###     'numCols' is the number of columns you want
###     'subplotIndex' is the index of the subplot (surprisingly 1-indexed)
### ex: say we have 3x4 subplots. Then the indexing will be as follows
###     1  2  3  4
###     5  6  7  8
###     9 10 11 12
# NOTE: you will want to include 'plt.subplots_adjust()'
###     to space out your subplots

plt.subplots_adjust(hspace=0.7)

plt.subplot(2, 1, 1)
plt.errorbar(x, y, sig, fmt='.')
plt.plot(x, yfit, 'r', lw=3)

plt.plot(x, np.ones(y.size)*60, 'b-')
plt.plot(np.ones(x.size)*100, y, 'g-')

plt.xlim(np.min(x), np.max(x))
plt.ylim(np.min(y), np.max(y))

### matplotlib can write text in latex by prefixing your string with r
### you'll want to include the following lines to enable this:
###     plt.rc('text', usetex=True)
###     plt.rc('font', family='sans-serif')

plt.text(101, 61, r" It's resting = $\Sigma / x$", verticalalignment='bottom')
plt.xlabel(r'$X$-Axis Label [units]', fontsize=12)
plt.ylabel(r'$Y$-Axis Label [units]', fontsize=12)
plt.title(r'Descriptive Title: e.g. \textit{Norwegian Blue Parrots vs Deaths}', fontsize=12)



    #### Plot Residuals ####

plt.subplot(2, 1, 2)
plt.plot(x, np.zeros(y.size), 'r-')
plt.errorbar(x, residuals, sig, fmt='b.')

plt.xlim(np.min(x), np.max(x))
plt.xlabel(r'$X$-Axis Label [units]', fontsize=12)
plt.ylabel(r'$Y$-Axis Label [units]', fontsize=12)
plt.title(r'Descriptive Title: e.g. \textit{Parrot Residuals}', fontsize=12)

plt.show()
