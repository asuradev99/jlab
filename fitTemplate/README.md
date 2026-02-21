# fitTemplate
An example template for a common Junior Lab work flow: importing data from a file, defining a function to fit that data to, performing the fit, and then outputting results, including plots.

Run fittemplate.py. 

By default, it loads the data file gauss3.dat and fits it using the algorithm in levmar.py. You could also try the other example data files, bev61.txt and bev81.txt; or try one of the other fitting routines, fitlin.py and gradsearch.py.

Note, the fitting routines included here have the advantage that the follow along with the description of the relevant algorithms given in the textbook of Bevington & Robinson. However, they are not in any way optimized code. For more professional results, try scipy.optimize.leastsq().
