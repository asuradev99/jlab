import numpy as np

def gradsearch(x, y, sig, func, p0):
    '''
    Performs a gradient search nonlinear fit

    Parameters
    ----------
    x : array
        The independent data
    y : array
        The dependent data
    sig : array
        The uncertainty in y
    func : callable
        The model function to fit to
    p0 : array
        Initial guess for parameters

    Returns
    ----------
    popt : array
        Optimal values for the parameters
    perr : array
        The uncertainty on the fit parameters
    chisq : float
        The value of chi-squared
    yfit : array
        The values of the fitted line at each x
    '''

    ### See Bevington and Robinson Ch. 8 (p. 153)

    # TLDR: we minimize the chi-squared by traveling opposite the direction
    ###     of the gradient. (you might recognize this as simple gradient descent)
    ###     we use a simplified formula for the partial derivative from Bevington
    ###         dX2                   dX2/db_i
    ###         ---- = dp_i * -------------------------
    ###         dp_i          sqrt(sum( (dX2/db_i)^2 ))
    ###
    ###         dX2    X2(p + f*dp_i) - X2(p)
    ###         ---- = ----------------------
    ###         db_i             f
    ###     where f is a small number (in this case f = 0.01)

    #### YOU MAY NEED TO MODIFY THESE PARAMETERS ####
    stepsize = np.abs(p0)*0.01  # small variation for numerical calculation of gradient
    stepdown = 0.1              # how large of a step is taken for gradient descent
    chicut = 0.1                # the maximum change in chi-squared allowed

    p = p0
    chi2 = calcchi2(x, y, sig, func, p)
    chi1 = chi2 + chicut*2      # placeholder value for chi1

    #### Perform Gradient Descent ####

    i = 0
    print('i \t\t Chisqr \t p0 \t p1 \t...')
    while(np.abs(chi2-chi1) > chicut):
        i = i + 1
        print('{0:3d}{1:12.1f}'.format(i, chi2),
            *['{0:7.1f}'.format(p) for p in p])
        pnew, stepsum = gradstep(x, y, sig, func, p, stepsize, stepdown)
        p = pnew
        stepdown = stepsum
        chi1 = chi2
        chi2 = calcchi2(x, y, sig, func, p)

    print('Final{1:10.1f}'.format(i, chi2),
            *['{0:7.1f}'.format(p) for p in p])

    #### Calculate Uncertainties ####

    perr = sigparab(x, y, sig, func, p, stepsize)
    yfit = func(x, *p)

    return p, perr, chi2, yfit



def gradstep(x, y, sig, func, p, stepsize, stepdown):
    '''
    Moves opposite the gradient of chi-squared. Calculates
    the new parameters and the total distance traveled

    Parameters
    ----------
    x : array
        The independent data
    y : array
        The dependent data
    sig : array
        The uncertainty in y
    func : callable
        The model function to fit to
    p : array
        Current parameters
    stepsize : array
        Small variation for numerical calculation of gradient
    stepdown : float
        How large of a step is taken for gradient descent

    Returns
    ----------
    pnew : array
        The new parameters found by traveling opposite the gradient
    stepsum : float
        Stepdown required to barely pass minimum in parameter space
    '''

    chi2 = calcchi2(x, y, sig, func, p)
    grad = calcgrad(x, y, sig, func, p, stepsize)
    chi3 = chi2 * 1.1           # placeholder value for chi3
    chi1 = chi2 * 0.9           # placeholder value for chi1
    stepdown = stepdown * 2
    while(chi3 > chi2):         # keep decreasing the stepdown until grad > 0
        stepdown = stepdown/2
        pnew = p + stepdown*grad
        chi3 = calcchi2(x, y, sig, func, pnew)

    stepsum = 0
    while(chi3 < chi2):         # keep taking steps until minimum is passed
        stepsum = stepsum + stepdown
        chi1 = chi2
        chi2 = chi3
        pnew = pnew + stepdown*grad
        chi3 = calcchi2(x, y, sig, func, pnew)

    ### approximating the minimum as a parabola, find the minimum in p
    ### See Bevington and Robinson Ch. 8 (p. 147)

    step1 = stepdown * ( (chi3-chi2) / (chi1-2*chi2+chi3) + 0.5 )
    pnew = pnew - step1*grad

    return pnew, stepsum



def calcchi2(x, y, sig, func, p):
    '''
    Calculate chi-squared at current parameters

    Parameters
    ----------
    x : array
        The independent data
    y : array
        The dependent data
    sig : array
        The uncertainty in y
    func : callable
        The model function to fit to
    p : array
        Current parameters

    Returns
    ----------
    chisq : float
        The value of chi-squared
    '''

    ### we specify the axis that we sum over because of "broadcasting"
    ### magic employed when calculating the gradient
    ### in the case you want to use for loops instead, you can omit 'axis=0'

    return np.sum( ( (y - func(x,*p)) / sig )**2, axis=0)



def calcgrad(x, y, sig, func, p, stepsize):
    '''
    Calculate negative gradient of chi-squared

    Parameters
    ----------
    x : array
        The independent data
    y : array
        The dependent data
    sig : array
        The uncertainty in y
    func : callable
        The model function to fit to
    p : array
        Current parameters
    stepsize : array
        Small variation for numerical calculation of gradient

    Returns
    ----------
    grad : float
        The negative gradient of chi-squared
    '''

    ### See Bevington and Robinson Ch. 8 (p. 154)

    f = 0.01
    chisq2 = calcchi2(x, y, sig, func, p)

    ### the original algorithm uses 'for loops' to calculate each partial derivative.
    ### in numpy/matlab, we can be more efficient by using 'broadcasting'
    ###
    ### normally, you can only operate on arrays with the same shape and size
    ### however, if you operate on arrays of different shape/size, then numpy
    ### will automatically expand the arrays such that they become the same size
    ###
    ### thus, you never use a for loop, which is often less efficient than vector math
    ### 
    ### ex:           (4)     (1 2 3)   (4 4 4)     (5 6 7)
    ###     (1 2 3) + (5)  =  (1 2 3) + (5 5 5)  =  (6 7 8)
    ###               (6)     (1 2 3)   (6 6 6)     (7 8 9)
    ###
    ### in our specific example, we want to broadcast all N variations in the
    ### parameters simultaneously, rather than doing it one at a time
    ###
    ###          (a0+da0 a0     a0    )
    ###     p2 = (a1     a1+da1 a1    ) = (p2+da0 p2+da1 p2+da2)
    ###          (a2     a2     a2+da2)
    ###
    ### x begins as a 1D array, which we can't broadcast to
    ### so we use x[:,np.newaxis] to turn it into a 2D vector
    ###
    ###                       (x0 )
    ###     x[:,np.newaxis] = (x1 )
    ###                       (...)
    ###
    ### thus, when we call calcchi2, x gets broadcast to match the shape of p2
    ### the result is that we calculate every partial derivative simultaneously
    ###
    ### if you are confused, try printing out p2 and x[:,np.newaxis]

    p2 = (p + f*np.diag(stepsize)).T
    chisq1 = calcchi2(x[:,np.newaxis], y[:,np.newaxis], sig[:,np.newaxis], func, p2)
    grad = chisq2 - chisq1

    ### NOTE: if you want to use a for loop, the following code will suffice
    # nparam = p.size
    # grad = np.zeros(nparam)
    # for n in range(nparam):
    #     p2 = p.copy()               # reset p2
    #     dp = f * stepsize[n]
    #     p2[n] = p2[n] + dp          # add a small variation to the nth param
    #     chisq1 = calcchi2(x, y, sig, func, p2)
    #     grad[n] = chisq2 - chisq1

    t = np.sum(grad**2)

    return stepsize * grad / t



def sigparab(x, y, sig, func, p, stepsize):
    '''
    Calculate negative gradient of chi-squared

    Parameters
    ----------
    x : array
        The independent data
    y : array
        The dependent data
    sig : array
        The uncertainty in y
    func : callable
        The model function to fit to
    p : array
        Current parameters
    stepsize : array
        Small variation for numerical calculation of gradient

    Returns
    ----------
    perr : array
        The uncertainty on the fit parameters
    '''

    ### read the comments on 'broadcasting' from the calcgrad function
    ### this time i won't go easy on you

    chisq2 = calcchi2(x, y, sig, func, p)

    p2 = (p + np.diag(stepsize)).T
    chisq3 = calcchi2(x[:,np.newaxis], y[:,np.newaxis], sig[:,np.newaxis], func, p2)

    p2 = p2 - 2*np.diag(stepsize)
    chisq1 = calcchi2(x[:,np.newaxis], y[:,np.newaxis], sig[:,np.newaxis], func, p2)

    perr = stepsize * np.sqrt( 2 / (chisq1-2*chisq2+chisq3) )

    ### NOTE: if you want to use a for loop, the following code will suffice
    # nparam = p.size
    # perr = np.zeros(nparam)
    # for n in range(nparam):
    #     p2 = p.copy()               # reset p2
    #     dp = stepsize[n]
    #     p2[n] = p2[n] + dp          # add a small variation to the nth param
    #     chisq3 = calcchi2(x, y, sig, func, p2)
    #     p2[n] = p2[n] - 2*dp
    #     chisq1 = calcchi2(x, y, sig, func, p2)
    #     perr[n] = dp * np.sqrt( 2 / (chisq1-2*chisq2+chisq3) )

    return perr