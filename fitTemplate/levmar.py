import numpy as np

def levmar(x, y, sig, func, p0):
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
    pcov : 2d array
        The estimated covariance matrix
    chisq : float
        The value of chi-squared
    yfit : array
        The values of the fitted line at each x
    '''

    ### See Bevington and Robinson Ch. 8 (p. 162)

    # TLDR: the marquardt method combines two different methods into one:
    ###         1) gradient search (see gradsearch.py)
    ###         2) parabolic method (where we approximate the minimum as a parabola)
    ###     we introduce the curvature matrix alpha (briefly mentioned in fitlin.py),
    ###     proportional to the hessian matrix of chi-squared wrt parameter space
    ###
    ###     the parabolic method is neatly written as a matrix equation
    ###         beta = alpha * da
    ###         beta_i \propto -dX2/dp_i
    ###         alpha_ij \propto d^2X2/dp_idp_j
    ###     we solve for da and let popt = p0 + dp
    ###     unfortunately, the parabolic method only works close to a minimum
    ###
    ###     by adding a term (1+lambda) to the diagonal of alpha, the behaviour of
    ###     the marquardt method changes depending on the size of lambda
    ###         lambda > 1   =>   diagonals dominate, emulating gradient search
    ###         lambda ~ 0   =>   diagonals unchanged, emulating parabolic method
    ###     thus, far from the minimum we want lamgda > 1
    ###     then, closer to the minimum we want lambda ~ 0

    #### YOU MAY NEED TO MODIFY THESE PARAMETERS ####
    stepsize = np.abs(p0)*0.001 # small variation for numerical calculation of derivatives
    chicut = 0.01               # the maximum change in chi-squared allowed

    ### exact algorithm steps given on pg 162 of Bevington

    p = p0
    chi2 = calcchi2(x, y, sig, func, p) ### ALGORITHM STEP 1
    lam = 0.00001                       ### ALGORITHM STEP 2
    chi1 = chi2 + chicut*2      # placeholder value for chi1

    print('Marquardt gradient-expansion algorithm')
    i = 0
    print('  i\tChisqr \t    lambda\tp0\tp1\t...')
    while(np.abs(chi2-chi1) > chicut): 
        i = i + 1
        print('{0:3d}{1:12.1f}{2:12.1e}'.format(i, chi2, lam),
            *['{0:7.1f}'.format(p) for p in p])

        chinew = chi2 + 1       # placeholder value for chinew
        while(chinew > chi2 + chicut):  ### ALGORITHM STEP 3
            dp = calcdp(x, y, sig, func, p, stepsize, lam)
            pnew = p + dp
            chinew = calcchi2(x, y, sig, func, pnew)
            if(chinew > chi2):          ### ALGORITHM STEP 4
                lam = lam * 10          ### if chi-squared increases, increase lambda

        lam = lam / 10                  ### ALGORITHM STEP 5
        p = pnew                        ### if chi-squared decreases, decrease lambda
        chi1 = chi2
        chi2 = chinew

    pcov = calcinvalpha(x, y, sig, func, p, stepsize, lam)

    ### the original algorithm uses a 'for loop' to get errors from the cov matrix
    ### numpy allows us to do so directly, without any for loops

    perr = np.sqrt( np.diag(pcov) )

    ### NOTE: if you want to use a for loop, the following code will suffice
    # nparam = p.size
    # perr = np.zeros(nparam)
    # for n in range(nparam):
    #     perr[n] = np.sqrt( pcov[n,n] )

    yfit = func(x, *p)

    return p, perr, pcov, chi2, yfit



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



def calcdp(x, y, sig, func, p, stepsize, lam):
    '''
    Calculate parameter stepdown for given lambda

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
        Small variation for numerical calculation of derivatives
    lam : float
        Degree to which diagonals of curvature matrix dominate

    Returns
    ----------
    dp : array
        The optimal change in parameters
    '''

    pcov = calcinvalpha(x, y, sig, func, p, stepsize, lam)
    der = calcderiv(x, y, sig, func, p, stepsize)

    ### the original algorithm uses TWO FOR LOOPS (double the sin)
    ### please don't do this
    ### IFF you are using python 3.5+, multiplying a vector and a matrix
    ### is as simple as b = A @ v
    ### otherwise, use b = A.dot(v)

    # beta = ( (y-func(x,*p)) / sig**2 ) @ der    # python 3.5+
    beta = ( (y-func(x,*p)) / sig**2 ).dot(der) # else

    ### NOTE: if you care not for sin, then the following code suffices
    # nparam = p.size
    # ndata = x.size
    # beta = np.zeros(nparam)
    # for m in range(nparam):
    #     for n in range(ndata):
    #         beta[m] = beta[m] + der[n,m] * (y[n]-func(x[n],*p)) / sig[n]**2

    # dp = pcov @ beta                            # python 3.5+
    dp = pcov.dot(beta)                         # else

    return dp



def calcinvalpha(x, y, sig, func, p, stepsize, lam):
    '''
    Calculate inverse of curvature matrix, which is the covariance matrix

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
        Small variation for numerical calculation of derivatives
    lam : float
        Degree to which diagonals of curvature matrix dominate

    Returns
    ----------
    pcov : 2d array
        The covariance matrix at current parameters
    '''

    der = calcderiv(x, y, sig, func, p, stepsize)

    ### the original algorithm uses THREE FOR LOOPS
    ### seriously please. one is ok, but three is too many
    ### for an explanation of what is going on here, please read the comments
    ### in the calcgrad function in gradsearch.py

    # alpha = (der/sig[:,np.newaxis]).T @ (der/sig[:,np.newaxis]) # python 3.5+
    alpha = (der/sig[:,np.newaxis]).T.dot(der/sig[:,np.newaxis])# else

    alpha = alpha * (np.ones((p.size,p.size)) + lam)

    ### NOTE: if you truly want to use for loops, then the following code suffices
    nparam = p.size
    ndata = x.size
    alpha = np.zeros((nparam,nparam))
    for m in range(nparam):
        for n in range(nparam):
            for l in range(ndata):
                alpha[m,n] = alpha[m,n] + der[l,m]*der[l,n]/sig[l]**2
    for n in range(nparam):
        alpha[n,n] = (1+lam) * alpha[n,n]

    pcov = np.linalg.inv(alpha)

    return pcov



def calcderiv(x, y, sig, func, p, stepsize):
    '''
    Numerically calculate derivative of func(x_i, *p) wrt p_j

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
        Small variation for numerical calculation of derivatives
    lam : float
        Degree to which diagonals of curvature matrix dominate

    Returns
    ----------
    der : 2d array
        Derivative of fit function wrt parameters
    '''

    ### the original algorithm uses for loops
    ### again, read comments from the calcgrad function in gradsearch.py
    ### here, we must transpose stepsize[:,np.newaxis] because it is a
    ### vector in parameter space, as opposed to data space

    p2 = (p + np.diag(stepsize)).T
    y0 = func(x, *p)
    y1 = func(x[:,np.newaxis], *p2)
    der = (y1 - y0[:,np.newaxis]) / stepsize[:,np.newaxis].T

    ### NOTE: if you want to use for loops, then the following code suffices
    # nparam = p.size
    # ndata = x.size
    # der = np.zeros((ndata,nparam))
    # for m in range(ndata):
    #     y0 = func(x[m],*p)
    #     for n in range(nparam):
    #         p[n] = p[n] + stepsize[n]
    #         y1 = func(x[m],*p)
    #         p[n] = p[n] - stepsize[n]
    #         der[m,n] = (y1 - y0) / stepsize[n]

    return der