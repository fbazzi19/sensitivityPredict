#libraries
import pandas as pd
import numpy as np
import sklearn
import scipy


from sklearn.neighbors import KernelDensity
from scipy.integrate import simpson
from scipy.stats import norm

def lobico_calc(y, z=1.96, t=0.15):
    """Calculation of a binary threshold on the continuous IC50 values. 
    For more information, see here. (https://doi.org/10.1038/srep36812)    

    Parameters
    ----------
    y : pandas dataframe
        IC50 values for cancer cell lines treated with the specified drug
    z : float
        Used to calculate the standard deviation during upsampling of the data, a value of
        1.96 is associated with a 95% confidence interval.
    t : float
        Used to determine how strict the threshold should be. A value of 0.05 indicates that the
        integral of the resistant cell lines distribution from negative infinity to the threshold
        should be equal to 0.05.
    
    Output
    ------
    threshold : float
        The determined binary threshold, IC50 values below are considered sensitive/
    distb : numpy matrix
        Distribution of upsampled IC50 values, returned only for further visualization purposes 
    """
    #for each cell line, sample 1000 points from a normal distribution with mean=IC50
    #and std=Z*RMSE
    distb=np.zeros(1000*y.shape[0])
    for i in range(y.shape[0]):
        start=0+i*1000
        stop=1000+i*1000
        distb[start:stop]=np.random.normal(loc=y['LN_IC50'][y.index[i]], scale=z*y['RMSE'][y.index[i]], size=1000)

    #kernel density estimation
    kde=KernelDensity(kernel="gaussian", bandwidth=0.5)
    distb = distb[:, np.newaxis]
    kde.fit(distb)

    #range for evaluation
    fmin, fmax = distb.min(), distb.max()
    x_vals = np.linspace(fmin, fmax, 1000)
    log_dens = kde.score_samples(x_vals[:, np.newaxis])
    f_vals = np.exp(log_dens)
    #Normalize f such that the integral of f(x)dx from fmin to fmax equals 1
    area_under_curve = simpson(f_vals, x=x_vals)  # Integrate using the trapezoidal rule
    f_vals /= area_under_curve  # Normalize the density values

    #set mu as the highest point of f
    mu_index = np.argmax(f_vals) #where mu is
    mu = x_vals[mu_index] #the value of mu

    # Compute the first and second derivatives
    f_prime = np.gradient(f_vals, x_vals)
    f_double_prime = np.gradient(f_prime, x_vals)
    f_triple_prime = np.gradient(f_double_prime, x_vals)

    #rule i Find θ where f' = 0, θ < μ, and f(θ) < 0.8 * f(μ)
    theta_candidates_i = x_vals[(f_prime == 0) & (x_vals < mu) & (f_vals < 0.8 * f_vals[mu_index])]
    #check the integral
    theta = None
    for candidate in reversed(theta_candidates_i): #reversed to ensure we are getting the largest theta
        integral_val = simpson(f_vals[x_vals <= candidate], x=x_vals[x_vals <= candidate])
        if integral_val > 0.05:
            theta = candidate #this is the final theta if rule i holds
            break

    #rule ii
    if theta is None:
        theta_candidates_ii = x_vals[(f_double_prime == 0) & (f_triple_prime > 0) & (x_vals < mu) & (f_vals < 0.8 * f_vals[mu_index])]
        for candidate in reversed(theta_candidates_ii):  # Iterate from largest to smallest
            integral_val = simpson(f_vals[x_vals <= candidate], x=x_vals[x_vals <= candidate])
            if integral_val > 0.05:
                theta = candidate
                break

    #rule iii
    if theta is None:
        theta = fmin

    #calculate the std of the resistant cell lines
    valid_data = distb[(distb >= theta) & (distb <= mu)]
    sigma = np.median(np.abs(valid_data - mu)) #stdev
    #population of resistant cell lines
    g=norm(loc=mu, scale=sigma**2)
    
    threshold = g.ppf(t) #the value of the threshold given t
    return threshold, distb