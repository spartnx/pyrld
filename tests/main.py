import os
import sys
fpath = os.path.join(os.pardir, "")
sys.path.append(fpath)

from pyrld import BayesRLD
import numpy as np

if __name__ == "__main__":
    # random number generator
    seed = 0
    rng = np.random.Generator(np.random.MT19937(seed))

    # input degradation model parameters (prior) - take from Gebraeel's PhD thesis (2003)
    phi = 0 # constant in degradation model
    mu0 = 0.02 # mean of theta (prior)
    sig0 = 1e-3 # standard deviation of theta (prior)
    mu1 = 0.02 # mean of beta (prior)
    sig1 = 1e-3 # standard deviation of beta (prior)
    sig = 1e-2 # standard deviation of error terms (dim = 1/minute**.5)
    D = 0.03 # threshold for failure detection - from Gebraeel (2007)

    # sample theta and beta
    theta = rng.lognormal(mu0, sig0)
    while theta <= 0:
        theta = rng.lognormal(mu0, sig0)
    beta = rng.normal(mu1, sig1)
    while beta <= 0:
        beta = rng.normal(mu1, sig1)

    # generate synthetic data from sampled theta and beta
    n = 1000 # number of samples
    dt = 2 # time interval between two data recording events
    signal = []
    for i in range(n):
        t = (i+1)*dt
        err = rng.normal(0, sig) # synthetic error term
        s = phi + theta*np.exp(beta*t)*np.exp(err - sig**2*t/2)

    # plot synthetic data (check)
    

    # create RLD object
    rld = BayesRLD(phi, mu0, sig0, mu1, sig1, sig, D)

    # simulate real-time degradation signal recording
    # first record
    t1 = dt
    s = signal[0]
    rld.push(t, s)
    tau = 100
    rld.rld_cdf(tau)

    # plot rld pdf after one data record


    # a few more records...
    for i in range(1, 500):
        t = (i+1)*dt
        s = signal[i]
        rld.push(t, s)

    # ... one more record
    t = t + dt
    s = signal[i+1]
    rld.push(t, s)
    tau = 100
    rld.rld_cdf(tau)

    # plot rld pdf after 500 data records
    
    
        




