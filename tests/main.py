import os
import sys
fpath = os.path.join(os.pardir, "")
sys.path.append(fpath)

from pyrld import BayesRLD
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # random number generator
    seed = 1000
    rng = np.random.Generator(np.random.MT19937(seed))

    # input degradation model parameters (prior) - take from Gebraeel's PhD thesis (2003)
    phi = 0.005 # constant in degradation model
    mu0 = -7.6 # mean of log(theta) (prior) - from Fig. 3.24 in Gebraeel's PhD thesis (2003)
    sig0 = 5e-2 # standard deviation of log(theta) (prior) - ??
    mu1 = 0.0037 # mean of beta (prior) - from Fig 3.24 in Gebraeel's PhD thesis (2003)
    sig1 = 1e-3 # standard deviation of beta (prior) - ??
    sig = 5e-3 # standard deviation of error terms (dim = 1/minute**.5) - ??
    D = 0.03 # threshold for failure detection - from Gebraeel (2007) - from Gebraeel et al. (2007)

    # sample theta and beta
    theta = rng.lognormal(mu0, sig0)
    while theta <= 0:
        theta = rng.lognormal(mu0, sig0)
        print(theta)
    beta = rng.normal(mu1, sig1)
    while beta <= sig**2/2:
        beta = rng.normal(mu1, sig1)

    # generate synthetic data from sampled theta and beta
    n = 170 # number of degradation signal samples
    t_offset = 630 # [min]
    dt = 2 # time interval between two data recording events [min]
    time = [dt*k for k in range(n)] # timeline during degradation [min]
    signal = []
    err = 0
    for t in time:
        err = err + rng.normal(0, sig*np.sqrt(t)) # synthetic error term - brownian motion
        s = phi + theta*np.exp(beta*(t+t_offset))*np.exp(err - sig**2*t/2)
        signal.append(s)

    # plot synthetic data (check)
    fig, ax = plt.subplots()
    ax.plot(time,signal,"b-")
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Degradation signal")
    ax.set_title("Synthetic degradation signal data")

    # create RLD object
    rld = BayesRLD(phi, mu0, sig0, mu1, sig1, sig, D)

    # simulate real-time degradation signal recording
    # first record and plot cdf and pdf
    rld.push(0, signal[0])
    interval = 300 # interval over which to plot cdf and pdf [min]
    rld.plot_cdf(interval)
    rld.plot_pdf(interval)

    # a few more records...
    N = 60
    for i in range(1, N):
        rld.push(i*dt, signal[i])

    # ... one more record with plots
    rld.push(N*dt, signal[N])
    interval = 300 # interval over which to plot cdf and pdf [min]
    rld.plot_cdf(interval)
    rld.plot_pdf(interval)

    # Show the plots
    plt.show()

    
        




