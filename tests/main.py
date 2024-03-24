"""
Example script creating an instance of the BayesRLD class with input parameters
taken from data found in Gebraeel's PhD thesis (2003) and one of his publications (2007).

The class instance then generates synthetic data using the provided input parameters
before simulating the sampling of new observations which are used to update the 
Residual Life Distribution (RLD) under the hood.

Reference inputs:
 > phi  = 0                : constant in degradation model                         : taken directly from [1] (p. 145)
 > sig  = np.sqrt(0.07721) : standard deviation of error terms (dim = 1/sqrt(min)) : taken directly from [1] (p. 146)
 > mu0  = -7.6764738       : mean of log(theta) (prior)                            : computed from Table 6.1 p.140 in [1] (cf. Excel file in 'tests' folder)
 > sig0 = 0.56172343       : standard deviation of log(theta) (prior)              : computed from Table 6.1 p.140 in [1] (cf. Excel file in 'tests' folder)
 > mu1  = 0.046585         : mean of beta (prior)                                  : computed from Table 6.1 p.140 in [1] (cf. Excel file in 'tests' folder)
 > sig1 = 0.0071362        : standard deviation of beta (prior)                    : computed from Table 6.1 p.140 in [1] (cf. Excel file in 'tests' folder)
 > D    = 0.03             : threshold for failure detection                       : taken directly from [2]
 > dt   = 2                : data recording interval, every dt minutes             : taken directly from [1] or [2]

References:
    > [1] Gebraeel's PhD thesis, "Real-time degradation modeling an residual life prediction for component maintenance and replacement" (2003)
    > [2] Gebraeel et al., "Residual-life distributions from component degradation signals: a Bayesian approach" (2007)
"""

import os
import sys
fpath = os.path.join(os.pardir, "")
sys.path.append(fpath)

from pyrld import BayesRLD
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # input parameters
    phi = 0
    sig = np.sqrt(0.07721)
    mu0 = -7.6764738
    sig0 = 0.56172343
    mu1 = 0.046585 
    sig1 = 0.0071362
    D = 0.03
    dt = 2
    interval = 3000 # interval over which to plot cdf and pdf [min]

    # create BayesRLD object
    rld = BayesRLD(phi, mu0, sig0, mu1, sig1, sig, D)

    # generate synthetic data
    time, signal, actual_fail_time, fig = rld.synthetic_data(dt, seed=0, n_extra=5)
    n = len(time)

    # simulation
    # first data record and RLD update
    rld.push(time[0], signal[0]) 
    rld.plot_cdf(interval)
    rld.plot_pdf(interval)
    t_10 = rld.percentile(p=0.1) # 10% chance the device fails before t_10 (10th percentile)
    t_50 = rld.percentile(p=0.5) # 50% chance the device fails before t_50 (median)
    t_90 = rld.percentile(p=0.9) # 90% chance the device fails before t_90 (90th percentile)
    print(f"t = {time[0]}")
    print(f"   [10%, 90%] : [{time[0]+round(t_10,3)}, {time[0]+round(t_90,3)}]")
    print(f"   med : {time[0]+round(t_50, 3)}\n")

    # a few more records and RLD updates...
    N = int(n/2)
    for i in range(1, N):
        rld.push(time[i], signal[i])

    # ... one more record and RLD update
    rld.push(time[N], signal[N])
    rld.plot_cdf(interval)
    rld.plot_pdf(interval)
    t_10 = rld.percentile(p=0.1)
    t_50 = rld.percentile(p=0.5)
    t_90 = rld.percentile(p=0.9)
    print(f"t = {time[N]}")
    print(f"   [10%, 90%] : [{time[N]+round(t_10,3)}, {time[N]+round(t_90,3)}]")
    print(f"   med : {time[N]+round(t_50, 3)}\n")

    # a few more records with RLD updates...
    N2 = int(n/4)
    for i in range(N+1, N+N2):
        rld.push(time[i], signal[i])

    # ... one more record RLD updates
    rld.push(time[N+N2], signal[N+N2])
    rld.plot_cdf(interval)
    rld.plot_pdf(interval)
    t_10 = rld.percentile(p=0.1)
    t_50 = rld.percentile(p=0.5)
    t_90 = rld.percentile(p=0.9)
    print(f"t = {time[N+N2]}")
    print(f"   [10%, 90%] : [{time[N+N2]+round(t_10,3)}, {time[N+N2]+round(t_90,3)}]")
    print(f"   med : {time[N+N2]+round(t_50, 3)}\n")

    # Plot several RLDs at different update times
    rld.multi_pdf_plots(interval=800)

    # Show the plots
    plt.show()

    
        




