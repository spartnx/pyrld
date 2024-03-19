import os
import sys
fpath = os.path.join(os.pardir, "")
sys.path.append(fpath)

from pyrld import BayesRLD
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # input degradation model parameters (prior) - take from Gebraeel's PhD thesis (2003)
    phi = 0 # constant in degradation model - from p. 145 of Gebraeel's PhD thesis (2003)
    sig = np.sqrt(0.07721) # standard deviation of error terms (dim = 1/minute**.5) - p/. 146 from Gebraeel's PhD thesis (2003)
    mu0 = -7.6764738 # mean of log(theta) (prior) - from Table 6.1 p.140 in Gebraeel's PhD thesis (2003)
    sig0 = 0.56172343 # standard deviation of log(theta) (prior) - from Table 6.1 p.140 in Gebraeel's PhD thesis (2003)
    mu1 = 0.046585 # mean of beta (prior) - from Table 6.1 p.140 in Gebraeel's PhD thesis (2003)
    sig1 = 0.0071362 # standard deviation of beta (prior) - from Table 6.1 p.140 in Gebraeel's PhD thesis (2003)
    D = 0.03 # threshold for failure detection - from Gebraeel et al. (2007)
    dt = 2 # data recording interval, every dt minutes

    # create RLD object
    rld = BayesRLD(phi, mu0, sig0, mu1, sig1, sig, D)

    # generate synthetic data
    time, signal, actual_fail_time, fig = rld.synthetic_data(dt, seed=0, n_extra=5)
    n = len(time)

    # simulate real-time degradation signal recording
    # first record and plot cdf and pdf
    rld.push(time[0], signal[0])
    interval = 3000 # interval over which to plot cdf and pdf [min]
    rld.plot_cdf(interval)
    rld.plot_pdf(interval)
    t_10 = rld.percentile(p=0.1)
    t_50 = rld.percentile(p=0.5)
    t_90 = rld.percentile(p=0.9)
    print(f"t = {time[0]}")
    print(f"   [10%, 90%] : [{time[0]+round(t_10,3)}, {time[0]+round(t_90,3)}]")
    print(f"   med : {time[0]+round(t_50, 3)}\n")

    # a few more records...
    N = int(n/2)
    for i in range(1, N):
        rld.push(time[i], signal[i])

    # ... one more record with plots
    rld.push(time[N], signal[N])
    interval = 3000 # interval over which to plot cdf and pdf [min]
    rld.plot_cdf(interval)
    rld.plot_pdf(interval)
    t_10 = rld.percentile(p=0.1)
    t_50 = rld.percentile(p=0.5)
    t_90 = rld.percentile(p=0.9)
    print(f"t = {time[N]}")
    print(f"   [10%, 90%] : [{time[N]+round(t_10,3)}, {time[N]+round(t_90,3)}]")
    print(f"   med : {time[N]+round(t_50, 3)}\n")

    # a few more records...
    N2 = int(n/4)
    for i in range(N+1, N+N2):
        rld.push(time[i], signal[i])

    # ... one more record with plots
    rld.push(time[N+N2], signal[N+N2])
    interval = 3000 # interval over which to plot cdf and pdf [min]
    rld.plot_cdf(interval)
    rld.plot_pdf(interval)
    t_10 = rld.percentile(p=0.1)
    t_50 = rld.percentile(p=0.5)
    t_90 = rld.percentile(p=0.9)
    print(f"t = {time[N+N2]}")
    print(f"   [10%, 90%] : [{time[N+N2]+round(t_10,3)}, {time[N+N2]+round(t_90,3)}]")
    print(f"   med : {time[N+N2]+round(t_50, 3)}\n")

    # Show the plots
    plt.show()

    
        




