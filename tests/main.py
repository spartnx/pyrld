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
    D = 0.03 # threshold for failure detection - from Gebraeel (2007) - from Gebraeel et al. (2007)
    dt = 2 # data recording interval, every dt minutes

    # create RLD object
    rld = BayesRLD(phi, mu0, sig0, mu1, sig1, sig, D)

    # generate synthetic data
    time, signal, fig = rld.synthetic_data(dt)
    n = len(time)

    # simulate real-time degradation signal recording
    # first record and plot cdf and pdf
    rld.push(time[0], signal[0])
    interval = 300 # interval over which to plot cdf and pdf [min]
    rld.plot_cdf(interval)
    rld.plot_pdf(interval)

    # a few more records...
    N = int(n/2)
    for i in range(1, N):
        rld.push(time[i], signal[i])

    # ... one more record with plots
    rld.push(time[N], signal[N])
    interval = 300 # interval over which to plot cdf and pdf [min]
    rld.plot_cdf(interval)
    rld.plot_pdf(interval)

    # Show the plots
    plt.show()

    
        




