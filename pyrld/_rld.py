import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

class BayesRLD(object):
    """Object to record degradation data and infer RLD.
    The stochastic degradation model used to compute the RLD is an increasing exponential 
    as defined in Gebraeel et al., "Residual-life distributions from component 
    degradation signals: a Bayesian approach" (2007).  

    The model implemented is that assuming Bownian motion error.
    """

    def __init__(self, phi, mu0, sig0, mu1, sig1, sig, D):
        # convert arguments into (private) attributes
        self.phi_ = phi # constant in degradation model
        self.mu0_ = mu0 # mean of theta (prior)
        self.sig0_ = sig0 # standard deviation of theta (prior)
        self.mu1_ = mu1 # mean of beta (prior)
        self.sig1_ = sig1 # standard deviation of beta (prior)
        self.sig_ = sig # standard deviation of error terms
        self.D_ = D # threshold for failure detection

        # bivariate normal distribution parameters (mean and variance)
        self.muT_ = None # mean of theta (posterior)
        self.sigT_ = None # standard deviation of theta (posterior)
        self.muB_ = None # mean of beta (posterior)
        self.sigB_ = None # standard deviation of beta (posterior)
        self.corr_ = None # correlation b/w beta and theta given observations

        # containers for observations of signal degradation signal
        self.timestamps_ = []
        self.signal_ = []
        self.logged_signal_ = []
        self.logged_signal_diff_ = []
        return
    
    def push(self, timestamp, signal):
        # to record real-time device data
        self.timestamps_.append(timestamp)
        self.signal_.append(signal)
        self.logged_signal_.append(np.log(signal - self.phi_))
        if len(self.signal_) == 1:
            self.logged_signal_diff_.append(self.logged_signal_[0])
        else:
            self.logged_signal_diff_.append(self.logged_signal_[-1] - self.logged_signal_[-2])
        self._update_rld()
        return
    
    def _update_rld(self):
        # proposition 2 from Gebraeel et al. (2007)
        mu1_p = self.mu1_ - (self.sig_**2/2)
        den = (self.sig0_**2 + self.sig_**2*self.timestamps_[0]) * (self.sig1_**2*self.timestamps_[-1] + self.sig_**2) \
              - self.sig0_**2*self.sig1_**2*self.timestamps_[0]
        self.muT_ = ((self.logged_signal_diff_[0]*self.sig0_**2 + self.mu0_*self.sig_**2*self.timestamps_[0]) \
                    * (self.sig1_**2*self.timestamps_[-1] + self.sig_**2) \
                    - self.sig0_**2*self.timestamps_[0] * (self.sig1_**2*np.sum(self.logged_signal_diff_) + mu1_p*self.sig_**2)) \
                    / den
        self.muB_ = ((self.sig1_**2*np.sum(self.logged_signal_diff_) + mu1_p*self.sig_**2) * (self.sig0_**2 + self.sig_**2*self.timestamps_[0])) \
                    - self.sig1_**2 * (self.logged_signal_diff_[0]*self.sig0_**2 + self.mu0_*self.sig_**2*self.timestamps_[0]) \
                    / den
        self.sigT_ = np.sqrt(self.sig_**2*self.sig0_**2*self.timestamps_[0]*(self.sig1_**2*self.timestamps_[-1] + self.sig_**2) / den)
        self.sigB_ = np.sqrt(self.sig_**2*self.sig1_**2*(self.sig0_**2 + self.sig_**2*self.timestamps_[0]) / den)
        self.corr_ = - self.sig0_*self.sig1_*np.sqrt(self.timestamps_[0]) / np.sqrt(den + self.sig0_**2*self.sig1_**2*self.timestamps_[0])
        return
    
    def rld_cdf(self, t):
        # the cumulative distribution function of the residual life distribution
        if len(self.signal_) == 0:
            raise Exception("Please first record a data point before calling rld_cdf().")
        else:
            mu = self.logged_signal_[-1] + self.muB_*t
            sig = np.sqrt(self.sigB_**2*t**2 + self.sig_**2*t)
            g = (mu - self.D_) / sig
        return scp.stats.norm.cdf(g)
    
    def rld_pdf(self, t):
        # the probability density function of the residual life distribution
        if len(self.signal_) == 0:
            raise Exception("Please first record a data point before calling rld_pdf().")
        else:
            mu = self.logged_signal_[-1] + self.muB_*t
            sig = np.sqrt(self.sigB_**2*t**2 + self.sig_**2*t)
            g = (mu - self.D_) / sig
            g_prime = (2*self.muB_*sig**2 - (mu-self.D_)*(2*self.sigB_**2*t+self.sig_**2)) / (2*sig**3)
        return scp.stats.norm.pdf(g) * g_prime
    
    def plot_cdf(self, interval, coeff=5):
        time = np.linspace(0.1,interval,num=int(coeff*interval))
        cdf = [self.rld_cdf(t) for t in time]
        fig, ax = plt.subplots()
        ax.plot(time, cdf, 'k-')
        ax.set_xlabel("Time [min]")
        ax.set_title(f"Residual Life Distribution CDF (update time: {self.timestamps_[-1]} min)")
        return fig
    
    def plot_pdf(self, interval, coeff=5):
        time = np.linspace(0.1,interval,num=int(coeff*interval))
        cdf = [self.rld_pdf(t) for t in time]
        fig, ax = plt.subplots()
        ax.plot(time, cdf, 'k-')
        ax.set_xlabel("Time [min]")
        ax.set_title(f"Residual Life Distribution PDF (update time: {self.timestamps_[-1]} min)")
        return fig