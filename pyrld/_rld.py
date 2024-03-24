import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

class BayesRLD(object):
    """Class to record degradation data and update RLD on the fly.
    The stochastic degradation model used to compute the RLD is an increasing exponential 
    with Brownian motion error as defined in Gebraeel et al., "Residual-life distributions 
    from component degradation signals: a Bayesian approach" (2007).  

    The degradation signal assumed in Gebraeel et al. (2007) is:

        S(t) = phi + theta*exp(beta*t)*exp(err(t) - (sig**2/2)*t)

    where:
        > S(t)   : degradation signal
        > phi    : a constant 
        > theta  : distributed as a lognormal random variable
        > beta   : distributed as a normal random variable
        > err(t) : Brownian motion error with mean 0 and variance sig**2*t

    Inputs:
        > phi  : a constant in the degradation model
        > mu0  : mean of log(theta) (prior)
        > sig0 : standard deviation of log(theta) (prior)
        > mu1  : mean of beta (prior) 
        > sig1 : standard deviation of beta (prior)
        > sig  : standard deviation of error terms (dim = 1/sqrt(min))
        > D    : threshold for failure detection
    """

    def __init__(self, phi, mu0, sig0, mu1, sig1, sig, D):
        # convert arguments into attributes
        self.phi_ = phi 
        self.mu0_ = mu0 
        self.sig0_ = sig0 
        self.mu1_ = mu1
        self.sig1_ = sig1
        self.sig_ = sig 
        self.D_ = D
        self.logD_ = np.log(D) # logged failure threshold (used to compute RLDs)

        # bivariate normal distribution parameters (mean and variance) - updated each time new data is recorded
        self.muT_ = None # mean of ln(theta) (posterior distribution)
        self.sigT_ = None # standard deviation of ln(theta) (posterior distribution)
        self.muB_ = None # mean of beta (posterior distribution)
        self.sigB_ = None # standard deviation of beta (posterior distribution)
        self.corr_ = None # correlation b/w beta and theta given the observations

        # containers for observations of signal degradation signal
        self.timestamps_ = []
        self.signal_ = []
        self.logged_signal_ = []
        self.logged_signal_diff_ = []

        # Multiple pdf plots
        self.fig, self.ax = plt.subplots()
        return
    
    def push(self, timestamp, signal):
        """Record a new observation and automatically update RLD.

        Inputs:
            > timestamp : time at which the observation is made
            > signal    : value of the made observation
        """
        # record observation
        self.timestamps_.append(timestamp)
        self.signal_.append(signal)
        # transform observation
        self.logged_signal_.append(np.log(signal - self.phi_))
        if len(self.signal_) == 1:
            self.logged_signal_diff_.append(self.logged_signal_[0])
        else:
            self.logged_signal_diff_.append(self.logged_signal_[-1] - self.logged_signal_[-2])
        # update RLD
        self._update_posterior()
        return
    
    def _update_posterior(self):
        """Update joint posterior distribution of ln(theta) and beta using all the observations recorded so far.
        Formulas taken from Proposition 2 in Gebraeel et al., "Residual-life distributions from component 
        degradation signals: a Bayesian approach" (2007)."""
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
        """Compute Cumulative Distribution Function (CDF) of the Residual Life Distribution (RLD).
        Formula taken from Gebraeel et al., "Residual-life distributions from component 
        degradation signals: a Bayesian approach" (2007).
        
        Input:
            > t : time at which to evaluate the CDF
        """
        if len(self.signal_) == 0:
            raise Exception("Please first record a data point before calling rld_cdf().")
        mu = self.logged_signal_[-1] + self.muB_*t
        sig = np.sqrt(self.sigB_**2*t**2 + self.sig_**2*t)
        g = (mu - self.logD_) / sig
        return scp.stats.norm.cdf(g)
    
    def rld_pdf(self, t):
        """Compute Probability Density Function (PDF) of the Residual Life Distribution (RLD).
        Formula taken from Gebraeel et al., "Residual-life distributions from component 
        degradation signals: a Bayesian approach" (2007).
        
        Input:
            > t : time at which to evaluate the PDF
        """
        if len(self.signal_) == 0:
            raise Exception("Please first record a data point before calling rld_pdf().")
        mu = self.logged_signal_[-1] + self.muB_*t
        sig = np.sqrt(self.sigB_**2*t**2 + self.sig_**2*t)
        g = (mu - self.logD_) / sig
        g_prime = (2*self.muB_*sig**2 - (mu-self.logD_)*(2*self.sigB_**2*t+self.sig_**2)) / (2*sig**3)
        return scp.stats.norm.pdf(g) * g_prime
    
    def plot_cdf(self, interval, coeff=5):
        """Plot the Cumulative Distribution Function (CDF) of the Residual Life Distribution (RLD).
        
        Inputs:
            > interval : time interval over which to plot the CDF
            > coef     : coeff by which interval is scaled to get the number of points to plot
        """
        time = np.linspace(0.1,interval,num=int(coeff*interval))
        cdf = [self.rld_cdf(t) for t in time]
        fig, ax = plt.subplots()
        ax.plot(time, cdf, 'k-')
        ax.set_xlabel("Time [min]")
        ax.set_title(f"Residual Life Distribution CDF (update time: {self.timestamps_[-1]} min)")
        return fig
    
    def plot_pdf(self, interval, coeff=5):
        """Plot the Probability Density Function (PDF) of the Residual Life Distribution (RLD).
        
        Inputs:
            > interval : time interval over which to plot the PDF
            > coef     : coeff by which interval is scaled to get the number of points to plot
        """
        time = np.linspace(0.1,interval,num=int(coeff*interval))
        pdf = [self.rld_pdf(t) for t in time]
        fig, ax = plt.subplots()
        ax.plot(time, pdf, 'k-')
        ax.set_xlabel("Time [min]")
        ax.set_title(f"Residual Life Distribution PDF (update time: {self.timestamps_[-1]} min)")
        self.ax.plot(time, pdf, label=f"Update time = {round(self.timestamps_[-1])}")
        return fig
    
    def synthetic_data(self, dt, seed=10, n_extra=20):
        """Generate synthetic data assuming the increasing exponential model.
        The function stops generating data points when the threshold self.D_ is reached
        plus n_extra additional points.
        
        Inputs:
            > dt      : time interval between any two observations
            > seed    : seed for random number geenrator
            > n_extra : number of additional points to generates
        """
        # initialize random number generator 
        rng = np.random.Generator(np.random.MT19937(seed))
        # sample stochastic parameters theta and beta
        theta = rng.lognormal(self.mu0_, self.sig0_)
        beta = rng.normal(self.mu1_, self.sig1_)
        while beta <= self.sig_**2/2:
            beta = rng.normal(self.mu1_, self.sig1_)
        # initialize containers
        time = []
        signal = []
        # initialize variables to update
        t = 0 # time counter
        n = 0 # sample counter
        err = 0 # Brownian error
        run = True
        actual_fail_time = None
        # run until failure threshold is reached
        while run:
            t = dt*n
            n += 1
            err = err + rng.normal(0, self.sig_*np.sqrt(dt)) # synthetic error term - brownian motion
            s = self.phi_ + theta*np.exp(beta*t)*np.exp(err - self.sig_**2*t/2)
            signal.append(s)
            time.append(t)
            if signal[-1] > self.D_:
                run = False
                actual_fail_time = t
        # run a few more samples
        for k in range(1,n_extra+1):
            t = dt*(n + k)
            err = err + rng.normal(0, self.sig_*np.sqrt(dt)) # synthetic error term - brownian motion
            s = self.phi_ + theta*np.exp(beta*t)*np.exp(err - self.sig_**2*t/2)
            signal.append(s)
            time.append(t)
        # plot synthetic data (check)
        fig, ax = plt.subplots()
        ax.plot(time,signal,"b-")
        ax.set_xlabel("Time [min]")
        ax.set_ylabel("Degradation signal")
        ax.set_title("Synthetic degradation signal data")
        return time, signal, actual_fail_time, fig
    
    def percentile(self, p=0.5, x0=400):
        """ Compute the (100*p)th percentile of the Cumulative Distribution Function (CDF) of the Residual Life Distribution (RLD).
        
        Inputs:
            > p  : probability, must be between 0 and 1
            > x0 : initial value for the fsolve function
        """
        # compute time at specified percentile given current RLD
        if len(self.signal_) == 0:
            raise Exception("Please first record a data point before calling rld_pdf().")
        if p>1 or p<0:
            raise Exception("Please input a percentile value between 0 and 1.")
        else:
            def func(x):
                return self.rld_cdf(x) - p
            return scp.optimize.fsolve(func, x0)[0]
        
    def multi_pdf_plot(self, interval=1500):
        self.ax.legend()
        self.ax.set_xlabel("Time [min]")
        self.ax.set_title(f"Evolution of the Residual Life Distribution PDF")
        self.ax.set_xlim((0,interval))
        return self.fig