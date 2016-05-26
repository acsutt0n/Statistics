# Bayesian linear regression, linear modeling and generalized linear models


# Imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pymc3 as pm



########################################################################
# Helper functions





########################################################################
# Regression models

"""
Linear regression follows a simple linear model:
            y ~ ax + b
        or  y = ax + b + epsilon
sampled from a probability distribution y ~ N(ax + b, sigma**2). We use
pymc3 to estimate the parameters a, b and sigma. 
"""

def bayes_linregress(y, x=None, nsamples=1000, showtrace=False, ):
  """
  Linear regression. Regress y onto x (or linspace(0,1,len(y)) if None).
  """
  with pm.Model() as model:
    a = pm.Normal('a', mu=0, sd=20)
    b = pm.Normal('b', mu=0, sd=20)
    sigma = pm.Uniform('sigma', lower=0, upper=20)
    # y_estimate
    y_est = a*x + b
    likelihood = pm.Normal('y', mu=y_est, sd=sigma, observed=y)
    # Inference and MCMC sampling
    start = pm.find_MAP() # Max a post. inference
    step = pm.NUTS() # Hamiltonian MCMC with No U-Turn Sampler
    trace = pm.sample(nsamples, step, start, random_seed=123,
                      progressbar=True)
  
  # Show/print the results
    if showtrace:
      pm.traceplot(trace)
      plt.show()
  print('\n') # Report the findings
  for obj in trace.varnames:
    print('%s: %.3f +/- %.3f' %(obj, np.mean(trace[obj]), np.std(trace[obj])))
  
  return trace





def 


########################################################################
# Demo functions

def linregress_demo(N=11, a_=6, b_=2):
  """
  Do a bayes linregress demo.
  """
  # Set up sample data
  x = np.linspace(0,1,N)
  y = a_*x + b_ + np.random.random(N)
  # Get the trace, run mcmc
  trace = bayes_linregress(y, x, showtrace=True)
  return




  








