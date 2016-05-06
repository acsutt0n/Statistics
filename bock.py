# Davi Bock's 2010 quality/quantity trade-off paper


import numpy as np
import matplotlib.pyplot as plt
import statsmodels as sm
import scipy.stats as stats
import copy



"""
Quality-Quantity trade-off assessment.
h -- function (z) of epsilon which describes how many z (edges)
per worker per day with error rate epsilon.
pEE, pII, pIE=pEI -- prob of E-E, I-I and E-I/I-E connections
lam -- proportion of neurons that are excitatory
n -- number of (true) nodes
alpha -- testing level
 --> if no values provided, Bock's toy model is used.
"""
class QQ():
  
  def __init__(self, params=None):
    """
    params is a dict with keys (types):
    'h' (function)              pEE, pEI, pII (floats, [0,1))
    lam (float)                 n (int)
    """
    self.params = params
    self.set_params() # Replace Nones with defaults
    self
    
  
  def set_defaults(self):
    """
    """
    def h_(epp):
      return 50 + 200/(np.sin(np.pi/4))*(np.sin(epp*np.pi/2.))
    pdefaults = {'h': h_, 'pEE': 0.1, 'pII': 0.1, 'pEI': 0.2, 'pIE': 0.2,
                 'lam': .9, 'n': 10000, 'alpha': 0.05}
    return pdefaults
  
  
  def set_params(self):
    """
    """
    pdefaults = self.set_defaults()
    if self.params is None:
      self.params = pdefaults
    for pk in pkeys:
      try:
        t_ = self.params[pk] # A value exists
      except:
        print('Parameter %s not defined; using defaults' %pk)
        self.params[pk] = pdefaults[pk]
    return self
  
  
  def pX_(self, epp):
    """
    """
    nE = self.params['lam']*self.params['n']
    nI = (1-self.params['lam'])*self.params['n']
    in1 = (self.params['lam']*self.params['pEE']*nE) /
          (self.params['pEE']*nE + self.params['pEI']*nI)
    in2 = ((1-self.params['lam'])*self.params['pII']*nI) /
          (self.params['pII']*nI + self.params['pIE']*nE)
    epproot = epp*(2*self.params['lam']**2 - 2*self.params['lam'] + 1)
    return (1-epp) * ( in1 + in2 ) + epproot
  

  def beta(self, epp):
    """
    Calculate everything that goes into the phi calculation.
    """
    # Multiple epp here?? No -- called multiple times
    # epp = np.linspace(0,1,100)
    # Alternative hypothesis: pEE < pIE (as in defaults)
    pXA = self.pX_(self, epp)
    savedparams = copy.deepcopy(self.params)
    # Null hypothesis: pEI==PEE
    self.params['pEE'] = self.params['pIE']
    pX0 = self.pX_(self, epp)
    print('For alternative hypothesis, pXA: %.3f' %np.mean(pXA))
    print('For null hypothesis, pX0: %.3f' %np.mean(pX0))
    self.params = copy.deepcopy(savedparams) # Revert back to old params
    # True Z
    nE = self.params['lam']*self.params['n']
    nI = (1-self.params['lam'])*self.params['n']
    z_ = (nE**2 * self.params['pEE'])/2. + (nI**2 * self.params['pII'])/2. + \
         (nE*nI*self.params['pEI']) # since there is a pIE and a pEI (but pIE=pEI), don't divide by 2
    numerat = pX0*(1-pX0)*stats.norm.ppf(self.params['alpha'] + \
              np.sqrt(z_)*(pX0-pXA)
    denom = pXA*(1-pXA)
    return stats.norm.cdf(numerat/denom)
  
  
  
    
  
  






def toy_model():
  """
  """
  return
  
  

##########################################################################
# Testing stuff

def px_(P, epp):
  # Calculate pX0 or pXA
  nE = P['lam']*P['n']
  nI = (1-P['lam'])*P['n']
  in1 = (P['lam']*P['pEE']*nE) / \
        (P['pEE']*nE + P['pEI']*nI)
  in2 = ((1-P['lam'])*P['pII']*nI) / \
        (P['pII']*nI + P['pIE']*nE)
  epproot = epp*(2*(P['lam']**2) - 2*P['lam'] + 1)
  print('  For epp=%.3f, epproot=%.3f ' %(epp, epproot))
  return (1-epp) * ( in1 + in2 ) + epproot



def beta_(P, epp):
  # P is for params
  nE = P['lam']*P['n']
  nI = (1-P['lam'])*P['n']
  pXA = px_(P, epp)
  savedparams = copy.deepcopy(P)
  # Null hypothesis: pEI==PEE
  savedparams['pEE'], savedparams['pII'] = P['pIE'], P['pIE']
  pX0 = px_(savedparams, epp)
  
  print('For alternative hypothesis, pXA: %.3f' %np.mean(pXA))
  print('For null hypothesis, pX0: %.3f' %np.mean(pX0))
  # P = copy.deepcopy(savedparams) # Revert back to old params
  try: 
    z_ = P['z']
  except: # 'True' z
    z_ = (nE**2 * P['pEE'])/2. + (nI**2 * P['pII'])/2. + \
       (nE*nI*P['pEI']) # since there is a pIE and a pEI (but pIE=pEI), don't divide by 2
  numerat = pX0*(1-pX0)*stats.norm.ppf(P['alpha']) + \
            np.sqrt(z_)*(pX0-pXA)
  denom = pXA*(1-pXA)
  print('PHI (%.3f / %.3f)' %(numerat, denom))
  return stats.norm.cdf(numerat/denom)



def g_(P, epp):
  # This *also* returns a beta, but composite with the _h_ function of P
  # P is for params
  nE = P['lam']*P['n']
  nI = (1-P['lam'])*P['n']
  pXA = px_(P, epp)
  savedparams = copy.deepcopy(P)
  # Null hypothesis: pEI==PEE
  savedparams['pEE'], savedparams['pII'] = P['pIE'], P['pIE']
  pX0 = px_(savedparams, epp)
  
  # Calculate g(epp)
  num1 = pX0 * (1-pX0) * stats.norm.ppf(P['alpha'])
  num2 = np.sqrt(P['h'](epp)) * (pX0 - pXA)
  denom = pXA * (1 - pXA)
  return stats.norm.cdf((num1+num2)/denom)



def simple_test():
  """
  Simple test of above prototype functions
  """
  epx = np.linspace(.0,1.,100)
  pdefaults = {'h': h_, 'pEE': 0.1, 'pII': 0.1, 'pEI': 0.2, 'pIE': 0.2,
               'lam': .9, 'n': 10000, 'alpha': 0.05}
  
















