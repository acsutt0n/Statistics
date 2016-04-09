# logistic_regression.py
# Generate data and explore the effects of regularization
# influenced by: http://blog.smellthedata.com/2009/06/python-logistic-regression-with-l2.html


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize.optimize import fmin_cg, fmin_bfgs, fmin



"""
Generate some logistical training and testing data, assign it a class (1,-1).
N - samples per data set
D - dimensions (number of datasets)
"""
class LogitData():

  def __init__(self, N=20, d=5):
    """
    Generate the data.
    """
    # Training data
    means = 0.05 * np.random.randn(2,d)
    self.X_train = np.zeros( (N,d) )
    self.Y_train = np.zeros( N )
    y = np.random.randint(0,2,N)
    for i in range(N):
      self.X_train[i, :] = np.random.random(d) + means[y, :]
      self.Y_train[i] = 2. * y[i] - 1
    
    # Testing data
    means = 0.05 * np.random.randn(2,d)
    self.X_test = np.zeros( (N,d) )
    self.Y_test = np.zeros( N )
    y = np.random.randint(0,2,N)
    for i in range(N):
      self.X_test[i, :] = np.random.random(d) + means[y, :]
      self.Y_test[i] = 2. * y[i] - 1



"""

"""
class LogitRegression():
  
  def __init__(self, x_train=None, y_train=None, 
                     x_test=None,  y_test=None,
                     alpha=0.1, synthetic=False):
    # L2 regularization coefficient
    self.alpha = alpha
    # Generate the data if it doens't exist
    self.set_data(x_train, y_train, x_test, y_test)
    # Initialize params to zero
    self.betas = np.zeros(self.x_train.shape[1])
  
  
  def negative_like(self, betas):
    return -1 * self.like(betas)
  
  
  def like(self, betas):
    """
    Likelihood given the current parameters.
    """
    # Date likelihood
    l = 0
    for i in range(self.n):
      
    
    
  def set_data(self, x_train, y_train, x_test, y_test):
    """
    Assign the values of the data.
    """
    if x_train == y_train == x_test == y_test == None:
      dat = LogitData()
      x_train, y_train = dat.X_train, dat.Y_train
      x_test, y_test = dat.X_test, dat.Y_test
    self.x_train, self.y_train = x_train, y_train
    self.x_test, self.y_test = x_test, y_test
    return self
    
  


















