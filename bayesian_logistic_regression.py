# bayesian_logistic_regression.py
# Adapted from http://blog.smellthedata.com/2009/06/really-bayesian-logistic-regression-in.html



from scipy.optimize.optimize import fmin_cg, fmin_bfgs, fmin, check_grad
from scipy.optimize import fsolve, bisect
import numpy as np
import matplotlib.pyplot as plt




""" 
Create N instances of d dimensional input vectors and a 1D
class label (-1 or 1).
"""
class LogitSynData():
  
  def __init__(self, N=20, d=5):
    means = .25 * np.array([[-1] * (d-1), [1] * (d-1)])
    
    self.X_train = np.zeros((N, d))
    self.Y_train = np.zeros(N)    
    for i in range(N):
      if np.random.random() > .5:
        y = 1
      else:
        y = 0
      self.X_train[i, 0] = 1
      # Note, this time Tarlow does add a column of 1's to the X-data;
      #   In the previous non-Bayesian code he omitted this and so there were
      #   fewer betas than expected (no intercept beta)
      self.X_train[i, 1:] = np.random.random(d-1) + means[y, :]
      self.Y_train[i] = 2.0 * y - 1
    
    self.X_test = np.zeros((N, d))
    self.Y_test = np.zeros(N)    
    for i in range(N):
      if np.random.randn() > .5:
        y = 1
      else:
        y = 0
      self.X_test[i, 0] = 1
      self.X_test[i, 1:] = np.random.random(d-1) + means[y, :]
      self.Y_test[i] = 2.0 * y - 1



""" 
A simple logistic regression model with L2 regularization (zero-mean
Gaussian priors on parameters). 
"""
class BayesLogitRegression():

  def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None,
         alpha=.1, synthetic=False):
    self.failures = []
    # Set L2 regularization strength
    self.alpha = alpha
    
    # Set the data.
    self.set_data(x_train, y_train, x_test, y_test)
    
    # Initialization only matters if you don't call train().
    self.all_betas = []
    self.betas = np.random.randn(self.x_train.shape[1])


  def negative_lik(self, betas):
    return -1 * self.lik(betas)


  def lik(self, betas):
    """ Likelihood of the data under the current settings of parameters. """
    
    # Data likelihood
    l = 0
    for i in range(self.n):
      l += np.log(sigmoid(self.y_train[i] * \
               np.dot(betas, self.x_train[i,:])))
    
    # Prior likelihood
    for k in range(0, self.x_train.shape[1]):
      l -= (self.alpha / 2.0) * self.betas[k]**2
      
    return l


  def lik_k(self, beta_k, k):
    """ The likelihood only in terms of beta_k. """

    new_betas = self.betas.copy()
    new_betas[k] = beta_k
    
    return self.lik(new_betas)
  

  def train(self):
    """ Define the gradient and hand it off to a scipy gradient-based
    optimizer. """
    
    # Define the derivative of the likelihood with respect to beta_k.
    # Need to multiply by -1 because we will be minimizing.
    # Note that intercept beta IS included in the fmin search!!
    dB_k = lambda B, k : (k > -1) * self.alpha * B[k] - np.sum([ \
                   self.y_train[i] * self.x_train[i, k] * \
                   sigmoid(-self.y_train[i] *\
                       np.dot(B, self.x_train[i,:])) \
                   for i in range(self.n)])
    
    # The full gradient is just an array of componentwise derivatives
    dB = lambda B : np.array([dB_k(B, k) \
                 for k in range(self.x_train.shape[1])])
    
    # Optimize
    self.betas = fmin_bfgs(self.negative_lik, self.betas, fprime=dB)
    

  def resample(self):
    """ Use slice sampling to pull a new draw for logistic regression
    parameters from the posterior distribution on beta. """
    
    failures = 0
    for i in range(10):
      try:
        new_betas = np.zeros(self.betas.shape[0])
        order = range(self.betas.shape[0])
        order.reverse()
        for k in order:
          new_betas[k] = self.resample_beta_k(k)
      except:
        failures += 1
        continue

      for k in range(self.betas.shape[0]):
        self.betas[k] = new_betas[k]
        
      print(self.betas)
      print(self.lik(self.betas))
      
      self.all_betas.append(self.betas.copy())
      
    if failures > 0:
      self.failures.append(
            "Warning: %s root-finding failures" % (failures))
    


  def resample_beta_k(self, k):
    """ 
    Resample beta_k conditional upon all other settings of beta.
    This can be used in the inner loop of a Gibbs sampler to get a
    full posterior over betas.
    Uses slice sampling (Neal, 2001). 
    """

    #print "Resampling %s" % k

    # Sample uniformly in (0, f(x0)), but do it in the log domain
    lik = lambda b_k : self.lik_k(b_k, k)
    x0 = self.betas[k]
    g_x0 = lik(x0)
    e = np.random.exponential()
    z = g_x0 - e
    
    # Find the slice of x where z < g(x0) (or where y < f(x0))
    #print "y=%s" % exp(z)
    lik_minus_z = lambda b_k : (self.lik_k(b_k, k) - z)

    # Find the zeros of lik_minus_k to give the interval defining the slice
    r0 = fsolve(lik_minus_z, x0)

    # Figure out which direction the other root is in
    eps = .001
    look_right = False
    if lik_minus_z(r0 + eps) > 0:
      look_right = True

    if look_right:
      r1 = bisect(lik_minus_z, r0 + eps, 1000)
    else:
      r1 = bisect(lik_minus_z, -1000, r0 - eps)

    L = min(r0, r1)
    R = max(r0, r1)
    x = (R - L) * np.random.random() + L

    #print "S in (%s, %s) -->" % (L, R),
    #print "%s" % x
    return x   
    

  def set_data(self, x_train, y_train, x_test, y_test):
    """ Take data that's already been generated. """
    
    self.x_train = x_train
    self.y_train = y_train
    self.x_test = x_test
    self.y_test = y_test
    self.n = y_train.shape[0]


  def training_reconstruction(self):
    p_y1 = np.zeros(self.n)
    for i in range(self.n):
      p_y1[i] = sigmoid(np.dot(self.betas, self.x_train[i,:]))
    
    return p_y1


  def test_predictions(self):
    p_y1 = np.zeros(self.n)
    for i in range(self.n):
      p_y1[i] = sigmoid(np.dot(self.betas, self.x_test[i,:]))
    
    return p_y1
  
  
  def plot_training_reconstruction(self):
    plt.plot(np.arange(self.n), .5 + .5 * self.y_train, 'bo')
    plt.plot(np.arange(self.n), self.training_reconstruction(), 'rx')
    plt.ylim([-.1, 1.1])


  def plot_test_predictions(self):
    plt.plot(np.arange(self.n), .5 + .5 * self.y_test, 'yo')
    plt.plot(np.arange(self.n), self.test_predictions(), 'rx')
    plt.ylim([-.1, 1.1])


##########################################################################
# Helpers



def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))
  


#########################################################################\

if __name__ == "__main__":

  data = LogitSynData(50, 2)

  alphas = [.1]
  for j, a in enumerate(alphas):
    plt.figure()
    
    # Create a new learner, but use the same data for each run
    lr = BayesLogitRegression(x_train=data.X_train, y_train=data.Y_train,
                x_test=data.X_test, y_test=data.Y_test,
                alpha=a)
    
    print("Initial likelihood:")
    print(lr.lik(lr.betas))
    
    # Train the model
    lr.train()
    map_betas = lr.betas.copy()
    
    plt.subplot(10, 2, 1)
    lr.plot_training_reconstruction()
    plt.ylabel("Alpha=%s" % a)
    if j == 0:
      plt.title("Training set reconstructions")
    plt.axis('off')
    
    plt.subplot(10, 2, 2)
    lr.plot_test_predictions()
    if j == 0:
      plt.title("Test set predictions")
    plt.axis('off')
    
    # Display execution info
    print("Final betas:")
    print(lr.betas)
    print("Final lik:")
    print(lr.lik(lr.betas))

    for l in range(1, 10000):
      lr.resample()

      # Plot the results
      if l % 1000 == 0:
        plt.subplot(10, 2, 2*(l/1000) + 1)
        lr.plot_training_reconstruction()
        plt.axis('off')
        plt.subplot(10, 2, 2*(l/1000) + 2)
        lr.plot_test_predictions()
        plt.axis('off')
    
    plt.figure()
    all_betas = np.array(lr.all_betas)

    plt.hexbin(all_betas[:,0], all_betas[:,1], bins='log')
    plt.plot([map_betas[0]], [map_betas[1]], 'rx', markersize=10)
  plt.show()




