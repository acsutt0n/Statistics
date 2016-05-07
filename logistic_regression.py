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
class LogitSynData():

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
      self.X_train[i, :] = np.random.random(d) + means[y[i], :]
      self.Y_train[i] = 2. * y[i] - 1
    
    # Testing data
    means = 0.05 * np.random.randn(2,d)
    self.X_test = np.zeros( (N,d) )
    self.Y_test = np.zeros( N )
    y = np.random.randint(0,2,N)
    for i in range(N):
      self.X_test[i, :] = np.random.random(d) + means[y[i], :]
      self.Y_test[i] = 2. * y[i] - 1



"""
Logistic regression with BFGS minimization for L2 regularization.
L2 penalty coefficient is alpha.
"""
class LogitRegression():
  
  def __init__(self, data=None, x_train=None, y_train=None, 
                     x_test=None,  y_test=None,
                     alpha=0.1, synthetic=False):
    # L2 regularization coefficient
    self.alpha = alpha
    # Generate the data if it doens't exist
    if data is None:
      self.set_data(x_train, y_train, x_test, y_test)
    else:
      self.set_data(data.X_train, data.Y_train, data.X_test, data.Y_test)
    # Initialize params to zero
    self.betas = np.zeros(self.d+1)
    self.log = {'betas': [], 'train_prob': [], 'test_prob': []}
  
  
  def negative_like(self, betas):
    return -1 * self.like(betas)
  
  
  def like(self, betas):
    """
    Likelihood given the current parameters.
    """
    # Date likelihood - increase log likelihood 
    l = 0
    for i in range(self.n): # For each data point
      l += np.log(sigmoid(self.y_train[i] * \
                          np.dot(betas, self.x_train[i,:])))
    
    # Prior likelihood - penalize each extra dimension/coefficient (not 0)
    for k in range(1, self.d): # For each dimension/feature
      l -= (self.alpha / 2.) * self.betas[k]**2
    
    return l
    
    
  def set_data(self, x_train, y_train, x_test, y_test):
    """
    Assign the values of the data.
    """
    if x_train is None and y_train is None and x_test is None and y_test is None:
      dat = LogitSynData()
      x_train, y_train = dat.X_train, dat.Y_train
      x_test, y_test = dat.X_test, dat.Y_test
    self.x_train, self.y_train = x_train, y_train
    self.x_test, self.y_test = x_test, y_test
    self.n = self.x_train.shape[0]
    try:
      self.d = self.x_train.shape[1] # Set number of dimensions
    except:
      self.d = 1
    newTrain = np.ones((self.n, self.d+1))
    newTest = np.ones((self.x_test.shape[0], self.d+1))
    if self.d > 1: # For adding an intercept beta, need a data col of 1's
      newTrain[:,1:] = self.x_train
      newTest[:,1:] = self.x_test
    else:
      newTrain[:,1] = self.x_train
      newTest[:,1] = self.x_test
    self.x_train = newTrain
    self.x_test = newTest
    return self
    
  
  def train(self):
    """
    Set gradient and let BFGS optimizer find min of neg log likelihood
    B - -log(likelihood) given betas
    """
    # Set derivative of likelihood w.r.t. beta[k], -1 to minimize -log(likelihood)
    if self.d > 1:
      dB_k = lambda B, k : (k > -1) * self.alpha * B[k] - \
                            np.sum([self.y_train[i] * self.x_train[i,k] * \
                            sigmoid(-self.y_train[i] * \
                                    np.dot(B, self.x_train[i,:]))
                            for i in range(self.n)])
    else:
      dB_k = lambda B, k : (k > -1) * self.alpha * B[k] - \
                          np.sum([self.y_train[i] * self.x_train[i,] * \
                          sigmoid(-self.y_train[i] * \
                                  np.dot(B, self.x_train[i,]))
                          for i in range(self.n)])
    
    # The full gradient is just an array of componentwise derivatives
    dB = lambda B : np.array([dB_k(B, k) for k in range(self.d+1)])
    
    # Optimize
    self.betas = fmin_bfgs(self.negative_like, self.betas, fprime=dB,
                           disp=True)
    return self
  
  
  def training_reconstruction(self):
    """
    ?s
    """
    p_y1 = np.zeros(self.n)
    for i in range(self.n):
      p_y1[i] = sigmoid(np.dot(self.betas, self.x_train[i,:]))
    self.log['train_prob'] = p_y1
    return p_y1
  
  
  def test_predictions(self):
    """
    """
    p_y1 = np.zeros(self.x_test.shape[0])
    for i in range(self.x_test.shape[0]):
      p_y1[i] = sigmoid(np.dot(self.betas, self.x_test[i,:]))
    self.log['test_prob'] = p_y1
    return p_y1
  
  
  def plot_training_reconstructions(self):
    """
    """
    plt.plot(np.arange(self.x_train.shape[0]), .5 + .5 * self.y_train, 'b', marker='o', linewidth=0.)
    plt.plot(np.arange(self.x_train.shape[0]), self.training_reconstruction(), 'r', marker='x', linewidth=0.)
    plt.ylim([-.1, 1.1])
  
  
  def plot_test_predictions(self):
    """
    """
    plt.plot(np.arange(self.x_test.shape[0]), .5 + .5 * self.y_test, 'y', marker='o', linewidth=0.)
    plt.plot(np.arange(self.x_test.shape[0]), self.test_predictions(), 'r', marker='x', linewidth=0.)
    plt.ylim([-.1,1.1])
  
  
  def show_all(self):
    """
    """
    plt.subplot(2,1,1)
    self.plot_training_reconstructions()
    plt.ylabel('Alpha=%s' %self.alpha)
    plt.xlabel('Training set reconstructions')
    plt.subplot(2,1,2)
    self.plot_test_predictions()
    plt.ylabel('Alpha=%s' %self.alpha)
    plt.xlabel('Test set predictions')
    plt.show()
  
  
  # End class LogitRegression




##########################################################################
# Helper functions

def sigmoid(x):
  return 1. / (1. + np.exp(-x))



def different_alphas(dataset, alphas=[0., 0.001, 0.01, 0.1]):
  """
  Evaluate and plot this for multiple alpha regularization levels
  """
  for j, a in enumerate(alphas):
    # Create a new model each time, but use the same data
    LR = LogitRegression(data=dataset, alpha=a)
    print('Inital likelihood:')
    print(LR.like(LR.betas))
    print('Initial betas:')
    print(LR.betas)
    
    # Train the model
    LR.train()
    # Display execution info
    print('Final betas:')
    print(LR.betas)
    print('Final likelihood:')
    print(LR.like(LR.betas))
    
    # Plot results
    plt.subplot(len(alphas), 2, 2*j + 1)
    LR.plot_training_reconstructions()
    plt.ylabel('Alpha=%s' %LR.alpha)
    if j == 0:
      plt.title('Training set reconstructions')
    plt.subplot(len(alphas), 2, 2*j+2)
    LR.plot_test_predictions()
    if j == 0:
      plt.title('Test set predictions')
  plt.show()
  return




##########################################################################

if __name__ == "__main__":
  # Create a 5-dimensional data set with 20 points
  data = LogitSynData(20, 5)
  # Try several different alphas, which regularize (penalize) higher order fits
  alphas = [0., 0.001, 0.01, 0.1]
  
  for j, a in enumerate(alphas):
    # Create a new model each time, but use the same data
    lr = LogitRegression(x_train=data.X_train, y_train=data.Y_train,
                         x_test=data.X_test, y_test=data.Y_test,
                         alpha=a)
    print('Inital likelihood:')
    print(lr.like(lr.betas))
    
    # Train the model
    lr.train()
    # Display execution info
    print('Final betas:')
    print(lr.betas)
    print('Final likelihood:')
    print(lr.like(lr.betas))
    
    # Plot results
    plt.subplot(len(alphas), 2, 2*j + 1)
    lr.plot_training_reconstructions()
    plt.ylabel('Alpha=%s' %a)
    if j == 0:
      plt.title('Training set reconstructions')
    plt.subplot(len(alphas), 2, 2*j+2)
    lr.plot_test_predictions()
    if j == 0:
      plt.title('Test set predictions')
  
  plt.show()







