# logistic_data.py
# Several (small) logistic regression data sets, returns as data objects
# (see below)



import numpy as np
import matplotlib.pyplot as plt



"""
Data here are broken into 4 objects:
  data.X_train - N x dim training data
  data.Y_train - N x 1 training targets {-1,1}
  data.X_test - M x dim test data (M not necessarily == N)
  data.Y_test - M x 1 test data targets (optional)
data.n also gives the sample size (equal to data.X_train.shape[0]
"""
class LogitData():
  
  def __init__(self, x_train, y_train, x_test, y_test,
               norm=False):
    if min(y_train) > -1: # Should range {-1,1}, not {0,1}
      y_train = np.array([1 if i == 1 else -1 for i in y_train])
    if min(y_test) > -1: # Should range {-1,1}, not {0,1}
      y_test = np.array([1 if i == 1 else -1 for i in y_test])
    self.X_train, self.Y_train = x_train, y_train
    self.X_test, self.Y_test = x_test, y_test
    self.n = self.X_train.shape[0]

  
  def train_plot(self):
    """
    Show the training data.
    """
    plt.plot(np.arange(len(self.Y_train)), self.Y_train, 'b', marker='o', linewidth=0.)
    plt.ylabel('Training targets')
    plt.ylim([-1.1,1.1])
    plt.show()
  
  
  def test_plot(self):
    """
    Show the test data.
    """
    plt.plot(np.arange(len(self.Y_test)), self.Y_test, 'y', marker='o', linewidth=0.)
    plt.ylabel('Test targets')
    plt.ylim([-1.1,1.1])
    plt.show()



########################################################################
# Small data sets

def surgical_deaths(train_n=None):
  # Choose where train and test data end
  deaths = [[50,0], [50,0], [51,0], [51,0], [53,0], [54,0], [54,0],
  [54,0], [55,0], [55,0], [56,0], [56,0], [56,0], [57,1], [57,1],
  [57,0], [57,0], [58,0], [59,1], [60,0], [61,0], [61,1], [61,1],
  [62,1], [62,1], [62,0], [62,1], [63,0], [63,0], [63,1], [64,0],
  [64,1], [65,0], [67,1], [67,1], [68,0], [68,1], [69,0], [70,1],
  [71,0]]
  if train_n is None:
    train_n = int(np.random.random()*(len(deaths)-10))+5
  x, y = [i[0] for i in deaths], [i[1] for i in deaths]
  train_inds, cnt = [], 0
  while len(train_inds) < train_n and cnt < 10000: 
    cnt += 1
    t_ = int(np.random.random()*len(deaths))
    if t_ not in train_inds:
      train_inds.append(t_)
  test_inds = [i for i in range(len(deaths)) if i not in train_inds]
  x_train = np.array([x[u] for u in train_inds])
  y_train = np.array([y[u] for u in train_inds])
  x_test = np.array([x[u] for u in test_inds])
  y_test = np.array([y[u] for u in test_inds])
  # Use a LogitData object to store this
  return LogitData(x_train=x_train, y_train=y_train,
                   x_test=x_test, y_test=y_test)




  





