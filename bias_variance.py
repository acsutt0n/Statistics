# bias_variance.py 
# Includes some treatment of loss functions


import numpy as np
import matplotlib.pyplot as plt



def gen_data(N=100, jitterx=False, jittery=True, show=False):
  x = np.linspace(-1,1,N*10)
  y_ = [1 - (i**2 - 2*np.exp(-100*i**2)) for i in x]
  y_ = y_[::10]
  y = [i + np.random.random(1)*0.2 for i in y_]
  for k in [x, y, y_]:
    print(len(k), type(k))
  if show:
    plt.plot(x[::10],y, 'blue', marker='x', label='With noise')
    plt.plot(x[::10], y_, 'red', label='Underlying fn')
    plt.legend(loc='best')
    plt.show()
  return np.array([x[::10],y])



def polyWithLoss(y, x=None, order=2, lam=0., show=False):
  """
  Fit a polynomial of _order_ with a regularization coefficient _lam_.
  """
  if x is None:
    x = np.arange(len(y))
  print('Order is %i (plus a constant)' %int(order))
  def minthis(k, x, y): # Function to minimize
    coeffs = k[:-1]
    lam = k[-1] # That's the regularization coefficient lambda
    x, y = np.array(x), np.array(y)
    fn = np.array([ sum([xi**u for u in range(len(coeffs[:-1]))]) # Higher order terms
                    + coeffs[-1] for xi in x])
    A = (y - fn)**2
    return A + sum([coef**2*lam for coef in coeffs[:-1]])
  
  
  
  


#








