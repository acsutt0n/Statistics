## likelihood functions -- mostly from Pawitan (2001)

import numpy as np
import matplotlib.pyplot as plt


########################################################################
#                              Chapter 2
########################################################################

def discrete_example(samples=5):
  """
  Creates a sample normalized distribution of likelihood functions.
  """
  def like(theta, x, n=100):
    L = theta**x * (1 - theta)**(n-x)
    return L*n, L*x
  successes = np.linspace(0,10,samples)
  successes = [int(i)*10 for i in successes]
  x = np.linspace(0,1,100)
  y = []
  for s in successes:
    t = [like(i,s)[0] for i in x]
    t = t/max(t)
    y.append(t)
  
  # plot
  plt.figure()
  for Y in y:
    plt.plot(x,Y)
  plt.show()



def continuous_example():
  """
  p. 24 (Pawitan, 2001) claims likelihood (L)
  L(theta) = P(theta) {X in (x-ep/2, x+ep/2)}
           = integral[x-ep/2, x+ep/2](p_theta(x)dx)
           ~= ep * p_theta(x)
  Where p_theta = p_sub_theta and ep=epsilon, some small nonzero value.
  This function tests this claim by evaluating the error between
  the absolute integral and the approximation for different distributions.
  """
  x = np.linspace(0,np.pi,200)
  y = np.sin(x)
  def choose_x(x): return int(np.random.random(1)*len(x))
  
  












