# slopefield.py - plots a slope field for a function and some initial
#                 conditions

import matplotlib.pyplot as plt
from scipy import *
from scipy import integrate
from scipy.integrate import ode
import numpy as np



## Vector field function
def vf(t,x):
  dx=np.zeros(2)
  dx[0]=1
  # ex: dx[1]=x[0]**2-x[0], or dx[1]= 1 - x[1]
  dx[1]= -2*x[1] + 2*np.sin(2*np.pi*x[0]) ## first change here for colored lines # <----------
  return dx


def plot_solution_curves(ax):
  # Solution curves
  t0=0; tEnd=10; dt=0.01;
  r = ode(vf).set_integrator('vode', method='bdf',max_step=dt) # ode call
  ic=[[-2,-10], [2,-10], [-4,-10]] # initial conditions
  color=['r','b','g']
  for k in range(len(ic)):
      Y=[];T=[];S=[];
      r.set_initial_value(ic[k], t0).set_f_params()
      while r.successful() and r.t +dt < tEnd:
          r.integrate(r.t+dt)
          Y.append(r.y)

      S=np.array(np.real(Y))
      ax.plot(S[:,0],S[:,1], color = color[k], lw = 1.25)


def make_mesh(ax):
  # Vector field
  X,Y = np.meshgrid( np.linspace(-5,5,20),np.linspace(-10,10,20) )
  U = 1
  ## change here for arrows ####### <--------------------------------------
  # ex: V = X**2-X; or 1-Y
  V = -2*Y + 2*np.sin(2*np.pi*X)
  # Normalize arrows
  N = np.sqrt(U**2+V**2)  
  U2, V2 = U/N, V/N
  ax.quiver( X,Y,U2, V2)


def run_plot():
  fig = plt.figure(num=1)
  ax=fig.add_subplot(111)
  plot_solution_curves(ax)
  make_mesh(ax)
  # plotting
  plt.xlim([-5,5])
  plt.ylim([-10,10])
  plt.xlabel(r"$x$")
  plt.ylabel(r"$y$")
  plt.show()

if __name__ == '__main__':
  run_plot()

