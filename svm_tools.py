# svm_tools.py
# All are adapted from sklearn demos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def svm_regression(y, x=None, ker='rbf', opt=0.1, show=False):
  """
  Pass an array, with or without x-axis values, and this returns the SVM.
  A kernel (ker) can also be specified: 'rbf' 'linear', 'poly'
  """
  from sklearn.svm import SVR
  if x is None: # Assume linearly spaced points
    x = np.arange(0,len(y))
  # Fit the regression model
  if ker == 'linear':
    svr = SVR(kernel=ker, C=1e3)
  elif ker == 'poly':
    if type(opt) is not int:
      print('Need a degree for a polynomial fit, not' + str(opt))
      return None
    svr = SVR(kernel=ker, C=1e3, degree=opt)
  else:
    svr = SVR(kernel='rbf', C=1e3, gamma=opt) # default is radial basis func
  y_svr = svr.fit(x, y).predict(x) # Fit
  
  # And plot if requested
  if show:
    plt.scatter(x, y, c='k', label='data')
    plt.plot(x, y_svr, c='b', label='SVR model')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
  return y_svr




def svm_novelty(xtrain, ytrain, xtest, ytest, show=False):
  """
  Novelty detection (svm_oneclass). mesh shows confidence area if show==True.
  """
  from sklearn import svm
  # Fit the model
  clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1) # radial basis func
  X_train, X_test = np.array([xtrain,ytrain]).T, np.array([xtest,ytest]).T
  clf.fit(X_train)
  y_pred_train = clf.predict(X_train)
  y_pred_test = clf.predict(X_test)
  n_error_train = y_pred_train[y_pred_train == -1]
  n_error_test = y_pred_test[y_pred_test == -1]
  
  # Show
  if show:
    min_x, max_x = min([min(xtrain),min(xtest)]), max([max(xtrain),max(xtest)])
    min_y, max_y = min([min(ytrain),min(ytest)]), max([max(ytrain),max(ytest)])
    xx, yy = np.meshgrid(np.linspace(min_x,max_x,500), 
                         np.linspace(min_y,max_y,500))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Contours
    plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), 
                 cmap=plt.cm.Blues_r)
    a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange') # Learned frontier
    tr1, tr0 = X_train[y_pred_train==1,:], X_train[y_pred_train==-1]
    te1, te0 = X_test[y_pred_test==1,:], X_test[y_pred_test==-1]
    b1 = plt.scatter(tr1[:,0], tr1[:,1], c='royalblue', s=20,
                     edgecolor='royalblue', alpha=0.8) # Train
    b2 = plt.scatter(tr0[:,0], tr0[:,1], marker='x', c='royalblue', s=40) # Train
    c1 = plt.scatter(te1[:,0], te1[:,1], c='red', edgecolor='red', 
                     alpha=0.8, s=20) # Test
    c2 = plt.scatter(te0[:,0], te0[:,1], marker='x', c='red', s=40) # Test
    plt.xlim([min_x-1, max_x+1])
    plt.ylim([min_y-1, max_y+1])
    plt.axis('tight')
    plt.legend([a.collections[0], b1, c1], 
               ['learned frontier', 'train data', 'test data'],
               loc='upper left')
    plt.xlabel('error train: %d/%d ; error test: %d/%d'
               %(len(n_error_train), len(xtrain), len(n_error_test), len(xtest)))
    plt.show()
  return




##########################################################################
# Demos

def novelty_demo():
  """
  Simple novelty demo.
  """
  X = 0.3 * np.random.randn(100,2)
  X_tr = np.r_[X+2, X-2]
  X = 0.4 * np.random.randn(20,2)
  X_te = np.r_[X+2, X-2]
  print(X_tr.shape, X_te.shape)
  svm_novelty(X_tr[:,0], X_tr[:,1], X_te[:,0], X_te[:,1], True)
  return
  
  



def svr_demo():
  """
  Simple regression demo.
  """
  x = np.sort(5*np.random.rand(40,1), axis=0)
  y = np.sin(x).ravel()
  y[::5] += 3 * (0.5 - np.random.rand(8)) # Random noise
  # Test 3 kernels
  y_rbf = svm_regression(y,x, ker='rbf', opt=0.1, show=False)
  y_lin = svm_regression(y,x, ker='linear')
  y_poly = svm_regression(y, x, ker='poly', opt=2)
  # Plot
  plt.scatter(x, y, c='k', label='data')
  plt.plot(x, y_rbf, c='b', label='RBF model')
  plt.plot(x, y_lin, c='r', label='linear model')
  plt.plot(x, y_poly, c='g', label='poly model')
  plt.xlabel('data')
  plt.ylabel('target')
  plt.title('Support Vector Regression')
  plt.legend()
  plt.show()
  return




















