"""
The bulk of this content is taken from the sklearn example
http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
I have added code and comments
## Not finished yet ! ##
"""

"""
===============================================================
Model selection with Probabilistic PCA and Factor Analysis (FA)
===============================================================

Probabilistic PCA and Factor Analysis are probabilistic models.
The consequence is that the likelihood of new data can be used
for model selection and covariance estimation.
Here we compare PCA and FA with cross-validation on low rank data corrupted
with homoscedastic noise (noise variance
is the same for each feature) or heteroscedastic noise (noise variance
is the different for each feature). In a second step we compare the model
likelihood to the likelihoods obtained from shrinkage covariance estimators.

One can observe that with homoscedastic noise both FA and PCA succeed
in recovering the size of the low rank subspace. The likelihood with PCA
is higher than FA in this case. However PCA fails and overestimates
the rank when heteroscedastic noise is present. Under appropriate
circumstances the low rank models are more likely than shrinkage models.

The automatic estimation from
Automatic Choice of Dimensionality for PCA. NIPS 2000: 598-604
by Thomas P. Minka is also compared.

"""
print(__doc__)

# Authors: Alexandre Gramfort
#          Denis A. Engemann
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

###############################################################################
# Create the data




class GenData:
  """ Generate random data. """
  def __init__(self, n_samples=1000, n_features=50, rank=10):
    print('Making random data ...')
    self.n_samples = n_samples
    self.n_features = n_features
    self.rank = rank
    self.sigma = 1.
    self.rng = np.random.RandomState(42) # Generate rand state seed
    U, _, _ = linalg.svd(self.rng.randn(self.n_features, self.n_features))
    self.X = np.dot(self.rng.randn(self.n_samples, self.rank), 
                    U[:, :self.rank].T)
    self.X_homo, self.X_hetero = None, None
    # Adding homoscedastic noise
    self.X_homo = self.X + self.sigma * self.rng.randn(self.n_samples, 
                                                       self.n_features)
    # Adding heteroscedastic noise
    self.sigmas = self.sigma * self.rng.rand(self.n_features) + self.sigma / 2.
    self.X_hetero = self.X + self.rng.randn(self.n_samples, self.n_features) \
                             * self.sigmas


  

###############################################################################
# Fit the models




def compute_scores(X, n_components):
  """
  This is the "y" data of the plots -- the CV scores.
  """
  pca = PCA()
  fa = FactorAnalysis()
  
  pca_scores, fa_scores = [], []
  for n in n_components:
    pca.n_components = n
    fa.n_components = n
    pca_scores.append(np.mean(cross_val_score(pca, X)))
    fa_scores.append(np.mean(cross_val_score(fa, X)))
  
  return pca_scores, fa_scores



def shrunk_cov_score(X):
  shrinkages = np.logspace(-2, 0, 30)
  cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages})
  return np.mean(cross_val_score(cv.fit(X).best_estimator_, X))



def lw_score(X):
  return np.mean(cross_val_score(LedoitWolf(), X))



def run_CV_and_plot():
  """
  """
  for X, title in [(X_homo, 'Homoscedastic Noise'),
                 (X_hetero, 'Heteroscedastic Noise')]:
    # Adding 5 components at a time, calculate the CV score for each component combination
    n_components = np.arange(0, n_features, 5)  # options for n_components
    pca_scores, fa_scores = compute_scores(X)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    pca = PCA(n_components='mle')
    pca.fit(X)
    n_components_pca_mle = pca.n_components_

    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
    print("best n_components by PCA MLE = %d" % n_components_pca_mle)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
    plt.axvline(n_components_pca, color='b',
                label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r',
                label='FactorAnalysis CV: %d' % n_components_fa, linestyle='--')
    plt.axvline(n_components_pca_mle, color='k',
                label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

    # compare with other covariance estimators
    plt.axhline(shrunk_cov_score(X), color='violet',
                label='Shrunk Covariance MLE', linestyle='-.')
    plt.axhline(lw_score(X), color='orange',
                label='LedoitWolf MLE' % n_components_pca_mle, linestyle='-.')

    plt.xlabel('num of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    plt.title(title)
  plt.show()





########################################################################
if __name__ == "__main__":
  run_CV()














