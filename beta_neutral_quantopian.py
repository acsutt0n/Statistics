## All of this is lifted directly from a quantopian notebook at
# https://www.quantopian.com/posts/quantopian-lecture-series-the-art-of-not-following-the-market

# Import libraries
import numpy as np
from statsmodels import regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
import sys
sys.path.append('/home/alex/Dropbox/Personal/stocks/')
from pairs_trading import *



# Get data for the specified period and stocks
start = 20140101
end = 20150101
asset = pull_series('TSLA', start)
benchmark = pull_series('SPY', start)
# Get rid of the first (0th) element because it is NAN
r_a = asset.pct_change()[1:]
r_b = benchmark.pct_change()[1:]
# Let's plot them just for fun
r_a.plot()
r_b.plot()
plt.ylabel("Daily Return")
plt.legend();
plt.show()


# Let's define everything in familiar regression terms
X = r_b.values # Get just the values, ignore the timestamps
Y = r_a.values

def linreg(x,y):
  # We add a constant so that we can also fit an intercept (alpha) to the model
  # This just adds a column of 1s to our data
  x = sm.add_constant(x)
  model = regression.linear_model.OLS(y,x).fit()
  # Remove the constant now that we're done
  x = x[:, 1]
  return model.params[0], model.params[1]



alpha, beta = linreg(X,Y)
print('alpha: ' + str(alpha))
print('beta: ' + str(beta))
X2 = np.linspace(X.min(), X.max(), 100)
Y_hat = X2 * beta + alpha
plt.scatter(X, Y, alpha=0.3) # Plot the raw data
plt.xlabel("SPY Daily Return")
plt.ylabel("TSLA Daily Return")
plt.plot(X2, Y_hat, 'r', alpha=0.9);  # Add the regression line, colored in red
plt.show()



# Construct a portfolio with beta hedging
portfolio = -1*beta*r_b + r_a
portfolio.name = "TSLA + Hedge"
# Plot the returns of the portfolio as well as the asset by itself
portfolio.plot(alpha=0.9)
r_b.plot(alpha=0.5);
r_a.plot(alpha=0.5);
plt.ylabel("Daily Return")
plt.legend()
print("means: ", portfolio.mean(), r_a.mean())
print("volatilities: ", portfolio.std(), r_a.std())
P = portfolio.values
alpha, beta = linreg(X,P)
print('alpha: ' + str(alpha))
print('beta: ' + str(beta))
plt.show()


def validate_beta_est(sym='TSLA', bench='SPY'):
  # Get the alpha and beta estimates over the last year
  start = 20140101
  end = 20150101
  asset = pull_series(sym, start)
  benchmark = pull_series(bench, start)
  r_a = asset.pct_change()[1:]
  r_b = benchmark.pct_change()[1:]
  X = r_b.values
  Y = r_a.values
  historical_alpha, historical_beta = linreg(X,Y)
  print('Asset Historical Estimate:')
  print('alpha: ' + str(historical_alpha))
  print('beta: ' + str(historical_beta))
  
  # Get data for a different time frame:
  start = 20150101
  end = 20150601
  asset = pull_series(sym, start)
  benchmark = pull_series(bench, start)

  # Repeat the process from before to compute alpha and beta for the asset
  r_a = asset.pct_change()[1:]
  r_b = benchmark.pct_change()[1:]
  X = r_b.values
  Y = r_a.values
  alpha, beta = linreg(X,Y)
  print('Asset Out of Sample Estimate:')
  print('alpha: ' + str(alpha))
  print('beta: ' + str(beta))

  # Create hedged portfolio and compute alpha and beta
  portfolio = -1*historical_beta*r_b + r_a
  P = portfolio.values
  alpha, beta = linreg(X,P)
  print('Portfolio Out of Sample:')
  print('alpha: ' + str(alpha))
  print('beta: ' + str(beta))

  # Plot the returns of the portfolio as well as the asset by itself
  portfolio.name = str(sym)+" + Hedge"
  portfolio.plot(alpha=0.9)
  r_a.plot(alpha=0.5);
  r_b.plot(alpha=0.5)
  plt.ylabel("Daily Return")
  plt.legend()
  plt.show()
  return



