# This tests some python stats and machine learning stuff


import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("darkgrid")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
mat_perf = read_csv('/home/alex/Dropbox/stats/data/student-mat.csv', delimiter=';')


sns.set_palette("deep", desat=.6)
sns.set_context(context='poster', font_scale=1)
sns.set_context(rc={"figure.figzise": (8,4)})
plt.hist(mat_perf.G3) # G3 is the final test score
plt.xticks(range(0,22,2))


def show_correlations(mat_perf):
  # Cycle through the variables (except previous test scores (G1,G2))
  # and see if any are significantly correlated with G3 (G1-3 are last 3 cols)
  test_stats = {'variable': [], 'test_type' : [], 'test_value' : []}
  for col in mat_perf.columns[:-3]:
    test_stats['variable'].append(col)
    if mat_perf[col].dtype == 'O': # 'O' is 'object', means string
      # Do ANOVA
      aov = smf.ols(formula='G3 ~ C(' + col + ')', data=mat_perf, missing='drop').fit()
      test_stats['test_type'].append('F Test')
      test_stats['test_value'].append(round(aov.fvalue,2))
    else: # Else it's a number (here int64)
      # Do correlation
      print(col + '\n')
      model = smf.ols(formula='G3 ~ ' + col, data=mat_perf, missing='drop').fit()
      value = round(model.tvalues[1],2)
      test_stats['test_type'].append('t Test')
      test_stats['test_value'].append(value)
  test_stats = pd.DataFrame(test_stats)
  test_stats.sort(columns='test_value', ascending=False, inplace=True)

  # Plotting the significant correlations
  f, (ax1, ax2) = plt.subplots(2,1, figsize=(24,9), sharex=False)
  sns.set_context(context='poster', font_scale=1)
  # F Values
  sns.barplot(x='variable', y='test_value', data=test_stats.query("test_type == 'F Test'"), hline=.1, ax=ax1, x_order=[x for x in test_stats.query("test_type == 'F Test'")['variable']])
  ax1.set_ylabel('F Values')
  ax1.set_xlabel('')
  # t Values
  sns.barplot(x='variable', y='test_value', data=test_stats.query("test_type == 't Test'"), hline=.1, ax=ax2, x_order=[x for x in test_stats.query("test_type == 't Test'")['variable']])
  ax2.set_ylabel('t Values')
  ax2.set_xlabel('')
  sns.despine(bottom=True)
  sns.set_palette("deep", desat=.6)
  plt.tight_layout(h_pad=3)
  plt.show()
  return



def rand_forest1():
  """
  Training the first random forest model
  """
  # Keep the "significant" variables
  usevars =  [x for x in test_stats.query("test_value >= 3.0 | test_value <= -3.0")['variable']]
  # 
  mat_perf['randu'] = np.array([np.random.uniform(0,1) for x in
                               range(0,mat_perf.shape[0])])
  # 
  mp_X = mat_perf[usevars]
  mp_X_train = mp_X[mat_perf['randu'] <= .67]
  mp_X_test = mp_X[mat_perf['randu'] > .67]
  mp_Y_train = mat_perf.G3[mat_perf['randu'] <= .67]
  mp_Y_test = mat_perf.G3[mat_perf['randu'] > .67]
  # for the training set
  cat_cols = [x for x in mp_X_train.columns if mp_X_train[x].dtype == "O"]
  for col in cat_cols:
    new_cols = pd.get_dummies(mp_X_train[col])
    new_cols.columns = col + '_' + new_cols.columns
    mp_X_train = pd.concat([mp_X_train, new_cols], axis=1)
  # for the testing set
  cat_cols = [x for x in mp_X_test.columns if mp_X_test[x].dtype == "O"]
  for col in cat_cols:
    new_cols = pd.get_dummies(mp_X_test[col])
    new_cols.columns = col + '_' + new_cols.columns
    mp_X_test = pd.concat([mp_X_test, new_cols], axis=1)
  mp_X_train.drop(cat_cols, inplace=True, axis=1)
  mp_X_test.drop(cat_cols, inplace=True, axis=1)
  rf = RandomForestRegressor(bootstrap=True,
         criterion='mse', max_depth=2, max_features='auto',
         min_density=None, min_samples_leaf=1, min_samples_split=2,
         n_estimators=100, n_jobs=1, oob_score=True, random_state=None,
         verbose=0)
  rf.fit(mp_X_train, mp_Y_train)
  return









