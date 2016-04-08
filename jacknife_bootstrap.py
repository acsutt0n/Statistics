# some stuff for checking distributions and bootstrapping

import numpy as np
import itertools as it
from scipy.misc import comb as comb



def gen_data(n1=10, n2=15, mean1=200, diff=-10):
  a = np.random.normal(mean1, np.sqrt(mean1*.2), n1)
  b = np.random.normal(mean1+diff, np.sqrt((mean1+diff)*.2), n2)
  return a, b



def randomize(a,b, replacement=False):
  # Retain respective sizes of samples but scramble with/ without replacement.
  pool = [i for i in a]
  for i in b:
    pool.append(i)
  np.random.shuffle(pool)
  new_a, new_b = pool[:len(a)], pool[len(a):]
  return new_a, new_b




######################### stats tests here ###############################

# wilcoxon distribution-free rank rum test
def wilcoxon(a,b, N=10):
  """
  - The two distributions are combined and rank-sorted then assigned to the
  items from a and b. If both samples are comprised of observations
  of similar magnitude, the ranks assigned to a and b should be similar,
  as should their sums.
  - There are len(a)+len(b)-Choose-len(a) possible combinations of the ranks.
  The sum of the ranks for each possible combination w forms a 
  distribution W, which can range from sum(range(len(a))) to 
  sum(range(C-len(a),C)).
  - The number of arrangements where observed rank w/C >> W/C 
  or w/C << W/C gives an implicit probability P(w=W|H0).
  """
  r1 = list(a)
  r2 = list(b)
  ranks = [i for i in r1]
  for i in r2:
    ranks.append(i)
  ranks.sort()
  ranks1 = [ranks.index(i) for i in r1]
  ranks2 = [ranks.index(i) for i in r2]
  
  # using normal distribution of ranged distributions
  Wmin, Wmax = sum(range(len(r1))), sum(range(len(ranks)-len(r1),
                                              len(ranks)))
  sample_w = sum(ranks1)
  numcombs = comb(len(ranks), len(r1))
  all_p = []
  # repeat for stability and to discount for seed
  for _j in range(10):
    W_dist = np.random.normal((Wmin+Wmax)/2, ((Wmin+Wmax)/2)*.2, numcombs)
    W_dist.sort()
    count = 0.
    for w in W_dist:
      if w < sample_w:
        count = count + 1
    all_p.append(count/numcombs)
  return 1. - np.mean(all_p)



def jacknife(a):
  """
  Return the collection of subsets acquired by removing one value from 
  the main set. A set of 12 values returns 12 subsets of 11 samples each.
  """
  sets = []
  for i in range(len(a)):
    curr_set = [k for k in a]
    curr_set.pop(i)
    sets.append(curr_set)
  if len(sets) != len(a):
    print('Warning: did not yield expected number of sets / %i vs %i'
          %(len(sets), len(a)))
  return sets
  


def jacknife_estimator(a):
  """
  'Estimate of the variance of an estimator' calculated using the jacknife.
  """
  a_var = np.var(a)
  sets_a = jacknife(a)
  var_of_sets = [np.var(i) for i in sets_a]
  jack_var = np.mean(var_of_sets)
  #jack_var_a = ((len(a)-1)/len(a)) * sum([(np.var(i)-jack_var)**2 for i in sets_a])
  jack_var_a = (1/len(a)) * sum([(np.var(i)-jack_var)**2 for i in sets_a])
  print(' Var(a) = %.5f.\n Jacknife Var(a) = %.5f.\n Estimated Var(a) = %.5f'
        %(a_var, jack_var, jack_var_a))
  return jack_var_a, jack_var, a_var
  



############################ randomization stuff here #####################

def wilcoxon_rand(a,b, N=1000):
  """
  Performs a wilcoxon rank sums test on N randomizations without replacement
  of a and b.
  """
  p_stat = []
  for k in range(N):
    new_a, new_b = randomize(a,b)
    p_stat.append(wilcoxon(new_a, new_b))
  p_sample = wilcoxon(a,b)
  p_stat.sort()
  count = 0.
  for p in p_stat:
    if p < p_sample:
      count = count+1
  
  return count/N, p_stat
  
  


################################ simulations ##############################






