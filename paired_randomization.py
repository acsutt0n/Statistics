# simple randomization test to check two samples


import numpy as np
import matplotlib.pyplot as plt
import itertools as it



def gen_data():
  # data from Manly 2007 p. 108
  cf = [92,0,72,80,57,76,81,67,50,77,90,72,81,88,0]
  sf = [43,67,64,64,51,53,53,26,36,48,34,48,6,28,48]
  return {'cf': cf, 'sf': sf}



def subtract_means(x):
  return [i-np.mean(x) for i in x]



def perm(n, seq): # seq must be a str, i.e.: '01'
  d = []
  for p in it.product(seq, repeat=n):
    d.append(''.join(p))
  return [int(i) for i in d]



def sums_difference(data):
  def sums(l):
    s = []
    for i in l:
      s.append(sum(int(j) for j in str(i)))
    return s
    
  n = min(len(data[k]) for k in data.keys())
  perms = perm(n, '01') # 
  dist = sums([int(i) for i in perms])
  dist.sort()
  dat, c = {}, 0
  for k in data.keys():
    dat[c] = data[k]
    c = c +1
  # paired comparison?
  if len(dat[0]) == len(dat[1]):
    real = 0
    for i in range(len(dat[0])):
      if dat[0][i]-dat[1][i] > 0:
        real = real + 1 # group 0 larger than group 1
  ind = None
  for i in range(len(dist)):
    if real <= dist[i]: # find the first dist > real
      if not ind:
        ind = i
  probability = (float(len(dist))-ind)/float(len(dist))
  print('Probability of data resulting from a random distribution: %.4f'
         %probability)
  return dist, ind, probability
  


if __name__ == '__main__':
  data = gen_data()
  d, i, p = sums_difference(data)
