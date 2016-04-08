import numpy as np
import matplotlib.pyplot as plt


def gendata(n=20, d=2):
  # n = num or samples, d = dimensions
  data = dict.fromkeys(range(d),[])
  for k in data.keys():
    data[k] = np.concatenate((np.random.random((n)/2),
                              np.random.random((n)/2)+1))
  return data

def jackaldata():
  data = [[120, 107, 110, 116, 114, 111, 113, 117, 114, 112], # males
          [110, 111, 107, 108, 110, 105, 107, 106, 111, 111]] # females
  return data


def plot_scatter(data):
  # data is a dict with keys for each dimension
  if len(data)>2:
    print('WARNING: too many dimensions!!!!!!1111')
  else:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data[0], data[1], 'o')
    plt.show()
  return 


def randdata(data, n=1000, d=2):
  # generate random d-dimension data fit to scale of 'data' input
  
  def one_sample(nn=20, dd=2, rrange=[[0,2],[0,2]]):
    ddata = dict.fromkeys(range(dd), [])
    for k in ddata.keys():
      scale, offset = rrange[k][1]-rrange[k][0], rrange[k][0]
      ddata[k] = np.random.random(nn)*scale + offset
    return ddata
    
  maxes, mins = [], []
  for k in data.keys():
    maxes.append(max(data[k]))
    mins.append(min(data[k]))
  rdata = dict.fromkeys(range(n))
  for d in rdata.keys():
    rdata[d] = one_sample()
  return rdata


def nearest_pt(data, p_ind, i=1):
  # finds the ith nearest point to the given point index 
  i=i+1
  def dist(pt1, pt2):
    return np.sqrt((pt2[0]-pt1[0])**2 + (pt2[1]-pt1[1])**2)
  pt1 = [data[0][p_ind],data[1][p_ind]]
  inds, dists, count = [], [np.inf for i in range(i)], 0
  while count < i:
    for p in range(len(data[0])):
      curr_dist = dist(pt1, [data[0][p],data[1][p]])
      #print('(%f, %f), (%f, %f) -- %f' %(pt1[0],pt1[1],data[0][p],data[1][p],
      #                                  curr_dist))
      if curr_dist<dists[count] and p not in inds and curr_dist>0.0:
        if len(inds) < count+1:
          inds.append(p)
        else:
          inds[count] = p
        if len(dists) < count+1:
          dists.append(curr_dist)
        else:
          dists[count] = curr_dist
    #print(count)
    count = count + 1
  return inds, dists[:-1] # last dist is always inf somehow


def distribution_distances(data, i=1):
  # return the mean distance to the ith closest point for a distribution
  dists = np.zeros((len(data[0]),i))
  for p in range(len(data[0])):
    _, out = nearest_pt(data, p, i)
    for k in range(len(out)):
      dists[p,k] = out[k]
    #print(dists)
  return [np.mean(dists[:,j]) for j in range(i)]
    


def random_distribution_distances(data, n=1000, i=1):
  # calls randdata, which returns a dict of keys=dims
  # data is passed to get the dimensions
  d = len(data.keys())
  l = len(data[0])
  D = np.zeros((i,n)) # each colum is a random sample of data
                      # each row i is the ith point distance
  rdata = randdata(data, n, d)
  for k in rdata.keys():
    D[:,k] = distribution_distances(rdata[k],i)
  return D, [np.mean(D[j,:]) for j in range(i)]
  


def plot_random_distribution(data, n=1000, i=1):
  D, _ = random_distribution_distances(data, n, i)
  fig = plt.figure()
  ax = fig.add_subplot(111)
  j = i-1
  dists = []
  for d in range(len(D[0,:])):
    dists.append(D[j,d])
  dists.sort()
  ax.plot(range(n),dists,'o', color='b', markeredgecolor='b')
  real = distribution_distances(data, i)[i-1]
  ind = 0
  while dists[ind] < real:
    ind = ind +1
  ax.plot(ind, real, '*',color='r', markeredgecolor='r', markersize=20)
  print('%.5f chance that sample distribution is random.' %(float(ind)/len(dists)))
  plt.show()
  
  return
  


def serial_monte_carlo(data, it=100, mode='paired'):
  """ Serially point-scramble data (type=dict) and plot differences
  between the means of the different dict elements.  """
  
  def rand1_n(n=10): # return a random int 1-10 (0-9)
    return int(np.random.random(1)*n)
  
  def mean_diff(data, split=10, d=1):
    if d == 1:
      return np.mean(data[0][:split]) - np.mean(data[0][split:])
  
  def swap(data, p0, p1): # return synthetic data with position0 swapped
                          # with position1
    newdata = [i for i in data]
    newdata[p0]=data[p1]
    newdata[p1]=data[p0]
    print(data==newdata)
    return newdata

  #print(serial_data)
  fig1 = plt.figure()
  ax1 = fig1.add_subplot(111)
    
  if mode == 'distance': # default for 2D data; still BUGGY
    serial_data = dict.fromkeys(range(it+1),{})
    count = 0
    serial_data[count]=data
    count = 1
    while count < it+1:
      p0, p1 = rand1_n(), rand1_n()+10
      serial_data[count] = swap(serial_data[count-1], p0, p1)
      # print(serial_data[count]==serial_data[count-1])
      print(serial_data[count-1])
      print(serial_data[count])
      # print(curr_data)
      count = count + 1
    i = 1 # change this for ith distance, 1 as default
    D = np.zeros((i,it+1)) # each row is a distance, each col is a trial (it+1)
    for d in serial_data.keys():
      D[:,d] = distribution_distances(serial_data[d], i) # default 
      # ith nearest neighbor is 1 here, must add arg (i.e.: 2,3,10) for others
    dists = D[0,:]
    dists.sort()
    ax1.plot(range(it),dists[1:], 'o', color='b', markeredgecolor='b')
    real = distribution_distances(data)
    ind = 0
    while dists[ind] < real:
      ind = ind +1
    ax1.plot(ind, real, '*',color='r', markeredgecolor='r', markersize=20)
    print('%.5f chance that sample distribution is random.' %(float(ind)/len(dists)))
    plt.show()
  
  if mode == 'paired':
    serial_data = dict.fromkeys(range(it+1),[])
    for k in serial_data.keys():
      pass  
    
    add = len(data[0])/2
    # synthetic data already created; else can use this template
    """ count = 0
    curr_data = data
    while count < it+1
    p0, p1 = ... """
    diffs = []
    for k in serial_data.keys():
      print(serial_data[k])
      diffs.append(mean_diff(serial_data[k]))
    real = diffs[0]
    diffs.sort()
    ax1.plot(range(it+1),diffs,'o',color='b',markeredgecolor='b')
    ind = 0
    while diffs[ind] < real:
      ind = ind + 1
    ax1.plot(ind, real, '*', color='r', markeredgecolor='r', markersize=20)
    print('%.5f chance that sample distribution is random.' %(float(ind)/len(diffs)))
    plt.show()
  # print(serial_data)
  return serial_data
  





##############################################

if __name__ == "__main__":
  data = gendata()
  plot_random_distribution(data)













