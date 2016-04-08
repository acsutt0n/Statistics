data1 = [3.56, 0.69, 0.10, 1.84, 3.93, 1.25, 0.18, 1.13, 0.27, 0.50,
        0.67, 0.01, 0.61, 0.82, 1.70, 0.39, 0.11, 1.20, 1.21, 0.72]
data2 = [431, 450, 431, 453, 481, 449, 441, 476, 460, 482, 472, 465, 
         421, 452, 451, 430, 458, 446, 466, 476]
data = np.array(data2)


def resample_data(data, n=1000):
  # bootstrap resample data n times to create distributions
  dists = dict.fromkeys(range(n), [])
  d = np.random.random(n*len(data))
  d = [data[int(i)] for i in d]
  c=0
  for k in dists.keys():
    dists[k] = make_dist(data)
  return dists
  


def make_dist(data):
  d = []
  for i in range(len(data)):
    d.append(data[int(np.random.random(1)*len(data))])
  return d



def plot_dist(data, nbins=5, scat=True):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  if scat==False:
    ax.hist(data, bins=nbins)
    """ # Another histogram option:
    hist, binedges = np.histogram(data, bins=nbins)
    binedges = [(binedges[i]+binedges[i+1])/2 for i in 
                range(len(binedges)-1)]
    ax.bar(binedges, hist)
    """
  else:
    data.sort()
    ax.plot(range(len(data)), data, 'o')
  plt.show()
  return
    
  

def bootstrap_std(data, show=True):
  dists = resample_data(data)
  stds = [np.std(dists[k]) for k in dists.keys()]
  plot_dist(stds)
  return stds


def conf_int(dist, alpha=0.05):
  l, u = int(len(dist)*alpha*0.5), len(dist)-int(len(dist)*alpha*0.5)
  print('95 percent confidence interval: %.5f - %.5f' %(dist[l], dist[u]))
  return dist[l], dist[u]
  
  


