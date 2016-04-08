########### Stats for Journal Club


import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
try:
  import pymc as pm
except:
  print('PyMC not found!')
import sys
if sys.version_info.major != 2:
  print('\n\n\n- - - Needs to be run with Version 2.7 for now! - - -\n\n\n')




########################################################################
# Helpers

def gen_data(N=40, ngroups=2, mus=[15, 15.7], sigmas=[2,2]):
  """
  Generate some random data.
  """
  data = []
  for n in range(ngroups):
    data.append(np.random.normal(mus[n], sigmas[n], N))
  return data




########################################################################
# Fickle p-value

def fickle(groups=None, N=10, sims=4, ret_p=False, show=True):
  """
  Two-group example of why p-value is unreliable.
  """
  if groups is None:
    A, B = gen_data(N*4, mus=[15,15]) # Dists must be moderately larger than samples
  # Both samples are drawn from the same dist.
  else:
    A, B = groups
  
  groups = []
  for s in range(sims):
    # Draw the samples
    a = [A[i] for i in [int(x) for x in np.random.random(N)*len(A)]]
    b = [B[i] for i in [int(x) for x in np.random.random(N)*len(B)]]
    groups.append([a,b])
  
  # Get the p-values
  p_vals = [stats.ttest_ind(t[0], t[1])[1] for t in groups]
  
  # If got more than 10 sims, find the largest, smallest, and 2 randos
  if sims > 10:
    keep = [p_vals.index(min(p_vals)), p_vals.index(max(p_vals)),
            int(np.random.random()*len(p_vals)),
            int(np.random.random()*len(p_vals))]
    groups = [groups[i] for i in keep]
    sims = len(keep)
  else:
    keep = [i for i in range(sims)]
    groups = [groups[i] for i in keep]
    sims = len(keep)
  
  # Plot the findings
  if show:
    fig = plt.figure()
    for s in range(sims):
      ax = fig.add_subplot(1,sims, s+1)
      ax.plot(np.random.random(N)*.25, groups[s][0], 'o', color='b', 
              alpha=0.5, )
      ax.plot([i + 1 for i in np.random.random(N)*.25], groups[s][1], 
              'o', color='r', alpha=0.5, )
      ax.plot([0,.5], [np.mean(groups[s][0]), np.mean(groups[s][0])],
              linewidth=1.5, color='b', alpha=0.8)
      ax.plot([.75,1.25], [np.mean(groups[s][1]), np.mean(groups[s][1])],
              linewidth=1.5, color='r', alpha=0.8)
      ax.set_title('Sim: %i, P=%.3f' %(s+1, p_vals[keep[s]]))
      plt.xticks([0,1], ['A', 'B'])
      ax.set_xlim([-.25,1.5])
      ax.set_ylim([0,5])
      if s != 0:
        ax.yaxis.set_visible(False)
    
    plt.show()
  
  if ret_p:
    return p_vals
  return

#


def sample_size(Ns=[10,30,64,100], sims=1000):
  """
  Examine the effect of sample size on p value (as a distribution).
  """
  dists = [fickle(N=i, sims=sims, ret_p=True, show=False) for i in Ns]
  
  # Plot the histograms of p values
  fig = plt.figure()
  for n in range(len(Ns)):
    ax = fig.add_subplot(1,len(Ns), n+1)
    logbins = np.logspace(-4, 0, 20)
    cnts, bins, patches = ax.hist(dists[n], bins=logbins, color='gray',
                                  edgecolor='white')
    
    # Figure out which bins are significant
    # cnts, _ = np.histogram(dists[n], bins=np.logspace(-4,0,20))
    bincents = 0.5 * (bins[:-1] + bins[1:])
    cols = []
    for b in bincents:
      if b <= 0.05:
        if b <= 0.01:
          if b <= 0.001:
            cols.append('red')
          else:
            cols.append('orange')
        else:
          cols.append('yellow')
      else:
        cols.append('gray')
    
    for c, p in zip(cols, patches):
      plt.setp(p, 'facecolor', c)
    
    ax.plot([0.05, 0.05], [0,max(cnts)], '-', color='r')
    ax.plot([0.01, 0.01], [0,max(cnts)], '-', color='gray')
    ax.plot([0.001, 0.001], [0,max(cnts)], '-', color='black')
    plt.gca().set_xscale('log')
    ax.set_title('Sample size: %i' %Ns[n])
    plt.xticks([0.0001, 0.001, 0.01, 0.1, 1],
                  ['0.0001', '0.001', '0.01', '0.1', '1'])
    ax.set_xlabel('P')
    
  plt.show()
  return

#



#########################################################################
# Bayesian p-values

def bayes_ttest(groups=None, N=40, show=False):
  """
  Run a Bayesian t-test on sample or true data.
  """
  if groups is None: # Generate some data
    group1, group2 = gen_data(N=40)
  elif len(groups) != 2:
    print('T-test requires only 2 groups, not %i' %len(groups))
    return None
  else:
    group1, group2 = groups
  
  pooled = np.concatenate((group1, group2)) # Pooled data
  # Establish priors
  mu1 = pm.Normal("mu_1", mu=pooled.mean(), tau=1.0/pooled.var()/N)
  mu2 = pm.Normal("mu_2", mu=pooled.mean(), tau=1.0/pooled.var()/N)
  sig1 = pm.Uniform("sigma_1",lower=pooled.var()/1000.0,upper=pooled.var()*1000)  
  sig2 = pm.Uniform("sigma_2",lower=pooled.var()/1000.0,upper=pooled.var()*1000)
  v = pm.Exponential("nu", beta=1.0/29)
  
  # Set up posterior distribution
  t1 = pm.NoncentralT("t_1", mu=mu1, lam=1.0/sig1, nu=v, value=group1,
                      observed=True)
  t2 = pm.NoncentralT("t_1", mu=mu2, lam=1.0/sig2, nu=v, value=group2,
                      observed=True)
  
  # Generate the model
  model = pm.Model( [t1, mu1, sig1, t2, mu2, sig2, v] ) # Push priors
  mcmc = pm.MCMC(model) # Generate MCMC object
  mcmc.sample(40000, 10000, 2) # Run MCMC sampler # "trace"
  
  # Get the numerical results
  mus1 = mcmc.trace('mu_1')[:]  
  mus2 = mcmc.trace('mu_2')[:]  
  sigmas1 = mcmc.trace('sigma_1')[:]  
  sigmas2 = mcmc.trace('sigma_2')[:]  
  nus = mcmc.trace('nu')[:] 
  diff_mus = mus1-mus2  # Difference in mus
  diff_sigmas = sigmas1-sigmas2  
  normality = np.log(nus)  
  effect_size = (mus1-mus2)/np.sqrt((sigmas1**2+sigmas2**2)/2.)  
  print('\n   Group 1 mu: %.4f\n   Group 2 mu: %.4f\n   Effect size: %.4f'
        %(mus1.mean(), mus2.mean(), effect_size.mean()))
  
  if show: # Plot some basic metrics if desired
    from pymc.Matplot import plot as mcplot
    # mcplot(mcmc) # This plots 5 graphs, only useful as a benchmark.
    
    # Finally, what can this tell is about the null hypothesis?
    # Split distribution
    fig2 = plt.figure() 
    ax2 = fig2.add_subplot(121)
    minx = min(min(mus1),min(mus2))  
    maxx = max(max(mus1),max(mus2))  
    xs = np.linspace(minx,maxx,1000)
    gkde1 = stats.gaussian_kde(mus1)  
    gkde2 = stats.gaussian_kde(mus2)
    ax2.plot(xs,gkde1(xs),label='$\mu_1$')  
    ax2.plot(xs,gkde2(xs),label='$\mu_2$')  
    ax2.set_title('$\mu_1$ vs $\mu_2$')
    ax2.legend()
    
    # Difference of mus
    ax3 = fig2.add_subplot(122)
    minx = min(diff_mus)  
    maxx = max(diff_mus)  
    xs = np.linspace(minx,maxx,1000)  
    gkde = stats.gaussian_kde(diff_mus)
    ax3.plot(xs,gkde(xs),label='$\mu_1-\mu_2$')  
    ax3.legend()
    ax3.axvline(0, color='#000000',alpha=0.3,linestyle='--')
    ax3.set_title('$\mu_1-\mu_2$')
    plt.show()
  
  return

#








