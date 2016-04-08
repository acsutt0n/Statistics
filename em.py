import numpy as np
import math

#### E-M Coin Toss Example as given in the EM tutorial paper by Do and Batzoglou* #### 




def get_mn_log_likelihood(obs,probs):
  """ Return the (log)likelihood of obs, given the probs"""
  # Multinomial Distribution Log PMF
  # ln (pdf)      =        multinomial coeff      *   product of probabilities
  # ln[f(x|n, p)] = [ln(n!) - (ln(x1!)+ln(x2!)+...+ln(xk!))] + [x1*ln(p1)+x2*ln(p2)+...+xk*ln(pk)]     

  multinomial_coeff_denom= 0
  prod_probs = 0
  for x in range(0,len(obs)): # loop through state counts in each observation
    multinomial_coeff_denom = multinomial_coeff_denom + math.log(math.factorial(obs[x]))
    prod_probs = prod_probs + obs[x]*math.log(probs[x])

  multinomial_coeff = math.log(math.factorial(sum(obs))) -  multinomial_coeff_denom
  likelihood = multinomial_coeff + prod_probs
  return likelihood



def em_web_sample(show=True):
  # Sample em from http://stackoverflow.com/questions/11808074/what-is-an-intuitive-explanation-of-expectation-maximization-technique
  head_counts = np.array([5,9,8,4,7]) # Number of heads
  tail_counts = 10-head_counts # Number of tails
  experiments = list(zip(head_counts,tail_counts))

  # initialise the pA(heads) and pB(heads)
  pA_heads = np.zeros(100); pA_heads[0] = 0.60
  pB_heads = np.zeros(100); pB_heads[0] = 0.50
  ll_As, ll_Bs = [], []
  wgtsA, wgtsB = [], []

  # E-M begins!
  delta = 0.001  
  j = 0 # iteration counter
  improvement = float('inf')
  while (improvement>delta):
    expectation_A = np.zeros((5,2), dtype=float) 
    expectation_B = np.zeros((5,2), dtype=float)
    for i in range(0,len(experiments)):
      e = experiments[i] # i'th experiment
      ll_A = get_mn_log_likelihood(e,np.array([pA_heads[j],1-pA_heads[j]])) # loglikelihood of e given coin A
      ll_B = get_mn_log_likelihood(e,np.array([pB_heads[j],1-pB_heads[j]])) # loglikelihood of e given coin B
      ll_As.append(ll_A)
      ll_Bs.append(ll_B)
      
      weightA = math.exp(ll_A) / ( math.exp(ll_A) + math.exp(ll_B) ) # corresponding weight of A proportional to likelihood of A 
      weightB = math.exp(ll_B) / ( math.exp(ll_A) + math.exp(ll_B) ) # corresponding weight of B proportional to likelihood of B                            
      wgtsA.append(weightA)
      wgtsB.append(weightB)
      expectation_A[i] = np.dot(weightA, e) 
      expectation_B[i] = np.dot(weightB, e)
      
    pA_heads[j+1] = sum(expectation_A)[0] / sum(sum(expectation_A)); 
    pB_heads[j+1] = sum(expectation_B)[0] / sum(sum(expectation_B)); 
    improvement = max( abs(np.array([pA_heads[j+1],pB_heads[j+1]]) - np.array([pA_heads[j],pB_heads[j]]) ))
    j = j+1
  
  if show:
    plt.subplot(3,1,1) # First is convergence of thetas
    plt.plot([i for i in pA_heads if i != 0.], color='cornflowerblue', 
             alpha=0.5, lw=5, label='p(A heads)')
    plt.plot([i for i in pB_heads if i != 0.], color='tomato', 
             alpha=0.5, lw=5, label='p(B heads)')
    plt.plot([i for i in pA_heads if i != 0.], color='cornflowerblue', lw=1)
    plt.plot([i for i in pB_heads if i != 0.], color='tomato', lw=1)
    plt.legend()
    plt.ylabel('Probability of heads')
    plt.subplot(3,1,2) # Then log likelihood
    plt.plot(ll_As, color='cornflowerblue', lw=2)
    plt.plot(ll_Bs, color='tomato', lw=2)
    plt.ylabel('log likelihood')
    plt.subplot(3,1,3) # Then weights
    plt.plot(wgtsA, color='cornflowerblue', lw=2)
    plt.plot(wgtsB, color='tomato', lw=2)
    plt.ylabel('Weights')
    plt.show()
  
  return pA_heads, pB_heads





#########################################################################
# Homebrew EM




def em_alex_sample(show=True):
  # A home-made em example
  head_counts = np.array([5,9,8,4,7]) # Number of heads
  tail_counts = 10-head_counts # Number of tails
  experiments = list(zip(head_counts,tail_counts))

  # initialise the pA(heads) and pB(heads)
  pA_heads = np.zeros(100); pA_heads[0] = 0.60
  pB_heads = np.zeros(100); pB_heads[0] = 0.50
  ll_As, ll_Bs = [], []
  wgtsA, wgtsB = [], []
  
  




