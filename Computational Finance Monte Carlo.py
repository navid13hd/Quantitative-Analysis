
# Seyed Navid Hosseini




import math
import numpy as np
import scipy.stats

def Eu_Option_BS_MC(S0, r, Sigma, T, K, M, f):
 
    ST = S0 * np.exp((r - 0.5*math.pow(sigma, 2))*T+ sigma*np.sqrt(T)*np.random.normal(size=M))
    
    payoff= f(ST, K)
    
    
    # compute the Monte-Carlo estimator of the option price
    V0 = np.exp(-r*T) * np.mean(payoff)
    
    # compute the standard error and the confidence interval
    se = np.std(payoff, ddof=1) / np.sqrt(M)
    z = scipy.stats.norm.ppf(0.975)
    c1 = V0 - z*se
    c2 = V0 + z*se
    
    return V0, c1, c2

# define the payoff function for a call option
def f(ST, K):
    return np.maximum(ST - K, 0)


# set the parameters of the option and the simulation
S0 = 110
r = 0.04
sigma = 0.2
T = 1
K = 100
M = 10000
t= 0

# compute the option price and confidence interval using Monte-Carlo
V0, c1, c2 = Eu_Option_BS_MC(S0, r, sigma, T, K, M, f)

# compute the Black-Scholes price of the option
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
BS_price = S0*scipy.stats.norm.cdf(d1) - K*np.exp(-r*T)*scipy.stats.norm.cdf(d2)

# print the results
print(f"Monte-Carlo price: {V0:.4f} [{c1:.4f}, {c2:.4f}]")
print(f"Black-Scholes price: {BS_price:.4f}")