
# Seyed Navid Hosseini

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

#defining function CRR for European call with given parameters
def CRR_EuCall(S_0, r, sigma, T, K, M):
    delta_T= T/M
#set u,d and Beta according to 2.4 and 2.7
    beta = 0.5 * (math.exp(-r*delta_T) + math.exp(r+math.pow(sigma, 2)*delta_T))
    u = beta + math.sqrt((beta**2)-1)
    d = math.pow(u, -1)
    
#making matrix S of stock prices in CRR model #zeroes or empty makes no difference.The point is the Loop from c_exercise 01 

    S = np.zeros((M+1, M+1))

    for i in range(1, M + 2, 1):
        for j in range(1, i + 1, 1):
            S[j - 1, i - 1] = S_0 * math.pow(u, j - 1) * math.pow(d, i - j)
            
    
#defining function V to calculate the value of the call option 
    def V(S):
        return np.maximum(0, S-K)
#allocating option values in Matrix V_0 same as the stock prices in C_exercise01
    V_0=np.zeros((M+1, M+1))
    
    V_0[:, M] = V(S[:, M])
    
    return V_0
    
    
    
#Black Sholes d1, d2, ...
def BlackScholes_EuCall (t, S_t, r, sigma, T, K):
    d_1= (math.log(S_t / K)+ (r + (math.pow(sigma, 2))/2)*(T-t))/(sigma * math.sqrt(T-t))
    d_2= d_1 - (sigma * math.sqrt(T-t))
    #defining phi d1 and d2 (normal continouse random variable)
    Phi_d1= scipy.stats.norm.cdf(d_1)
    Phi_d2= scipy.stats.norm.cdf(d_2)
    #BS formula
    C = S_t * Phi_d1 - ((K*math.exp(-r * (T-t))*Phi_d2))
    return C
    
#testing 
S_0=100
r=0.03
sigma=0.3
T=1
M=100
K=range(70,201)




# plot comparison and do not forget to label everything


plt.show()




