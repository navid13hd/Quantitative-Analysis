
# Seyed Navid Hosseini


import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#3.21 Option price in the BS model by numerical integration 
def f(S0, Sigma, T,x): 
   return ((S0 * math.exp(r - 0.5)* Sigma**2)*T+ Sigma * math.sqrt(T) * x)
    

def BS_Price_Int(S0, r, Sigma, T, f, x):
    
    return 1/ math.sqrt(2*math.pi) * f * math.exp(-r * T) * math.exp(-0.5 * math.pow(x, 2))
    
#for integration 
    I= integrate.quad(BS_Price_Int, -np.inf , np.inf)
    return I[0]


def BS_Greeks_num(r, sigma, S0, T, g, eps):
    
    #f(x, y)
    VBS = BS_Price_Int(r, sigma, S0, T, g)
    
    #f(x+εx, y)
    delta0 = BS_Price_Int(r, sigma,(1 + eps) * S0, T, g)
    
    #f(x-εx, y)
    delta1 = BS_Price_Int(r, sigma,(1 - eps) * S0, T, g)
    
    #not sure if we have to multiply by (1+ eps) or just +- eps for epsilon augmentation in sigma 
    vega0 = BS_Price_Int(r, sigma + eps , S0, T, g) 
    
             
    #f(x+εx, y)−2 f(x, y) + f(x−εx, y))
    gamma = (delta0 - (2 * VBS) + delta1) / (math.pow(eps * S0, 2))
    #f(x+εx, y)− f(x, y) / εx
    delta = (vega0 - VBS)/ (eps * S0)  
    vega = vega0  / (eps * sigma)
    
    return delta, vega, gamma



# Parameters

r = 0.05
sigma = 0.3
T = 1
eps = 0.001
S0 = range (60, 141, 1)

#Payoff function g
def g(x):
    return np.maximum(x - 110, 0)


# Compute Delta for different S0 values from 60 to 140
delta = np.empty(len(S0))
vega = np.empty(len(S0))
gamma = np.empty(len(S0))

for i in range(len(S0)):
    Value = BS_Greeks_num(r, sigma, S0, T, g, eps)
    delta[i] = Value[0]
    vega[i] = Value[1]
    gamma[i] = Value[2]

# Plot Delta
plt.plot(S0, delta)
plt.xlabel('S0')
plt.ylabel('Delta')
plt.title('Delta for European Call Option')
plt.show()