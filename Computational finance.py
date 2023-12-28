
# Seyed Navid Hosseini


import numpy as np
import matplotlib.pyplot as plt


def MC_integration (N, f, a, b):
    x= np.random.uniform(a, b, N)
    
    MC_estimator= (b-a)/N * np.sum(f(x))
    
    return MC_estimator


def f(x):
    return np.square(1-x**2)


a=0
b=1
# compute the definite integral with different values of N
N_values = [10, 100, 1000, 10000]

# initialize empty list to fill up the values for different sample size 
plt_vals = [] 

for N in N_values:
    integral = MC_integration(N, f, a, b)
    plt_vals.append(integral) 
    print(f"For N = {N}, the MC-estimator of the integral is: {integral:.4f}")

# just extra plot histogram of plt_vals
plt.hist(plt_vals, ec="black") 
plt.show()