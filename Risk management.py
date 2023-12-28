
#Navid Hosseini stu232675



#call packages
import numpy as np
import scipy.stats as stat
import matplotlip.pyplot as plt

#function to compute VaR
def VaR_log_normal(s,alpha):
    #Log returns X same as exercise 1
    x = np.diff(np.log(s)) #differentiate
    sigma = np.sqrt(np.var(x))
    mu = np.mean(x)
    #qL as quantile, stats.norm.ppf Returns a 95% significance interval for a one-tail test on a standard normal distribution
    #qL as in page 11
    qL = stat.norm.ppf(alpha) #quantile with respect to alpha of VaR
    VaR = s[-1] * (1 - np.exp(mu - sigma * qL))  #computes VaR from last entry of s i.e. s_n
    return VaR

#retrieve dax data
data = np.genfromtxt("dax_data.csv", delimiter=";", usecols=4, skip_header=1)
#ensure data starts with n0 end with n8368
dax = np.flip(data)

m = 252 #days after which to consider
alpha = [0.9, 0.95]

n = len(dax)
loss = np.zeros(n)
loss[1:] = -np.diff(dax) #diff fn computed s(i) - s(i-1) so inserted negative at
# the beginning to ensure it is the reverse, initial value remains at zero


VaR = np.zeros((len(alpha), n)) #matrix for VaR
violations = np.zeros((len(alpha), n)) #Matrix for violations


dax2 = dax[m:n]

def VaR_log_normal(s, alpha):
    for j in range(0, len(alpha)):
        for i in range(m, n):
        
            VaR[j, i] = VaR_log_normal(dax[i - m:i], alpha[j])
            violations[j, i] = loss[i] > VaR[j, i]


plt.plot(range(m, n), loss[m:n], '+')  
plt.plot(range(m, n), VaR[0, m:n])  
plt.plot(range(m, n), VaR[1, m:n])  
plt.show()



