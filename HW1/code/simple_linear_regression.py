## copyright, Keith Chugg
##  EE599, 2019

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt 

x = np.arange(10)
y = 3*x+4
y =  y + np.random.normal(0,2,10)
slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
y_hat = intercept + slope * x

fig = plt.figure()
plt.plot(x,y_hat, color='r')
plt.scatter(x,y)
plt.xlabel("x")
plt.ylabel("estimate of y")