import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import quad
from scipy.integrate import odeint

OM = 0.29
#gamma = 0.545
sigma8 = 0.8
z_star = 0.537


def E(z):
    if z > z_star:
        Ez = np.sqrt( 1-((0.75*OM)/(1+z_star))-(0.25*OM*((1+z_star)**3))+(OM * np.power(1 + z,3)))
    else:
        Ez = np.sqrt(1-((0.75*OM)/(1+z_star))+(((0.75*OM)/(1+z_star))*((1+z)**4)))
    return Ez

def dlnEdz(z):
    if z > z_star:
        dlnEdz = (1.5*((1+z)**2))/(1-((0.75*OM)/(1+z_star))-(0.25*OM*((1+z_star)**3))+ (OM * np.power(1 + z,3)))
    else:
        dlnEdz = (1.5*(OM/(1+z_star))*((1+z)**3))/(1-((0.75*OM)/(1+z_star))+((0.75*OM)/(1+z_star))*((1+z)**4))
    
    return dlnEdz


def f(x,t):
    
    return ((x**2)/(1+t))-(dlnEdz(t)-(2/(1+t)))*x - (2*OM*((1+t)**2))/((E(t))**2)

time = np.linspace(0,5,1000)
z = np.linspace(0,3,1000)

z2 = odeint(f,0.56,time)

a = 0
b = 10
N = 1000
h1 = (b-a)/N  
z1=1
sigma = np.zeros(N)
fsigma = np.zeros(N)
    
for i in range(0,N):
    sigma[i] = sigma8*np.sum(np.exp(-((z2[i])/(1+i*h1))*h1))

for i in range(0,N):
    
    fsigma[i] = z2[i]*sigma[i]

plt.plot(time,fsigma)

plt.xlim(0, 0.9)
plt.ylim(0.0, 0.8)

