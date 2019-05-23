#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:56:25 2019

@author: layaparkavousi
"""

from scipy.integrate import odeint

import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import quad

OM = 0.29
#gamma = 0.545
sigma8 = 0.8
z_star = 0.52

def E(z):
    if z > z_star:
        Ez = np.sqrt( 1-((0.75*OM)/(1+z_star))-(0.25*OM*((1+z_star)**3))+(OM * np.power(1 + z,3)))
    else:
        Ez = np.sqrt(1-((0.75*OM)/(1+z_star))+((0.75*OM)/(1+z_star))*((1+z)**4))
    return Ez

def dlnEdz(z):
    if z > z_star:
        dlnEdz = (1.5*((1+z)**3))/((1-OM)+ (OM * np.power(1 + z,3)))
    else:
        dlnEdz = (1.5*(OM/(1+z_star))*((1+z)**3))/(1-((0.75*OM)/(1+z_star))+((0.75*OM)/(1+z_star))*((1+z)**4))
    
    return dlnEdz


def f(x,t):
    
    return ((x**2)/(1+t))-(dlnEdz(t)-(2/(1+t)))*x - (2*OM*((1+t)**2))/((E(t))**2)

"""
a = 0
b =1
N = 1000
h = (b-a)/N
tpoints = np.arange(0,1,h)
"""
x0 = 1             # initial value
a =10        # integration limits for t
b = 0

t = np.arange(a, b, -0.001)  # values of t for
                          # which we require
                          # the solution y(t)
x = odeint(f, x0, t)  # actual computation of y(t)

N = len(t)

h1 = -(b-a)/N  
z1=10
N1 = -z1/h1
sigma = np.zeros(int(N))
fsigma = np.zeros(int(N))
xpoints = np.zeros(int(N))
for i in range(0,int(N)):
    
    xpoints[i] = x[int(N)-1-i]
    
for i in range(0,int(N)):
    sigma[i] = sigma8*np.sum(np.exp(-((xpoints[i])/(1+i*h1))*i*h1))
    
for i in range(0,int(N)):
    
    fsigma[i] = xpoints[i]*sigma[i]

import pylab          # plotting of results
pylab.plot(t, fsigma)
pylab.xlabel('z'); pylab.ylabel('fsigma8')

plt.xlim(0, 1)
plt.ylim(0.0, 1)