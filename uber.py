#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:34:41 2019

@author: layaparkavousi
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import quad

OM = 0.29
#gamma = 0.545
sigma8 = 0.8
z_star = 0.413

def E(z):
    if z > z_star:
        Ez = np.sqrt( 1-((0.75*OM)/(1+z))-(0.25*OM*((1+z_star)**3))+(OM * np.power(1 + z,3)))
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

a = 0
b =1
N = 1000
h = -(b-a)/N
tpoints = np.arange(10,0,h)
x = 1
ff = np.zeros(N)
xpoints1 = []

for t in tpoints:
    xpoints1.append(x)
    k1 = h*f(x,t)
    k2 = h*f(x+0.5*k1,t+0.5*h)
    k3 = h*f(x+0.5*k2,t+0.5*h)
    k4 = h*f(x+k3,t+h)
    x += (k1+2*k2+2*k3+k4)/6

    
h1 = (b-a)/N  
z1=1
N1 = z1/h1
sigma = np.zeros(int(N1))
fsigma = np.zeros(int(N1))
xpoints = np.zeros(int(N1))

for i in range(0,int(N)):
    
    xpoints[i] = xpoints1[int(N)-i]
    
for i in range(0,int(N1)):
    sigma[i] = sigma8*np.sum(np.exp(-((xpoints[i])/(1+i*h1))*i))
    
for i in range(0,int(N1)):
    
    fsigma[i] = xpoints[i]*sigma[i]


z = np.arange(0,10,10*h1)
    
plt.plot(z,fsigma,label = "Runge-Kutta",color='red')
plt.xlim(0, 1)
#plt.ylim(0.0, 2)