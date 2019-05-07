#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  5 23:47:39 2019

@author: layaparkavousi
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

OmgM = 0.29
gamma = 0.545
sigma8 = 0.8

def E_inverse(z):
    Ez = np.sqrt((1-OmgM)+ (OmgM * np.power(1 + z,3)))
    return 1. / Ez



def jj(z):
    m = (-(omega_M(z))**gamma)/(1+z)
    
    return m

def integral2(z):
    
    q, _ = quad(jj,0,z ,  args=())
    
    return q 

def omega_M(z):
    omega = OmgM*((1+z)**3)* ((E_inverse(z))**2)
    return omega

#(x**2+(2-1.5*omega_M(t))*x-1.5*omega_M(t))/((1+t)**3)
#(x**2+(2-(1.5*omega_M(t)/((OmgM*(1+t)**3)+(1-OmgM))))*x-1.5*OmgM*((1+t)**3)/((OmgM*(1+t)**3)+(1-OmgM))))/((1+t)**3)
def f(x,t):
    
    return (x**2+(2-1.5*omega_M(t))*x-1.5*omega_M(t))/((1+t)**1)


a = 0
b =2
N = 200
h = -(b-a)/N
#tpoints= np.linspace(10., 0, 500)

tpoints = np.arange(10,0,h)
xpoints = []
x = 1

for t in tpoints:
    xpoints.append(x)
    k1 = h*f(x,t)
    k2 = h*f(x+0.5*k1,t+0.5*h)
    k3 = h*f(x+0.5*k2,t+0.5*h)
    k4 = h*f(x+k3,t+h)
    x += (k1+2*k2+2*k3+k4)/6


#z = np.linspace(10., 0, 500)
#N = len(z)
ff = np.zeros(N)




z = np.linspace(0, 2, 500)
N = len(z)
ff = np.zeros(N)
f1 = np.zeros(N)
f2 = np.zeros(N)

for i in range (0,N):
    f1[i] = ((OmgM*((1+z[i])**3))/((OmgM*((1+z[i])**3))+1-OmgM))**0.545

for i in range (0,N):
    
    f2[i] = (omega_M(z[i]))**0.545
    
    

for i in range (0,N):
    ff[i] = sigma8 * f2[i] * np.exp(integral2(z[i]))
    
#plt.plot(z,ff,label="gamma" )
#plt.plot(tpoints,xpoints,label = "Runge-Kutta",color='red')
#plt.plot(z,f1,label="gamma" )
plt.plot(z,f2,label="gamma" )

plt.plot(tpoints,xpoints,label = "Runge-Kutta",color='red')
plt.xlabel("redshift(z)")
plt.ylabel("f(z)")
#plt.plot(z,delta)
plt.title("LCDM model")
plt.legend()

#plt.ylim(0.29, 0.6)