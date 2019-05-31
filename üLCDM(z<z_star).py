#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 02:18:17 2019

@author: layaparkavousi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 01:59:29 2019

@author: layaparkavousi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:48:59 2019

@author: layaparkavousi
"""


import matplotlib.pyplot as plt
import numpy as np

OM = 0.28
sigma8_0 = 0.8
z_star = 0.537

def E(z):
    
    Ez = np.sqrt(1-((0.75*OM)/(1+z_star))+(((0.75*OM)/(1+z_star))*((1+z)**4)))
        
    return Ez

def dlnEdz(z):
    
    dlnEdz = (1.5*(OM/(1+z_star))*((1+z)**3))/(1-((0.75*OM)/(1+z_star))+((0.75*OM)/(1+z_star))*((1+z)**4))
    return dlnEdz

def RungeKutta4(x0, xn, y0, z0, h):

    n = int((xn - x0)/h)
    # Containers for solutions
    xlist = [0] * (n + 1)
    ylist = [0] * (n + 1)
    zlist = [0] * (n + 1)

    xlist[0] = x = x0
    ylist[0] = y = y0
    zlist[0] = z = z0

    for i in range(1, n + 1):
        k1 = h * f(x, y, z)
        l1 = h * g(x, y, z)
        k2 = h * f(x + 0.5 * h, y + 0.5 * k1, z + 0.5*l1)
        l2 = h * g(x + 0.5 * h, y + 0.5 * k1, z + 0.5*l1)
        k3 = h * f(x + 0.5 * h, y + 0.5 * k2, z + 0.5*l2)
        l3 = h * g(x + 0.5 * h, y + 0.5 * k2, z + 0.5*l2)
        k4 = h * f(x + h, y + k2, z + l2)
        l4 = h * g(x + h, y + k2, z + l2)
        xlist[i] = x = x0 + i * h
        ylist[i] = y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        zlist[i] = z = z + (l1 + 2*l2 + 2*l3 + l4) / 6

    return xlist, ylist


def g(x, y, z):

    return -(dlnEdz(x)-(1/(1+x)))*z + ((2*OM*(1+x))/(E(x)**2))*y


def f(x, y, z):
    # y' = z
    return z


def plot(x1, y1, color1, linestyle1, h):


    plt.plot(x1, y1, color=color1, linestyle=linestyle1)
    
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':

    x0 = 0
    xn = 0.536
    h1 = 0.001
    y0 = 1
    z0 = -0.562

    xlist1, ylist1 = RungeKutta4(x0, xn, y0, z0, h1)
    
N = len(xlist1)
sigma8 = np.zeros(N)

for i in range(0,N):
    
    sigma8[i] = sigma8_0*ylist1[i]
    
xpoints1 = []

def fz(x,z):
    a = ((x**2)/(1+z))-((1.5*(OM/(1+z_star))*((1+z)**3))/(1-((0.75*OM)/(1+z_star))+((0.75*OM)/(1+z_star))*((1+z)**4))-(2/(1+z)))*x - (2*OM*((1+z)**2))/((E(z))**2)
    
    return a

tpoints = np.arange(0,0.537,0.001)

h = 0.001
x = 0.562
fsigma8 = np.zeros(N)
x1 = np.zeros(N)

for t in tpoints:
    xpoints1.append(x)
    k1 = h*fz(x,t)
    k2 = h*fz(x+0.5*k1,t+0.5*h)
    k3 = h*fz(x+0.5*k2,t+0.5*h)
    k4 = h*fz(x+k3,t+h)
    x += (k1+2*k2+2*k3+k4)/6

for i in range(0,N):
    x1[i] = xpoints1[N-i-1]

for i in range(0,N):
    
    fsigma8[i] = xpoints1[i]*sigma8[i]
    
    

    
plt.plot(xlist1, fsigma8)

plt.xlim(0, 0.9)
plt.ylim(0.3, 0.6)
