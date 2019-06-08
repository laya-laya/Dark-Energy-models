#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 00:16:28 2019

@author: layaparkavousi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:12:45 2019

@author: layaparkavousi
"""

import matplotlib.pyplot as plt
import numpy as np

OM = 0.29
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
    xn = 0.537
    h1 = 0.001
    y0 = 1
    z0 = -0.562

    xlist1, ylist1 = RungeKutta4(x0, xn, y0, z0, h1)
    
N = len(xlist1)
sigma8_1 = np.zeros(N)

for i in range(0,N):
    
    sigma8_1[i] = sigma8_0*ylist1[i]
    
xpoints1 = []

def fz(x,z):
    a = ((x**2)/(1+z))-((1.5*(OM/(1+z_star))*((1+z)**3))/(1-((0.75*OM)/(1+z_star))+((0.75*OM)/(1+z_star))*((1+z)**4))-(2/(1+z)))*x - (2*OM*((1+z)**2))/((E(z))**2)
    
    return a

tpoints1 = np.arange(0,0.538,0.001)
h = 0.001
x = 0.562
fsigma8_1 = np.zeros(N)

for t in tpoints1:
    xpoints1.append(x)
    k1 = h*fz(x,t)
    k2 = h*fz(x+0.5*k1,t+0.5*h)
    k3 = h*fz(x+0.5*k2,t+0.5*h)
    k4 = h*fz(x+k3,t+h)
    x += (k1+2*k2+2*k3+k4)/6


for i in range(0,N):
    
    fsigma8_1[i] = xpoints1[i]*sigma8_1[i]
    

def E1(z):

    Ez = np.sqrt( (1-((0.75*OM)/(1+z_star))-(0.25*OM*((1+z_star)**3)))+(OM * np.power(1 + z,3)))
        
    return Ez

def dlnEdz1(z):

    dlnEdz = (1.5*OM*((1+z)**2))/(1-((0.75*OM)/(1+z_star))-(0.25*OM*((1+z_star)**3))+ (OM * np.power(1 + z,3)))
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
        k1 = h * f1(x, y, z)
        l1 = h * g1(x, y, z)
        k2 = h * f1(x + 0.5 * h, y + 0.5 * k1, z + 0.5*l1)
        l2 = h * g1(x + 0.5 * h, y + 0.5 * k1, z + 0.5*l1)
        k3 = h * f1(x + 0.5 * h, y + 0.5 * k2, z + 0.5*l2)
        l3 = h * g1(x + 0.5 * h, y + 0.5 * k2, z + 0.5*l2)
        k4 = h * f1(x + h, y + k2, z + l2)
        l4 = h * g1(x + h, y + k2, z + l2)
        xlist[i] = x = x0 + i * h
        ylist[i] = y = y + (k1 + 2*k2 + 2*k3 + k4) / 6
        zlist[i] = z = z + (l1 + 2*l2 + 2*l3 + l4) / 6

    return xlist, ylist


def g1(x, y, z):

    return -(dlnEdz1(x)-(1/(1+x)))*z + ((2*OM*(1+x))/(E1(x)**2))*y


def f1(x, y, z):
    # y' = z
    return z


if __name__ == '__main__':

    x01 = 10
    xn1 = 0.537
    h11 = -0.001
    y01= 0.05
    z01 = 0.01

    xlist2, ylist2 = RungeKutta4(x01, xn1, y01, z01, h11)

N1 = len(ylist2)    
sigma8_2 = np.zeros(N1)

for i in range(0,N1):
    
    sigma8_2[i] = sigma8_0*ylist2[i]
    
xpoints1 = []

def fz1(x,z):
    a = ((x**2)/(1+z))-((1.5*OM*((1+z)**2))/(1-((0.75*OM)/(1+z_star))-(0.25*OM*((1+z_star)**3))+ (OM * np.power(1 + z,3)))-(2/(1+z)))*x - (2*OM*((1+z)**2))/((E1(z))**2)
    return a

tpoints2 = np.arange(10.001,0.537,-0.001)

h = -0.001
x = 1
fsigma8_2 = np.zeros(N1)
xpoints2=[]
for t in tpoints2:
    xpoints2.append(x)
    k1 = h*fz1(x,t)
    k2 = h*fz1(x+0.5*k1,t+0.5*h)
    k3 = h*fz1(x+0.5*k2,t+0.5*h)
    k4 = h*fz1(x+k3,t+h)
    x += (k1+2*k2+2*k3+k4)/6


for i in range(0,N):
    
    fsigma8_2[i] = xpoints2[N1-1-i]*sigma8_2[N1-1-i]
    
    
N2 =N+N1
fsigma8 = np.zeros(N2)
xlist3 = np.zeros(N2)

for i in range(0,N):
    fsigma8[i] = fsigma8_1[i]
    
for i in range(0,N1):
    
    fsigma8[i+N] = fsigma8_2[i]
    
for i in range(0,N):
    xlist3[i] = xlist1[i]
    
for i in range(0,N1):
    
    xlist3[i+N] = xlist2[i]
 
z_vec = [0.067,0.170,0.220 ,0.250 ,0.370 ,0.410 ,0.570 ,0.600 ,0.770 ,0.780 ,0.800,0.300,0.440,0.730,0.320,0.350,0.800]
print(fsigma8[67])  
n = len(z_vec)
fsigma8_theo = np.zeros(n)
for i in range(0,n):
    fsigma8_theo[i] = fsigma8[int((z_vec[i])*100)]
plt.plot(xlist3, fsigma8,label = "üLCDM")

plt.xlim(0, 0.9)
plt.ylim(0.3, 0.6)
plt.legend()