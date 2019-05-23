#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:34:56 2019

@author: layaparkavousi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 23:36:35 2019

@author: layaparkavousi
"""

import corner
import emcee
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cte
from scipy.integrate import quad
from scipy.optimize import minimize


z_sn , m_obs, dm_obs = np.loadtxt("SCPUnion2.1_mu_vs_z.dat", unpack=True)
z_star = 0.413
l_cmb = 302.
err_l_cmb = 0.2
z_ls = 1090.

theta_cmb = np.pi / l_cmb
err_theta_cmb = err_l_cmb / l_cmb * theta_cmb
print("z = %.1f, Theta = %.7f +/- %.7f" % (z_ls, theta_cmb, err_theta_cmb))

def E(z,omgM):
    if z > z_star:
        Ez = np.sqrt((1-omgM)+ (omgM * np.power(1 + z,3)))
    else:
        Ez = np.sqrt(1-omgM + 0.25*omgM*np.power(1+z_star,3)+(((0.75*omgM)/(1+z_star))*((1+z)**4)))
    return Ez

def integrand1(z,omgM):
    return np.sqrt(1-((0.75*omgM)/(1+z_star))+((0.75*omgM)/(1+z_star))*((1+z)**4))

def integrand2(z,omgM):
    return np.sqrt((1-omgM)+ (omgM * np.power(1 + z,3)))


def luminosity_distance1(z,omgM,h):
    return (((quad(integrand1,0, z_star,args=(omgM))[0])/(1+z)))*4.3*(10**3)

def luminosity_distance2(z,omgM,h):
    return (((quad(integrand2,z_star, z,args=(omgM))[0])/(1+z)))*4.3*(10**3)
"""
def luminosity_distance(z,omgM,h):
    
    return luminosity_distance1(z,omgM,h)+luminosity_distance2(z,omgM,h)
    
"""
def luminosity_distance(z,omgM,h):
    #h = h_0(omgM)
    integral, _ = quad(E, 0, z, epsrel=1e-8, args=(omgM))
    return (cte.c /10**5) /h*(1+z) *integral

def distance_modulus(z,omgM,h):
    #h = h_0(omgM)
    return 5. * np.log10(luminosity_distance(z,omgM,h)) +25

def jj(z,omgM,gamma0,gamma1):
    m = -((omgM)**(gamma0+(gamma1*z/(1+z))))/(1+z)
    
    return m

def integral2(z,omgM,gamma0,gamma1):
    
    q, _ = quad(jj,0,z , epsrel=1e-8, args=(omgM,gamma0,gamma1))
    
    return q 


def fsigma8_model(z,sigma8,omgM,gamma0,gamma1):
    fsigma = sigma8 * np.power(omgM,gamma0+((gamma1*z)/(1+z)))*np.exp(integral2(z,omgM,gamma0,gamma1))
    return fsigma

def chisq_sn(omgM,h):
    
    m_model = np.array([distance_modulus(z,omgM,h) for z in z_sn])
    chisq_vec = np.power(((m_model - m_obs) / dm_obs), 2)
    return chisq_vec.sum()

file = "fsigma8.dat"

z_f, fsigma8_obs, err_fsigma8_obs = np.loadtxt(file, unpack=True)


def chisq_fsigma8(sigma8,omgM,gamma0,gamma1):

    fsigma8_theo = np.array([fsigma8_model(z,sigma8,omgM,gamma0,gamma1) for z in z_f])
    chisq = np.sum( np.power( (fsigma8_theo - fsigma8_obs) / err_fsigma8_obs, 2 ) )
    return chisq


omgM_fid = 0.28
h_fid = np.sqrt(0.1426/omgM_fid)
gamma0_fid = 0.6
sigma8_fid = 0.8
gamma1_fid = 0

#z_star_fid = 0.413

omgMh2_cmb = 0.1426
omgMh2_cmb_error = 0.0020

def chisq_cmb2(omgM,h):
    omgMh2_cmb_model = omgM*(h**2)
    chisq = np.power( (omgMh2_cmb_model - omgMh2_cmb) / omgMh2_cmb_error, 2 )
    return chisq
    
    
def chisq_sn_cmb(pars):
    
    omgM,h,gamma0,gamma1,sigma8 = pars
    
    return chisq_sn(omgM,h)+ chisq_fsigma8(sigma8,omgM,gamma0,gamma1)+chisq_cmb2(omgM,h)
           
result_sn_cmb = minimize(chisq_sn_cmb, [omgM_fid,h_fid,gamma0_fid,gamma1_fid,sigma8_fid], bounds=((0.01, 1.), (0.01, 1.),(0.1,1),(-1,1),(0.25,1.25)))
omgM_sn_cmb,h_sn_cmb,gamma0_sn_cmb ,gamma1_sn_cmb,sigma8_sn_cmb = result_sn_cmb.x

print("omegaM = ",  omgM_sn_cmb)
print("h = ", h_sn_cmb)
print("gamma0 = ", gamma0_sn_cmb)
print("gamma1 = ", gamma1_sn_cmb)
print("sigma8(0) = ", sigma8_sn_cmb)

def lnprior(pars):
    h, omgM,gamma0,gamma1,sigma8 = pars
    if  0.0 < omgM < 1.0 and  0.0 < h and 0.0 < gamma0 < 1 and 0 < sigma8 < 2 and -1 < gamma1 < 1:
        return 0.0
    return -np.inf

def lnlike_sn_bao(pars):
    omgM,h,gamma0,gamma1,sigma8 = pars
    return -0.5 * chisq_sn_cmb([omgM,h, gamma0,gamma1 , sigma8])

def lnprob_sn_bao(pars):
    lp = lnprior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_sn_bao(pars)


ndim, nwalkers, nsteps = 5, 50, 1000
pos = [[omgM_sn_cmb,h_sn_cmb,gamma0_sn_cmb,gamma1_sn_cmb,sigma8_sn_cmb] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# MCMC chain with 50 walkers and 1000 steps
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_sn_bao, threads=4)
sampler.run_mcmc(pos, nsteps)

# Getting chains
omgM_sn_cmb_chain = sampler.chain[:,:,0]
h_sn_cmb_chain = sampler.chain[:,:,1]
gamma0_sn_cmb_chain = sampler.chain[:,:,2]
gamma1_sn_cmb_chain = sampler.chain[:,:,3]
sigma8_sn_cmb_chain = sampler.chain[:,:,4]

# Average and standard deviation between chains
h_sn_cmb_chain_mean = np.mean(h_sn_cmb_chain, axis=0)
h_sn_cmb_chain_std = np.std(h_sn_cmb_chain, axis=0) / np.sqrt(nwalkers)

# Reshaping
omgM_sn_cmb_chain_flat = np.reshape(omgM_sn_cmb_chain, (nwalkers*nsteps,))
h_sn_cmb_chain_flat = np.reshape(h_sn_cmb_chain, (nwalkers*nsteps,))
gamma0_sn_cmb_chain_flat = np.reshape(gamma0_sn_cmb_chain, (nwalkers*nsteps,))
gamma1_sn_cmb_chain_flat = np.reshape(gamma1_sn_cmb_chain, (nwalkers*nsteps,))
sigma8_sn_cmb_chain_flat = np.reshape(sigma8_sn_cmb_chain, (nwalkers*nsteps,))

labels = [r"$\Omega_{m0}$", r"$h$",r"$gamma_0$",r"$gamma_1$",r"$sigma8$"]
samples = np.c_[omgM_sn_cmb_chain_flat,h_sn_cmb_chain_flat,gamma0_sn_cmb_chain_flat,gamma1_sn_cmb_chain_flat,sigma8_sn_cmb_chain_flat].T

omgM_sn_cmb_chain_mean = np.mean(omgM_sn_cmb_chain, axis=0)
omgM_sn_cmb_chain_err = np.std(omgM_sn_cmb_chain, axis=0) / np.sqrt(nwalkers)
h_sn_cmb_chain_mean = np.mean(h_sn_cmb_chain, axis=0)
h_sn_cmb_chain_err = np.std(h_sn_cmb_chain, axis=0) / np.sqrt(nwalkers)
gamma0_sn_cmb_chain_mean = np.mean(gamma0_sn_cmb_chain, axis=0)
gamma0_sn_cmb_chain_err = np.std(gamma0_sn_cmb_chain, axis=0) / np.sqrt(nwalkers)
gamma1_sn_cmb_chain_mean = np.mean(gamma1_sn_cmb_chain, axis=0)
gamma1_sn_cmb_chain_err = np.std(gamma1_sn_cmb_chain, axis=0) / np.sqrt(nwalkers)
sigma8_sn_cmb_chain_mean = np.mean(sigma8_sn_cmb_chain, axis=0)
sigma8_sn_cmb_chain_err = np.std(sigma8_sn_cmb_chain, axis=0) / np.sqrt(nwalkers)


burn = 1000
samples_burned = np.c_[[par[burn:] for par in samples]]
fig = corner.corner(samples_burned.T,labels=labels,
                    quantiles=[0.16, 0.5, 0.84], 
                    levels=(1-np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-9./2)), #1sigma, 2sigma and 3sigma contours
                    show_titles=False, title_kwargs={"fontsize": 12},
                    smooth1d=None, plot_contours=True,
                    no_fill_contours=False, plot_density=True,)



