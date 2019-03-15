#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:20:45 2019

@author: layaparkavousi
"""

import corner
import emcee
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cte
from scipy.integrate import quad
from scipy.optimize import minimize

## CMB data 
l_cmb = 302.
err_l_cmb = 0.2
z_ls = 1090.

theta_cmb = np.pi / l_cmb
err_theta_cmb = err_l_cmb / l_cmb * theta_cmb
print("z = %.1f, Theta = %.7f +/- %.7f" % (z_ls, theta_cmb, err_theta_cmb))

##Supernova data
# Loading data
z_sn , m_obs, dm_obs = np.loadtxt("SCPUnion2.1_mu_vs_z.dat", unpack=True)

## BAO data

bao_file = "theta_bao.dat"

z_bao, theta_deg, err_theta_deg = np.loadtxt(bao_file, unpack=True)

theta_bao = theta_deg * np.pi / 180.
err_theta_bao = err_theta_deg * np.pi / 180.

"""
plt.xlabel(r"$z$")
plt.ylabel(r"$\mu = m - M$")
plt.errorbar(z_obs, m_obs, dm_obs, fmt='r.' )
"""
## w != -1  general parametrization for equation of state
### functions for friedmann equation and DM and chi

def luminosity_integrand(z, w_0,w_a,omgM,omgR):
    Ez = np.sqrt(((1-omgM)*np.power(1+z, 3*(1+(w_0+w_a*(z/(1+z)))))+ (omgM * np.power(1 + z,3))+ omgR * np.power(1 + z, 4)))
    return 1. / Ez

def h_0(omgM):
    h = np.sqrt(0.1426/omgM)
    return h


def luminosity_distance(z,w_0,w_a,omgM,h,omgR):
    #h = h_0(omgM)
    integral, _ = quad(luminosity_integrand, 0, z, epsrel=1e-8, args=(w_0,w_a,omgM,omgR))
    return (cte.c /10**5) /h*(1+z) *integral

## distance_modulus
def distance_modulus(z,w_0,w_a,omgM,h,omgR):
    #h = h_0(omgM)
    return 5. * np.log10(luminosity_distance(z,w_0,w_a,omgM,h,omgR)) +25

def E_inverse(z,w_0,w_a, omgM, omgR):
    Ez = np.sqrt(((1 - omgM)*np.power(1+z, 3*(1+(w_0+w_a*(z/(1+z))))))+ omgM * np.power(1 + z, 3) + omgR * np.power(1 + z, 4))
    return 1. / Ez


def sound_horizon_integrand(z,w_0,w_a, omgM, omgR, omgB, omgG):
    cs = np.power(3 + 9./4 * omgB / (omgG * z), -0.5)
    return cs * E_inverse(z, w_0,w_a,omgM, omgR)

def sound_horizon(z_star, w_0,w_a,omgM, omgR, omgB, omgG):
    integral, _ = quad(sound_horizon_integrand, z_star, np.inf, args=(w_0,w_a,omgM, omgR, omgB, omgG))
    return integral

def angular_distance(z,w_0,w_a, omgM, omgR):
    integral, _ = quad(E_inverse, 0, z, args=(w_0,w_a,omgM, omgR))
    return integral

def theta_model(z, z_star, w_0,w_a,omgM, omgR, omgB, omgG):
    return sound_horizon(z_star, w_0,w_a,omgM, omgR, omgB, omgG) / angular_distance(z, w_0,w_a,omgM, omgR)

def chisq_sn(w_0,w_a,omgM,h,omgR):
    
    m_model = np.array([distance_modulus(z,w_0,w_a,omgM,h,omgR) for z in z_sn])
    chisq_vec = np.power(((m_model - m_obs) / dm_obs), 2)
    return chisq_vec.sum()

def chisq_cmb(w_0,w_a,omgM, omgR, omgB, omgG):
    z_star_cmb = 1090
    theta_cmb_model = theta_model(z_ls, z_star_cmb, w_0,w_a,omgM, omgR, omgB, omgG)
    chisq = np.power( (theta_cmb_model - theta_cmb) / err_theta_cmb, 2 )
    return chisq

def chisq_bao(w_0,w_a,omgM, omgR, omgB, omgG):
    z_star_bao = 1060
    theta_bao_model = np.array([theta_model(z, z_star_bao, w_0,w_a,omgM, omgR, omgB, omgG) for z in z_bao])
    chisq = np.sum( np.power( (theta_bao_model - theta_bao) / err_theta_bao, 2 ) )
    return chisq

def chisq_cmb_bao(w_0,w_a,omgM,h,omgR, omgB, omgG):
    
    chisq = chisq_cmb(w_0,w_a,omgM, omgR, omgB, omgG) + \
            chisq_bao(w_0,w_a,omgM, omgR, omgB, omgG)
    return chisq

omgM_fid = 0.3
h_fid = np.sqrt(0.1426/omgM_fid)
omgB_fid = 0.05
omgG_fid = 5e-5
omgR_fid = 8.4e-5
w_0fid = -1
w_afid = 0

omgMh2_cmb = 0.1426
omgMh2_cmb_error = 0.0020

def chisq_cmb2(omgM,h):
    omgMh2_cmb_model = omgM*(h**2)
    chisq = np.power( (omgMh2_cmb_model - omgMh2_cmb) / omgMh2_cmb_error, 2 )
    return chisq
    
    

def chisq_sn_cmb(pars):
    w_0,w_a,omgM,h = pars
    return chisq_cmb(w_0,w_a,omgM, omgR_fid, omgB_fid, omgG_fid) + \
           chisq_sn(w_0,w_a,omgM,h, omgR_fid)+ chisq_cmb2(omgM,h)+chisq_bao(w_0,w_a,omgM, omgR_fid, omgB_fid, omgG_fid)
           
result_sn_cmb = minimize(chisq_sn_cmb, [w_0fid,w_afid,omgM_fid,h_fid], bounds=((-2, 0),(-2, 1),(0.01, 1.), (0.01, 1.)))
w0_sn_cmb,wa_sn_cmb,omgM_sn_cmb,h_sn_cmb = result_sn_cmb.x
#print("Convergiu?: ", result_sn_bao.success)
#print("chisq / dof = ", result_sn_bao.fun / (len(z_sn) + len(z_bao) - 2))
print("w_0 = ", w0_sn_cmb)
print("w_a = ", wa_sn_cmb)
print("omegaM = ",  omgM_sn_cmb)
print("h = ", h_sn_cmb)

def lnprior(pars):
    w_0,w_a,h, omgM = pars
    if  -2 < w_0 < 0 and -2 < w_a < 0.8 and 0.0 < omgM < 1.0 and  0.0 < h:
        return 0.0
    return -np.inf

def lnlike_sn_bao(pars):
    w_0,w_a,omgM,h= pars
    return -0.5 * chisq_sn_cmb([w_0,w_a,omgM,h])

def lnprob_sn_bao(pars):
    lp = lnprior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_sn_bao(pars)



ndim, nwalkers, nsteps = 4, 50, 1000
pos = [[w0_sn_cmb,wa_sn_cmb,omgM_sn_cmb,h_sn_cmb] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# MCMC chain with 50 walkers and 1000 steps
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_sn_bao, threads=4)
sampler.run_mcmc(pos, nsteps)

# Getting chains
w0_sn_cmb_chain = sampler.chain[:,:,0]
wa_sn_cmb_chain = sampler.chain[:,:,1]
omgM_sn_cmb_chain = sampler.chain[:,:,2]
h_sn_cmb_chain = sampler.chain[:,:,3]

# Average and standard deviation between chains
h_sn_cmb_chain_mean = np.mean(h_sn_cmb_chain, axis=0)
h_sn_cmb_chain_std = np.std(h_sn_cmb_chain, axis=0) / np.sqrt(nwalkers)

# Reshaping
w0_sn_cmb_chain_flat = np.reshape(w0_sn_cmb_chain, (nwalkers*nsteps,))
wa_sn_cmb_chain_flat = np.reshape(wa_sn_cmb_chain, (nwalkers*nsteps,))
omgM_sn_cmb_chain_flat = np.reshape(omgM_sn_cmb_chain, (nwalkers*nsteps,))
h_sn_cmb_chain_flat = np.reshape(h_sn_cmb_chain, (nwalkers*nsteps,))

labels = [r"$w_{0}$",r"$w_{a}$",r"$\Omega_{m0}$", r"$h$"]
samples = np.c_[w0_sn_cmb_chain_flat,wa_sn_cmb_chain_flat,omgM_sn_cmb_chain_flat,h_sn_cmb_chain_flat].T

w0_sn_cmb_chain_mean = np.mean(w0_sn_cmb_chain, axis=0)
w0_sn_cmb_chain_err = np.std(wa_sn_cmb_chain, axis=0) / np.sqrt(nwalkers)
wa_sn_cmb_chain_mean = np.mean(wa_sn_cmb_chain, axis=0)
wa_sn_cmb_chain_err = np.std(wa_sn_cmb_chain, axis=0) / np.sqrt(nwalkers)
omgM_sn_cmb_chain_mean = np.mean(omgM_sn_cmb_chain, axis=0)
omgM_sn_cmb_chain_err = np.std(omgM_sn_cmb_chain, axis=0) / np.sqrt(nwalkers)
h_sn_cmb_chain_mean = np.mean(h_sn_cmb_chain, axis=0)
h_sn_cmb_chain_err = np.std(h_sn_cmb_chain, axis=0) / np.sqrt(nwalkers)


burn = 4000
samples_burned = np.c_[[par[burn:] for par in samples]]
fig = corner.corner(samples_burned.T,labels=labels,
                    quantiles=[0.16, 0.5, 0.84], 
                    levels=(1-np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-9./2)), #1sigma, 2sigma and 3sigma contours
                    show_titles=True, title_kwargs={"fontsize": 12},
                    smooth1d=None, plot_contours=True,
                    no_fill_contours=False, plot_density=True,)


