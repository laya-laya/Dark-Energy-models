#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 02:17:32 2019

@author: layaparkavousi
"""

import corner
import emcee
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cte
from scipy.integrate import quad
from scipy.optimize import minimize, brentq

z_obs, m_obs, dm_obs_stat = np.loadtxt("Legacy.dat", unpack=True)

# Systematic error
dm_syst = 0.0169

# Combining errors
dm_obs = np.sqrt(dm_syst + dm_obs_stat ** 2)

def luminosity_integrand(z, omgM, omgk):
    Ez = np.sqrt((1 - omgM-omgk) + omgM * np.power(1 + z, 3) + omgk * np.power(1 + z, 2))
    return 1. / Ez

def luminosity_distance(z, h, omgM, omgk):
    integral, _ = quad(luminosity_integrand, 0, z, epsrel=1e-8, args=(omgM, omgk))
    return (cte.c / 10. ** 5) / h * (1 + z) * integral

def distance_modulus(z, h, omgM, omgk):
    return 5. * np.log10(luminosity_distance(z, h, omgM, omgk)) + 25.

def chisq(h, omgM, omgk):
    m_model = np.array([distance_modulus(z, h, omgM, omgk) for z in z_obs])
    chisq_vec = np.power(((m_model - m_obs) / dm_obs), 2)
    return chisq_vec.sum()

h_fid = 0.73
omgM_fid = 0.3
omgk_fid = 0.003


# Maximum likelihood solution for omg_M and omg_R fixed at fiducial values
chisq_h = np.vectorize(lambda h: chisq(h, omgM_fid, omgk_fid))
result_h = minimize(chisq_h, h_fid, bounds=[(0.5, 1.)])
h_best, = result_h.x
print("Convergiu?: ", result_h.success)
print("chisq / dof = ", result_h.fun[0] / (len(z_obs) - 1))
print("h = ", h_best)

chisq_omgM = np.vectorize(lambda omgM: chisq(h_fid, omgM, omgk_fid))
result_omgM = minimize(chisq_omgM, omgM_fid, bounds=[(0.01, 1.)])
omgM_best, = result_omgM.x
print("Convergiu?: ", result_omgM.success)
print("chisq / dof = ", result_omgM.fun[0] / (len(z_obs) - 1))
print("omegaM = ",  omgM_best)

chisq_omgk = np.vectorize(lambda omgk: chisq(h_fid, omgM_fid, omgk))
result_omgk = minimize(chisq_omgk, omgk_fid, bounds=[(0.01, 1.)])
omgk_best, = result_omgk.x
print("Convergiu?: ", result_omgk.success)
print("chisq / dof = ", result_omgk.fun[0] / (len(z_obs) - 1))
print("omegak = ",  omgk_best)


chisq_joint = lambda x: chisq(x[0], x[1], x[2])
result_joint = minimize(chisq_joint, [h_fid, omgM_fid, omgk_fid], bounds=((0.5, 1.), (0.01, 1),(0.01, 1.)),)
h_joint, omgM_joint , omgk_joint= result_joint.x
print("Convergiu?: ", result_joint.success)
print("chisq / dof = ", result_joint.fun / (len(z_obs) - 2))
print("h = ", h_joint)
print("omegaM = ",  omgM_joint)
print("omegak = ",  omgk_joint)


def lnprior(pars):
    h, omgM,omgk = pars
    if 0.0 < h and 0.0 < omgM < 1.0 and 0 < omgk < 1.0:
        return 0.0
    return -np.inf

def lnlike(pars):
    h, omgM, omgk = pars
    return -0.5 * chisq(h, omgM, omgk)

def lnprob(pars):
    lp = lnprior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars)

# initial position at maximum likelihood values
ndim, nwalkers, nsteps = 3, 50, 1000
pos = [result_joint.x + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# MCMC chain with 50 walkers and 1000 steps
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4)
sampler.run_mcmc(pos, nsteps)

# Getting chains
h_chain = sampler.chain[:,:,0]
omgM_chain = sampler.chain[:,:,1]
omgk_chain = sampler.chain[:,:,2]

# Average and standard deviation between chains
h_chain_mean = np.mean(h_chain, axis=0)
h_chain_std = np.std(h_chain, axis=0) / np.sqrt(nwalkers)

# Reshaping
h_chain_flat = np.reshape(h_chain, (nwalkers*nsteps,))
omgM_chain_flat = np.reshape(omgM_chain, (nwalkers*nsteps,))
omgk_chain_flat = np.reshape(omgk_chain, (nwalkers*nsteps,))

labels = [r"$h$", r"$\Omega_{m}$" , r"$\Omega_{k}$"]
samples = np.c_[h_chain_flat, omgM_chain_flat,omgk_chain_flat].T

h_chain_mean = np.mean(h_chain, axis=0)
h_chain_err = np.std(h_chain, axis=0) / np.sqrt(nwalkers)
omgM_chain_mean = np.mean(omgM_chain, axis=0)
omgM_chain_err = np.std(omgM_chain, axis=0) / np.sqrt(nwalkers)
omgk_chain_mean = np.mean(omgk_chain, axis=0)
omgk_chain_err = np.std(omgk_chain, axis=0) / np.sqrt(nwalkers)

burn = 500
samples_burned = np.c_[[par[burn:] for par in samples]]
fig = corner.corner(samples_burned.T, labels=labels,
                    quantiles=[0.16, 0.5, 0.84], 
                    levels=(1-np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-9./2)), #1sigma, 2sigma and 3sigma contours
                    show_titles=True, title_kwargs={"fontsize": 12},
                    smooth1d=None, plot_contours=True,
                    no_fill_contours=False, plot_density=True,)

samples[:, 2] = np.exp(samples[:, 2])
h_perc = np.percentile(samples[0], [16,50,84])
omgM_perc = np.percentile(samples[1], [16,50,84])

print(h_perc[1], h_perc[0] - h_perc[1], h_perc[2] - h_perc[1])
print(omgM_perc[1], omgM_perc[0] - omgM_perc[1], omgM_perc[2] - omgM_perc[1])
print(omgk_perc[1], omgk_perc[0] - omgk_perc[1], omgk_perc[2] - omgk_perc[1])

