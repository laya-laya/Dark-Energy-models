#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 15:39:40 2019

@author: layaparkavousi
"""

import corner
import emcee
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cte
from scipy.integrate import quad
from scipy.optimize import minimize 


z_obs, m_obs, dm_obs = np.loadtxt("Legacy.dat", unpack=True)


"""
# Checking
plt.xlabel(r"$z$")
plt.ylabel(r"$\mu = m - M$")
plt.errorbar(z_obs, m_obs, dm_obs, fmt='r.' )

"""
## w != -1
def luminosity_integrand(z, omgM, omgde):
    Ez = np.sqrt((omgde) + (omgM * np.power(1 + z, 3))+ ((1-omgM- omgde)*np.power(1+z, 2)))
    return 1. / Ez

def luminosity_distance(z, h, omgM,omgde):
    integral, _ = quad(luminosity_integrand, 0, z, epsrel=1e-8, args=(omgM, omgde))
    return (cte.c / 10. ** 5) / h * (1 + z) * integral

def distance_modulus(z, h, omgM, omgde):
    return 5. * np.log10(luminosity_distance(z, h, omgM, omgde)) + 25.

def chisq(h, omgM, omgde):
    m_model = np.array([distance_modulus(z, h, omgM, omgde) for z in z_obs])
    chisq_vec = np.power(((m_model - m_obs) / dm_obs), 2)
    return chisq_vec.sum()


h_fid = 0.7
omgM_fid = 0.3
omgde_fid = 0.7

chisq_h = np.vectorize(lambda h: chisq(h, omgM_fid, omgde_fid))
result_h = minimize(chisq_h, h_fid, bounds=[(0.5, 1.)])
h_best, = result_h.x
print("Convergiu?: ", result_h.success)
print("chisq / dof = ", result_h.fun[0] / (len(z_obs) - 1))
print("h = ", h_best)

chisq_omgM = np.vectorize(lambda omgM: chisq(h_fid, omgM, omgde_fid))
result_omgM = minimize(chisq_omgM, omgM_fid, bounds=[(0.01, 1)])
omgM_best, = result_omgM.x
print("Convergiu?: ", result_omgM.success)
print("chisq / dof = ", result_omgM.fun[0] / (len(z_obs) - 1))
print("omegaM = ",  omgM_best)



chisq_omgde = np.vectorize(lambda omgde: chisq(h_fid, omgM_fid, omgde))
result_omgde = minimize(chisq_omgde, omgde_fid, bounds=[(0.01, 1)])
omgde_best, = result_omgde.x
print("Convergiu?: ", result_omgde.success)
print("chisq / dof = ", result_omgde.fun[0] / (len(z_obs) - 1))
print("omegak = ",  omgde_best)


chisq_joint = lambda x: chisq(x[0], x[1], x[2])
result_joint = minimize(chisq_joint, [h_fid, omgM_fid, omgde_fid], bounds=((0.5, 1.), (0.001, 1),(0.001, 1)),)
h_joint, omgM_joint , omgde_joint= result_joint.x
print("Convergiu?: ", result_joint.success)
print("chisq / dof = ", result_joint.fun / (len(z_obs) - 2))
print("h = ", h_joint)
print("omegaM = ",  omgM_joint)
print("omgde = ",  omgde_joint)


def lnprior(pars):
    h, omgM, omgde = pars
    if 0.0 < h and 0.0 < omgM < 1 and 0 < omgde < 1 :
        return 0.0
    return -np.inf

def lnlike(pars):
    h, omgM, omgde = pars
    return -0.5 * chisq(h, omgM, omgde)

def lnprob(pars):
    lp = lnprior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars)

ndim, nwalkers, nsteps = 3, 50, 1000
pos = [result_joint.x + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4)
sampler.run_mcmc(pos, nsteps)

h_chain = sampler.chain[:,:,0]
omgM_chain = sampler.chain[:,:,1]
omgde_chain = sampler.chain[:,:,2]

h_chain_mean = np.mean(h_chain, axis=0)
h_chain_std = np.std(h_chain, axis=0) / np.sqrt(nwalkers)
h_chain_flat = np.reshape(h_chain, (nwalkers*nsteps,))
omgM_chain_flat = np.reshape(omgM_chain, (nwalkers*nsteps,))
omgde_chain_flat = np.reshape(omgde_chain, (nwalkers*nsteps,))

labels = [r"$h$", r"$\Omega_{m}$", r"$\Omega_{de}$"]
samples = np.c_[h_chain_flat, omgM_chain_flat,omgde_chain_flat].T

h_chain_mean = np.mean(h_chain, axis=0)
h_chain_err = np.std(h_chain, axis=0) / np.sqrt(nwalkers)
omgM_chain_mean = np.mean(omgM_chain, axis=0)
omgM_chain_err = np.std(omgM_chain, axis=0) / np.sqrt(nwalkers)
omgde_chain_mean = np.mean(omgde_chain, axis=0)
omgde_chain_err = np.std(omgde_chain, axis=0) / np.sqrt(nwalkers)

burn = 5000
samples_burned = np.c_[[par[burn:] for par in samples]]
fig = corner.corner(samples_burned.T, labels=labels,
                    quantiles=[0.05, 0.5, 0.95], 
                    levels=(1-np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-9./2)), #1sigma, 2sigma and 3sigma contours
                    smooth1d=None, plot_contours=True,
                    no_fill_contours=False, plot_density=True,)