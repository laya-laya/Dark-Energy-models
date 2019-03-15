#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 16:48:20 2019

@author: layaparkavousi
"""

import corner
import emcee
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as cte
from scipy.integrate import quad
from scipy.optimize import minimize


z_obs, m_obs, dm_obs = np.loadtxt("SCPUnion2.1_mu_vs_z.dat", unpack=True)
"""
# plot redshift vs distance modulus
plt.xlabel(r"$z$")
plt.ylabel(r"$\mu = m - M$")
plt.errorbar(z_obs, m_obs, dm_obs, fmt='r.' )

"""
### functions for friedmann equation and DM and chi

def luminosity_integrand(z, omgM, omgde):
    Ez = np.sqrt((omgde) + omgM * np.power(1 + z, 3))
    
    return 1. / Ez

def luminosity_distance(z, h, omgM, omgde):
    integral, _ = quad(luminosity_integrand, 0, z, epsrel=1e-8, args=(omgM, omgde))
    f = (cte.c / 10**5) / h*(1+z)*integral
    return f
## distance_modulus
    
def DM(z, h, omgM, omgde):
    DM = 5. * np.log10(luminosity_distance(z, h, omgM, omgde)) + 25
    return DM

def chi2(h, omgM, omgde):
    m_model = np.array([DM(z, h, omgM, omgde) for z in z_obs])
    chisq_vec = np.power(((m_model - m_obs) / dm_obs), 2)
    return chisq_vec.sum()

## initials(needed for minimize tool in scipy lib)
h_fid = 0.67
omgM_fid = 0.3
omgDE_fid = 0.69
chi2_h = np.vectorize(lambda h: chi2(h, omgM_fid, omgDE_fid))
result_h = minimize(chi2_h, h_fid, bounds=[(0.01, 1.)])
h_best, = result_h.x
print("h = ", h_best)

chi2_omgM = np.vectorize(lambda omgM: chi2(h_fid, omgM, omgDE_fid))
result_omgM = minimize(chi2_omgM, omgM_fid, bounds=[(0.01, 1.)])
omgM_best, = result_omgM.x
print("omegaM = ",  omgM_best)


chi2_t = lambda x: chi2(x[0], x[1], omgDE_fid)
result_t = minimize(chi2_t, [h_fid, omgM_fid], bounds=((0.001, 1.), (0.001, 1.)),)
h_t, omgM_t = result_t.x

def lnprior(pars):
    h, omgM = pars
    if 0.0 < h and 0.0 < omgM < 1.0:
        return 0.0
    return -np.inf

def lnlike(pars):
    h, omgM = pars
    return -0.5 * chi2(h, omgM, omgDE_fid)

def lnprob(pars):
    lp = lnprior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(pars)

#likelihood values
    # MCMC chain 

ndim, nwalkers, nsteps = 2, 50, 1000
pos = [result_t.x + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, threads=4)
sampler.run_mcmc(pos, nsteps)

h_chain = sampler.chain[:,:,0]
omgM_chain = sampler.chain[:,:,1]

h_chain_mean = np.mean(h_chain, axis=0)
h_chain_std = np.std(h_chain, axis=0) / np.sqrt(nwalkers)

h_chain_flat = np.reshape(h_chain, (nwalkers*nsteps,))
omgM_chain_flat = np.reshape(omgM_chain, (nwalkers*nsteps,))

labels = [r"$h$", r"$\Omega_{m0}$"]
samples = np.c_[h_chain_flat, omgM_chain_flat].T
h_chain_mean = np.mean(h_chain, axis=0)
h_chain_error = np.std(h_chain, axis=0) / np.sqrt(nwalkers)
omgM_chain_mean = np.mean(omgM_chain, axis=0)
omgM_chain_error = np.std(omgM_chain, axis=0) / np.sqrt(nwalkers)

# corner plot
burn = 5000
samples_burned = np.c_[[par[burn:] for par in samples]]
fig = corner.corner(samples_burned.T, labels=labels,
                    quantiles=[0.16, 0.5, 0.84], 
                    levels=(1-np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-9./2)),
                    show_titles=True, title_kwargs={"fontsize": 12},
                    smooth1d=None, plot_contours=True,
                    no_fill_contours=False, plot_density=True,)

