#! /usr/bin/env python
'''
  Simulating Cluster Infall
  writen by Conor McPartland
  May 21 2014
'''
import numpy as np
from constants import *
from cluster import *
from galaxy import *
from orbit import *
from astropy.io import fits

'''
internal units set to 1=:
  L - 1 Mpc
  T - 1 Gyr
  M - 1e10 M_solar
'''
class Simulation:
  def __init__(self, res=60, b=0.1, v0=200, vt0=0, r0=2.5, tmax=8., tsteps=1000, r_plt_range=((0,np.pi),(0,1)), d_plt_range=None, bins=60, alpha=0.1):
    self.clus = Cluster()
    orb = Orbit(self.clus, b, v0=v0, vt0=vt0, r0=r0, tmax=tmax, tsteps=tsteps)
    gal = Galaxy(orb=orb)
    rho = orb.local_density[gal.event] * (1.e10*u.solMass/u.Mpc**3).to(1.e-4*(1.67372e-24*u.g)/u.cm**3).value

    self.ri_hist = np.zeros((bins,bins))
    self.di_hist = np.zeros((bins,bins))
    self.rd_hist = np.zeros((bins,bins))
    if d_plt_range == None:
      d_plt_range = ((0,np.pi),(np.log10(rho.min()),np.log10(rho.max())))
    self.d_plt_range = d_plt_range

    for i in np.arccos(2*np.linspace(0,1.,res)-1):
      for j in np.linspace(0,2*np.pi,res):
        for k in np.linspace(0,2*np.pi,res):
          rorb = orb.projection(phi=i, inclination=k, theta=j)
          # Weighting terms for probability
          weight = gal.dm_t * (alpha + (1-alpha) * np.sin(rorb.visibility()[gal.event]) )
          self.ri_hist += np.histogram2d(rorb.incidence()[gal.event], 
                                         rorb.r[gal.event], range=r_plt_range,
                                         bins=bins, weights=weight)[0]
          self.di_hist += np.histogram2d(rorb.incidence()[gal.event], np.log10(rho),
                                    range=d_plt_range, bins=bins, weights=weight)[0]
          self.rd_hist += np.histogram2d(rho, rorb.r[gal.event], 
                                    range=(d_plt_range[1],r_plt_range[1]), bins=bins, weights=weight)[0]

def sim_run(bins=60, steps=50, res=50, v0=200., vt0=0, bmax=1, bmin=0., tmax=5, tsteps=1000, save=False, alpha=0.1):
  dens = np.zeros((steps,bins,bins))
  rad = np.zeros((steps,bins,bins))
  rad_dens = np.zeros((steps,bins,bins))
  # Sample on even steps in r^2 to get even areal coverage
  bc = np.sqrt(np.linspace(bmin,bmax, steps))
  for i in range(steps):
    print i
    if i == 0: d_range = None
    else: d_range = sim.d_plt_range
    sim = Simulation(res=res, b=bc[i], v0=v0, vt0=vt0, tmax=tmax, tsteps=tsteps, bins=bins, alpha=alpha, d_plt_range=d_range)
    dens[i] = sim.di_hist
    rad[i] = sim.ri_hist
    rad_dens[i] = sim.rd_hist
    if i > 0:
      dens[i] += dens[i-1]
      rad[i] += rad[i-1]
      rad_dens[i] += rad_dens[i-1]

  for i in range(steps):
    rad[i] /= rad[i][np.isfinite(rad[i])].sum()
    dens[i] /= dens[i][np.isfinite(dens[i])].sum()
    rad_dens[i] /= rad_dens[i][np.isfinite(rad_dens[i])].sum()
  if save:
    fits.writeto('model_bins%i_steps%i_res%i_v%i_vt%i_nodm_bmax%2.3f.fits' % (bins, steps, res, v0, vt0, bmax), np.array([rad,dens,rad_dens]), clobber=True)
  return rad, dens, rad_dens


