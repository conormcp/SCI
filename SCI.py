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
from multiprocessing import Pool
'''
internal units set to 1=:
  L - 1 Mpc
  T - 1 Gyr
  M - 1e10 M_solar
'''
def function(simObj):
  simObj.integrate()

class Simulation:
  def __init__(self, res=60, b=0.1, v0=200, vt0=0, r0=2.5, tmax=5., tsteps=1000, r_plt_range=((0,np.pi),(0,1)), d_plt_range=None, bins=60, alpha=0.1, save=False):
    print 'Calculating orbit for b = ', b
    self.b = b
    self.v0 = v0
    self.vt0 = vt0
    self.res = res
    self.bins = bins
    self.save = save
    self.alpha = alpha
    self.clus = Cluster()
    self.orb = Orbit(self.clus, b, v0=v0, vt0=vt0, r0=r0, tmax=tmax, tsteps=tsteps)
    self.gal = Galaxy(orb=self.orb)
    self.rho = self.orb.local_density[self.gal.event] * (1.e10*u.solMass/u.Mpc**3).to(1.e-4*(1.67372e-24*u.g)/u.cm**3).value
    self.ri_hist = np.zeros((bins,bins))
    self.di_hist = np.zeros((bins,bins))
    self.rd_hist = np.zeros((bins,bins))
    self.r_plt_range = r_plt_range
    if d_plt_range == None:
      d_plt_range = ((0,np.pi),(np.log10(self.rho.min()),np.log10(self.rho.max())))
    self.d_plt_range = d_plt_range

  def integrate(self):
    if self.save: print 'Integrating b = ', self.b
    for i in np.arccos(2*np.linspace(0,1.,self.res)-1):  # Integrate over elevation angle
      for j in np.linspace(0,2*np.pi,self.res):          # Integrate over azimuthal angle
        for k in np.linspace(0,2*np.pi,self.res):        # Integrate over cylinder
          rorb = self.orb.projection(phi=i, theta=j, inclination=k)
          # Weighting terms for probability
          weight = self.gal.dm_t * (self.alpha + (1-self.alpha) * np.sin(rorb.visibility()[self.gal.event]) )
          self.ri_hist += np.histogram2d(rorb.incidence()[self.gal.event], 
                                         rorb.r[self.gal.event], range=self.r_plt_range,
                                         bins=self.bins, weights=weight)[0]
          self.di_hist += np.histogram2d(rorb.incidence()[self.gal.event], np.log10(self.rho),
                                    range=self.d_plt_range, bins=self.bins, weights=weight)[0]
          self.rd_hist += np.histogram2d(self.rho, rorb.r[self.gal.event], 
                                    range=(self.d_plt_range[1],self.r_plt_range[1]), bins=self.bins, weights=weight)[0]
    out = np.array([self.ri_hist,self.di_hist,self.rd_hist])
    fits.writeto('steps/model_step_bins%i_res%i_v%i_vt%i_nodm_b%2.3f.fits' % (self.bins, self.res, self.v0, self.vt0, self.b), out, clobber=True)


def sim_run(bins=60, steps=50, res=50, v0=200., vt0=0, bmax=1, bmin=0., tmax=5, tsteps=1000, save=False, alpha=0.1):
  dens = np.zeros((steps,bins,bins))
  rad = np.zeros((steps,bins,bins))
  rad_dens = np.zeros((steps,bins,bins))
  # Sample on even steps in r^2 to get even areal coverage
  bc = np.sqrt(np.linspace(bmin**2,bmax**2, steps))
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

def run_mp(bins=60, steps=50, res=50, v0=200., vt0=0, bmax=1, bmin=0., tmax=5, tsteps=1000, save=False, alpha=0.1, nproc=4):
  # Sample on even steps in r^2 to get even areal coverage
  bc = np.sqrt(np.linspace(bmin**2,bmax**2, steps))
  # Run first simulation outside of multiprocessing loop 
  # to grab density limits from orbit with the smallest impact parameter b
  sim = Simulation(res=res, b=bc[0], v0=v0, vt0=vt0, tmax=tmax, tsteps=tsteps, bins=bins, alpha=alpha, d_plt_range=None, save=save)
  sim.integrate()
  d_range = sim.d_plt_range
  objs = []
  for i in np.arange(steps-1)+1:
    objs.append(Simulation(res=res, b=bc[i], v0=v0, vt0=vt0, tmax=tmax, tsteps=tsteps, bins=bins, alpha=alpha, d_plt_range=d_range, save=save) )
  # Initialize Pool of workers
  p = Pool(processes=nproc)
  p.map_async(function, objs)
  p.close()
  p.join()

def combine_mp(bins=60, steps=50, res=50, v0=200., vt0=0, bmax=1, bmin=0., tmax=5, tsteps=1000, save=False, alpha=0.1):
  # Sample on even steps in r^2 to get even areal coverage
  bc = np.sqrt(np.linspace(bmin,bmax, steps))
  # Initialize Output arrays
  dens = np.zeros((steps,bins,bins))
  rad = np.zeros((steps,bins,bins))
  rad_dens = np.zeros((steps,bins,bins))
  # Combine cylindrical shell results from above into single fits file with steps in radius
  for i in range(steps):
    print i
    sim = fits.getdata('steps/model_step_bins%i_res%i_v%i_vt%i_nodm_b%2.3f.fits' % (bins, res, v0, vt0, bc[i]) )
    dens[i] = sim[0]
    rad[i] = sim[1]
    rad_dens[i] = sim[2]
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


