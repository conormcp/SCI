import numpy as np
from scipy.special import hyp2f1
from orbit import *
from constants import *

class BetaModel:
  def __init__(self, rs=0.2, beta=1., rho0=5.e5):
    '''
    Class implementation of the isothermal King model or Beta model
    ref. Cavaliere & Fusco-Femiano 1976
    intialized with inputs:
        r0 - scale radius [Mpc]
      beta - power law index
      rho0 - central density
    '''
    self.rs = rs
    self.beta = beta
    self.rho0 = rho0

  def scale(self, r):
    return r/float(self.rs)

  def density(self, r, number_dens=False):
    x = self.scale(r)
    if number_dens:
      rho0 = 10**(np.log10(self.rho0) - np.log10( (1.67372e-24*u.g).to(1.e10*u.solMass).value ) )
    else: rho0 = self.rho0
    return rho0 * ( 1. + x**2 )**-(3.*self.beta/2.)

  def mod_density(self, r, rsteep=0.57, rmax=1.2, beta2=0.6, number_dens=False):
    x = self.scale(r)
    if number_dens:
      rho0 = 10**(np.log10(self.rho0) - np.log10( (1.67372e-24*u.g).to(1.e10*u.solMass).value ) )
    else: rho0 = self.rho0
    dens = rho0 * ( 1. + x**2 )**-(3.*self.beta/2.)
    dens[r > rsteep] = rho0 * ( 1. + x[r > rsteep]**2 )**-(3.*beta2/2.)
    dens[r > rmax] = 0
    return dens

  def surface_density(self, r, S0=None):
    x = self.scale(r)
    if S0 == None: S0 = 1    
    return 10**(np.log10(S0) + (0.5 - 3.*self.beta)*np.log10( 1. + x**2 ) )

  def mass(self, r, rho0=None):
    x = self.scale(r)
    if rho0 == None: rho0 = self.rho0
    fact = 4/3.*np.pi * rho0 * r**3
    F = hyp2f1(3./2., 3*self.beta/2., 5./2., -x**2)
    return fact*F

  def EI(self, r):
    x = self.scale(r)
    fact = 4/3.*np.pi * self.rho0**2 * r**3
    F = hyp2f1(3./2., 3*self.beta, 5./2., -x**2)
    return fact*F*nH_ne

  def phi(self, r):
    x = self.scale(r)
    fact = 4*np.pi * G * self.rho0 * self.rs**2
    term = np.log(x + np.sqrt(1. + x**2) )
    return -fact * term / x

  def f(self, r):
    return -G * self.mass(r) * r**-2

class BetaEllip:
  def __init__(self, rs=0.2, beta=2./3., ey=0., ez=0., rho0=None):
    if rho0 == None: self.beta = BetaModel(rs=rs, beta=beta, rho0=1.)
    else: self.beta = BetaModel(rs=rs, beta=beta, rho0=rho0)
    self.ey = float(ey)
    self.ez = float(ez)

  def radius(self, x, y, z=None, mode='ell'):
    if mode == 'ell':
      if z == None: return np.sqrt( x**2 + (y/(1-self.ey))**2 )
      else: return np.sqrt( x**2 + (y/float(1-self.ey))**2 + (z/float(1-self.ez))**2)
    if mode == 'cir':
      if z == None: return np.sqrt( x**2 + y**2 )
      else: return np.sqrt( x**2 + y**2 + z**2)
    else:
      print "Aperture Shape Not Recognized!"

  def integrate(self, rmax, steps=100, mod=False):
    if mod: density_f = self.beta.mod_density
    else: density_f = self.beta.density
    xmax = rmax
    ymax = rmax * (1 - self.ey)
    zmax = rmax * (1 - self.ez)
    dx = xmax / float(steps)
    dy = ymax / float(steps)
    dz = zmax / float(steps)
    total_n2 = 0
    total_m = 0
    x, y = np.indices((steps, steps), dtype=float)
    x = (x + 0.5)*dx 
    y = (y + 0.5)*dy
    z = np.linspace(0, zmax, steps+1)
    aper = self.radius(x, y, mode='ell')
    for i in np.arange(steps)+1:
      dens = density_f(self.radius(x, y, (z[i-1]+z[i])/2, mode='ell') )
      total_n2 += (dens[aper<=rmax]**2 ).sum() *dx*dy*dz
      total_m += (dens[aper<=rmax] ).sum() *dx*dy*dz
    self.EI = 8*total_n2*nH_ne
    self.MI = 8*total_m
    return self.MI, self.EI

  def get_ne0(self, EI_spec):
    '''
    Calculate central electron density given the EI from the spectrum
    '''
    self.ne0 = 10**(0.5*( np.log10((EI_spec*u.cm**-3).to(u.Mpc**-3)) - np.log10(self.EI) ) )
    self.nH0 = self.ne0 * nH_ne
    self.rho0 = 1.347 * 10**(np.log10(self.nH0) + np.log10((1.67372e-24*u.g).to(1.e10*u.solMass).value) )
    return self.ne0, self.nH0, self.rho0

  def getMass(self, rmax, fgas=0.082, steps=200, mod=False):
    if mod: density_f = self.beta.mod_density
    else: density_f = self.beta.density
    xmax = rmax
    ymax = rmax * (1 - self.ey)
    zmax = rmax * (1 - self.ez)
    dx = xmax / float(steps)
    dy = ymax / float(steps)
    dz = zmax / float(steps)
    total_m = 0.
    x, y = np.indices((steps, steps), dtype=float)
    x = (x + 0.5)*dx 
    y = (y + 0.5)*dy
    z = np.linspace(0, zmax, steps+1)
    for i in np.arange(steps)+1:
      dens = density_f(self.radius(x, y, (z[i-1]+z[i])/2., mode='ell') )
      aper = self.radius(x, y, (z[i-1]+z[i])/2., mode='cir')
      total_m += (dens[aper<=rmax] ).sum() *dx*dy*dz

    self.M_gas = 8*total_m * self.rho0
    self.M_grav = self.M_gas / float(fgas)
    return self.M_gas, self.M_grav

  def extrapolate(self, rmax, rmes, mode='ell', steps=1000):
    xmax = rmax
    ymax = rmax 
    dx = xmax / float(steps)
    dy = ymax / float(steps)
    x, y = np.indices((steps, steps), dtype=float)
    x = (x + 0.5)*dx 
    y = (y + 0.5)*dy
    r_cir = self.radius(x, y, mode='cir')
    r_ell = self.radius(x, y, mode='ell')
    if mode == 'ell':
      ratio = self.beta.surface_density(r_ell[r_cir<=rmax]).sum()/self.beta.surface_density(r_ell[r_ell<=rmes]).sum()
    if mode == 'cir':
      ratio = self.beta.surface_density(r_ell[r_cir<=rmax]).sum()/self.beta.surface_density(r_ell[r_cir<=rmes]).sum()

    return ratio

class Cluster:
  # Gas frac comes from Mantz et al. 2014 MNRAS 440
  def __init__(self, rho0=7.64e4, beta=0.59, gas_frac=0.074, rs=0.18):
    self.gas = BetaModel(rs=rs, beta=beta, rho0=rho0*gas_frac)
    self.dm = BetaModel(rs=rs, beta=beta, rho0=rho0)
    self.rho0 = rho0 *(1.e10*u.solMass/u.Mpc**3).to(1.e-4*(1.67372e-24*u.g)/u.cm**3).value
    self.rho0_gas = rho0*gas_frac *(1.e10*u.solMass/u.Mpc**3).to(1.e-4*(1.67372e-24*u.g)/u.cm**3).value
    self.gas_frac = gas_frac
    self.beta = beta
    self.rs = rs

  def stripping_radius(self, v0=200, r0=2.5):
    v0 = v0 * (u.km/u.s).to(u.Mpc/u.Gyr)
    orb = Orbit(self, b=0, v0=v0, r0=r0)
    gal = Galaxy()
    gal.mkOrbit(orb)
    radius = orb.r[gal.event][0]
    density = orb.local_density[gal.event][0] * (1.e10*u.solMass/u.Mpc**3).to(1.e-4*(1.67372e-24*u.g)/u.cm**3).value
    return radius, density
    

