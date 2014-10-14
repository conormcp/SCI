from constants import *
import numpy as np
from scipy.integrate import simps, quad, cumtrapz

class Galaxy:
  def __init__(self, orb=None, radius=3e-2, Mstar=10., astar=4.e-3, bstar=2.5e-4,
               Mbulge=1., rbulge=4.e-4, r200=0.245, c=10.,
               gas_frac=0.1, agas=7.e-3):
    self.radius = radius
    self.Mstar = Mstar
    self.Mgas = Mstar*gas_frac
    self.Mgas0 = self.Mgas
    self.agas = agas
    self.astar = astar
    self.bstar = bstar
    self.Mbulge = Mbulge
    self.rbulge = rbulge
    self.r0 = r200/c
    delta_c = (200./3) * c**3 /(np.log(1+c)-c/float(1+c))
    self.rho0 = delta_c * cosmo.critical_density(0.3).to(1e10*u.solMass/u.Mpc**3).value
    self.prof()
    if orb != None: self.mkOrbit(orb)

  def gas_SD(self, r):
    const = self.Mgas/(8*self.agas**2)
    sech = 1./np.cosh(r/self.agas)
    return const*sech

  def prof(self):
    r = np.linspace(0,self.radius, 10001)
    mr = self.Mgas - cumtrapz(2*np.pi*r*self.gas_SD(r), x=r)
    rc = (r[:-1]+r[1:])/2.
    self.radius = np.interp(0, -mr, rc)
    r = np.linspace(0,self.radius, 10001)
    mr = self.Mgas - cumtrapz(2*np.pi*r*self.gas_SD(r), x=r)
    rc = (r[:-1]+r[1:])/2.
    self.rc, self.mr = rc, mr

  def NFW_potential(self, r, z):
    rsph = np.sqrt(r**2 + z**2)
    const = -4*np.pi*G*self.rho0*self.r0**2
    ln = np.log(1 + rsph/self.r0)/(rsph/self.r0)
    return const*ln    

  def PK_potential(self,r,z):  
    const = -G*self.Mstar
    sqrt = np.sqrt(r**2 + (self.astar + np.sqrt(z**2 + self.bstar**2) )**2 )
    return const/sqrt

  def SB_potential(self, r, z):
    rsph = np.sqrt(r**2 + z**2)
    const = -G*self.Mbulge
    return const/(rsph + self.rbulge)

  def DM_potential(self, r, z):
    rsph = np.sqrt(r**2 + z**2)
    factor = np.pi * G * self.rho0 * self.r0**2
    term1 = np.pi - 2*(1 + self.r0/rsph) * np.arctan2(self.r0,rsph)
    term2 = 2*(1 + self.r0/rsph)*np.log(1 + rsph/self.r0)
    term3 = -(1 + self.r0/rsph) * np.log(1 + (rsph/self.r0)**2)
    return factor*(term1+term2+term3)

  def potential(self,r,z):
    return self.PK_potential(r,z) + self.SB_potential(r,z) #+ self.NFW_potential(r,z)

  def v_esc(self, r):
    phi = self.potential(r,0)
    return np.sqrt(-2*phi)

  def mass_cdm(self, r, z):
    rsph = np.sqrt(r**2 + z**2)
    factor = np.pi * self.rho0 * self.r0**3
    term1 = - 2*np.arctan2(rsph,self.r0)
    term2 = 2*np.log(1 + rsph/self.r0)
    term3 = np.log(1 + (rsph/self.r0)**2)
    return factor*(term1+term2+term3)

  def dphidz(self, r, z):
    # derivative of bulge potential
    rsph = np.sqrt(z**2 + r**2)
    sb = (z/rsph) * G * self.Mbulge/ (rsph + self.rbulge)**2
    # derivative of disk potential
    bz = np.sqrt(self.bstar**2 + z**2)
    top = G * self.Mstar * z * (self.astar + bz)
    bottom = bz * (r**2 + (self.astar + bz)**2 ) **1.5
    kp = top/bottom
    const = 4*np.pi*G * self.rho0 * self.r0 * z/rsph
    term1 = np.log(1 + rsph/self.r0)
    term2 = (rsph/self.r0)/(1 + rsph/self.r0)
    dm = const*(term1+term2)
    return sb + kp# + dm 

  def f_restore(self, r):
    rc = np.logspace(-8,0,101)
    zc = np.logspace(-8,0,101)
    rr, zz = np.meshgrid(rc,zc)
    arr =  self.dphidz(rr,zz)
    f_r = arr.max(axis=0)
    return np.interp(r, rc, f_r)*self.gas_SD(r)

  def R_strip(self, pram):
    r = np.linspace(0,0.5,1001)
    return np.interp(pram, self.f_restore(r)[::-1], r[::-1])

  def tau_strip(self, rstr, rp):
    return self.v_esc(rstr) * self.gas_SD(rstr) / rp

  def m_t(self, deltam):
    """
    Calculates new radius given a change mass
    """
    rc = self.rc
    mr = self.mr - (self.Mgas0 - self.Mgas)
    return np.interp(deltam, mr[::-1], rc[::-1])

  def strip(self, pram, dt):
    """
    Calculate mass loss and change in radius and update attributes
    """
    rstr = self.R_strip(pram)
    # Integrate over anulus in gas surface density
    dm = quad(lambda r: 2*np.pi*r*self.gas_SD(r), rstr,self.radius)[0]
    tau = self.tau_strip(rstr,pram)
    deltam = dm/tau*dt
    self.radius = self.m_t(deltam=deltam)        
    self.Mgas -= deltam
    if self.Mgas < 0: self.Mgas = 0
    
    return deltam

  def mkOrbit(self, orbObj):
    self.orb = orbObj
    dt = orbObj.tmax/float(orbObj.tsteps)
    t = []
    rp_f = []
    fs = []
    rp = []
    mgas_t = []
    dm_t = []
    radius_t = []
    r_strip_t = []
    event = np.zeros(len(orbObj.t), dtype=bool)
    for i in range(len(orbObj.t)):
      pram = orbObj.ram_pressure[i]
      if (self.R_strip(pram) < self.radius) & (self.Mgas > 0):
        event[i] = True
#        fs.append(self.f(self.orb.r[i]))
#        rp_f.append(pram/self.f(self.orb.r[i]))
        rp.append(pram)
        t.append(orbObj.t[i])
        mgas_t.append(self.Mgas)
        radius_t.append(self.radius)
        r_strip_t.append(self.R_strip(pram))
        dm_t.append(self.strip(pram=pram, dt=dt))

    self.t = np.array(t)
#    self.rp_f = np.array(rp_f)
#    self.fs = np.array(fs)
    self.rp = np.array(rp)
    self.mgas_t = np.array(mgas_t)
    self.event = event
    self.dm_t = np.array(dm_t)
    self.r_strip_t = np.array(r_strip_t)
    self.radius_t = np.array(radius_t)


