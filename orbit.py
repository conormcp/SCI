import numpy as np
import matplotlib.pyplot as plt
from constants import *

class Orbit:
  def __init__(self, clusterObj, b=0.1, v0=200, vt0=0, r0=2.5, tmax=5., tsteps=1000):
    self.cluster = clusterObj
    self.b = b
    self.v0 = v0 * (u.km/u.s).to(u.Mpc/u.Gyr)
    if vt0 != 0: self.vt0 = vt0 * (u.km/u.s).to(u.Mpc/u.Gyr)
    else: self.vt0 = 0
    self.r0 = r0
    self.tmax = tmax    
    self.tsteps = tsteps
    self.i = 0
    self.theta = 0
    self.integrate()    

  def copy(self):
    from copy import copy
    return copy(self)

  def integrate(self):
    # initialize vector attributes to hold position and velocity
    self.t = np.linspace(0, self.tmax, self.tsteps+1)
    self.x = np.zeros(len(self.t))
    self.y = np.zeros(len(self.t))
    self.vx = np.zeros(len(self.t))
    self.vy = np.zeros(len(self.t))
    self.r = np.zeros(len(self.t))
    self.theta = np.zeros(len(self.t))

    # set initial conditions
    self.r[0] = self.r0
    self.y[0] = self.b
    self.x[0] = self.r0 #np.sqrt(self.r0**2 - self.b**2)
    self.vx[0] = -self.v0
    self.vy[0] = self.vt0
    self.theta[0] = np.arctan2(self.y[0],self.x[0])

    for i in range(self.tsteps):
      delt = self.t[i+1] - self.t[i]
      # update position
      self.x[i+1] = self.x[i] + self.vx[i]*delt
      self.y[i+1] = self.y[i] + self.vy[i]*delt
      # calculate position in polar coords
      self.r[i+1] = np.sqrt(self.x[i+1]**2 + self.y[i+1]**2)
      self.theta[i+1] = np.arctan2(self.y[i+1],self.x[i+1])
      #if self.x[i+1] < 0: self.theta[i+1] += np.pi
      # calculate force and update velocities
      f = self.cluster.dm.f(self.r[i+1])
      self.vx[i+1] = self.vx[i] + np.cos(self.theta[i+1])*f*delt
      self.vy[i+1] = self.vy[i] + np.sin(self.theta[i+1])*f*delt

    self.speed = np.sqrt(self.vx**2+self.vy**2)
    self.local_density = self.cluster.gas.density(self.r)
    self.ram_pressure =  self.local_density * self.speed**2

  def p_norm(self):
    rho_fid = (1.e-4*(1.67372e-24*u.g)/u.cm**3).to(1.e10*u.solMass/u.Mpc**3).value
    v_fid = (1.e3*u.km/u.s).to(u.Mpc/u.Gyr).value
    return 10**( np.log10(self.ram_pressure) - np.log10(rho_fid) - 2*np.log10(v_fid) )

  def incidence(self):
    r = np.sqrt(self.x**2 + self.y**2)
    speed = np.sqrt(self.vx**2 + self.vy**2)
    dotprod = (-self.x*self.vx - self.y*self.vy)
    cosang = dotprod/speed/r
    cosang[cosang > 1] = 1
    cosang[cosang < -1] = -1
    angle = np.arccos(cosang)
    return angle

  def projection(self, inclination=0, phi=0., theta=None):
    phi = np.pi/2 - phi
    newObj = self.copy()
    newObj.i = inclination
    newObj.theta = theta
    # Cast 2D orbit into 3D with phi and inclination rotations
    newObj.x = self.x.copy()*np.cos(phi) - self.y.copy() * np.sin(inclination) * np.sin(phi)
    newObj.vx = self.vx.copy()*np.cos(phi) - self.vy.copy() * np.sin(inclination) * np.sin(phi)
    newObj.y = self.y.copy() * np.cos(inclination)
    newObj.vy = self.vy.copy() * np.cos(inclination)
    newObj.z = self.x.copy() * np.sin(phi) + self.y.copy() * np.sin(inclination) * np.cos(phi)
    newObj.vz = self.vx.copy() * np.sin(phi) + self.vy.copy() * np.sin(inclination) * np.cos(phi)
    # Calcualte PROJECTED radius from cluster center
    newObj.r = np.sqrt(newObj.x.copy()**2+newObj.y.copy()**2)  

    if theta != None:    # Rotate in theta
      x = newObj.x * np.cos(theta) - newObj.y*np.sin(theta) 
      y = newObj.x * np.sin(theta) + newObj.y*np.cos(theta)
      vx = newObj.vx * np.cos(theta) - newObj.vy*np.sin(theta) 
      vy = newObj.vx * np.sin(theta) + newObj.vy*np.cos(theta)
      newObj.x = x
      newObj.y = y
      newObj.vx = vx
      newObj.vy = vy
      # Calcualte PROJECTED radius from cluster center
      newObj.r = np.sqrt(newObj.x**2+newObj.y**2)
    
    return newObj

  def visibility(self):
    if self.i > 0:
      vz = np.sqrt(self.speed**2 - self.vx**2 - self.vy**2)
      vz[np.isnan(vz)] = 0 
      v_angle = np.arccos(vz/self.speed)
      v_angle[np.isnan(v_angle)] = np.pi/2.
      v_angle[v_angle > np.pi/2] = np.pi - v_angle[v_angle > np.pi/2]
    else:
      v_angle = np.ones(len(self.vx))*np.pi/2.
    return v_angle

  def show(self, polar=False, plot=True):
    if plot: plt.figure(figsize=(6,6))
    if polar:
      plt.subplot(111, polar=True)
      plt.plot(self.theta,self.r)
      plt.ylim(-self.r0, self.r0)
    else:
      plt.plot(self.x, self.y)
      plt.xlim(-self.r0, self.r0)
      plt.ylim(-self.r0, self.r0)
      plt.plot([0],[0], 'ko')
    if plot: plt.show()

