import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM

G = const.G.to(u.Mpc**3/u.Gyr**2/(1.e10*u.solMass)).value
sig0 = (1.e-3 *u.g/u.cm**2).to(1.e10*u.solMass/u.Mpc**2).value
pram0 = (1.e-11 * u.erg/u.cm**3).to(1.e10*u.solMass/u.Mpc/u.Gyr**2).value

nH_ne = 0.852

H0 = 70.
cosmo = FlatLambdaCDM(H0=H0, Om0=0.3)

DA = lambda z: cosmo.angular_diameter_distance(z).to(u.cm).value
# Return proper distance [Mpc] for a given redshift and sep. in arcsec
properDist = lambda arcsec, z: arcsec*cosmo.kpc_proper_arcmin(z)/60./1000.

def EI(norm, z):
  return 10 ** (np.log10(norm) + np.log10(4*np.pi) + 2*np.log10( DA(z)*(1 + z) )  + 14)

def R_delta(delta, z, kT, beta_T=1.05):
  delta_z = (delta * cosmo.Om0) / (18*np.pi**2 * cosmo.Om(z)) 
  return 3.8 * np.sqrt(beta_T* (kT/10.) / (delta_z*(1+z)**3) ) / (H0/50.)

def calcLum(flux, z):
  return 4*np.pi*10**(2*np.log10(cosmo.luminosity_distance(z).to(u.cm).value) + np.log10(flux) )

