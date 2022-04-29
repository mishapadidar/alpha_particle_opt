from stella import *    
import numpy as np
import sys
sys.path.append("../field")
from make_LandremanPaul_field import load_LandremanPaul_field, load_LandremanPaul_bounds
from make_LandremanPaul_field import compute_plasma_volume,make_surface_classifier


PROTON_MASS = 1.67262192369e-27  # kg
NEUTRON_MASS = 1.67492749804e-27  # kg
ONE_EV = 1.602176634e-19  # J
ALPHA_PARTICLE_MASS = 2*PROTON_MASS + 2*NEUTRON_MASS
FUSION_ALPHA_PARTICLE_ENERGY = 3.52e6 *ONE_EV # Ekin
FUSION_ALPHA_SPEED_SQUARED = 2*FUSION_ALPHA_PARTICLE_ENERGY/ALPHA_PARTICLE_MASS

vmec_input = "../field/input.LandremanPaul2021_QA"
plasma_vol = compute_plasma_volume(vmec_input)
classifier = make_surface_classifier(vmec_input)

vpar0_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
vpar0_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)
vpar_vol = vpar0_ub - vpar0_lb
prob0 = 1/plasma_vol*vpar_vol

def cyl_to_cart(r,phi,z):
  """ cylindrical to cartesian coordinates """
  return np.array([r*np.cos(phi),r*np.sin(phi),z])

def u0(r,phi,z,vpar):
  xyz = cyl_to_cart(r,phi,z).reshape(1,-1)
  if classifier.evaluate(xyz) >=0:
    if vpar >= vpar0_lb and vpar <= vpar0_ub:
      return prob0
  return 0

# biot-savart field
field_path="../field/bs.LandremanPaul2021_QA"
bs = load_LandremanPaul_field(vmec_input,field_path)

def bfield(r,phi,z):
  xyz = cyl_to_cart(r,phi,z).reshape((1,-1))
  bs.set_points(xyz)
  return bs.B().flatten()
def gradAbsB(r,phi,z):
  xyz = cyl_to_cart(r,phi,z).reshape((1,-1))
  bs.set_points(xyz)
  return bs.GradAbsB().flatten()

rmin,rmax,zmin,zmax = load_LandremanPaul_bounds(vmec_input,field_path)
dr  = (rmax-rmin)/10
nfp = 2
dphi = 2*np.pi/nfp/10
dz  = (zmax-zmin)/10
vparmin = vpar0_lb
vparmax = vpar0_ub
dvpar = (vparmax-vparmin)/10
dt = 1e-8
tmax = 1e-4
integration_method='midpoint'

solver = STELLA(u0,bfield,gradAbsB,
    rmin,rmax,dr,dphi,nfp,
    zmin,zmax,dz,
    vparmin,vparmax,dvpar,
    dt,tmax,integration_method)

solver.solve()
