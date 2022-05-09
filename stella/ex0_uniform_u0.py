from stella import *    
import numpy as np
from bfield import load_field,compute_rz_bounds,compute_plasma_volume,make_surface_classifier

# load constants
PROTON_MASS = 1.67262192369e-27  # kg
NEUTRON_MASS = 1.67492749804e-27  # kg
ONE_EV = 1.602176634e-19  # J
ALPHA_PARTICLE_MASS = 2*PROTON_MASS + 2*NEUTRON_MASS
FUSION_ALPHA_PARTICLE_ENERGY = 3.52e6 *ONE_EV # Ekin
FUSION_ALPHA_SPEED_SQUARED = 2*FUSION_ALPHA_PARTICLE_ENERGY/ALPHA_PARTICLE_MASS

# load the plasma volume, classifier and bfield
ntheta=nphi=128
plasma_vol = compute_plasma_volume(ntheta=ntheta,nphi=nphi)
classifier = make_surface_classifier(ntheta=ntheta,nphi=nphi)
bs = load_field(ntheta=ntheta,nphi=nphi)

# set the bounds
rmin,rmax,zmin,zmax = compute_rz_bounds(ntheta=ntheta,nphi=nphi)
delta_r = rmax - rmin 
# expand the r,z bounds to fully enclose the plasma
rmax = rmax + 0.3*delta_r
rmin = max(rmin - 0.3*delta_r,0.0)
delta_z = zmax - zmin 
zmax = zmax + 0.3*delta_z
zmin = zmin - 0.3*delta_z
vpar0_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
vpar0_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)
vparmin = 8*vpar0_lb
vparmax = 8*vpar0_ub
nfp = 2 # number field periods
# set discretization sizes
n_r = 64
n_phi = 32
n_z = 64
n_vpar = 32
dt = 1e-8
tmax = 1e-5
integration_method='midpoint'
# compute the initial state volume
vpar_vol = vpar0_ub - vpar0_lb
prob0 = 1/plasma_vol/vpar_vol

def cyl_to_cart(r_phi_z):
  """ cylindrical to cartesian coordinates 
  input:
  r_phi_z: (N,3) array of points (r,phi,z)

  return
  xyz: (N,3) array of point (x,y,z)
  """
  r = r_phi_z[:,0]
  phi = r_phi_z[:,1]
  z = r_phi_z[:,2]
  return np.vstack((r*np.cos(phi),r*np.sin(phi),z)).T
def u0(X):
  """ 
  u0(r,phi,z,vpar) 
  input: X, 2d array of points in cylindrical (r,phi,z,vpar)
  """
  xyz = cyl_to_cart(X[:,:-1])
  c= np.array([classifier.evaluate(x.reshape((1,-1)))[0].item() for x in xyz])
  idx_feas = c >= 0
  idx_feas = np.logical_and(idx_feas, X[:,-1] >= vpar0_lb)
  idx_feas = np.logical_and(idx_feas, X[:,-1] <= vpar0_ub)
  prob = np.zeros(len(xyz))
  # det(Jac) = r
  det_jac = X[:,0][idx_feas]
  prob[idx_feas] = prob0*det_jac
  return prob

def bfield(xyz):
  # add zero to shut simsopt up
  bs.set_points(xyz + np.zeros(np.shape(xyz)))
  return bs.B()
def gradAbsB(xyz):
  # add zero to shut simsopt up
  bs.set_points(xyz + np.zeros(np.shape(xyz)))
  return bs.GradAbsB()

solver = STELLA(u0,bfield,gradAbsB,
    rmin,rmax,n_r,n_phi,nfp,
    zmin,zmax,n_z,
    vparmin,vparmax,n_vpar,
    dt,tmax,integration_method)

solver.solve(classifier=classifier)
