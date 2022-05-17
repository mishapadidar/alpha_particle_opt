from stella import *    
import numpy as np
import time 
from simsopt.field.magneticfieldclasses import ToroidalField

"""
Track a blob of probability mass through a Toroidal B field
WITHOUT drift computations. 

The toroidal field is not divergence free so we do not include
drifts.

The expected behavior is that probability will follow field 
lines.
"""

# load constants
PROTON_MASS = 1.67262192369e-27  # kg
NEUTRON_MASS = 1.67492749804e-27  # kg
ONE_EV = 1.602176634e-19  # J
ALPHA_PARTICLE_MASS = 2*PROTON_MASS + 2*NEUTRON_MASS
FUSION_ALPHA_PARTICLE_ENERGY = 3.52e6 *ONE_EV # Ekin
FUSION_ALPHA_SPEED_SQUARED = 2*FUSION_ALPHA_PARTICLE_ENERGY/ALPHA_PARTICLE_MASS

# define the bfield.
R0 = 10.0 # meters
B0 = 5.0 # Tesla
bs = ToroidalField(R0,B0)

# set discretization sizes
n_r = 128
n_phi = 4
n_z = 128
n_vpar = 128
dt = 1e-8
tmax = 1e-5
integration_method='midpoint'
mesh_type = "chebyshev"
include_drifts = False
interp_type = "cubic"
# set the bounds, these define the plasma
nfp = 128 # large b/c axisymmetry
R_minor = 2.0
rmax = R0 + R_minor
rmin = R0 - R_minor
zmax = R_minor 
zmin = 0.0
# plot the field
bs.to_vtk('magnetic_field_ex2',nphi=32,rmin=rmin,rmax=rmax,zmin=zmin,zmax=zmax)
# set the initial density bounds for vpar
vpar0_lb = -1.0*np.sqrt(FUSION_ALPHA_SPEED_SQUARED)
vpar0_ub = 1.0*np.sqrt(FUSION_ALPHA_SPEED_SQUARED)
# set the initial density bounds
r0_lb = R0 - 0.25*R_minor
r0_ub = R0 + 0.25*R_minor
phi0_lb = np.pi/nfp - 2*np.pi/nfp/n_phi
phi0_ub = np.pi/nfp + 2*np.pi/nfp/n_phi
z0_lb = (zmin+zmax)/2 - 4*(zmax-zmin)/n_z
z0_ub = (zmin+zmax)/2 + 4*(zmax-zmin)/n_z
vparmin = vpar0_lb
vparmax = vpar0_ub
# compute the initial state volume
vpar_vol = vpar0_ub - vpar0_lb
spatial_vol = 0.5*(phi0_ub-phi0_lb)*(z0_ub-z0_lb)*(r0_ub**2 - r0_lb**2)
prob0 = 1/spatial_vol/vpar_vol


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
  idx_feas = X[:,0] >= r0_lb
  idx_feas = np.logical_and(idx_feas, X[:,0] <= r0_ub)
  idx_feas = np.logical_and(idx_feas, X[:,-1] >= vpar0_lb)
  idx_feas = np.logical_and(idx_feas, X[:,-1] <= vpar0_ub)
  idx_feas = np.logical_and(idx_feas, X[:,1] >= phi0_lb)
  idx_feas = np.logical_and(idx_feas, X[:,1] <= phi0_ub)
  idx_feas = np.logical_and(idx_feas, X[:,2] >= z0_lb)
  idx_feas = np.logical_and(idx_feas, X[:,2] <= z0_ub)
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
    dt,tmax,integration_method,
    mesh_type=mesh_type,
    include_drifts=include_drifts,
    interp_type=interp_type)

solver.solve()
