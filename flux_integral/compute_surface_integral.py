import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from guiding_center_eqns_cartesian import *
from trace_particles import *
import sys
sys.path.append("../stella")
from bfield import load_field,compute_rz_bounds,compute_plasma_volume,make_surface_classifier
sys.path.append("../utils")
from constants import *
import vtkClass
import matplotlib.pyplot as plt

# TODO: work out the math for this integral before continuing
# on the numerics. Are we sure that the integral reduces to using
# the time inside the plasma.
# We can also compute the integral the other way by integrating 
# the length of characteristics.

# tracing parameters
tmax = 1e-6
dt = 1e-9
n_skip = np.inf
method = 'midpoint' # euler or midpoint
include_drifts = True

# surface discretization
ntheta=nphi=64

# TODO: Warning: spacing is non-uniform since interval
# TODO: size varies. Ask David about this.
# vpar discretization
n_vpar = 128
assert n_vpar % 2 == 0, "must use even number of points"


# load the plasma 
vmec_input="../stella/input.new_QA_scaling"
surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="half period", nphi=nphi, ntheta=ntheta)

# multiplier for number of field periods
nfp = surf.nfp
period_mult = 2 # because we are using a half period
# multiply integrals by this to get total over entire stellarator
symmetry_mult = nfp*period_mult 

# load the plasma volume, classifier
plasma_vol = compute_plasma_volume(vmec_input,ntheta=ntheta,nphi=nphi)

# TODO: use a classifier for a slightly enlarged surface
# TODO: this will improve the accuracy in the ODE tracing.
classifier = make_surface_classifier(vmec_input=vmec_input, rng="full torus",ntheta=ntheta,nphi=nphi)

# compute the initial state volume
vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)
vpar_vol = vpar_ub - vpar_lb
prob_const = 1/plasma_vol/vpar_vol
print(plasma_vol)
print(vpar_vol)
print(prob_const)

# load the bfield
bs_path="../stella/bs.new_QA_scaling"
bs = load_field(vmec_input,bs_path,ntheta=ntheta,nphi=nphi)
def bfield(xyz):
  # add zero to shut simsopt up
  bs.set_points(xyz + np.zeros(np.shape(xyz)))
  return bs.B()
def gradAbsB(xyz):
  # add zero to shut simsopt up
  bs.set_points(xyz + np.zeros(np.shape(xyz)))
  return bs.GradAbsB()

dim_xyz = 3

# build a guiding center object
GC = GuidingCenter(bfield,gradAbsB,include_drifts=include_drifts)

# discretize the plasma boundary
xyz = surf.gamma() # shape (nphi, ntheta,3)
xyz = xyz.reshape((-1,dim_xyz)) # reshape to (nphi*ntheta,3)

# get the quadrature points
thetas = surf.quadpoints_theta
phis = surf.quadpoints_phi
dtheta = thetas[1] - thetas[0]
dphi = phis[1] - phis[0]

# get the surface normals (should be outward facing)
normals = surf.normal() # shape (nphi, ntheta,3)
normals = normals.reshape((-1,dim_xyz)) # reshape to (nphi*ntheta,3)
area_elements = np.linalg.norm(normals,axis=1)
normals = (normals.T/area_elements).T # unit normals

# sanity check: compute the area from the area_elements
print("")
print('surface area check')
print(np.sum(area_elements*dtheta*dphi)*symmetry_mult,surf.area())


def compute_vpar_roots(xyz,normals,flag_plot=False):
  # compute B on grid
  Bb = bfield(xyz) # shape (N,3)
  # compute Bg on grid
  Bg = gradAbsB(xyz) # shape (N,3)
  # field values
  B = np.linalg.norm(Bb,axis=1) # shape (N,)
  b = (Bb.T/B).T # shape (N,3)
  c = ALPHA_PARTICLE_MASS /ALPHA_PARTICLE_CHARGE/B/B/B # shape (N,)
  # compute first coeff in quadratic
  coeff1 = c*np.sum(np.cross(Bb,Bg) * normals,axis=1)/ 2
  # b.n
  coeff2 = np.sum(b*normals,axis=1)
  # coeff1* v**2
  coeff3 = coeff1*FUSION_ALPHA_SPEED_SQUARED
  # get the discriminant
  discriminant = coeff2**2 - 4*coeff1*coeff3

  # get the sign of the convexity coefficient
  sgn_a = np.sign(coeff1)
  # get the number of roots
  n_roots = np.sign(discriminant) + 1
  
  # get the roots
  root1 = (-coeff2 - np.sqrt(discriminant))/2/coeff1
  root2 = (-coeff2 + np.sqrt(discriminant))/2/coeff1
  roots = np.vstack((root1,root2)).T
  
  # dont keep points that are concave down and have 1 or less intercepts
  idx_drop = (coeff1 < 0) & (n_roots <= 1)

  # drop some points
  xyz = xyz[(~idx_drop)]
  normals = normals[(~idx_drop)]
  roots = roots[(~idx_drop)]
  sgn_a = sgn_a[(~idx_drop)]
  n_roots = n_roots[(~idx_drop)]

  if flag_plot:
    #ls = np.logspace(0,np.log(vpar_ub),100)
    #xxx = np.hstack((-ls,ls))
    quadratic = lambda xx: coeff1*xx**2 + coeff2*xx + coeff3
    #plt.plot(xxx,quadratic(xxx))
    #plt.yscale('symlog')
    #plt.show()
    print(roots)
    print(vpar_lb,vpar_ub)
    print(quadratic(np.linspace(root2[0],vpar_ub,100)))
    print(quadratic(np.linspace(vpar_lb,root1[0],100)))
  
  return xyz,normals,roots,sgn_a,n_roots

# compute the roots of the vpar quadratic
xyz,normals,roots,sgn_a,n_roots = compute_vpar_roots(xyz,normals)



"""
Loop through the boundary points xyz.
Discretize the vpar intervals.
Backtrace the trajectories.
"""

# accumulator for the loss fraction
loss_fraction = 0.0

for ii,xx in enumerate(xyz):

  # TODO: use a chebyshev grid or other grid that does not
  # include the endpoint, b/c the endpoints are roots of the 
  # quadratic which dot not have positive flux.

  # determine the vpar interval
  if sgn_a[ii]>0:
    # convex quadratic 
    if n_roots[ii] == 2:
      # intervals are [vpar_lb,roots[ii][0]] and [roots[ii][1],vpar_ub]

      # restrict the roots to the interval
      left_root = roots[ii][0]
      right_root = roots[ii][1]
      if left_root <= vpar_lb and right_root >= vpar_ub:
        # no points to integrate
        continue
      elif left_root <= vpar_lb and right_root <=vpar_lb:
        # no points to integrate
        continue
      elif left_root >= vpar_ub and right_root >=vpar_ub:
        # no points to integrate
        continue
      elif left_root <= vpar_lb and right_root >=vpar_lb:
        # single interval [right_root,vpar_ub]
        vpar_disc = np.linspace(right_root,vpar_ub,n_vpar)
      elif left_root <= vpar_ub and right_root >=vpar_ub:
        # single interval [vpar_lb,left_root]
        vpar_disc = np.linspace(vpar_lb,left_root,n_vpar)
      else:
        # two intervals 
        T1 = np.linspace(vpar_lb,left_root,int(n_vpar/2))
        T2 = np.linspace(right_root,vpar_ub,int(n_vpar/2))
        vpar_disc = np.hstack((T1,T2))

    # TODO: consider the case with 1 root.
    else:
      # single interval [vpar_lb,vpar_ub]
      vpar_disc = np.linspace(vpar_lb,vpar_ub,n_vpar)

  elif sgn_a[ii] == 0:
    # quadratic is linear
    print("")
    print("Warning: Quadratic is Linear")
    raise ValueError

  else:
    # concave quadratic 

    if n_roots[ii] == 2:
      left_root = roots[ii][0]
      right_root = roots[ii][1]
      
      # restrict roots to interval
      if left_root <= vpar_lb and right_root >= vpar_ub:
        # single interval [vpar_lb,vpar_ub]
        vpar_disc = np.linspace(vpar_lb,vpar_ub,n_vpar)
      elif left_root <= vpar_lb and right_root <=vpar_lb:
        # no points to integrate
        continue
      elif left_root >= vpar_ub and right_root >=vpar_ub:
        # no points to integrate
        continue
      elif left_root <= vpar_lb and right_root >=vpar_lb:
        # single interval [vpar_lb,right_root]
        vpar_disc = np.linspace(vpar_lb,right_root,n_vpar)
      elif left_root <= vpar_ub and right_root >=vpar_ub:
        # single interval [left_root,vpar_ub]
        vpar_disc = np.linspace(left_root,vpar_ub,n_vpar)
      else:
        # single interval [left_root,right_root]
        vpar_disc = np.linspace(left_root,right_root, n_vpar)

  # form the set of points X in 4d state space
  yy = np.tile(xx,len(vpar_disc)).reshape((-1,dim_xyz))
  X = np.hstack((yy,vpar_disc.reshape((-1,1)) ))

  # get the guiding center velocity at the points
  v_gc = GC.GC_rhs(X)[:,:-1] # just keep the spatial part

  # v_gc * normal
  v_gcn = v_gc @ normals[ii]

  # safety check that we are integrating outward flux
  if np.any(v_gcn < -1e-8):
    print("")
    print('negative vgcn')
    print(v_gcn[v_gcn<0])
    print(n_roots[ii]) 
    print(sgn_a[ii]) 
    compute_vpar_roots(np.atleast_2d(xx),np.atleast_2d(normals[ii]),flag_plot=True)
    quit()

  # area element
  dA = area_elements[ii]

  # line element
  dvpar = vpar_disc[1] - vpar_disc[0]

  # volume element
  dV = dA*dtheta*dphi*dvpar

  ## trace the particles.
  tau_in_plasma  = trace_particles(X,GC,tmax,dt,classifier=classifier,
              method=method,n_skip=n_skip,direction='backward')

  # TODO: use higher order integration
  # accumulate the loss fraction
  loss_fraction += prob_const*np.sum(tau_in_plasma*v_gcn*dV)*symmetry_mult
  print(loss_fraction)
  
