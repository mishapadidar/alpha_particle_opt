import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surface import signed_distance_from_surface
from guiding_center_eqns_cartesian import *
from trace_particles import *
import sys
sys.path.append("../stella")
from bfield import load_field,compute_rz_bounds,compute_plasma_volume,make_surface_classifier
sys.path.append("../utils")
from constants import *
from grids import *
from concentricSurfaceClassifier import concentricSurfaceClassifier
import vtkClass
import matplotlib.pyplot as plt

# TODO: ask David about setting the interval spacing.
# TODO: ask David about floating point errors in the roots computation.


# tracing parameters
tmax = 1e-6
dt = 1e-8
n_skip = np.inf
method = 'midpoint' # euler or midpoint
include_drifts = True
eps_classifier = 1e-3

# surface discretization
ntheta=nphi=32
# vpar discretization
n_vpar = 32
assert n_vpar % 2 == 0, "must use even number of points"


# load the plasma 
vmec_input="../stella/input.new_QA_scaling"
surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="field period", nphi=nphi, ntheta=ntheta)

# multiplier for number of field periods
nfp = surf.nfp
period_mult = 1 # 2 if we are using a half period
# multiply integrals by this to get total over entire stellarator
symmetry_mult = nfp*period_mult 

# load the plasma volume
plasma_vol = compute_plasma_volume(vmec_input,ntheta=ntheta,nphi=nphi)

# build the surface classifier
classifier = concentricSurfaceClassifier(vmec_input, nphi=512, ntheta=512,eps=eps_classifier)

# compute the initial state volume
vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)
vpar_vol = vpar_ub - vpar_lb
prob_const = 1/plasma_vol/vpar_vol
print('plasma vol:',plasma_vol)
print('vpar vol:',vpar_vol)
print('prob const:',prob_const)

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
print('our surface area',np.sum(area_elements*dtheta*dphi)*symmetry_mult)
print('simsopt surface area:',surf.area())
print("")

def compute_quadratic_coeffs(xyz,normals):
  """
  Compute the coefficients of the outward flux quadratic. 

  xyz: array point points, shape (N,3)
  normals: array of outward facing normals, shape (N,3)
  return: three (N,) arrays of coefficients: coeff1,coeff2,coeff3.
  """
  # compute B on grid
  Bb = bfield(xyz) # shape (N,3)
  # compute Bg on grid
  Bg = gradAbsB(xyz) # shape (N,3)
  # field values
  B = np.linalg.norm(Bb,axis=1) # shape (N,)
  b = np.copy((Bb.T/B).T) # shape (N,3)
  c = np.copy(ALPHA_PARTICLE_MASS /ALPHA_PARTICLE_CHARGE/B/B/B) # shape (N,)
  # compute first coeff in quadratic
  coeff1 = np.copy(c*np.sum(np.cross(Bb,Bg) * normals,axis=1)/ 2)
  # b.n
  coeff2 = np.copy(np.sum(b*normals,axis=1))
  # coeff1* v**2
  coeff3 = np.copy(coeff1*FUSION_ALPHA_SPEED_SQUARED)

  # zero out small coefficients
  coeff1_tol = 1e-14
  coeff2_tol = 1e-14
  idx_no_coeff1 = np.abs(coeff1) <= coeff1_tol
  idx_no_coeff2 = np.abs(coeff2) <= coeff2_tol
  coeff1[idx_no_coeff1] = 0.0
  coeff2[idx_no_coeff2] = 0.0

  return coeff1,coeff2,coeff3

def compute_vpar_roots(coeff1,coeff2,coeff3):
  """
  Compute the roots of the outward flux quadratic.
  
  coeff1,coeff2,coeff3: (N,) arrays of quadratic coefficients

  return: 
    roots: (N,2) array of possible roots. 
          np.nan will be used is less than 2 roots exist
    n_roots: (N,) array containing the number of roots.
  """
  # TODO: stabilize root computation.

  # get the discriminant
  discriminant = np.copy(coeff2**2 - 4*coeff1*coeff3)

  # get the number of roots
  n_roots = np.sign(discriminant) + 1

  # get the roots of the quadratic polynomials
  root1 = (-coeff2 - np.sqrt(discriminant))/2/coeff1
  root2 = (-coeff2 + np.sqrt(discriminant))/2/coeff1

  # correct the roots of the linear polynomials
  idx_no_coeff1 = (coeff1 == 0.0)
  idx_no_coeff2 = (coeff2 == 0.0)
  idx_linear = idx_no_coeff1 & (~idx_no_coeff2) # linear with non-zero slope
  idx_constant = idx_no_coeff1 & idx_no_coeff2
  # 1 root
  n_roots[idx_linear] = 1
  root1[idx_linear] = -coeff3[idx_linear]/coeff2[idx_linear]
  root2[idx_linear] = np.nan
  # 0 roots
  n_roots[idx_constant] = 0
  root1[idx_constant] = np.nan
  root2[idx_constant] = np.nan

  # stack the roots
  roots = np.vstack((root1,root2)).T

  # sort the roots, smallest to largest
  roots = np.sort(roots,axis=1)

  # check n_roots
  n_roots_calc = 2 - np.isnan(roots).sum(axis=1) 
  assert np.sum(np.abs(n_roots - n_roots_calc)) == 0.0, "n roots computed incorrectly"
  
  return roots,n_roots

def compute_vpar_bounds(roots,n_roots,coeff1,coeff2,coeff3):
  """
  For each spatial points xyz compute the integration bounds 
  on vpar. We return a list containing lists of bounds for each point. 

  return: 
  List containing list of intervals for each point. An interval
  contains a start point, end point.
          [
           [[a1,b1],[a2,b2]], # point 1; 2 intervals
           [[a1,b1]], # point 2; 1 interval
           [], # point 3; no intervals
           ...
           [[a1,b1],[a2,b2]], # point n; 2 intervals
          ]
  """
  # storage
  vpar_bounds = []

  sgn_coeff1 = np.sign(coeff1)

  for ii in range(len(roots)):

    if n_roots[ii] == 2:
      left_root = roots[ii][0]
      right_root = roots[ii][1]

      # convex quadratic 
      if sgn_coeff1[ii]>0:
        if left_root <= vpar_lb and right_root >= vpar_ub:
          # no points to integrate
          intervals = []
        elif left_root <= vpar_lb and right_root <=vpar_lb:
          # no points to integrate
          intervals = []
        elif left_root >= vpar_ub and right_root >=vpar_ub:
          # no points to integrate
          intervals = []
        elif left_root <= vpar_lb and right_root >=vpar_lb:
          # single interval [right_root,vpar_ub]
          intervals = [[right_root,vpar_ub]]
        elif left_root <= vpar_ub and right_root >=vpar_ub:
          # single interval [vpar_lb,left_root]
          intervals = [[vpar_lb,left_root]]
        else:
          # two intervals 
          intervals = [[vpar_lb,left_root],[right_root,vpar_ub]]

      # concave quadratic
      else:
        if left_root <= vpar_lb and right_root >= vpar_ub:
          # single interval [vpar_lb,vpar_ub]
          intervals = [[vpar_lb,vpar_ub]]
        elif left_root <= vpar_lb and right_root <=vpar_lb:
          # no points to integrate
          intervals = []
        elif left_root >= vpar_ub and right_root >=vpar_ub:
          # no points to integrate
          intervals = []
        elif left_root <= vpar_lb and right_root >=vpar_lb:
          # single interval [vpar_lb,right_root]
          intervals = [[vpar_lb,right_root]]
        elif left_root <= vpar_ub and right_root >=vpar_ub:
          # single interval [left_root,vpar_ub]
          intervals = [[left_root,vpar_ub]]
        elif left_root >= vpar_lb and right_root <= vpar_ub:
          # single interval [left_root,right_root]
          intervals = [[left_root,right_root]]
        else:
          print("")
          print("WARNING: we hit an unknown case")
          print("What case is this catching?")
          print(left_root,right_root)
          print(vpar_lb,vpar_ub)
          quit()

    elif n_roots[ii] == 1:
      one_root = roots[ii][0]

      # convex quadratic with 1 root
      if sgn_coeff1[ii]>0:
        # single interval [vpar_lb,vpar_ub]
        intervals = [[vpar_lb,vpar_ub]]
      
      # linear function with nonzero slope
      elif coeff1[ii] == 0.0:
        slope = coeff2[ii]

        if one_root <= vpar_ub and slope>0:
          # one interval
          intervals = [[max(one_root,vpar_lb),vpar_ub]]
        elif one_root >= vpar_lb and slope<0:
          # one interval
          intervals = [[vpar_lb,min(one_root,vpar_ub)]]
      
      else: 
        # no interval
        intervals = []

    # no roots
    else:
      # convex quadratic 
      if sgn_coeff1[ii]>0:
        # single interval [vpar_lb,vpar_ub]
        intervals = [[vpar_lb,vpar_ub]]

      # concave quadratic
      elif sgn_coeff1[ii]< 0:
        # no interval
        intervals = []

      # constant quadratic
      else: 
        if np.sign(coeff3[ii])>0:
          # single interval [vpar_lb,vpar_ub]
          intervals = [[vpar_lb,vpar_ub]]
        else:
          # no interval
          intervals = []
     
    # save the intervals
    vpar_bounds.append(intervals)

  return vpar_bounds

# get the quadratic coeffs
coeff1,coeff2,coeff3 = compute_quadratic_coeffs(xyz,normals)

print('num linear funcs',np.sum(coeff1 == 0))
print('num constant funcs',np.sum((coeff1 == 0) & (coeff2 == 0.0)))

# compute the roots of the vpar quadratic
roots,n_roots = compute_vpar_roots(coeff1,coeff2,coeff3)

# get the vpar integration bounds
vpar_bounds = compute_vpar_bounds(roots,n_roots,coeff1,coeff2,coeff3)

# accumulator for the loss fraction
loss_fraction = 0.0

for ii,xx in enumerate(xyz):
  print("")
  print(f"{ii})")

  # get the vpar integration bounds
  bounds = vpar_bounds[ii]
  n_bounds = len(bounds)

  #np.set_printoptions(precision=16)
  #print(xx)
  #print(normals[ii])
  #np.set_printoptions(precision=8)
  print('coeff1',coeff1[ii])
  print('coeff2',coeff2[ii])
  print('coeff3',coeff3[ii])
  print('n_roots',n_roots[ii])
  print('roots',roots[ii])

  # skip points with no integration intervals.
  if n_bounds == 0:
    continue

  for jj,interval in enumerate(bounds):
    print(f"point {ii}, interval {jj})")
  
    ### TODO: some points return nan for roots. 
    ### They should have a lot of flux to contribute b/c the 
    ### entire interval is being integrated over.
    #quadratic = lambda xx: coeff1[ii]*(xx**2) + coeff2[ii]*xx + coeff3[ii]
    #print('quadratic(left root)',quadratic(roots[ii][0]))
    #print('quadratic(right root)',quadratic(roots[ii][1]))

    # discretize the interval
    vpar_disc = loglin_grid(interval[0],interval[1],int(n_vpar/n_bounds))

    # form the set of points X in 4d state space
    yy = np.tile(xx,len(vpar_disc)).reshape((-1,dim_xyz))
    X = np.hstack((yy,vpar_disc.reshape((-1,1)) ))

    # get the guiding center velocity at the points
    v_gc = GC.GC_rhs(X)[:,:-1] # just keep the spatial part

    # v_gc * normal
    v_gcn = v_gc @ normals[ii]

    # safety check that we are integrating outward flux
    vgcn_tol = -1e-10
    if np.any(v_gcn < vgcn_tol):
      print("")
      print('WARNING: Negative v_gc * normal')
      print(v_gcn[v_gcn<0])
      print('number of roots',n_roots[ii]) 
      print('roots',roots[ii])
      print('vpar interval',interval)
      print('vparlb, ub',vpar_lb,vpar_ub)
      print('coeff 1',coeff1[ii])
      print('coeff 2',coeff2[ii])
      print('coeff 3',coeff3[ii])
      quadratic = lambda x: coeff1[ii]*x**2 + coeff2[ii]*x + coeff3[ii]
      print(quadratic(roots[ii][0]))
      print(quadratic(roots[ii][1]))
      print(quadratic(vpar_lb))
      print(quadratic(vpar_ub))
      quit()

    # TODO: should my area element be for trapezoidal integration?
    # area element
    dA = area_elements[ii]*dtheta*dphi

    # line elements for trapezoidal integration
    dvpar = vpar_disc[1:] - vpar_disc[:-1]

    # volume element
    dV = dA*dvpar

    # TODO: verify accuracy of tau_in_plasma.
    _, tau_in_plasma  = trace_particles(X,GC,tmax,dt,classifier=classifier,
                method=method,n_skip=n_skip,direction='backward')

    print('tau in plasma')
    print(tau_in_plasma)

    # accumulate the loss fraction; trapezoidal integration 
    fx = v_gcn*tau_in_plasma
    fx_avg = (fx[1:] + fx[:-1])/2 # trapezoidal
    loss_fraction += prob_const*np.sum(fx_avg*dV)*symmetry_mult
    print('loss fraction:',loss_fraction)
  
