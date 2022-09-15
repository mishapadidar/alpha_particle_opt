import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surface import signed_distance_from_surface
from scipy.integrate import simpson
from guiding_center_eqns_cartesian import *
from trace_particles import *
import sys
sys.path.append("../stella")
from bfield import load_field,compute_rz_bounds,compute_plasma_volume,make_surface_classifier
sys.path.append("../utils")
from constants import *
from grids import *
import vtkClass

class FluxIntegrator:
  """
  Class for computing the flux integral.
  """

  def __init__(self,
                vmec_input,
                bs_path,
                nphi=32,
                ntheta=32,
                nvpar=32,
                nphi_classifier=512,
                ntheta_classifier=512,
                eps_classifier=1e-5,
                tmax=1e-6,
                dt=1e-8,
                ode_method='midpoint'
                ):
    assert nvpar % 2 == 0, "must use even number of points"
    # surface file
    self.vmec_input = vmec_input
    # ODE integration variables
    self.tmax = tmax
    self.dt = dt
    self.n_skip = np.inf
    self.ode_method = ode_method # euler or midpoint
    self.include_drifts = True
    # classifier relaxation tol
    self.eps_classifier = eps_classifier
    self.nphi_classifier = nphi_classifier
    self.ntheta_classifier = ntheta_classifier
    # discretizations
    self.nphi = nphi
    self.ntheta = ntheta
    self.nvpar = nvpar
    self.dim_xyz = 3

    # load the surface
    surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="field period", nphi=nphi, ntheta=ntheta)
    self.nfp = surf.nfp
  
    # load the plasma volume
    self.plasma_vol = compute_plasma_volume(vmec_input,nphi=nphi,ntheta=ntheta)
    # build the surface classifier
    self.classifier = make_surface_classifier(vmec_input=vmec_input,rng="full torus",
                      nphi=nphi_classifier,ntheta=ntheta_classifier)
    # load the bfield
    self.bs = load_field(vmec_input,bs_path,nphi=nphi,ntheta=ntheta)
  
    # compute the initial state volume
    self.vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
    self.vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)
    self.vpar_vol = self.vpar_ub - self.vpar_lb
    self.prob_const = 1/self.plasma_vol/self.vpar_vol

    # build a guiding center object
    self.GC = GuidingCenter(self.bfield,self.gradAbsB,include_drifts=self.include_drifts)

    # get the quadrature points
    self.quadpoints_phi = surf.quadpoints_phi # shape (nphi,)
    self.quadpoints_theta = surf.quadpoints_theta # shape (ntheta,)

    # discretize the plasma boundary
    self.xyz = surf.gamma() # shape (nphi, ntheta,3)
    
    # get the surface normals (should be outward facing)
    self.normals = surf.normal() # shape (nphi, ntheta,3)
    area_elements = np.linalg.norm(self.normals.reshape((-1,3)),axis=1)
    unit_normals = (self.normals.reshape((-1,3)).T/area_elements).T # unit normals
    self.unit_normals = unit_normals.reshape((self.nphi,self.ntheta,3))
    self.area_elements = area_elements.reshape((self.nphi,self.ntheta))

  
  def bfield(self,xyz):
    # add zero to shut simsopt up
    self.bs.set_points(xyz + np.zeros(np.shape(xyz)))
    return self.bs.B()

  def gradAbsB(self,xyz):
    # add zero to shut simsopt up
    self.bs.set_points(xyz + np.zeros(np.shape(xyz)))
    return self.bs.GradAbsB()

  def compute_quadratic_coeffs(self,xyz,normals):
    """
    Compute the coefficients of the outward flux quadratic. 
  
    xyz: array point points, shape (N,3)
    normals: array of outward facing normals, shape (N,3)
    return: three (N,) arrays of coefficients: coeff1,coeff2,coeff3.
    """
    # compute B on grid
    Bb = self.bfield(xyz) # shape (N,3)
    # compute Bg on grid
    Bg = self.gradAbsB(xyz) # shape (N,3)
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
  
  def compute_vpar_roots(self,coeff1,coeff2,coeff3):
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
  
  def compute_vpar_bounds(self,roots,n_roots,coeff1,coeff2,coeff3):
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
          if left_root <= self.vpar_lb and right_root >= self.vpar_ub:
            # no points to integrate
            intervals = []
          elif left_root <= self.vpar_lb and right_root <=self.vpar_lb:
            # no points to integrate
            intervals = []
          elif left_root >= self.vpar_ub and right_root >=self.vpar_ub:
            # no points to integrate
            intervals = []
          elif left_root <= self.vpar_lb and right_root >=self.vpar_lb:
            # single interval [right_root,vpar_ub]
            intervals = [[right_root,self.vpar_ub]]
          elif left_root <= self.vpar_ub and right_root >=self.vpar_ub:
            # single interval [vpar_lb,left_root]
            intervals = [[self.vpar_lb,left_root]]
          else:
            # two intervals 
            intervals = [[self.vpar_lb,left_root],[right_root,self.vpar_ub]]
  
        # concave quadratic
        else:
          if left_root <= self.vpar_lb and right_root >= self.vpar_ub:
            # single interval [vpar_lb,vpar_ub]
            intervals = [[self.vpar_lb,self.vpar_ub]]
          elif left_root <= self.vpar_lb and right_root <=self.vpar_lb:
            # no points to integrate
            intervals = []
          elif left_root >= self.vpar_ub and right_root >=self.vpar_ub:
            # no points to integrate
            intervals = []
          elif left_root <= self.vpar_lb and right_root >=self.vpar_lb:
            # single interval [vpar_lb,right_root]
            intervals = [[self.vpar_lb,right_root]]
          elif left_root <= self.vpar_ub and right_root >=self.vpar_ub:
            # single interval [left_root,vpar_ub]
            intervals = [[left_root,self.vpar_ub]]
          elif left_root >= self.vpar_lb and right_root <= self.vpar_ub:
            # single interval [left_root,right_root]
            intervals = [[left_root,right_root]]
          else:
            print("")
            print("WARNING: we hit an unknown case")
            print("What case is this catching?")
            print(left_root,right_root)
            print(self.vpar_lb,self.vpar_ub)
            quit()
  
      elif n_roots[ii] == 1:
        one_root = roots[ii][0]
  
        # convex quadratic with 1 root
        if sgn_coeff1[ii]>0:
          # single interval [vpar_lb,vpar_ub]
          intervals = [[self.vpar_lb,self.vpar_ub]]
        
        # linear function with nonzero slope
        elif coeff1[ii] == 0.0:
          slope = coeff2[ii]
  
          if one_root <= self.vpar_ub and slope>0:
            # one interval
            intervals = [[max(one_root,self.vpar_lb),self.vpar_ub]]
          elif one_root >= self.vpar_lb and slope<0:
            # one interval
            intervals = [[self.vpar_lb,min(one_root,self.vpar_ub)]]
        
        else: 
          # no interval
          intervals = []
  
      # no roots
      else:
        # convex quadratic 
        if sgn_coeff1[ii]>0:
          # single interval [vpar_lb,vpar_ub]
          intervals = [[self.vpar_lb,self.vpar_ub]]
  
        # concave quadratic
        elif sgn_coeff1[ii]< 0:
          # no interval
          intervals = []
  
        # constant quadratic
        else: 
          if np.sign(coeff3[ii])>0:
            # single interval [vpar_lb,vpar_ub]
            intervals = [[self.vpar_lb,self.vpar_ub]]
          else:
            # no interval
            intervals = []
       
      # save the intervals
      vpar_bounds.append(intervals)
  
    return vpar_bounds

  def integrate_vpar(self,xyz,normal,bounds):
    """
    Compute the vpar integral and time integral
      int_{I(x)} int_0^T f(t,x,vpar) dt v_{gc}*unit_normal dvpar
        =  int_{I(x)} min{T,tau(x,vpar)}*C_{P}*v_{gc}*unit_normal dvpar
    for a given Cartesian point x.

    xyz: array, Cartesian point shape (3,)
    normal: array, Normal vector at xyz, shape (3,)
    bounds: list of integration bounds
    """
    # TODO
    # [ ] switch to a scipy integrator that automates the discretization of vpar
    # [ ] switch to a scipy ODE integrator to solve the IVP

    tot = 0.0
  
    n_bounds = len(bounds)

    # integrate vpar
    for ii_bounds,interval in enumerate(bounds):
      #print(f"interval {ii_bounds})")
    
      # discretize the interval
      vpar_disc = loglin_grid(interval[0],interval[1],int(self.nvpar/n_bounds))
  
      # form the set of points X in 4d state space
      yy = np.tile(xyz,len(vpar_disc)).reshape((-1,self.dim_xyz))
      X = np.hstack((yy,vpar_disc.reshape((-1,1)) ))
  
      # get the guiding center velocity at the points
      v_gc = self.GC.GC_rhs(X)[:,:-1] # just keep the spatial part
  
      # v_gc * normal
      v_gcn = v_gc @ normal
  
      # trace; tau_in_plasma = min(T,tau)
      _, tau_in_plasma  = trace_particles(X,self.GC,self.tmax,self.dt,classifier=self.classifier,
                  eps=self.eps_classifier,method=self.ode_method,n_skip=np.inf,direction='backward')

      # line elements for trapezoidal integration
      dvpar = vpar_disc[1:] - vpar_disc[:-1]

      # function values for trapezoid
      integrand = self.prob_const*tau_in_plasma*v_gcn
      integrand_avg = (integrand[1:] + integrand[:-1])/2

      # accumulate
      tot += np.sum(integrand_avg*dvpar)
    
    return tot


  def solve(self):
  
    # get the quadrature points
    xyz = np.copy(self.xyz.reshape((-1,self.dim_xyz)))
    unit_normals = np.copy(self.unit_normals.reshape((-1,self.dim_xyz)))

    # get the quadratic coeffs
    coeff1,coeff2,coeff3 = self.compute_quadratic_coeffs(xyz,unit_normals)
    
    print('num linear funcs',np.sum(coeff1 == 0))
    print('num constant funcs',np.sum((coeff1 == 0) & (coeff2 == 0.0)))
    
    # compute the roots of the vpar quadratic
    roots,n_roots = self.compute_vpar_roots(coeff1,coeff2,coeff3)
    
    # get the vpar integration bounds
    vpar_bounds = self.compute_vpar_bounds(roots,n_roots,coeff1,coeff2,coeff3)

    # storage
    vpar_integrals = np.zeros((self.nphi,self.ntheta))
    
    print('starting integration')
    # compute the vpar integral over all theta and phi
    for ii_phi,phi in enumerate(self.quadpoints_phi):
      for ii_theta,theta in enumerate(self.quadpoints_theta):

        # convert the theta, phi indexes to linear
        ii_x = ii_phi*self.nphi + ii_theta
    
        # get the surface point
        xx = xyz[ii_x]
        unit_normal = unit_normals[ii_x]
    
        # get the vpar integration bounds
        bounds = vpar_bounds[ii_x]
        n_bounds = len(bounds)
    
        #print("")
        #print(f"{ii_x})")
        #np.set_printoptions(precision=16)
        #print(xx)
        #print(normals[ii_x])
        #np.set_printoptions(precision=8)
        #print('coeff1',coeff1[ii_x])
        #print('coeff2',coeff2[ii_x])
        #print('coeff3',coeff3[ii_x])
        #print('n_roots',n_roots[ii_x])
        #print('roots',roots[ii_x])
    
        # integrate over vpar
        if n_bounds == 0:
          vpar_integrals[ii_phi,ii_theta] = 0.0
        else:
          vpar_integrals[ii_phi,ii_theta] = self.integrate_vpar(xx,unit_normal,bounds)
    
    # integrate over theta with simpsons rule
    theta_integrals = np.zeros(self.nphi)
    for ii_phi in range(self.nphi):
      # integrate over theta; include the area element
      theta_integrals[ii_phi] = simpson(vpar_integrals[ii_phi]*self.area_elements[ii_phi],self.quadpoints_theta)
    
    # now integrate over phi with simpsons rule
    loss_fraction = simpson(theta_integrals,self.quadpoints_phi)

    # multiply by symmetry factor
    loss_fraction = loss_fraction*self.nfp
    
    return loss_fraction
    
