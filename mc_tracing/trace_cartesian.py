import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.tracing import trace_particles, LevelsetStoppingCriterion
import sys
from guiding_center_eqns_cartesian import *
sys.path.append("../stella")
from bfield import load_field,compute_rz_bounds,compute_plasma_volume,make_surface_classifier
sys.path.append("../utils")
from coords import cyl_to_cart,cart_to_cyl
from concentricSurfaceClassifier import concentricSurfaceClassifier
from constants import *
from grids import loglin_grid
import vtkClass


def uniform_sampler(surf,classifier):
  """
  Sample uniformly from the plasma.
  """
  # set the bounds
  rmin,rmax,zmin,zmax = compute_rz_bounds(surf)
  lb_vpar = -1.0*np.sqrt(FUSION_ALPHA_SPEED_SQUARED)
  ub_vpar = 1.0*np.sqrt(FUSION_ALPHA_SPEED_SQUARED)
  lb_r,ub_r = rmin,rmax
  lb_phi,ub_phi = 0,2*np.pi
  lb_z,ub_z = zmin,zmax
  
  # do rejection sampling
  X = np.zeros((0,4)) # [r,phi,z, vpar]
  while len(X) < n_particles:
    # sample r using the inverse transform r = F^{-1}(U)
    # where the CDF inverse if F^{-1}(u) = sqrt(2u/D + r0_lb^2) 
    # D = 2/(r0_ub^2 - r0_lb^2)
    U = np.random.uniform(0,1)
    D = 2.0/(ub_r**2 - lb_r**2)
    r = np.sqrt(2*U/D + lb_r**2)
    # sample uniformly from phi
    phi = np.random.uniform(lb_phi,ub_phi)
    # sample uniformly from z
    z = np.random.uniform(lb_z,ub_z)
    cyl  = np.array([r,phi,z])
    xyz = cyl_to_cart(np.atleast_2d(cyl))
    # check if particle is in plasma
    if classifier.evaluate(np.atleast_2d(xyz)) > 0:
      vpar = np.random.uniform(lb_vpar,ub_vpar,1)
      point = np.append(xyz.flatten(),vpar)
      X =np.vstack((X,point)) # [x,y,z, vpar]

  return xyz

  


def compute_losses(X,bs,classifier,tmax):
  
  """
  Sample from the initial distribution

  X: (n_particles,4) list of points (x,y,z,vpar)
  bs: biot savarat object
  classifier: surface classifier
  tmax: max trace time.
  """

  # storage
  exit_times = np.zeros(0)
  exit_states = np.zeros((0,4))
  
  n_particles = len(X)
  stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)]

  loss_count = 0
  for ii in range(n_particles):
    #print("tracing particle ",ii)
  
    # get the particle
    xyz = X[ii].reshape((1,-1))
    vpar = [X[ii,-1]]
  
    # trace
    res_tys, res_phi_hits= trace_particles(bs, xyz, vpar, tmax=tmax, mass=ALPHA_PARTICLE_MASS,
                 charge=ALPHA_PARTICLE_CHARGE, Ekin=FUSION_ALPHA_PARTICLE_ENERGY, 
                 tol=1e-10, stopping_criteria=stopping_criteria, mode='gc_vac',
                  forget_exact_path=False)
  
    # get the final state at end of trace
    txyz = res_tys[0] # trajectory [t,x,y,z,vpar]
    final_xyz = np.atleast_2d(txyz[-1,1:4])
  
    # loss fraction
    if len(res_phi_hits[0])>0:
      # exit time.
      tau = res_phi_hits[0].flatten()[0]
      # exit state
      state = res_phi_hits[0].flatten()[2:]
      
      # print some stuff
      print("lost particle ",ii)
      print(res_phi_hits)
      print('exit time',tau)
      loss_count += 1
      print('loss fraction', loss_count/(ii+1))
     
      # save it
      exit_times = np.append(exit_times,tau)
      exit_states = np.vstack((exit_states,state))
  
  print("")
  print('loss fraction', loss_count/n_particles)
  print('exit times')
  print(exit_times)

  return exit_states,exit_times
  

"""
Compute particle losses for a vmec surface
"""
np.random.seed(0)

n_particles = 10
tmax = 1e-5
vmec_input = "../vmec_input_files/input.new_QA_scaling"

# load the initial surface
nphi=ntheta=128
surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="field period", nphi=nphi, ntheta=ntheta)
# build coils
bs = compute_biot_savart_field(surf)
# scale the coil currents to reactor strength
bs = rescale_coil_currents(surf,bs)
# load the surface classifier
nphi=ntheta=512
surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="full torus", nphi=nphi, ntheta=ntheta)
classifier = make_surface_classifier(vmec_input=vmec_input, rng="full torus",ntheta=ntheta,nphi=nphi)
