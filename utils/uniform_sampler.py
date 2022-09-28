
import numpy as np
from coords import *
from constants import *
import sys
sys.path.append("../field")
from biot_savart_field import compute_rz_bounds

class UniformSampler:
  """
  Sample uniformly, in cartesian space, from a plasma 
  volume using rejection sampling.
  """

  def __init__(self,surf,classifier):
    self.surf = surf
    self.classifier = classifier
    # set the bounds
    rmin,rmax,zmin,zmax = compute_rz_bounds(surf)
    lb_vpar = -1.0*np.sqrt(FUSION_ALPHA_SPEED_SQUARED)
    ub_vpar = 1.0*np.sqrt(FUSION_ALPHA_SPEED_SQUARED)
    lb_r,ub_r = rmin,rmax
    lb_phi,ub_phi = 0,2*np.pi
    lb_z,ub_z = zmin,zmax

    # set the sampling bounds
    self.lb_vpar = lb_vpar
    self.ub_vpar = ub_vpar
    self.lb_r = lb_r
    self.ub_r = ub_r
    self.lb_phi = lb_phi
    self.ub_phi = ub_phi
    self.lb_z = lb_z
    self.ub_z = ub_z
  
  def sample(self,n_particles):
    # do rejection sampling
    X = np.zeros((0,4)) # [r,phi,z, vpar]
    while len(X) < n_particles:
      # sample r using the inverse transform r = F^{-1}(U)
      # where the CDF inverse if F^{-1}(u) = sqrt(2u/D + r0_lb^2) 
      # D = 2/(r0_ub^2 - r0_lb^2)
      U = np.random.uniform(0,1)
      D = 2.0/(self.ub_r**2 - self.lb_r**2)
      r = np.sqrt(2*U/D + self.lb_r**2)
      # sample uniformly from phi
      phi = np.random.uniform(self.lb_phi,self.ub_phi)
      # sample uniformly from z
      z = np.random.uniform(self.lb_z,self.ub_z)
      cyl  = np.array([r,phi,z])
      xyz = cyl_to_cart(cyl.reshape((1,-1)))
      # check if particle is in plasma
      if self.classifier.evaluate(xyz) > 0:
        vpar = np.random.uniform(self.lb_vpar,self.ub_vpar,1)
        point = np.append(xyz.flatten(),vpar)
        X =np.vstack((X,point)) # [x,y,z, vpar]
    return X
