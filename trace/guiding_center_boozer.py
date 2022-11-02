import numpy as np
import sys
sys.path.append("../utils")
from constants import *


class GuidingCenterVacuumBoozer:
    
    def __init__(self,field,y0):
      self.field = field
      self.y0 = y0 # [s,t,z,vpar]
      self.mass = ALPHA_PARTICLE_MASS 
      self.q = ALPHA_PARTICLE_CHARGE  # charge

      # compute mu = vperp^2/2B
      stz   = y0[:3]
      v_par = y0[3]
      field.set_points(np.reshape(stz,(1,3)))
      modB = field.modB_ref()[0][0]
      vperp_squared = FUSION_ALPHA_SPEED_SQUARED - v_par**2
      self.mu = vperp_squared/2/modB

    def GuidingCenterVacuumBoozerRHS(self,ys):
      """
      Guiding center right hand side for boozer vacuum tracing.
      ys: (4,) array [s,t,z,vpar]
      """
      stz   = ys[:3]
      v_par = ys[3]
      q = self.q
      mu = self.mu
      mass = self.mass
      field = self.field

      field.set_points(np.reshape(stz,(1,3)))
      psi0 = field.psi0
      modB = field.modB_ref()[0][0]
      G = field.G_ref()[0][0]
      iota = field.iota_ref()[0][0]
      #modB_derivs = field.modB_derivs_ref()[0]
      modB_derivs = field.modB_derivs()[0]
      dmodBds = modB_derivs[0]
      dmodBdtheta = modB_derivs[1]
      dmodBdzeta = modB_derivs[2]
      v_perp2 = 2*mu*modB;
      fak1 = mass*v_par*v_par/modB + mass*mu;

      dydt = np.zeros(4)
      dydt[0] = -dmodBdtheta*fak1/(q*psi0);
      dydt[1] = dmodBds*fak1/(q*psi0) + iota*v_par*modB/G;
      dydt[2] = v_par*modB/G;
      dydt[3] = -(iota*dmodBdtheta + dmodBdzeta)*mu*modB/G;
      return dydt
