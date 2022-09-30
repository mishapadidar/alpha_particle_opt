import numpy as np
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import trace_particles_boozer, MinToroidalFluxStoppingCriterion, \
    MaxToroidalFluxStoppingCriterion,  ToroidalTransitStoppingCriterion
from simsopt.mhd import Vmec
import sys
sys.path.append("../utils")
from constants import *

def trace_boozer(vmec,stz_inits,vpar_inits,tmax=1e-2):
  """
  Trace particles in boozer coordinates.

  vmec: a vmec object
  stz_inits: (n,3) array of (s,theta,zeta) points
  vpar_inits: (n,) array of vpar values
  tmax: max tracing time
  """

  # Construct radial interpolant of magnetic field
  order = 3
  bri = BoozerRadialInterpolant(vmec, order, enforce_vacuum=True)
  
  # Construct 3D interpolation
  nfp = vmec.wout.nfp
  degree = 3
  srange = (0, 1, 10)
  thetarange = (0, np.pi, 10)
  zetarange = (0, 2*np.pi/nfp, 10)
  field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)
  #print('Error in |B| interpolation', field.estimate_error_modB(1000), flush=True)


  #stopping_criteria = [MaxToroidalFluxStoppingCriterion(0.99), 
  #                     MinToroidalFluxStoppingCriterion(0.01),
  #                     ToroidalTransitStoppingCriterion(100,True)]
  stopping_criteria = [MaxToroidalFluxStoppingCriterion(0.99)]
  
  # storage
  exit_times = np.zeros((0))
  exit_states = np.zeros((0,4))
  loss_count = 0
 
  n_particles = len(stz_inits)
  for ii in range(n_particles):
    #print("tracing particle ",ii)
  
    # get the particle
    stz = stz_inits[ii].reshape((1,-1))
    vpar = [vpar_inits[ii]]

    # trace
    res_tys, res_zeta_hits = trace_particles_boozer(
        field, 
        stz, 
        vpar, 
        tmax=tmax, 
        mass=ALPHA_PARTICLE_MASS, 
        charge=ALPHA_PARTICLE_CHARGE,
        Ekin=FUSION_ALPHA_PARTICLE_ENERGY, 
        tol=1e-8, 
        mode='gc_vac',
        stopping_criteria=stopping_criteria,
        forget_exact_path=False
        )

    # get the final state at end of trace
    tstz = res_tys[0] # trajectory [t,x,y,z,vpar]
    final_stz = np.atleast_2d(tstz[-1,1:4])
  
    # loss fraction
    if len(res_zeta_hits[0])>0:
      # exit time.
      tau = res_zeta_hits[0].flatten()[0]
      # exit state
      state = res_zeta_hits[0].flatten()[2:]
      
      ## print some stuff
      #loss_count += 1
      #print("lost particle ",ii)
      #print(res_zeta_hits)
      #print('exit time',tau)
      #print('loss fraction', loss_count/(ii+1))
     
      # save it
      exit_times = np.append(exit_times,tau)
      exit_states = np.vstack((exit_states,state))
  
  #print("")
  #print('loss fraction', loss_count/n_particles)
  #print('exit times')
  #print(exit_times)

  return exit_states,exit_times

  #3print(gc_zeta_hits)
  #3exit_times  = np.array([xx.flatten()[0] for xx in gc_zeta_hits])

  #3return exit_states,exit_times
  

if __name__ == "__main__":

  vmec_input = '../vmec_input_files/input.nfp2_QA'
  vmec = Vmec(vmec_input)
  vmec.run()

  n_particles = 10
  stz_inits = np.array([np.random.uniform(0,1,n_particles),
                        np.random.uniform(0,2*np.pi,n_particles),
                        np.random.uniform(0,2*np.pi,n_particles)]).T
  vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
  vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)   
  vpar_inits = np.random.uniform(vpar_lb,vpar_ub,n_particles)

  exit_states,exit_times = trace_boozer(vmec,stz_inits,vpar_inits,tmax=1e-6)
  print(exit_times)
