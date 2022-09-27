import numpy as np
from simsopt.field.tracing import trace_particles, LevelsetStoppingCriterion
import sys
sys.path.append("../utils")
from constants import *


def compute_loss_times(X,bs,classifier,tmax):
  
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
  

