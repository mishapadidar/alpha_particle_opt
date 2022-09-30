import numpy as np
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import trace_particles_boozer, MinToroidalFluxStoppingCriterion, \
    MaxToroidalFluxStoppingCriterion,  ToroidalTransitStoppingCriterion
from simsopt.mhd import Vmec
from mpi4py import MPI
import sys
sys.path.append("../utils")
from constants import *
from divide_work import *

def trace_boozer(vmec,stz_inits,vpar_inits,tmax=1e-2):
  """
  Trace particles in boozer coordinates.

  vmec: a vmec object
  stz_inits: (n,3) array of (s,theta,zeta) points
  vpar_inits: (n,) array of vpar values
  tmax: max tracing time
  """
  n_particles = len(stz_inits)

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
   

  # divide the work
  comm = MPI.COMM_WORLD
  size = comm.Get_size()
  rank = comm.Get_rank()
  work_intervals,work_counts = divide_work(n_particles,size)
  my_work = work_intervals[rank]
  my_counts = work_counts[rank]
  print("")
  print(my_counts)

  my_times = np.zeros(my_counts)

  for idx,point_idx in enumerate(my_work):
  
    # get the particle
    stz = stz_inits[point_idx].reshape((1,-1))
    vpar = [vpar_inits[point_idx]]

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
  
    # lost particle
    if len(res_zeta_hits[0])>0:
      # exit time.
      tau = res_zeta_hits[0].flatten()[0]
      # exit state
      state = res_zeta_hits[0].flatten()[2:]
    else:
      tau = tmax
    # save it
    my_times[idx] = tau
  
  # broadcast the values
  exit_times = np.zeros(n_particles)
  comm.Gatherv(my_times,(exit_times,work_counts),root=0)
  comm.Bcast(exit_times,root=0)

  return exit_times

  

if __name__ == "__main__":
  from simsopt.util.mpi import MpiPartition

  vmec_input = '../vmec_input_files/input.nfp2_QA'
  n_partitions = 1
  comm = MpiPartition(n_partitions)
  vmec = Vmec(vmec_input, mpi=comm,keep_all_files=False,verbose=False)
  vmec.run()

  n_particles = 1000
  stz_inits = np.array([np.random.uniform(0,1,n_particles),
                        np.random.uniform(0,2*np.pi,n_particles),
                        np.random.uniform(0,2*np.pi,n_particles)]).T
  vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
  vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)   
  vpar_inits = np.random.uniform(vpar_lb,vpar_ub,n_particles)

  import time
  t0 = time.time()
  exit_times = trace_boozer(vmec,stz_inits,vpar_inits,tmax=1e-2)
  print(exit_times)
  print(time.time() - t0)
