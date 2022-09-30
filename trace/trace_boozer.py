import numpy as np
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import trace_particles_boozer, MinToroidalFluxStoppingCriterion, \
    MaxToroidalFluxStoppingCriterion,  ToroidalTransitStoppingCriterion
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
from mpi4py import MPI
import sys
sys.path.append("../utils")
from grids import symlog_grid
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

class TraceBoozer:
  """
  A class to make tracing from a vmec configuration a bit 
  more convenient.
  """

  def __init__(self,vmec_input,n_partitions=1,max_mode=1):
    
    self.vmec_input = vmec_input
    self.max_mode = max_mode

    # load vmec and mpi
    self.comm = MpiPartition(n_partitions)
    self.vmec = Vmec(vmec_input, mpi=self.comm,keep_all_files=False,verbose=False)
    
    # define parameters
    self.surf = self.vmec.boundary
    self.surf.fix_all()
    self.surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    self.surf.fix("rc(0,0)") # fix the Major radius
    
    # variables
    self.x0 = np.copy(self.surf.x) # nominal starting point
    self.dim_x = len(self.x0) # dimension

  def sync_seeds(self,sd=None):
    """
    Sync the np.random.seed of the various worker groups.
    The seed is a random number <1e6.
    """
    seed = np.zeros(1)
    if self.comm.proc0_world:
      if sd:
        seed = sd*np.ones(1)
      else:
        seed = np.random.randint(int(1e6))*np.ones(1)
    self.comm.comm_world.Bcast(seed,root=0)
    np.random.seed(int(seed[0]))
    return int(seed[0])
    
  def surface_grid(self,s_label,ntheta,nzeta,nvpar):
    """
    Builds a grid on a single surface.
    """
    # bounds
    vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
    vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)   
    # use fixed particle locations
    thetas = np.linspace(0, 2*np.pi, ntheta)
    zetas = np.linspace(0,2*np.pi/self.surf.nfp, nzeta)
    vpars = symlog_grid(vpar_lb,vpar_ub,nvpar)
    # build a mesh
    [thetas,zetas,vpars] = np.meshgrid(thetas, zetas,vpars)
    stz_inits = np.zeros((ntheta*nzeta*nvpar, 3))
    stz_inits[:, 0] = s_label
    stz_inits[:, 1] = thetas.flatten()
    stz_inits[:, 2] = zetas.flatten()
    vpar_inits = vpars.flatten()
    return stz_inits,vpar_inits
    
  # set up the objective
  def compute_confinement_times(self,x,stz_inits,vpar_inits,tmax):
    self.surf.x = np.copy(x)
    try:
      self.vmec.run()
    except:
      # VMEC failure!
      return -np.inf*np.ones(len(stz_inits)) 
    exit_times = trace_boozer(self.vmec,stz_inits,vpar_inits,tmax=tmax)
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
