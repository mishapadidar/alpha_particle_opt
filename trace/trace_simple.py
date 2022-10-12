import numpy as np
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
from mpi4py import MPI
import sys
sys.path.append("../../SIMPLE/build")
from pysimple import simple, simple_main, params as simple_params, new_vmec_stuff_mod as stuff, velo_mod
sys.path.append("../utils")
from grids import symlog_grid
from constants import *
from divide_work import *


class TraceSimple:
  """
  Class for tracing in simple using a VMEC boundary.
  """

  def __init__(self,vmec_input,n_partitions=1,max_mode=1,major_radius=None):
    
    self.vmec_input = vmec_input
    self.max_mode = max_mode

    # load vmec and mpi
    self.comm = MpiPartition(n_partitions)
    self.vmec = Vmec(vmec_input, mpi=self.comm,keep_all_files=True,verbose=False)
    
    # define parameters
    self.surf = self.vmec.boundary
    self.surf.fix_all()
    self.surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    if major_radius:
      # rescale the major radius
      factor = major_radius/self.surf.get("rc(0,0)")
      self.surf.x = self.surf.x*factor
      # rescale the toroidal flux by factor**2
      self.vmec.indata.phiedge = self.vmec.indata.phiedge*(factor**2)

    self.surf.fix("rc(0,0)") # fix the Major radius
    
    # variables
    self.x0 = np.copy(self.surf.x) # nominal starting point
    self.dim_x = len(self.x0) # dimension

    # pysimple stuff; see SIMPLE/examples/simple.in
    stuff.multharm = 5     # 3=fast/inacurate, 5=normal,7=very accurate
    simple_params.contr_pp = -1e10     # Trace all passing passing
    simple_params.notrace_passing = 0      # leave at 0! set to 1 to skip passing particles
    simple_params.startmode = -1       # -1 Manual, 1 generate on surface
    simple_params.sbeg = 0.5 # surface to generate on
    velo_mod.isw_field_type = 0 # 0 trace in canonical
    simple_params.ntimstep = 10000 # number of timesteps; increase for accuracy
    simple_params.npoiper2 = 256	 # points per period for integrator step; increase for accuracy
    self.tracy = simple_params.Tracer()

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

  def flux_grid(self,ns,ntheta,nzeta,nvpar):
    """
    Build a 4d grid over the flux coordinates and vpar.
    """
    # bounds
    vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
    vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)   
    # use fixed particle locations
    surfaces = np.linspace(0.01,0.98, ns)
    thetas = np.linspace(0, 1.0, ntheta)
    zetas = np.linspace(0,1.0, nzeta)
    vpars = symlog_grid(vpar_lb,vpar_ub,nvpar)
    # build a mesh
    [surfaces,thetas,zetas,vpars] = np.meshgrid(surfaces,thetas, zetas,vpars)
    stz_inits = np.zeros((ns*ntheta*nzeta*nvpar, 3))
    stz_inits[:, 0] = surfaces.flatten()
    stz_inits[:, 1] = thetas.flatten()
    stz_inits[:, 2] = zetas.flatten()
    vpar_inits = vpars.flatten()
    return stz_inits,vpar_inits
    
  def surface_grid(self,s_label,ntheta,nzeta,nvpar):
    """
    Builds a grid on a single surface.
    """
    # bounds
    vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
    vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)   
    # use fixed particle locations
    thetas = np.linspace(0, 1.0, ntheta)
    zetas = np.linspace(0,1.0, nzeta)
    #vpars = symlog_grid(vpar_lb,vpar_ub,nvpar)
    vpars = np.linspace(vpar_lb,vpar_ub,nvpar)
    # build a mesh
    [thetas,zetas,vpars] = np.meshgrid(thetas, zetas,vpars)
    stz_inits = np.zeros((ntheta*nzeta*nvpar, 3))
    stz_inits[:, 0] = s_label
    stz_inits[:, 1] = thetas.flatten()
    stz_inits[:, 2] = zetas.flatten()
    vpar_inits = vpars.flatten()
    return stz_inits,vpar_inits

  def compute_confinement_times(self,x,stp_inits,vpar_inits,tmax):
    """
    x: vmec surface description
    stp_inits: (s,theta,phi) vmec coords. shape (N,3)
    vpar_inits: parallel velocities. shape (N,)
    tmax: max trace time
    """
    n_particles = len(vpar_inits)

    self.surf.x = np.copy(x)
    try:
      self.vmec.run()
    except:
      # VMEC failure!
      return -np.inf*np.ones(len(stp_inits)) 

    # get the wout file
    wout_file = self.vmec.output_file.split("/")[-1]
    simple.init_field(self.tracy, wout_file,
        stuff.ns_s, stuff.ns_tp, stuff.multharm, simple_params.integmode)

    # set the trace time
    simple_params.trace_time = tmax

    # divide the particles
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    work_intervals,work_counts = divide_work(n_particles,size)
    my_work = work_intervals[rank]
    my_counts = work_counts[rank]

    # set n particles
    #simple_params.ntestpart = n_particles
    simple_params.ntestpart = my_counts

    simple_params.params_init()
    # normalize the velocities
    v_normalized = np.ones((n_particles,1))
    vpar_normalized = vpar_inits.reshape((-1,1))/np.sqrt(FUSION_ALPHA_SPEED_SQUARED) 
    # s, th_vmec, ph_vmec, v/v0, v_par/v
    zstart = np.hstack((stp_inits,v_normalized,vpar_normalized)).T
    simple_params.zstart = zstart[:,my_work]
    # trace
    simple_main.run(self.tracy)
    #return simple_params.times_lost

    my_times= simple_params.times_lost

    # correct the -1 values
    my_times[my_times < 0] = tmax

    # broadcast the values
    exit_times = np.zeros(n_particles)
    comm.Gatherv(my_times,(exit_times,work_counts),root=0)
    comm.Bcast(exit_times,root=0)
    return exit_times



if __name__ == "__main__":

  vmec_input = '../vmec_input_files/input.nfp2_QA'
  tracer = TraceSimple(vmec_input,n_partitions=1,max_mode=1,major_radius=5)
  tracer.sync_seeds(0)
  x0 = tracer.x0
  dim_x = tracer.dim_x
  tmax = 1e-4

  # tracing points
  ntheta=nzeta = 5
  nvpar=5
  stp_inits,vpar_inits = tracer.surface_grid(0.4,ntheta,nzeta,nvpar)

  import time
  t0  = time.time()
  c_times = tracer.compute_confinement_times(x0,stp_inits,vpar_inits,tmax)
  print('time',time.time() - t0)
  print('mean',np.mean(c_times))
  #print(c_times)
