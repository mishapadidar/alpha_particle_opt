import numpy as np
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import trace_particles_boozer, MaxToroidalFluxStoppingCriterion,MinToroidalFluxStoppingCriterion
from simsopt.geo.surfacegarabedian import SurfaceGarabedian
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
from mpi4py import MPI
import sys
sys.path.append("../utils")
sys.path.append("../sample")
from grids import symlog_grid
from radial_density import RadialDensity
from constants import *
from divide_work import *

class TraceBoozer:
  """
  A class to make tracing from a vmec configuration a bit 
  more convenient.
  """

  def __init__(self,vmec_input,
    n_partitions=1,
    max_mode=1,
    minor_radius=1.7,
    major_radius=13.6,
    target_volavgB=5.0,
    tracing_tol=1e-8,
    interpolant_degree=3,
    interpolant_level=8,
    bri_mpol=32,
    bri_ntor=32,
    x0=[],
    x0_max_mode=None,
    x0_major_radius=13.6):
    """
    vmec_input: vmec input file
    n_partitions: number of partitions used by vmec mpi.
    max_mode: number of modes used by vmec
    minor_radius: set Delta(0,0) to this value.
      Delta(0,0) would be approximately the minor radius of a torus.
      The minor radius is better approximated from the aspect ratio.
    major_radius: will rescale entire device so that this is the major radius.
              If the surface is purely a torus, then setting the major and minor radius
              like this will give you the right aspect ratio. But if the surface
              is not a normal torus, then the aspect ratio may be much smaller or larger.
    target_volavgB: will set phiedge so that this is the volume averge |B|.
                    phiedge= pi* a^2 * B approximately, so we try to rescale to 
                    achieve the target B value.
    tracing_tol:a tolerance used to determine the accuracy of the tracing
    interpolant_degree: degree of the polynomial interpolants used for 
      interpolating the field. 1 is fast but innacurate, 3 is slower but more accurate.
    interpolant_level: number of points used to interpolate the boozer radial 
      interpolant (per direction). 5=fast/innacurate, 8=medium, 12=slow/accurate
    bri_mpol,bri_ntor: number of poloidal and toroidal modes used in BoozXform,
        less modes. 16 is faster than 32.
    """
    
    self.vmec_input = vmec_input
    self.max_mode = max_mode

    # For Garabedian rep
    # load vmec and mpi
    #self.mpi = MpiPartition(n_partitions)
    #vmec = Vmec(vmec_input, mpi=self.mpi,keep_all_files=False,verbose=False)

    # build a garabedian surface
    #self.surf = SurfaceGarabedian.from_RZFourier(vmec.boundary)

    # tell vmec it has a garabedian surface now
    #vmec.boundary = self.surf
    #self.vmec = vmec

    # make the descision variables
    #self.surf.fix_all()
    #self.surf.fix_range(mmin=1-max_mode, mmax=1+max_mode,
    #                 nmin=-max_mode, nmax=max_mode, fixed=False)

    # For RZFourier rep
    self.mpi = MpiPartition(n_partitions)
    self.vmec = Vmec(vmec_input, mpi=self.mpi,keep_all_files=False,verbose=False)
    # get the boundary rep
    self.surf = self.vmec.boundary

    if len(x0) > 0:
      """
      Load a point with. We assume that x0_max_mode <= max_mode, that 
      the major radius of the configuration was fixed and is not represented
      in the array x0.
      """
      assert x0_max_mode <= max_mode,"we cannot decrease the max_mode"
      # load the starting point
      self.surf.fix_all()
      self.surf.fixed_range(mmin=0, mmax=x0_max_mode,
                       nmin=-x0_max_mode, nmax=x0_max_mode, fixed=False)
      # assume the point was scaled to the given major radius
      self.surf.set("rc(0,0)",x0_major_radius) 
      self.surf.fix("rc(0,0)") # fix the Major radius
      self.surf.x = np.copy(x0)
    
    # set the resolution
    self.surf.fix_all()
    self.surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    
    # rescale the surface by the major radius
    if major_radius is not None:
      # Delta(1,0) is major radius
      #factor = major_radius/self.surf.get("Delta(1,0)")
      # rc(0,0) is the major radius
      factor = major_radius/self.surf.get("rc(0,0)")
      self.surf.x = self.surf.x*factor

    ## set the approximate minor radius Delta(0,0)
    #if minor_radius is not None:
    #  self.surf.set('Delta(0,0)', minor_radius) # fix minor radius

    #print(self.surf.local_dof_names)
    #print(self.surf.x)
    #print(major_radius)
    #print(minor_radius)

    ### fix the radii
    #self.surf.fix("Delta(1,0)") # fix the Major radius
    #self.surf.fix("Delta(0,0)") # fix the Minor radius
    self.surf.fix("rc(0,0)") # fix the Major radius

    # fix the phiedge to get the correct scaling
    if target_volavgB is not None:
      #self.vmec.run()
      #major_radius = self.surf.get('Delta(1,0)')
      major_radius = self.surf.get('rc(0,0)')
      #avg_minor_rad = major_radius/self.vmec.aspect() # true avg minor radius
      avg_minor_rad = major_radius/self.surf.aspect_ratio() # true avg minor radius
      self.vmec.indata.phiedge = np.pi*(avg_minor_rad**2)*target_volavgB
      self.vmec.need_to_run_code = True
      #self.vmec.run()
    #print('aspect',self.vmec.aspect())
    #print('volavgB',self.vmec.wout.volavgB)
    #print('phiedge',self.vmec.indata.phiedge)

    # variables
    self.x0 = np.copy(self.surf.x) # nominal starting point
    self.dim_x = len(self.x0) # dimension

    # tracing params
    self.tracing_tol=tracing_tol
    self.interpolant_degree=interpolant_degree
    self.interpolant_level=interpolant_level
    self.bri_mpol=bri_mpol
    self.bri_ntor=bri_ntor

    # placeholders
    self.x_field = []
    self.field = None
    self.bri = None

  def sync_seeds(self,sd=None):
    """
    Sync the np.random.seed of the various worker groups.
    The seed is a random number <1e6.
    """
    # TODO: only sync across mpi group
    seed = np.zeros(1)
    #if self.mpi.proc0_world:
    if self.mpi.proc0_groups:
      if sd:
        seed = sd*np.ones(1)
      else:
        seed = np.random.randint(int(1e6))*np.ones(1)
    #self.mpi.comm_world.Bcast(seed,root=0)
    self.mpi.comm_groups.Bcast(seed,root=0)
    np.random.seed(int(seed[0]))
    return int(seed[0])

  def radial_grid(self,ns,ntheta,nzeta,nvpar,min_cdf=0.01,max_cdf=0.95,vpar_lb=-V_MAX,vpar_ub=V_MAX):
    """
    Build a 4d grid over the flux coordinates and vpar which is uniform in the 
    radial CDF.
    min_cdf,max_cdf: lower and upper bounds on the grid in CDF space
    """
    # uniformly grid according to the radial measure
    sampler = RadialDensity(1000)
    surfaces = np.linspace(min_cdf,max_cdf, ns)
    surfaces = sampler._cdf_inv(surfaces)
    # use fixed particle locations
    thetas = np.linspace(0, 2*np.pi, ntheta)
    zetas = np.linspace(0,2*np.pi/self.surf.nfp, nzeta)
    #vpars = symlog_grid(vpar_lb,vpar_ub,nvpar)
    vpars = np.linspace(vpar_lb,vpar_ub,nvpar)
    # build a mesh
    [surfaces,thetas,zetas,vpars] = np.meshgrid(surfaces,thetas, zetas,vpars)
    stz_inits = np.zeros((ns*ntheta*nzeta*nvpar, 3))
    stz_inits[:, 0] = surfaces.flatten()
    stz_inits[:, 1] = thetas.flatten()
    stz_inits[:, 2] = zetas.flatten()
    vpar_inits = vpars.flatten()
    return stz_inits,vpar_inits

  def flux_grid(self,ns,ntheta,nzeta,nvpar,s_min=0.01,s_max=1.0,vpar_lb=-V_MAX,vpar_ub=V_MAX):
    """
    Build a 4d grid over the flux coordinates and vpar.
    """
    # use fixed particle locations
    surfaces = np.linspace(s_min,s_max, ns)
    thetas = np.linspace(0, 2*np.pi, ntheta)
    zetas = np.linspace(0,2*np.pi/self.surf.nfp, nzeta)
    #vpars = symlog_grid(vpar_lb,vpar_ub,nvpar)
    vpars = np.linspace(vpar_lb,vpar_ub,nvpar)
    # build a mesh
    [surfaces,thetas,zetas,vpars] = np.meshgrid(surfaces,thetas, zetas,vpars)
    stz_inits = np.zeros((ns*ntheta*nzeta*nvpar, 3))
    stz_inits[:, 0] = surfaces.flatten()
    stz_inits[:, 1] = thetas.flatten()
    stz_inits[:, 2] = zetas.flatten()
    vpar_inits = vpars.flatten()
    return stz_inits,vpar_inits
    
  def surface_grid(self,s_label,ntheta,nzeta,nvpar,vpar_lb=-V_MAX,vpar_ub=V_MAX):
    """
    Builds a grid on a single surface.
    """
    # use fixed particle locations
    # theta is [0,pi] with stellsym
    thetas = np.linspace(0, 2*np.pi, ntheta)
    zetas = np.linspace(0,2*np.pi/self.surf.nfp, nzeta)
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

  def poloidal_grid(self,zeta_label,ns,ntheta,nvpar,s_max=0.98):
    """
    Builds a grid on a poloidal cross section
    """
    # bounds
    vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
    vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)   
    # use fixed particle locations
    surfaces = np.linspace(0.01,s_max, ns)
    thetas = np.linspace(0, 2*np.pi, ntheta)
    vpars = np.linspace(vpar_lb,vpar_ub,nvpar)
    # build a mesh
    [surfaces,thetas,vpars] = np.meshgrid(surfaces,thetas, vpars)
    stz_inits = np.zeros((ns*ntheta*nvpar, 3))
    stz_inits[:, 0] = surfaces.flatten()
    stz_inits[:, 1] = thetas.flatten()
    stz_inits[:, 2] = zeta_label
    vpar_inits = vpars.flatten()
    return stz_inits,vpar_inits

  def sample_volume(self,n_particles):
    """
    Sample the volume using the radial density sampler
    """
    # TODO: use MPI group comm

    # SAA sampling over (s,theta,zeta,vpar)
    s_inits = np.zeros(n_particles)
    theta_inits = np.zeros(n_particles)
    zeta_inits = np.zeros(n_particles)
    vpar_inits = np.zeros(n_particles)
    #if rank == 0:
    if self.mpi.proc0_groups:
      sampler = RadialDensity(1000)
      s_inits = sampler.sample(n_particles)
      # randomly sample theta,zeta,vpar
      # theta is [0,pi] with stellsym
      theta_inits = np.random.uniform(0,2*np.pi,n_particles)
      zeta_inits = np.random.uniform(0,2*np.pi/self.surf.nfp,n_particles)
      vpar_inits = np.random.uniform(-V_MAX,V_MAX,n_particles)
    # broadcast the points
    self.mpi.comm_groups.Bcast(s_inits,root=0)
    self.mpi.comm_groups.Bcast(theta_inits,root=0)
    self.mpi.comm_groups.Bcast(zeta_inits,root=0)
    self.mpi.comm_groups.Bcast(vpar_inits,root=0)
    # stack the samples
    stp_inits = np.vstack((s_inits,theta_inits,zeta_inits)).T
    return stp_inits,vpar_inits

  def sample_surface(self,n_particles,s_label):
    """
    Sample the volume using the radial density sampler
    """
    # TODO: use MPI group comm
    ## divide the particles
    #comm = MPI.COMM_WORLD
    #size = comm.Get_size()
    #rank = comm.Get_rank()
    # SAA sampling over (theta,zeta,vpar) for a fixed surface
    s_inits = s_label*np.ones(n_particles)
    theta_inits = np.zeros(n_particles)
    zeta_inits = np.zeros(n_particles)
    vpar_inits = np.zeros(n_particles)
    #if rank == 0:
    if self.mpi.proc0_groups:
      # randomly sample theta,zeta,vpar
      # theta is [0,pi] with stellsym
      theta_inits = np.random.uniform(0,2*np.pi,n_particles)
      zeta_inits = np.random.uniform(0,2*np.pi/self.surf.nfp,n_particles)
      vpar_inits = np.random.uniform(-V_MAX,V_MAX,n_particles)
    # broadcast the points
    self.mpi.comm_groups.Bcast(theta_inits,root=0)
    self.mpi.comm_groups.Bcast(zeta_inits,root=0)
    self.mpi.comm_groups.Bcast(vpar_inits,root=0)
    # stack the samples
    stp_inits = np.vstack((s_inits,theta_inits,zeta_inits)).T
    return stp_inits,vpar_inits


  def compute_boozer_field(self,x):
    # to save on recomputes
    if np.all(self.x_field == x):
      return self.field,self.bri

    self.surf.x = np.copy(x)
    try:
      self.vmec.run()
    except:
      # VMEC failure!
      return None,None

    # Construct radial interpolant of magnetic field
    bri = BoozerRadialInterpolant(self.vmec, order=self.interpolant_degree,
                      mpol=self.bri_mpol,ntor=self.bri_ntor, enforce_vacuum=True)
    
    # Construct 3D interpolation
    nfp = self.vmec.wout.nfp
    srange = (0, 1, self.interpolant_level)
    thetarange = (0, np.pi, self.interpolant_level)
    zetarange = (0, 2*np.pi/nfp,self.interpolant_level)
    field = InterpolatedBoozerField(bri, degree=self.interpolant_degree, srange=srange, thetarange=thetarange,
                       zetarange=zetarange, extrapolate=True, nfp=nfp, stellsym=True)
    self.field = field
    self.bri = bri
    self.x_field = np.copy(x)
    return field,bri


  def compute_modB(self,field,bri,ns=32,ntheta=32,nphi=32):
    """
    Compute |B| on a grid in Boozer coordinates.
    """
    stz_grid,_ = self.flux_grid(ns,ntheta,nphi,1)
    field.set_points(stz_grid)
    modB = field.modB().flatten()
    return modB

  def compute_mu(self,field,bri,stz_inits,vpar_inits):
    """
    Compute |B| on a grid in Boozer coordinates.
    """
    field.set_points(stz_inits+np.zeros(np.shape(stz_inits)))
    modB = field.modB().flatten()
    vperp_squared = FUSION_ALPHA_SPEED_SQUARED - vpar_inits**2
    mu = vperp_squared/2/modB
    return mu


  def compute_mu_crit(self,field,bri,ns=64,ntheta=64,nphi=64):
    """
    Compute |B| on a grid in Boozer coordinates.
    """
    modB = self.compute_modB(field,bri,ns,ntheta,nphi)
    Bmax = np.max(modB)
    mu_crit = FUSION_ALPHA_SPEED_SQUARED/2/Bmax
    return mu_crit


  # set up the objective
  def compute_confinement_times(self,x,stz_inits,vpar_inits,tmax,field=None,bri=None):
    """
    Trace particles in boozer coordinates according to the vacuum GC 
    approximation using simsopt.

    x: a point describing the current vmec boundary
    stz_inits: (n,3) array of (s,theta,zeta) points
    vpar_inits: (n,) array of vpar values
    tmax: max tracing time
    """
    n_particles = len(vpar_inits)

    if field is None:
      field,bri = self.compute_boozer_field(x)
    if field is None:
      # VMEC failure
      return -np.inf*np.ones(len(stz_inits)) 

    stopping_criteria = [MaxToroidalFluxStoppingCriterion(0.99),MinToroidalFluxStoppingCriterion(0.01)]
     

    # TODO: do we want the group comm here rather than world?
    #comm = MPI.COMM_WORLD

    # trace
    try:
      res_tys, res_zeta_hits = trace_particles_boozer(
          field, 
          stz_inits, 
          vpar_inits, 
          tmax=tmax, 
          mass=ALPHA_PARTICLE_MASS, 
          charge=ALPHA_PARTICLE_CHARGE,
          Ekin=FUSION_ALPHA_PARTICLE_ENERGY, 
          tol=self.tracing_tol, 
          mode='gc_vac',
          comm=self.mpi.comm_groups,
          stopping_criteria=stopping_criteria,
          forget_exact_path=True
          )
    except:
      # tracing failure
      return -np.inf*np.ones(len(stz_inits)) 

    exit_times = np.zeros(n_particles)
    for ii,res in enumerate(res_zeta_hits):

      # check if particle hit stopping criteria
      if len(res) > 0:
        if int(res[0,1]) == -1:
          # particle hit MaxToroidalFluxCriterion
          exit_times[ii] = res[0,0]
        if int(res[0,1]) == -2:
          # particle hit MinToroidalFluxCriterion
          exit_times[ii] = tmax
      else:
        # didnt hit any stopping criteria
        exit_times[ii] = tmax
   

    return exit_times


  #def jacobian_bulk(self,x,stz_inits,vpar_inits,tmax,h=1e-4,method='forward',ns=32,ntheta=32,nzeta=32):
  #  """
  #  Evaluate the objective jacobian by distributing the 
  #  finite difference over the worker groups. 
  #  Use this function if n_partitions > 1 AND
  #  if all worker groups are evaluating the jacobian 
  #  together at the same point, such as in a single
  #  optimization run. Do not use this function for 
  #  concurrent jacobian evaluations at distinct points.
  #  y: input variables describing surface
  #  y: array of size (self.dim_x,)
  #  h: step size
  #  h: float
  #  method: 'forward' or 'central'
  #  method: string
  #  return: jacobian of self.eval
  #  return: array of size (self.dim_F,self.dim_x)
  #  """
  #  assert method in ['forward','central']
  #  #assert self.n_partitions > 1

  #  # build a list of evals for finite difference
  #  h2   = h/2.0
  #  Ep   = x + h2*np.eye(self.dim_x)
  #  X = np.vstack((x.reshape((1,-1)),Ep))
  #  n_points = len(X)

  #  # divide the evals across groups
  #  idxs,counts = divide_work(len(X),self.mpi.ngroups)
  #  idx = idxs[self.mpi.group]

  #  # sizes
  #  n_points_loc = len(X[idx])
  #  n_c_times = len(vpar_inits)
  #  n_modB = ns*ntheta*nzeta
  #  print('number of local points',n_points_loc,'n groups',self.mpi.ngroups)

  #  t0 = time.time()
  #  # do the evals
  #  c_times_loc = np.zeros((n_points_loc,n_c_times))
  #  modB_loc = np.zeros((n_points_loc,n_modB))
  #  for ii,xx in enumerate(X[idx]):
  #    #field,bri = self.compute_boozer_field(xx)
  #    #c_times_loc[ii] = self.compute_confinement_times(xx,stz_inits,vpar_inits,tmax,field=field,bri=bri)
  #    #modB_loc[ii] = self.compute_modB(field,bri,ns=ns,ntheta=ntheta,nphi=nzeta)
  #    c_times_loc[ii] = self.compute_confinement_times(xx,stz_inits,vpar_inits,tmax)

  #  print(c_times_loc)
  #  print('evalulation complete',time.time()-t0)
  #  t0 = time.time()

  #  # barrier
  #  self.mpi.comm_world.Barrier()
  #  # head leader gathers all evals
  #  c_times = np.zeros(n_points*n_c_times)
  #  counts_c_times = n_c_times*np.array(counts).astype(int)
  #  self.mpi.comm_leaders.Gatherv(c_times_loc.flatten(),(c_times,counts_c_times),root=0)
  #  # broadcast to leaders
  #  self.mpi.comm_leaders.Bcast(c_times,root=0)
  #  # broadcast internally within group
  #  self.mpi.comm_groups.Bcast(c_times,root=0)
  #  # reshape to 2D-array
  #  c_times = np.reshape(c_times,(-1,n_c_times))

  #  print('first brodcast complete',time.time()-t0)

  #  # barrier
  #  self.mpi.comm_world.Barrier()
  #  # head leader gathers all evals
  #  modB = np.zeros(n_points*n_modB)
  #  counts_modB = n_modB*np.array(counts).astype(int)
  #  self.mpi.comm_leaders.Gatherv(modB_loc.flatten(),(modB,counts_modB),root=0)
  #  # broadcast to leaders
  #  self.mpi.comm_leaders.Bcast(modB,root=0)
  #  # broadcast internally within group
  #  self.mpi.comm_groups.Bcast(modB,root=0)
  #  # reshape to 2D-array
  #  modB = np.reshape(modB,(-1,n_modB))

  #  print('brodcast complete',time.time()-t0)

  #  # forward difference
  #  jac_c_times = (c_times[1:] - c_times[0])/(h2)
  #  jac_modB = (modB[1:] - modB[0])/(h2)
  #  
  #  return np.copy(jac_c_times.T),np.copy(jac_modB.T)
    

  ## set up the objective
  #def compute_confinement_times_manual(self,x,stz_inits,vpar_inits,tmax,tracing_tol=1e-6):
  #  """
  #  Trace particles in boozer coordinates according to the vacuum GC 
  #  approximation.

  #  vmec: a vmec object
  #  stz_inits: (n,3) array of (s,theta,zeta) points
  #  vpar_inits: (n,) array of vpar values
  #  tmax: max tracing time
  #  """
  #  n_particles = len(vpar_inits)

  #  self.surf.x = np.copy(x)
  #  try:
  #    self.vmec.run()
  #  except:
  #    # VMEC failure!
  #    return -np.inf*np.ones(len(stz_inits)) 

  #  # Construct radial interpolant of magnetic field
  #  order = 3
  #  mpol=ntor=16

  #  import time
  #  print("")
  #  print("forming interpolated boozer field")
  #  t0 = time.time()
  #  bri = BoozerRadialInterpolant(self.vmec, order=order,mpol=mpol,ntor=ntor, enforce_vacuum=True)
  #  print('time',time.time() - t0)
  #  
  #  # Construct 3D interpolation
  #  nfp = self.vmec.wout.nfp
  #  degree = 3
  #  srange = (0, 1, 8)
  #  thetarange = (0, np.pi, 8)
  #  zetarange = (0, 2*np.pi/nfp, 8)
  #  print("")
  #  print("forming interpolated boozer field")
  #  t0 = time.time()
  #  field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)
  #  print('time',time.time() - t0)
  #  print('Error in |B| interpolation', field.estimate_error_modB(1000), flush=True)

  #  # TODO: remove
  #  #field = bri

  #  def stopping_criteria(y):
  #    """
  #    MaxToroidalFluxStoppingCriteria
  #    Equals 0.0 when s = 0.99
  #    Transition from negative to postive indicates
  #    particle ejects.
  #    return s - 0.99
  #    """
  #    return y[0] - 0.99
  #  stopping_criteria.terminal=True
  #  stopping_criteria.direction=1.0
  #   
  #  # divide the work
  #  comm = MPI.COMM_WORLD
  #  size = comm.Get_size()
  #  rank = comm.Get_rank()
  #  work_intervals,work_counts = divide_work(n_particles,size)
  #  my_work = work_intervals[rank]
  #  my_counts = work_counts[rank]

  #  my_times = np.zeros(my_counts)

  #  from scipy.integrate import solve_ivp
  #  from guiding_center_boozer import GuidingCenterVacuumBoozer

  #  print("")
  #  print("tracing")

  #  for idx,point_idx in enumerate(my_work):
  #    
  #    # get the particle
  #    stz = stz_inits[point_idx]
  #    vpar = vpar_inits[point_idx]
  #    y0 = np.append(stz,vpar)
  #    # make a guiding center object
  #    GC = GuidingCenterVacuumBoozer(field,y0)
  #    gc_rhs = GC.GuidingCenterVacuumBoozerRHS
  #    t0 = time.time()

  #    # solve
  #    solve_ivp(gc_rhs,(0,tmax),y0,first_step=1e-6,atol=tracing_tol,events=[stopping_criteria])

  #  print('time',time.time() - t0)

  #  #
  #  #  # get the particle
  #  #  stz = stz_inits[point_idx].reshape((1,-1))
  #  #  vpar = [vpar_inits[point_idx]]
  #  
  #  ## broadcast the values
  #  #exit_times = np.zeros(n_particles)
  #  #comm.Gatherv(my_times,(exit_times,work_counts),root=0)
  #  #comm.Bcast(exit_times,root=0)

  #  return exit_times

  

if __name__ == "__main__":

  #vmec_input = '../vmec_input_files/input.nfp2_QA_cold_high_res'
  vmec_input = '../vmec_input_files/input.nfp4_QH_warm_start_high_res'
  #vmec_input = '../vmec_input_files/input.nfp4_QH_cold_high_res_quasysymmetry_opt'
  minor_radius = 1.7
  major_radius = 2.0*1.7
  target_volavgB = 5.0
  tracer = TraceBoozer(vmec_input,
                      n_partitions=8,
                      max_mode=1,
                      minor_radius=minor_radius,
                      major_radius=major_radius,
                      target_volavgB=target_volavgB,
                      tracing_tol=1e-8,
                      interpolant_degree=1,
                      interpolant_level=10,
                      bri_mpol=16,
                      bri_ntor=16)
  tracer.sync_seeds(0)
  x0 = tracer.x0
  dim_x = tracer.dim_x
  tmax = 1e-4

  # compute the field and 
  field,bri = tracer.compute_boozer_field(x0)
  #mu_crit = tracer.compute_mu_crit(field,bri)
  
  # compute the mirror ratio
  modB = tracer.compute_modB(field,bri,ns=64,ntheta=64,nphi=64)
  print('mirror ratio',np.max(modB)/np.min(modB))

  # gauss quadrature
  from gauss_quadrature import *
  ns = 2
  ntheta=2
  nzeta = 2
  nvpar = 2
  s_lin,s_weights = gauss_quadrature_nodes_coeffs(ns,0.0,1.0)
  theta_lin,theta_weights = gauss_quadrature_nodes_coeffs(ntheta,0,2*np.pi)
  zeta_lin,zeta_weights = gauss_quadrature_nodes_coeffs(nzeta,0,2*np.pi/tracer.surf.nfp)
  vpar_lin,vpar_weights = gauss_quadrature_nodes_coeffs(nvpar,-V_MAX,V_MAX)
  [surfaces,thetas,zetas,vpars] = np.meshgrid(s_lin,theta_lin,zeta_lin,vpar_lin,indexing='ij')
  [w1,w2,w3,w4] = np.meshgrid(s_weights,theta_weights,zeta_weights,vpar_weights,indexing='ij')
  quad_weights = w1*w2*w3*w4
  stz_inits = np.zeros((ns*ntheta*nzeta*nvpar, 3))
  stz_inits[:, 0] = surfaces.flatten()
  stz_inits[:, 1] = thetas.flatten()
  stz_inits[:, 2] = zetas.flatten()
  vpar_inits = vpars.flatten()
  # radial likelihood
  radial_sampler = RadialDensity(1000)
  likelihood = radial_sampler._pdf(stz_inits[:,0])
  likelihood *= (1/(2*np.pi))*(tracer.surf.nfp/(2*np.pi))*(1/(2*V_MAX))
  import time

  t0  = time.time()
  print('tracing')
  x0 = x0 + 1e-3*np.eye(len(x0))[tracer.mpi.group]
  c_times_loc = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
  print('time',time.time() - t0)
  # sizes
  n_c_times = len(vpar_inits)
  n_points = tracer.mpi.ngroups
  counts = np.ones(n_points)
  # barrier
  tracer.mpi.comm_world.Barrier()
  # head leader gathers all evals
  c_times = np.zeros(n_points*n_c_times)
  counts_c_times = n_c_times*np.array(counts).astype(int)
  tracer.mpi.comm_leaders.Gatherv(c_times_loc.flatten(),(c_times,counts_c_times),root=0)
  # broadcast to leaders
  tracer.mpi.comm_leaders.Bcast(c_times,root=0)
  # broadcast internally within group
  tracer.mpi.comm_groups.Bcast(c_times,root=0)
  # reshape to 2D-array
  c_times = np.reshape(c_times,(-1,n_c_times))
  print(c_times)
  quit()

  #jac = tracer.jacobian_bulk(x0,stz_inits,vpar_inits,tmax,h=1e-4,method='forward',ns=32,ntheta=32,nzeta=32)
  #print(jac)
  #quit()

  t0  = time.time()
  print('tracing')
  c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
  print('time',time.time() - t0)

  # quadrature
  feat = c_times
  int0 = feat*likelihood
  int0 = int0.reshape((ns,ntheta,nzeta,nvpar))
  int0 = int0*quad_weights
  res = np.sum(int0)
  print('mean',res)

  """
  Now compute statistics with monte carlo
  """

  # tracing points
  n_particles = 4000
  stz_inits,vpar_inits = tracer.sample_volume(n_particles)
  print('tracing')
  c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
  std_err = np.std(c_times)/np.sqrt(len(c_times))
  mu = np.mean(c_times)
  print('mean',mu-2*std_err,mu+2*std_err)
  print('loss fraction',np.mean(c_times < tmax))
