import numpy as np
from mpi4py import MPI
import sys
import pickle
from pdfo import pdfo,NonlinearConstraint
debug = False
if debug:
  sys.path.append("../../utils")
  sys.path.append("../../trace")
  sys.path.append("../../sample")
  sys.path.append("../../../SIMPLE/build/")
else:
  sys.path.append("../../../utils")
  sys.path.append("../../../trace")
  sys.path.append("../../../sample")
  sys.path.append("../../../../SIMPLE/build/")
from trace_simple import TraceSimple
from eval_wrapper import EvalWrapper
from radial_density import RadialDensity
from constants import V_MAX

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""
Optimize a configuration to minimize alpha particle losses

ex.
  mpiexec -n 1 python3 optimize.py SAA 0.5 mean_energy 10 10 10 10
"""


# tracing parameters
#tmax_list = [1e-5,1e-4,1e-3]
tmax_list = [1e-4,1e-3]
# configuration parmaeters
n_partitions = 1
max_mode = 1
major_radius = 5
#vmec_input="../../vmec_input_files/input.nfp2_QA_cold_high_res"
vmec_input="../../vmec_input_files/input.nfp2_QA_cold"
# optimizer params
maxfev = 1000
rhobeg = 0.1
rhoend = 1e-5

if not debug:
  vmec_input="../" + vmec_input

# read inputs
sampling_type = sys.argv[1] # SAA or grid
sampling_level = sys.argv[2] # "full" or a float surface label
objective_type = sys.argv[3] # mean_energy or mean_time
ns = int(sys.argv[4])  # number of surface samples
ntheta = int(sys.argv[5]) # num theta samples
nphi = int(sys.argv[6]) # num phi samples
nvpar = int(sys.argv[7]) # num vpar samples
assert sampling_type in ['SAA' or "grid"]
assert objective_type in ['mean_energy','mean_time'], "invalid objective type"

n_particles = ns*ntheta*nphi*nvpar

# build a tracer object
tracer = TraceSimple(vmec_input,n_partitions=n_partitions,max_mode=max_mode,major_radius=major_radius)
tracer.sync_seeds()
x0 = tracer.x0
dim_x = tracer.dim_x

if sampling_type == "grid" and sampling_level == "full":
  # grid over (s,theta,phi,vpar)
  stp_inits,vpar_inits = tracer.flux_grid(ns,ntheta,nzeta,nvpar)
elif sampling_type == "grid":
  # grid over (theta,phi,vpar) for a fixed surface label
  s_label = float(sampling_level)
  stp_inits,vpar_inits = tracer.surface_grid(s_label,ntheta,nzeta,nvpar)
elif sampling_type == "SAA" and sampling_level == "full":
  # SAA sampling over (s,theta,phi,vpar)
  s_inits = np.zeros(n_particles)
  theta_inits = np.zeros(n_particles)
  phi_inits = np.zeros(n_particles)
  vpar_inits = np.zeros(n_particles)
  if rank == 0:
    sampler = RadialDensity(1000)
    s_inits = sampler.sample(n_particles)
    # randomly sample theta,phi,vpar
    theta_inits = np.random.uniform(0,1,n_particles)
    phi_inits = np.random.uniform(0,1,n_particles)
    vpar_inits = np.random.uniform(-V_MAX,V_MAX,n_particles)
  # broadcast the points
  comm.Bcast(s_inits,root=0)
  comm.Bcast(theta_inits,root=0)
  comm.Bcast(phi_inits,root=0)
  comm.Bcast(vpar_inits,root=0)
  # stack the samples
  stp_inits = np.vstack((s_inits,theta_inits,phi_inits)).T

elif sampling_type == "SAA":
  # SAA sampling over (theta,phi,vpar) for a fixed surface
  s_inits = float(sampling_level)*np.ones(n_particles)
  theta_inits = np.zeros(n_particles)
  phi_inits = np.zeros(n_particles)
  vpar_inits = np.zeros(n_particles)
  if rank == 0:
    # randomly sample theta,phi,vpar
    theta_inits = np.random.uniform(0,1,n_particles)
    phi_inits = np.random.uniform(0,1,n_particles)
    vpar_inits = np.random.uniform(-V_MAX,V_MAX,n_particles)
  # broadcast the points
  comm.Bcast(theta_inits,root=0)
  comm.Bcast(phi_inits,root=0)
  comm.Bcast(vpar_inits,root=0)
  # stack the samples
  stp_inits = np.vstack((s_inits,theta_inits,phi_inits)).T


# double check
assert n_particles == len(stp_inits), "n_particles does not equal length of points"
# sync seeds again
tracer.sync_seeds()

# aspect constraint
def aspect_ratio(x):
  """
  Compute the aspect ratio
  """
  # update the surface
  tracer.surf.x = np.copy(x)

  # evaluate the objectives
  try:
    asp = tracer.surf.aspect_ratio()
  except:
    asp = np.inf

  # catch partial failures
  if np.isnan(asp):
    asp = np.inf

  return asp
aspect_lb = 4.0
aspect_ub = 8.0
aspect_constraint = NonlinearConstraint(aspect_ratio, aspect_lb,aspect_ub)

# wrap the tracer object
def get_ctimes(x):
  return tracer.compute_confinement_times(x,stp_inits,vpar_inits,tmax)

# set up the objective
def expected_negative_c_time(x,tmax):
  """
  Negative average confinement time, 
    f(w) = -E[T | w]
  Objective is for minimization.

  x: array,vmec configuration variables
  tmax: float, max trace time
  """
  #c_times = evw(x)
  c_times = get_ctimes(x)
  if np.any(~np.isfinite(c_times)):
    # vmec failed here; return worst possible value
    res = tmax
  else:
    # minimize negative confinement time
    res = tmax-np.mean(c_times)
  loss_frac = np.mean(c_times<tmax)
  if rank == 0:
    print('obj:',res,'E[tau]',np.mean(c_times),'P(loss):',loss_frac)
  sys.stdout.flush()
  return res

def expected_energy_retained(x,tmax):
  """
  Expected energy retained by a particle before ejecting
    f(w) = E[3.5exp(-2T/tau_s) | w]
  We use tmax, the max trace time, instead of the slowing down
  time tau_s to improve the conditioning of the objective.
  
  Objective is for minimization.

  x: array,vmec configuration variables
  tmax: float, max trace time
  """
  #c_times = evw(x)
  c_times = get_ctimes(x)
  if np.any(~np.isfinite(c_times)):
    # vmec failed here; return worst possible value
    E = 3.5
    res = E
  else:
    # minimize energy retained by particle
    E = 3.5*np.exp(-2*c_times/tmax)
    res = np.mean(E)
  loss_frac = np.mean(c_times<tmax)
  if rank == 0:
    print('obj:',res,'E[tau]',np.mean(c_times),'P(loss):',loss_frac)
  sys.stdout.flush()
  return res



for tmax in tmax_list:
  if rank == 0:
    print(f"optimizing with tmax = {tmax}")

  # define the objective with tmax
  if objective_type == "mean_energy":
    objective = lambda x: expected_energy_retained(x,tmax)
    ftarget = 3.5*np.exp(-2)
  elif objective_type == "mean_time":
    objective = lambda x: expected_negative_c_time(x,tmax)
    ftarget = 0.0
  evw = EvalWrapper(objective,dim_x,1)

  # optimize
  res = pdfo(evw, x0, method='cobyla', constraints=[aspect_constraint],options={'maxfev': maxfev, 'ftarget': ftarget,'rhobeg':rhobeg,'rhoend':rhoend})
  xopt = res.x

  # reset x0 for next iter
  x0 = np.copy(xopt)

  # evaluate the configuration
  aspect_opt = aspect_ratio(xopt)
  c_times_opt = get_ctimes(xopt)
  
  # save results
  if rank == 0:
    print(res)
    outfile = f"./data_opt_{objective_type}_surface_{sampling_type}_tmax_{tmax}.pickle"
    outdata = {}
    outdata['X'] = evw.X
    outdata['FX'] = evw.FX
    outdata['xopt'] = xopt
    outdata['aspect_opt'] = aspect_opt
    outdata['c_times_opt'] = c_times_opt
    outdata['major_radius'] = major_radius
    outdata['vmec_input'] = vmec_input
    outdata['max_mode'] = max_mode
    outdata['vmec_input'] = vmec_input
    outdata['objective_type'] = objective_type
    outdata['sampling_type'] = sampling_type
    outdata['sampling_level'] = sampling_level
    outdata['stp_inits'] = stp_inits
    outdata['vpar_inits'] = vpar_inits
    outdata['tmax'] = tmax
    pickle.dump(outdata,open(outfile,"wb"))
