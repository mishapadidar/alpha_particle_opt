import numpy as np
from mpi4py import MPI
import sys
import pickle
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
from finite_difference import finite_difference
from radial_density import RadialDensity
from eval_wrapper import EvalWrapper
from constants import V_MAX

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# resampling parameters
n_particles = 1000
#sampling_level = "full" # full, or float surface label
# manually override vmec input
vmec_input = "../../vmec_input_files/input.nfp2_QA_cold_high_res"

if not debug:
  vmec_input="../" + vmec_input

# load the point
infile = sys.argv[1] 
indata = pickle.load(open(infile,"rb"))
xopt = indata['xopt']
#vmec_input = indata['vmec_input']
major_radius = indata['major_radius']
max_mode = indata['max_mode']
objective_type = indata['objective_type'] 
tmax = indata['tmax']
sampling_level = indata['sampling_level']


# build a tracer object
n_partitions = 1
tracer = TraceSimple(vmec_input,n_partitions=n_partitions,max_mode=max_mode,major_radius=major_radius)
x0 = tracer.x0
tracer.sync_seeds()
dim_x = tracer.dim_x

# sample some points
if sampling_level == "full":
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

else:
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

# compute the confinement times
import time
t0 = time.time()
c_times = tracer.compute_confinement_times(xopt,stp_inits,vpar_inits,tmax)
print(time.time() - t0)

# dump a pickle file with the trace times
if rank == 0:
  out_of_sample = {}
  out_of_sample['x'] = xopt
  out_of_sample['stp_inits'] = stp_inits
  out_of_sample['vpar_inits'] = vpar_inits
  out_of_sample['c_times'] = c_times
  out_of_sample['tmax'] = tmax
  out_of_sample['n_particles'] = n_particles
  # append the new information to the indata
  indata['out_of_sample'] = out_of_sample
  pickle.dump(indata,open(infile,"wb"))



"""
Evaluate the gradients with finite difference
"""

# finite difference parameter
h_fdiff = 1e-4

# use the original particle positions
stp_inits = indata['stp_inits'] 
vpar_inits = indata['vpar_inits'] 
n_particles = len(stp_inits)
  
def get_ctimes(x):
  return tracer.compute_confinement_times(x,stp_inits,vpar_inits,tmax)
evw = EvalWrapper(get_ctimes,dim_x,n_particles)

# set up the objective
def expected_negative_c_time(x,tmax):
  """
  Negative average confinement time, 
    f(w) = -E[T | w]
  Objective is for minimization.

  x: array,vmec configuration variables
  tmax: float, max trace time
  """
  c_times = evw(x)
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
  c_times = evw(x)
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

# define the objective with tmax
if objective_type == "mean_energy":
  objective = lambda x: expected_energy_retained(x,tmax)
  ftarget = 3.5*np.exp(-2)
elif objective_type == "mean_time":
  objective = lambda x: expected_negative_c_time(x,tmax)
  ftarget = 0.0

def aspect(x):
    tracer.surf.x = np.copy(x)
    asp = tracer.surf.aspect_ratio()
    sys.stdout.flush()
    return asp

# aspect ratio gradient
grad_asp = finite_difference(aspect,xopt,h=1e-4)
if rank == 0:
  print("")
  print('gradient_aspect(xopt)')
  print(grad_asp)
  print(np.linalg.norm(grad_asp))

# evaluate the gradient 
grad = finite_difference(objective,xopt,h=h_fdiff)
if rank == 0:
  print("")
  print('gradient(xopt)')
  print(grad)
  print(np.linalg.norm(grad))
# get the gradient evals
X_opt = evw.X
FX_opt = evw.FX

# check the lagrange condition
lam = -grad @ grad_asp /(grad_asp @ grad_asp)
grad_lag = grad+lam*grad_asp
if rank == 0:
  print("")
  print('lagrange multiplier',lam)
  print('grad lagrangian')
  print(grad_lag)
  print(np.linalg.norm(grad_lag))

# evaluate the gradient 
grad0 = finite_difference(objective,x0,h=h_fdiff)
if rank == 0:
  print("")
  print('grad(x0)')
  print(grad0)
  print(np.linalg.norm(grad0))

# make a pickle file
if rank == 0:
  gradient_evals = {}
  gradient_evals['X'] = X_opt
  gradient_evals['FX'] = FX_opt
  gradient_evals['xopt'] = xopt
  gradient_evals['grad_xopt'] = grad
  gradient_evals['grad_aspect_xopt'] = grad_asp
  gradient_evals['grad_x0'] = grad0
  gradient_evals['h_fdiff'] = h_fdiff
  indata['gradient_at_xopt'] = gradient_evals
  pickle.dump(outdata,open(outfile,"wb"))

