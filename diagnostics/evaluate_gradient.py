import numpy as np
from mpi4py import MPI
import sys
import pickle
sys.path.append("../utils")
sys.path.append("../trace")
sys.path.append("../sample")
sys.path.append("../../SIMPLE/build/")
from trace_simple import TraceSimple
from finite_difference import finite_difference

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# finite difference parameter
h_fdiff = 1e-2
# manually override vmec input
vmec_input = "../vmec_input_files/input.nfp2_QA_cold_high_res"


# load the point
infile = sys.argv[1] 
indata = pickle.load(open(infile,"rb"))
xopt = indata['xopt']
stp_inits = indata['stp_inits'] 
vpar_inits = indata['vpar_inits'] 
#vmec_input = indata['vmec_input']
major_radius = indata['major_radius']
max_mode = indata['max_mode']
objective_type = indata['objective_type'] 
tmax = indata['tmax']


# build a tracer object
n_partitions = 1
tracer = TraceSimple(vmec_input,n_partitions=n_partitions,max_mode=max_mode,major_radius=major_radius)
x0 = tracer.x0
tracer.sync_seeds()
dim_x = tracer.dim_x

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

