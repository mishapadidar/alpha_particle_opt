import numpy as np
from mpi4py import MPI
import sys
import pickle
from pdfo import pdfo
debug = False
if debug:
  sys.path.append("../../utils")
  sys.path.append("../../trace")
  sys.path.append("../../../SIMPLE/build/")
else:
  sys.path.append("../../../utils")
  sys.path.append("../../../trace")
  sys.path.append("../../../../SIMPLE/build/")
from trace_simple import TraceSimple
from eval_wrapper import EvalWrapper

rank = MPI.COMM_WORLD.Get_rank()
"""
Compute particle losses with MC tracing
"""


quantile = 0.5
ns = 10
ntheta=nzeta = 10
nvpar=10
tmax = 1e-4
n_partitions = 1
max_mode = 1
major_radius = 5
vmec_input="../../vmec_input_files/input.nfp2_QA_cold_high_res"
# bounding away from sensitive paths
#vpar_lb = -0.5e7
#vpar_ub = 0.5e7

if not debug:
  vmec_input="../" + vmec_input

# optimizer params
maxfev = 1000
rhobeg = 1.0
rhoend = 1e-6
ftarget = -tmax


# build a tracer object
tracer = TraceSimple(vmec_input,n_partitions=n_partitions,max_mode=max_mode,major_radius=major_radius)
tracer.sync_seeds(0)
x0 = tracer.x0
dim_x = tracer.dim_x

s_label = sys.argv[1]
if s_label == "full":
  #stz_inits,vpar_inits = tracer.flux_grid(ns,ntheta,nzeta,nvpar,vpar_lb=vpar_lb,vpar_ub=vpar_ub)
  stz_inits,vpar_inits = tracer.flux_grid(s_label,ntheta,nzeta,nvpar)
else:
  s_label = float(s_label)
  #stz_inits,vpar_inits = tracer.surface_grid(s_label,ntheta,nzeta,nvpar,vpar_lb=vpar_lb,vpar_ub=vpar_ub)
  stz_inits,vpar_inits = tracer.surface_grid(s_label,ntheta,nzeta,nvpar)
n_particles = len(stz_inits)

# wrap the tracer
def get_ctimes(x):
  return tracer.compute_confinement_times(x,stz_inits,vpar_inits,tmax)
evw = EvalWrapper(get_ctimes,dim_x,n_particles)


# set up the objective
def objective(x):
  """
  maximize a quantile of confinement time
  """
  c_times = evw(x)
  if np.any(~np.isfinite(c_times)):
    # vmec failed here
    res = 0 # worst possible trace time
  else:
    res = np.quantile(c_times,q=quantile)
  loss_frac = np.mean(c_times<tmax)
  if rank == 0:
    print('Quantile[times]',res,'P(loss):',loss_frac)
  sys.stdout.flush()
  return - res # negative for minimization

# optimize
res = pdfo(objective, x0, method='bobyqa', options={'maxfev': maxfev, 'ftarget': ftarget,'rhobeg':rhobeg,'rhoend':rhoend})
#from scipy.optimize import minimize
#res = minimize(objective, x0, method='L-BFGS-B', options={'maxfun': maxfev, 'gtol':1e-8,
#'eps':1e-2})
xopt = res.x

if rank == 0:
  print(res)

  outfile = f"./data_opt_surface_{s_label}_tmax_{tmax}.pickle"
  outdata = {}
  outdata['X'] = evw.X
  outdata['FX'] = evw.FX
  outdata['xopt'] = xopt
  outdata['major_radius'] = major_radius
  outdata['vmec_input'] = vmec_input
  outdata['max_mode'] = max_mode
  outdata['vmec_input'] = vmec_input
  outdata['s_label'] = s_label
  outdata['tmax'] = tmax
  pickle.dump(outdata,open(outfile,"wb"))
