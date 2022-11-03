import numpy as np
from mpi4py import MPI
import sys
import pickle
from pdfo import pdfo,NonlinearConstraint as pdfo_nlc
from skquant.opt import minimize as skq_minimize
from scipy.optimize import differential_evolution, NonlinearConstraint as sp_nlc, minimize as sp_minimize
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
from trace_boozer import TraceBoozer
from eval_wrapper import EvalWrapper
from radial_density import RadialDensity
from constants import V_MAX

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""
Optimize a configuration to minimize alpha particle losses

ex.
  mpiexec -n 1 python3 optimize.py random 0.5 mean_energy pdfo 1 5 10 10 10 10
"""


# tracing parameters
tmax_list = [1e-5,1e-4,1e-3]
# configuration parmaeters
n_partitions = 1
vmec_input="../../vmec_input_files/input.nfp2_QA_cold_high_res"
#vmec_input="../../vmec_input_files/input.nfp2_QA_cold"
# optimizer params
maxfev = 2000
max_step = 0.1
min_step = 1e-6
# trace boozer params
tracing_tol = 1e-8
interpolant_degree = 3
interpolant_level  = 8
bri_mpol = 16
bri_ntor = 16

if not debug:
  vmec_input="../" + vmec_input

# read inputs
sampling_type = sys.argv[1] # random or grid
sampling_level = sys.argv[2] # "full" or a float surface label
objective_type = sys.argv[3] # mean_energy or mean_time
method = sys.argv[4] # optimization method
max_mode = int(sys.argv[5]) # max mode
major_radius = float(sys.argv[6]) # major radius
ns = int(sys.argv[7])  # number of surface samples
ntheta = int(sys.argv[8]) # num theta samples
nphi = int(sys.argv[9]) # num phi samples
nvpar = int(sys.argv[10]) # num vpar samples
assert sampling_type in ['random', "grid"]
assert objective_type in ['mean_energy','mean_time'], "invalid objective type"
assert method in ['pdfo','snobfit','diff_evol','nelder'], "invalid optimiztaion method"

n_particles = ns*ntheta*nphi*nvpar

# build a tracer object
#tracer = TraceSimple(vmec_input,n_partitions=n_partitions,max_mode=max_mode,major_radius=major_radius)
tracer = TraceBoozer(vmec_input,
                      n_partitions=n_partitions,
                      max_mode=max_mode,
                      major_radius=major_radius,
                      tracing_tol=tracing_tol,
                      interpolant_degree=interpolant_degree,
                      interpolant_level=interpolant_level,
                      bri_mpol=bri_mpol,
                      bri_ntor=bri_ntor)
tracer.sync_seeds()
x0 = tracer.x0
dim_x = tracer.dim_x


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


def get_ctimes(x,tmax):
  # sync seeds again
  tracer.sync_seeds()
  if sampling_type == "grid" and sampling_level == "full":
    # grid over (s,theta,phi,vpar)
    stp_inits,vpar_inits = tracer.flux_grid(ns,ntheta,nphi,nvpar)
  elif sampling_type == "grid":
    # grid over (theta,phi,vpar) for a fixed surface label
    s_label = float(sampling_level)
    stp_inits,vpar_inits = tracer.surface_grid(s_label,ntheta,nphi,nvpar)
  elif sampling_type == "random" and sampling_level == "full":
    # volume sampling
    stp_inits,vpar_inits = tracer.sample_volume(n_particles)
  elif sampling_type == "random":
    # surface sampling
    s_label = float(sampling_level)
    stp_inits,vpar_inits = tracer.sample_surface(n_particles,s_label)
  # trace
  c_times = tracer.compute_confinement_times(x,stp_inits,vpar_inits,tmax)
  return c_times

# set up the objective
def expected_negative_c_time(x,tmax):
  """
  Negative average confinement time, 
    f(w) = -E[T | w]
  Objective is for minimization.

  x: array,vmec configuration variables
  tmax: float, max trace time
  """
  c_times = get_ctimes(x,tmax)
  if np.any(~np.isfinite(c_times)):
    # vmec failed here; return worst possible value
    res = tmax
  else:
    # minimize negative confinement time
    res = tmax-np.mean(c_times)
  loss_frac = np.mean(c_times<tmax)
  if rank == 0:
    print('obj:',res,'E[tau]',np.mean(c_times),'std[tau]',np.std(c_times),'P(loss):',loss_frac)
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
  c_times = get_ctimes(x,tmax)
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
    print('obj:',res,'E[tau]',np.mean(c_times),'std[tau]',np.std(c_times),'P(loss):',loss_frac)
  sys.stdout.flush()
  return res


# TODO: move this to its own script
def find_vmec_bounds(lb_k,ub_k,batch_size = 100,maxiter=10,growth_factor=1.3,prob_lb=0.5):
  """
  Find bounds on the decision variables that contain the region
  where vmec passes. Will grow to contain the entire feasible region.

  lb,ub: (dim_x,) array, initial guess for the bounds
    can guess x0 +- small number
  batch_size: number of evaluations to take per update
  maxiter: number of iterations
  growth_factor: amount to increase the region per iteration.
  """
  for kk in range(maxiter):
    tracer.sync_seeds()

    # evaluate bounds
    X_k = np.random.uniform(lb,ub,(batch_size,dim_x))
    Y_k = np.zeros(batch_size,dtype=bool)
    for ii,x in enumerate(X_k):
      res = True
      tracer.surf.x = np.copy(x)
      try:
        tracer.vmec.run()
      except:
        # VMEC failure!
        res = False
      Y_k[ii] = res
    # get the passing samples
    X_pass = X[Y]
    # proposed new bounds
    lb_n = np.min(X_pass,axis=0)
    ub_n = np.max(X_pass,axis=0)
    # update bounds
    lb   = np.minimum(lb,lb_n)
    ub   = np.maximum(ub,ub_n)
    print("")
    print('success frac',np.mean(Y))
    print(lb)
    print(ub)
    if kk < maxiter -1 :
      # enlarge
      diff = (ub-lb)/4
      ub = np.copy(ub + growth_factor*diff)
      lb = np.copy(lb - growth_factor*diff)
  return lb,ub
#lb = x0 - 1e-2
#ub = x0 + 1e-2
#x_lb,x_ub = find_vmec_bounds(lb,ub,batch_size=100,maxiter=8,growth_factor=2.0)
x_lb = np.array([-0.9404309, -0.83193019, 0.28720149,-1.00806875,-0.79377344,-0.77146669, 0.16415827,-0.86378062])
x_ub = np.array([1.04723777,0.69724909,1.77739318,0.92464055,0.77225157,0.84754077,1.87259283,0.74216392])
  


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
  if method == "pdfo":
    rhobeg = max_step
    rhoend = min_step
    aspect_constraint = pdfo_nlc(aspect_ratio, aspect_lb,aspect_ub)
    res = pdfo(evw, x0, method='cobyla', constraints=[aspect_constraint],options={'maxfev': maxfev, 'ftarget': ftarget,'rhobeg':rhobeg,'rhoend':rhoend})
    xopt = np.copy(res.x)
  elif method == 'snobfit':
    # snobfit
    bounds = np.vstack((x_lb,x_ub)).T
    def penalty_objective(x):
      obj = evw(x)
      con = np.max([aspect_ratio(x) - aspect_ub,0.0])**2 + np.max([aspect_lb - aspect_ratio(x),0.0])**2
      pen = 1e3
      return obj + pen*con
    res, _ = skq_minimize(penalty_objective, x0, bounds, maxfev, method='SnobFit')
    xopt = np.copy(res.optpar)
  elif method == "diff_evol":
    # differential evolution
    bounds = np.vstack((x_lb,x_ub)).T
    constraints = [sp_nlc(aspect_ratio,aspect_lb,aspect_ub)]
    popsize = 10 # population is popsize*dim_x individuals
    maxiter = int(maxfev/dim_x/popsize)
    res = differential_evolution(evw,bounds=bounds,popsize=popsize,maxiter=maxiter,x0=x0,constraints = constraints)
    xopt = np.copy(res.x)
  elif method == "nelder":
    def extreme_barrier(x):
      obj = evw(x)
      con1 = aspect_ratio(x) - aspect_ub
      con2 = aspect_lb - aspect_ratio(x) 
      if (con1 > 0) or (con2 > 0):
        return np.inf
      else:
        return obj
    # nelder-mead
    xatol = min_step # minimum step size
    res = sp_minimize(extreme_barrier,x0,method='Nelder-Mead',options={'maxfev':maxfev,'xatol':xatol})
    xopt = np.copy(res.x)

  # reset x0 for next iter
  x0 = np.copy(xopt)

  # evaluate the configuration
  aspect_opt = aspect_ratio(xopt)
  c_times_opt = get_ctimes(xopt,tmax)
  if rank == 0:
    print('aspect(xopt)',aspect_opt)
    print('E[c_time(xopt)]',np.mean(c_times_opt))
    print('Loss fraction',np.mean(c_times_opt<tmax))
    print('E[Energy]',np.mean(3.5*np.exp(-2*c_times_opt/tmax)))
  
  # save results
  if rank == 0:
    print(res)
    outfile = f"./data_opt_{objective_type}_{sampling_type}_surface_{sampling_level}_tmax_{tmax}_{method}.pickle"
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
    outdata['method'] = method
    #outdata['stp_inits'] = stp_inits
    #outdata['vpar_inits'] = vpar_inits
    outdata['tmax'] = tmax
    pickle.dump(outdata,open(outfile,"wb"))
