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
  sys.path.append("../../opt")
  sys.path.append("../../../SIMPLE/build/")
else:
  sys.path.append("../../../utils")
  sys.path.append("../../../trace")
  sys.path.append("../../../sample")
  sys.path.append("../../../opt")
  sys.path.append("../../../../SIMPLE/build/")
from trace_simple import TraceSimple
from trace_boozer import TraceBoozer
from eval_wrapper import EvalWrapper
from radial_density import RadialDensity
from constants import V_MAX
from sid_psm import SIDPSM

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""
Optimize a configuration to minimize alpha particle losses

ex.
  mpiexec -n 1 python3 optimize.py random 0.5 mean_energy pdfo 1 nfp4_QH_warm_high_res 10 10 10 10
"""


# tracing parameters
#tmax_list = [1e-4,1e-3,1e-2]
tmax_list = [1e-3,1e-2]
# configuration parmaeters
n_partitions = 1
minor_radius = 1.7
aspect_target = 8.0
major_radius = aspect_target*minor_radius
target_volavgB = 5.0
# optimizer params
maxfev = 300
max_step = 0.1 
min_step = 1e-6
# trace boozer params
tracing_tol = 1e-8
interpolant_degree = 3
interpolant_level  = 8
bri_mpol = 16
bri_ntor = 16


# read inputs
sampling_type = sys.argv[1] # random or grid
sampling_level = sys.argv[2] # "full" or a float surface label
objective_type = sys.argv[3] # mean_energy or mean_time
method = sys.argv[4] # optimization method
max_mode = int(sys.argv[5]) # max mode
vmec_label = sys.argv[6] # vmec file
ns = int(sys.argv[7])  # number of surface samples
ntheta = int(sys.argv[8]) # num theta samples
nphi = int(sys.argv[9]) # num phi samples
nvpar = int(sys.argv[10]) # num vpar samples
assert sampling_type in ['random', "grid"]
assert objective_type in ['mean_energy','mean_time'], "invalid objective type"
assert method in ['pdfo','snobfit','diff_evol','nelder','sidpsm'], "invalid optimiztaion method"

n_particles = ns*ntheta*nphi*nvpar


if vmec_label == "nfp2_QA_cold_high_res":
  vmec_input="../../vmec_input_files/input.nfp2_QA_cold_high_res"
elif vmec_label == "nfp2_QA_high_res":
  vmec_input="../../vmec_input_files/input.nfp2_QA_high_res"
elif vmec_label == "nfp4_QH_warm_high_res":
  vmec_input="../../vmec_input_files/input.nfp4_QH_warm_start_high_res"

if not debug:
  vmec_input="../" + vmec_input

# build a tracer object
#tracer = TraceSimple(vmec_input,n_partitions=n_partitions,max_mode=max_mode,major_radius=major_radius)
tracer = TraceBoozer(vmec_input,
                      n_partitions=n_partitions,
                      max_mode=max_mode,
                      minor_radius=minor_radius,
                      major_radius=major_radius,
                      target_volavgB=target_volavgB,
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


def get_ctimes(x,tmax,sampling_type,sampling_level):
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
  try:
    c_times = tracer.compute_confinement_times(x,stp_inits,vpar_inits,tmax)
  except:
    return np.zeros(len(vpar_inits))
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
  c_times = get_ctimes(x,tmax,sampling_type,sampling_level)
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
  c_times = get_ctimes(x,tmax,sampling_type,sampling_level)
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
    res = pdfo(evw, x0, method='bobyqa',options={'maxfev': maxfev, 'ftarget': ftarget,'rhobeg':rhobeg,'rhoend':rhoend})
    xopt = np.copy(res.x)
  elif method == 'snobfit':
    # snobfit
    bounds = np.vstack((x_lb,x_ub)).T
    res, _ = skq_minimize(evw, x0, bounds, maxfev, method='SnobFit')
    xopt = np.copy(res.optpar)
  elif method == "diff_evol":
    # differential evolution
    bounds = np.vstack((x_lb,x_ub)).T
    popsize = 10 # population is popsize*dim_x individuals
    maxiter = int(maxfev/dim_x/popsize)
    res = differential_evolution(evw,bounds=bounds,popsize=popsize,maxiter=maxiter,x0=x0)
    xopt = np.copy(res.x)
  elif method == "nelder":
    init_simplex = np.zeros((dim_x+1,dim_x))
    init_simplex[0] = np.copy(x0)
    init_simplex[1:] = np.copy(x0 + max_step*np.eye(dim_x))
    def penalty_obj(x):
      obj = evw(x)
      asp = aspect_ratio(x)
      return obj + 1000*np.max([asp-aspect_target,0.0])**2
    # nelder-mead
    xatol = min_step # minimum step size
    res = sp_minimize(penalty_obj,x0,method='Nelder-Mead',
                options={'maxfev':maxfev,'xatol':xatol,'initial_simplex':init_simplex})
    xopt = np.copy(res.x)
  elif method == "sidpsm":
    def penalty_obj(x):
      obj = evw(x)
      asp = aspect_ratio(x)
      return obj + 1000*np.max([asp-aspect_target,0.0])**2
    sid = SIDPSM(penalty_obj,x0,max_eval=maxfev,delta=max_step,delta_min=min_step,delta_max=10*max_step)
    res = sid.solve()
    xopt = np.copyy(res['x'])

  # reset x0 for next iter
  x0 = np.copy(xopt)

  # evaluate the configuration
  c_times_opt = get_ctimes(xopt,tmax,sampling_type,sampling_level) 
  tracer.surf.x = np.copy(xopt)
  aspect_opt = tracer.vmec.aspect()
  if rank == 0:
    print('aspect(xopt)',aspect_opt)
    print('E[c_time(xopt)]',np.mean(c_times_opt))
    print('Loss fraction',np.mean(c_times_opt<tmax))
    print('E[Energy]',np.mean(3.5*np.exp(-2*c_times_opt/tmax)))

  # out of sample performance
  c_times_out_of_sample = get_ctimes(xopt,tmax,"random",sampling_level) # out of sample
  
  # save results
  if rank == 0:
    print(res)
    outfile = f"./data_opt_{vmec_label}_{objective_type}_{sampling_type}_surface_{sampling_level}_tmax_{tmax}_{method}_mmode_{max_mode}.pickle"
    outdata = {}
    outdata['X'] = evw.X
    outdata['FX'] = evw.FX
    outdata['xopt'] = xopt
    outdata['aspect_opt'] = aspect_opt
    outdata['c_times_opt'] = c_times_opt
    outdata['c_times_out_of_sample'] = c_times_out_of_sample
    outdata['major_radius'] = major_radius
    outdata['minor_radius'] =  minor_radius
    outdata['target_volavgB'] = target_volavgB
    outdata['vmec_input'] = vmec_input
    outdata['max_mode'] = max_mode
    outdata['vmec_input'] = vmec_input
    outdata['objective_type'] = objective_type
    outdata['sampling_type'] = sampling_type
    outdata['sampling_level'] = sampling_level
    outdata['method'] = method
    outdata['maxfev'] = maxfev
    outdata['max_step'] = max_step
    outdata['min_step'] = min_step
    outdata['tracing_tol'] = tracing_tol
    outdata['interpolant_degree'] = interpolant_degree
    outdata['interpolant_level'] = interpolant_level
    outdata['bri_mpol'] = bri_mpol
    outdata['bri_ntor'] = bri_ntor
    #outdata['stp_inits'] = stp_inits
    #outdata['vpar_inits'] = vpar_inits
    outdata['tmax'] = tmax
    pickle.dump(outdata,open(outfile,"wb"))
