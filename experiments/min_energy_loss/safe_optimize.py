import numpy as np
import sys
import pickle
from pdfo import pdfo,NonlinearConstraint as pdfo_nlc
from skquant.opt import minimize as skq_minimize
from scipy.optimize import differential_evolution, NonlinearConstraint as sp_nlc, minimize as sp_minimize
from safe_eval import SafeEval
debug = False
if debug:
  sys.path.append("../../utils")
  sys.path.append("../../opt")
else:
  sys.path.append("../../../utils")
  sys.path.append("../../../opt")
from eval_wrapper import EvalWrapper
from sid_psm import SIDPSM


"""
Optimize a configuration to minimize alpha particle losses

ex.
  python3 safe_optimize.py random 0.5 mean_energy pdfo 1 nfp4_QH_warm_high_res 10 10 10 10

For optimal speed use n_cores = 2, and request 2 cores from G2.
timing: 
- 10k particles @ 1e-5 --> 25 sec
- 10k particles @ 1e-4 --> 73 sec
- 2400 particles @ 1e-4 --> 31 sec
"""


# tracing parameters
#tmax_list = [1e-4,1e-3,1e-2]
tmax_list = [1e-3,1e-2]
# configuration parmaeters
n_partitions = 1
n_cores = 2
minor_radius = 1.7
aspect_target = 8.0
major_radius = aspect_target*minor_radius
target_volavgB = 5.0
# optimizer params
maxfev = 600
max_step = 1.2
init_step = 0.3
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

if vmec_label == "nfp2_QA_cold_high_res":
  vmec_input="../../vmec_input_files/input.nfp2_QA_cold_high_res"
elif vmec_label == "nfp2_QA_high_res":
  vmec_input="../../vmec_input_files/input.nfp2_QA_high_res"
elif vmec_label == "nfp4_QH_warm_high_res":
  vmec_input="../../vmec_input_files/input.nfp4_QH_warm_start_high_res"

if not debug:
  vmec_input="../" + vmec_input
if debug:
  n_cores = 1

# build a safe eval object
tracer_args = {}
tracer_args['vmec_input'] = vmec_input 
tracer_args['max_mode'] = max_mode 
tracer_args['tracing_tol'] = tracing_tol
tracer_args['interpolant_degree'] = interpolant_degree
tracer_args['interpolant_level'] = interpolant_level
tracer_args['bri_mpol'] = bri_mpol 
tracer_args['bri_ntor'] = bri_ntor 
tracer_args['major_radius'] = major_radius
tracer_args['minor_radius'] = minor_radius
tracer_args['target_volavgB'] = target_volavgB 
tracer_args['tmax'] = tmax_list[0] # dummy init
tracer_args['sampling_type'] = sampling_type
tracer_args['sampling_level'] = sampling_level
tracer_args['ns'] = ns
tracer_args['ntheta'] = ntheta
tracer_args['nphi'] = nphi
tracer_args['nvpar'] = nvpar
evaluator = SafeEval(eval_script = "safe_eval.py",default_F = np.inf,args=tracer_args,n_cores=n_cores)

# load a starting point
x0 = evaluator.eval([],["x0"])["x0"]
dim_x = len(x0)


# set up the objective
def expected_negative_c_time(x,tmax):
  """
  Negative average confinement time, 
    f(w) = -E[T | w]
  Objective is for minimization.

  x: array,vmec configuration variables
  tmax: float, max trace time
  """
  c_times = evaluator.eval(x,['c_times'])["c_times"]
  if np.any(~np.isfinite(c_times)):
    # vmec failed here; return worst possible value
    res = tmax
  else:
    # minimize negative confinement time
    res = tmax-np.mean(c_times)
  loss_frac = np.mean(c_times<tmax)
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
  c_times = evaluator.eval(x,['c_times'])["c_times"]
  if np.any(~np.isfinite(c_times)):
    # vmec failed here; return worst possible value
    E = 3.5
    res = E
  else:
    # minimize energy retained by particle
    E = 3.5*np.exp(-2*c_times/tmax)
    res = np.mean(E)
  loss_frac = np.mean(c_times<tmax)
  print('obj:',res,'E[tau]',np.mean(c_times),'std[tau]',np.std(c_times),'P(loss):',loss_frac)
  sys.stdout.flush()
  return res


#x_lb = np.array([-0.9404309, -0.83193019, 0.28720149,-1.00806875,-0.79377344,-0.77146669, 0.16415827,-0.86378062])
#x_ub = np.array([1.04723777,0.69724909,1.77739318,0.92464055,0.77225157,0.84754077,1.87259283,0.74216392])


for tmax in tmax_list:
  print(f"optimizing with tmax = {tmax}")

  tracer_args['tmax'] = tmax_list[0]
  evaluator = SafeEval(eval_script = "safe_eval.py",default_F = np.inf,args=tracer_args,n_cores=n_cores)

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
    rhobeg = init_step
    rhoend = min_step
    res = pdfo(evw, x0, method='bobyqa',options={'maxfev': maxfev, 'ftarget': ftarget,'rhobeg':rhobeg,'rhoend':rhoend})
    xopt = np.copy(res.x)
  #elif method == 'snobfit':
  #  # snobfit
  #  bounds = np.vstack((x_lb,x_ub)).T
  #  res, _ = skq_minimize(evw, x0, bounds, maxfev, method='SnobFit')
  #  xopt = np.copy(res.optpar)
  #elif method == "diff_evol":
  #  # differential evolution
  #  bounds = np.vstack((x_lb,x_ub)).T
  #  popsize = 10 # population is popsize*dim_x individuals
  #  maxiter = int(maxfev/dim_x/popsize)
  #  res = differential_evolution(evw,bounds=bounds,popsize=popsize,maxiter=maxiter,x0=x0)
  #  xopt = np.copy(res.x)
  elif method == "nelder":
    init_simplex = np.zeros((dim_x+1,dim_x))
    init_simplex[0] = np.copy(x0)
    init_simplex[1:] = np.copy(x0 + init_step*np.eye(dim_x))
    def penalty_obj(x):
      obj = evw(x)
      asp = evaluator.eval(x,['aspect'])['aspect']
      print('aspect',asp)
      return obj + 1000*np.max([asp-aspect_target,0.0])**2
    # nelder-mead
    xatol = min_step # minimum step size
    res = sp_minimize(penalty_obj,x0,method='Nelder-Mead',
                options={'maxfev':maxfev,'xatol':xatol,'initial_simplex':init_simplex})
    xopt = np.copy(res.x)
  elif method == "sidpsm":
    def penalty_obj(x):
      obj = evw(x)
      if objective_type == "mean_energy" and obj >= 3.5:
        return np.inf
      elif objective_type == "mean_time" and obj >=tmax:
        return np.inf
      asp = evaluator.eval(x,['aspect'])['aspect']
      print('aspect',asp)
      return obj + 1000*np.max([asp-aspect_target,0.0])**2
    sid = SIDPSM(penalty_obj,x0,max_eval=maxfev,delta=init_step,delta_min=min_step,delta_max=max_step)
    res = sid.solve()
    xopt = np.copy(res['x'])

  # reset x0 for next iter
  x0 = np.copy(xopt)

  # evaluate the configuration
  c_times_opt = evaluator.eval(xopt,['c_times'])["c_times"]
  aspect_opt = evaluator.eval(xopt,['aspect'])["aspect"]
  print('aspect(xopt)',aspect_opt)
  print('E[c_time(xopt)]',np.mean(c_times_opt))
  print('Loss fraction',np.mean(c_times_opt<tmax))
  print('E[Energy]',np.mean(3.5*np.exp(-2*c_times_opt/tmax)))

  # out of sample performance
  temp_args = tracer_args.copy()
  temp_args['sampling_type'] = "random"
  temp_ev = SafeEval(eval_script = "safe_eval.py",default_F = np.inf,args=temp_args,n_cores=n_cores)
  c_times_out_of_sample = temp_ev.eval(xopt,['c_times'])['c_times']
  
  # save results
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
