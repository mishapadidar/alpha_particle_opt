import numpy as np
import sys
import pickle
import subprocess

class SafeEval:
  """
  A class for performing crash-resilient evaluations of simsopt.
  Evalutations are performed by launching mpi processes through a python
  subprocess call, which read and write simsopt evaluation information to a file,
  to be read by the SafeEval class. 

  This class can only perform seriel evals.
  
  The driver script which calls this class should be run with `python3 ...` not `mpiexec`.
  """

  def __init__(self,eval_script = "safe_eval.py",default_F = np.inf,args={},n_cores=1):
    """
    eval_script: string, name of file that performs the evaluation
                  This file, safe_eval.py, contains an evaluation of qh_prob1 and is used
                  as the default eval script. Alternatively you can write your own 
                  eval script by modeling it after the `if __name__=="__main__" portion of 
                  this script.
    default_F: default value to return if an evaluation fails.
    args: a dictionary of args to be passed to the eval function.
    """
    self.eval_script = eval_script
    self.default_F = default_F
    self.args = args
    self.barcode = np.random.randint(int(1e8))
    self.n_cores = n_cores

  def eval(self,yy,requests = ['c_times','asp']):
    """
    Do a single evaluation safely
    
    1. write a pickle file with yy and vmec_input. Write the default function value as well
    2. call the safe evaluation python script to read that file and do the evaluation
       and write back to the file
    3. read the file and return the evaluation.
    """
    # prepare the pickle data
    outdata = {}
    outdata['x'] = yy
    outdata['requests'] = requests
    for r in requests:
      outdata[r] = self.default_F
    outdata['args'] = self.args
    # write a pickle file
    pickle.dump(outdata,open(f'_safe_eval_{self.barcode}.pickle','wb'))
    # subprocess call
    bashCommand = f"mpiexec -n {self.n_cores} python3 {self.eval_script} _safe_eval_{self.barcode}.pickle"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    # read the pickle
    indata = pickle.load(open(f'_safe_eval_{self.barcode}.pickle','rb'))
    return indata
  
  
if __name__=="__main__":  
  """
  Here is the actual function evaluation that is performed.
  """ 
  import sys
  import pickle
  sys.path.append("../../trace")
  sys.path.append("../../utils")
  sys.path.append("../../trace")
  sys.path.append("../../sample")
  #sys.path.append("../../../trace")
  #sys.path.append("../../../utils")
  #sys.path.append("../../../trace")
  #sys.path.append("../../../sample")
  from trace_boozer import TraceBoozer
  from radial_density import RadialDensity
  from constants import V_MAX
  
  # load the point and objective requests
  infile = sys.argv[1]
  indata = pickle.load(open(infile,'rb'))
  x = indata['x']
  requests = indata['requests']
  # load args for the tracing
  vmec_input = indata['args']['vmec_input']
  max_mode = indata['args']['max_mode']
  tracing_tol = indata['args']['tracing_tol'] 
  interpolant_degree = indata['args']['interpolant_degree'] 
  interpolant_level = indata['args']['interpolant_level'] 
  bri_mpol = indata['args']['bri_mpol']
  bri_ntor = indata['args']['bri_ntor']
  major_radius = indata['args']['major_radius']
  minor_radius = indata['args']['minor_radius']
  target_volavgB = indata['args']['target_volavgB']
  tmax = indata['args']['tmax']
  sampling_type = indata['args']['sampling_type']
  sampling_level = indata['args']['sampling_level']
  ns = indata['args']['ns']
  ntheta = indata['args']['ntheta']
  nphi = indata['args']['nphi']
  nvpar = indata['args']['nvpar']

  # build the tracer object
  tracer = TraceBoozer(vmec_input,
                      n_partitions=1,
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

  if 'x0' in requests:
    "special case: return the starting point"
    indata['x0'] = np.copy(tracer.x0)
    pickle.dump(indata,open(infile,"wb"))

  else:
    # update the surface
    tracer.surf.x = np.copy(x)
  
    if 'aspect' in requests:
      # evaluate the objectives
      try:
        asp = tracer.surf.aspect_ratio()
      except:
        asp = np.inf
      if np.isnan(asp):
        asp = np.inf
      indata['aspect'] = asp
  
    if 'c_times' in requests:
      n_particles = ns*ntheta*nphi*nvpar
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
      # sync seeds again
      tracer.sync_seeds()
      # trace
      c_times = tracer.compute_confinement_times(x,stp_inits,vpar_inits,tmax)
      # write to output
      indata['c_times'] = c_times
      indata['mean_c_time'] = np.mean(c_times)
      indata['mean_energy_retained'] = np.mean(3.5*np.exp(-2*c_times/tmax))
  
    pickle.dump(indata,open(infile,"wb"))

