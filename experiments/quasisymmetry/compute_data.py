import numpy as np
from mpi4py import MPI
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt._core import Optimizable
import sys
import pickle
sys.path.append("../../utils")
sys.path.append("../../trace")
sys.path.append("../../sample")
from finite_difference import forward_difference
from trace_boozer import TraceBoozer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""
Compute the value of d @ H @ d where H is the hessian of the energy objective
and d is the gradient of the quasisymmetry objective.

This can be computed with second order central difference on the energy objective
along the direction d.

Run with
  mpiexec -n 1 python3 compute_data.py
"""

# load a configuration
config = "A" # A or B
#config = "B" # A or B

if config == "A":
  infile = "./data_opt_nfp4_phase_one_aspect_7.0_iota_0.89_mean_energy_SAA_surface_full_tmax_0.01_bobyqa_mmode_3_iota_None.pickle"
elif config == "B":
  infile = "./data_opt_nfp4_phase_one_aspect_7.0_iota_-1.043_mean_energy_SAA_surface_0.25_tmax_0.01_bobyqa_mmode_3_iota_None.pickle"

indata = pickle.load(open(infile,"rb"))
vmec_input = indata['vmec_input']
vmec_input = vmec_input[3:] # remove the ../
aspect_target = indata['aspect_target']
major_radius = indata['major_radius']
target_volavgB = indata['target_volavgB']
max_mode = indata['max_mode']
x0 = indata['xopt']
if rank == 0:
  print(vmec_input)

# list of mn params
mn_list = [(0,1),(1,4),(1,-4)]
n_obj = len(mn_list)

# configuration params
#vmec_input = "../../vmec_input_files/input.nfp4_QH_warm_start_high_res"
#aspect_target = 7.0
#major_radius = 1.7*aspect_target
#target_volavgB = 5.0
#max_mode = 3 


# tracing params
s_label = 0.25 # 0.25 or full
tmax = 0.01 
n_particles = 10000 
h_fdiff_x = 1e-3 # finite difference
h_fdiff_qs = 1e-4 # finite difference quasisymmetry

# step sizes for use in finite differences
step_sizes = h_fdiff_x*np.array([4.0,2.0,1.0,0.5,0.25,0.225,0.2,0.017,0.15,0.1,0.085,0.07,0.05,-0.05,-0.1,-0.25,-0.5,-1.0])
n_steps = len(step_sizes)

# tracing accuracy params
tracing_tol=1e-8
interpolant_degree=3
interpolant_level=8
bri_mpol=32
bri_ntor=32

# set up a tracer
tracer = TraceBoozer(vmec_input,
                    n_partitions=1,
                    max_mode=max_mode,
                    major_radius=major_radius,
                    aspect_target=aspect_target,
                    target_volavgB=target_volavgB,
                    tracing_tol=tracing_tol,
                    interpolant_degree=interpolant_degree,
                    interpolant_level=interpolant_level,
                    bri_mpol=bri_mpol,
                    bri_ntor=bri_ntor)
#x0 = np.copy(tracer.x0)
dim_x = len(x0)

# sample particles via SAA
tracer.sync_seeds()
if s_label == "full":
  stz_inits,vpar_inits = tracer.sample_volume(n_particles)
else:
  stz_inits,vpar_inits = tracer.sample_surface(n_particles,s_label)

# save the parameters
if rank == 0:
  outfilename = f"./quasisymmetry_data_config_{config}_mmode_{max_mode}_tmax_{tmax}.pickle"
  outdata = {}
  outdata['mn_list'] = mn_list
  outdata['step_sizes'] = step_sizes
  outdata['tmax'] = tmax
  outdata['n_particles'] = n_particles
  outdata['s_label'] = s_label
  outdata['h_fdiff_x'] = h_fdiff_x
  outdata['h_fdiff_qs'] = h_fdiff_qs
  outdata['vmec_input'] = vmec_input
  outdata['aspect_target'] = aspect_target
  outdata['major_radius'] = major_radius
  outdata['target_volavgB'] = target_volavgB
  outdata['max_mode'] = max_mode
  outdata['s_label'] = s_label
  outdata['tracing_tol'] = tracing_tol
  outdata['interpolant_degree'] = interpolant_degree
  outdata['interpolant_level'] = interpolant_level
  outdata['bri_mpol'] = bri_mpol
  outdata['bri_ntor'] = bri_ntor
  outdata['stz_inits'] = stz_inits
  outdata['vpar_inits'] = vpar_inits
  # dump data
  pickle.dump(outdata,open(outfilename,"wb"))


def EnergyLoss(c_times,axis=None):
  """ Energy Loss Objective """
  if axis is None:
    return np.mean(3.5*np.exp(-2*c_times/tmax))
  else:
    return np.mean(3.5*np.exp(-2*c_times/tmax),axis=axis)

def confinement_times(x):
  """
  shortcut for computing the confinement times.
  """
  c_times = tracer.compute_confinement_times(x,stz_inits,vpar_inits,tmax) 
  return c_times


# compute energy at x0
c_times_x0 = confinement_times(x0)
energy_x0 = EnergyLoss(c_times_x0)

# save the gradient
if rank == 0:
  outdata['c_times_x0'] = c_times_x0
  outdata['energy_x0'] = energy_x0

# storage
c_times_plus_all_d = np.zeros((n_obj,n_steps,n_particles))
energy_plus_all_d = np.zeros((n_obj,n_steps))
qs_grads = np.zeros((n_obj,dim_x))
qs_plus_all_d = np.zeros((n_obj,n_steps))
qs0_all_d = np.zeros(n_obj)


for ii,(helicity_m,helicity_n) in enumerate(mn_list):

  # quasisymmetry objective with (m,n)
  qsrr = QuasisymmetryRatioResidual(tracer.vmec,
                                  np.arange(0, 1.01, 0.1),  # Radii to target
                                  helicity_m=helicity_m, helicity_n=helicity_n)  # (M, N) you want in |B|
  def qs_residuals(x):
    """
    Compute the QS residuals
    """
    tracer.surf.x = np.copy(x)
    try:
      qs = qsrr.residuals() # quasisymmetry
    except:
      qs = np.inf
    ret = qs
    if rank == 0:
      print(ret)
      sys.stdout.flush()
    return ret

  # print
  if rank == 0:
    print("")
    print('computing jacobian of QS with finite difference')
    print('dim_x',dim_x)
    sys.stdout.flush()
  
  # QS jacobian
  qs_jac,qs0,_,_ = forward_difference(qs_residuals,x0,h_fdiff_qs,return_evals=True)
  # QS gradient
  qs_grad = 2*qs_jac.T @ qs0

  # take c_times for central difference (we take some extra for redundancy)
  c_times_plus = np.zeros((n_steps,n_particles))
  qs_plus = np.zeros(n_steps)
  for jj,tt in enumerate(step_sizes):
    c_times_plus[jj] = confinement_times(x0 + tt*qs_grad)
    # save QS values as well
    qs_plus[jj] = np.sum(qs_residuals(x0 + tt*qs_grad)**2)

  # compute energy
  energy_plus = EnergyLoss(c_times_plus,axis=1)

  # save the data
  qs_grads[ii] = np.copy(qs_grad)
  qs_plus_all_d[ii] = np.copy(qs_plus)
  qs0_all_d[ii] = np.sum(qs0**2)
  c_times_plus_all_d[ii] = np.copy(c_times_plus)
  energy_plus_all_d[ii] = np.copy(energy_plus)

  # save data
  if rank == 0:
    outdata['qs_grads'] = qs_grads
    outdata['qs_plus_all_d'] = qs_plus_all_d
    outdata['qs0_all_d'] = qs0_all_d
    outdata['c_times_plus_all_d'] = c_times_plus_all_d
    outdata['energy_plus_all_d'] = energy_plus_all_d
    # dump data
    pickle.dump(outdata,open(outfilename,"wb"))


if rank == 0:
  print("")
  print("======================================")
  print("Done.")
  print("======================================")
