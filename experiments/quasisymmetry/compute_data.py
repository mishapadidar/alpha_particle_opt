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
Run with
  mpiexec -n 1 python3 compute_data.py
"""

# load a configuration
config = "A" # A or B

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
mn_list = [(0,1),(1,4),(1,2),(1,1),(2,1),(4,1)]
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
h_fdiff_x = 5e-3 # finite difference
h_fdiff_qs = 1e-4 # finite difference quasisymmetry

# tracing accuracy params
tracing_tol=1e-8
interpolant_degree=3
interpolant_level=4
bri_mpol=16
bri_ntor=16 

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


# c_times jacobian
c_times_jac_x0, c_times_x0, x_plus_h, c_times_plus_x0=forward_difference(confinement_times,x0,h=h_fdiff_x,return_evals=True)
# energy gradient
energy_x0 = EnergyLoss(c_times_x0)
energy_plus_x0 = EnergyLoss(c_times_plus_x0,axis=1)
energy_grad_x0 = (energy_plus_x0 - energy_x0)/h_fdiff_x

# save the gradient
if rank == 0:
  outdata['c_times_x0'] = c_times_x0
  outdata['c_times_plus_x0'] = c_times_plus_x0
  outdata['energy_grad_x0'] = energy_grad_x0

# storage
c_times_all_y = np.zeros((n_obj,n_particles))
c_times_plus_all_y = np.zeros((n_obj,dim_x,n_particles))
energy_grad_all_y = np.zeros((n_obj,dim_x))
qs_grads = np.zeros((n_obj,dim_x))


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
  qs_grad = 2*qs_jac.T @ qs0

  # take second directional derivative: H @ d
  d= qs_grad/np.linalg.norm(qs_grad)
  y0 = x0 + h_fdiff_x*d
  # compute grad_energy(x+hd)
  c_times_jac_y0, c_times_y0, y0_plus_h, c_times_plus_y0=forward_difference(confinement_times,y0,h=h_fdiff_x,return_evals=True)
  energy_y0 = EnergyLoss(c_times_y0)
  energy_plus_y0 = EnergyLoss(c_times_plus_y0,axis=1)
  energy_grad_y0 = (energy_plus_x0 - energy_y0)/h_fdiff_x
  # now compute the directional derivative
  directional_deriv = (energy_grad_y0 - energy_grad_x0)/h_fdiff_x

  # save the data
  qs_grads[ii] = np.copy(qs_grad)
  c_times_all_y[ii] = np.copy(c_times_y0)
  c_times_plus_all_y[ii] = np.copy(c_times_plus_y0)
  energy_grad_all_y[ii] = np.copy(energy_grad_y0)

  # save data
  if rank == 0:
    outdata['qs_grads'] = qs_grads
    outdata['c_times_all_y'] = c_times_all_y
    outdata['c_times_plus_all_y'] = c_times_plus_all_y
    outdata['energy_grad_all_y'] = energy_grad_all_y
    # dump data
    pickle.dump(outdata,open(outfilename,"wb"))


if rank == 0:
  print("")
  print("======================================")
  print("Done.")
  print("======================================")
