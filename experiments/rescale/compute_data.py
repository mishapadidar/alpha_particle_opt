import numpy as np
from mpi4py import MPI
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt._core import Optimizable
import sys
import pickle
sys.path.append("../../utils")
sys.path.append("../../trace")
sys.path.append("../../sample")
from trace_boozer import TraceBoozer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""
Run with
  mpiexec -n 1 python3 compute_data.py
"""

# load a configuration
infile = "data_opt_nfp4_phase_one_aspect_7.0_iota_0.89_mean_energy_SAA_surface_0.25_tmax_0.01_bobyqa_mmode_3_iota_None.pickle"
indata = pickle.load(open(infile,"rb"))
vmec_input = indata['vmec_input']
vmec_input = vmec_input[3:] # remove the ../
x0 = indata['xopt']
if rank == 0:
  print(vmec_input)

# configuration params
#vmec_input = "../../vmec_input_files/input.nfp4_QH_warm_start_high_res"
aspect_target = 7.0
major_radius = 1.7*aspect_target
target_volavgB = 5.0
max_mode = 3 


# tracing params
s_label = 0.25 # 0.25 or full
tmax = 0.01 
n_particles = 10000 
h_fdiff_x = 5e-3 # finite difference
h_fdiff_qs = 1e-4 # finite difference quasisymmetry
h_fdiff_y = 3e-2 # finite difference in scaled space
helicity_m = 1 # quasisymmetry M
helicity_n = -1 # quasisymmetry N

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
  outfilename = f"./hessian_data_mmode_{max_mode}_tmax_{tmax}.pickle"
  outdata = {}
  outdata['tmax'] = tmax
  outdata['n_particles'] = n_particles
  outdata['s_label'] = s_label
  outdata['h_fdiff_x'] = h_fdiff_x
  outdata['h_fdiff_y'] = h_fdiff_y
  outdata['h_fdiff_qs'] = h_fdiff_qs
  outdata['helicity_m'] = helicity_m
  outdata['helicity_n'] = helicity_n
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

"""
Compute the hessian diagonal of the energy objective
"""
# print
if rank == 0:
  print('computing hessian of energy with finite difference')
  print('dim_x',dim_x)

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

# central difference the confinement times
Ep   = x0 + h_fdiff_x*np.eye(dim_x)
Em   = x0 - h_fdiff_x*np.eye(dim_x)
c_times_plus   = np.array([confinement_times(e) for e in Ep])
c_times_minus   = np.array([confinement_times(e)  for e in Em])
c_times0 = confinement_times(x0) 
# compute energy
energy0 = EnergyLoss(c_times0)
energy_plus = EnergyLoss(c_times_plus,axis=1)
energy_minus =EnergyLoss(c_times_minus,axis=1) 
# hessian diagonal: central difference
hess_energy = (energy_plus -2*energy0 + energy_minus)/h_fdiff_x/h_fdiff_x

# save the data
if rank == 0:
  outdata['x0'] = x0
  outdata['Xp'] = Ep
  outdata['Xm'] = Em
  outdata['c_times_plus_x'] = c_times_plus
  outdata['c_times_minus_x'] = c_times_minus
  outdata['c_times0'] = c_times0
  outdata['energy0'] = energy0
  outdata['energy_plus_x'] = energy_plus
  outdata['energy_minus_x'] = energy_minus
  outdata['hess_energy_x'] = hess_energy
  # dump data
  pickle.dump(outdata,open(outfilename,"wb"))




"""
Compute the jacobian of the QS-residuals
"""

# quasisymmetry objective
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

# jacobian
Ep = x0 + h_fdiff_qs*np.eye(dim_x)
Fp = np.array([qs_residuals(e) for e in Ep])
F0 = qs_residuals(x0)
jac = (Fp - F0).T/h_fdiff_qs



"""
Rescale the variables
"""

# build the Gauss-Newton hessian approximation
jit = 1e-6*np.eye(dim_x) # jitter
Hess_qs = jac.T @ jac+jit
L = np.linalg.cholesky(Hess_qs)

if rank == 0:
  print('')
  print('QS Hessian eigenvalues')
  print(np.linalg.eigvals(Hess_qs))
  sys.stdout.flush()

# rescale the variables y = L.T @ x
def to_scaled(x):
  """maps to new variables"""
  return L.T @ x
def from_scaled(y):
  """maps back to old variables"""
  return np.linalg.solve(L.T,y)

# map x0 to y0
y0 = to_scaled(x0)

# save data
if rank == 0:
  outdata['qs0'] = F0
  outdata['Hess_qs'] = Hess_qs
  outdata['L_scale'] = L
  # dump data
  pickle.dump(outdata,open(outfilename,"wb"))



"""
Compute the hessian diagonal of the energy objective
"""
# print
if rank == 0:
  print('computing hessian of energy with finite difference')
  print('dim_x',dim_x)

def confinement_times(y):
  """
  Shortcut for computing confinement times from scaled space
  """
  x = from_scaled(y)
  c_times = tracer.compute_confinement_times(x,stz_inits,vpar_inits,tmax) 
  return c_times

# central difference the confinement times
Ep   = y0 + h_fdiff_y*np.eye(dim_x)
Em   = y0 - h_fdiff_y*np.eye(dim_x)
c_times_plus   = np.array([confinement_times(e) for e in Ep])
c_times_minus   = np.array([confinement_times(e)  for e in Em])
# compute energy
energy_plus = EnergyLoss(c_times_plus,axis=1)
energy_minus =EnergyLoss(c_times_minus,axis=1) 
# hessian diagonal: central difference
hess_energy = (energy_plus -2*energy0 + energy_minus)/h_fdiff_y/h_fdiff_y

# save data
if rank == 0:
  outdata['y0'] = y0
  outdata['Yp'] = Ep
  outdata['Ym'] = Em
  outdata['c_times_plus_y'] = c_times_plus
  outdata['c_times_minus_y'] = c_times_minus
  outdata['energy_plus_y'] = energy_plus
  outdata['energy_minus_y'] = energy_minus
  outdata['hess_energy_y'] = hess_energy
  # dump data
  pickle.dump(outdata,open(outfilename,"wb"))


if rank == 0:
  print("")
  print("======================================")
  print("Done.")
  print("======================================")
