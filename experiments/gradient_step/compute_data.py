import numpy as np
from mpi4py import MPI
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt._core import Optimizable
import sys
import pickle
debug = False
if debug:
  sys.path.append("../../utils")
  sys.path.append("../../trace")
  sys.path.append("../../sample")
else:
  sys.path.append("../../../utils")
  sys.path.append("../../../trace")
  sys.path.append("../../../sample")
from trace_boozer import TraceBoozer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""
Run 
  mpiexec -n 1 python3 compute_data.py ../min_energy_loss/data_phase_one_tmax_0.0001_SAA_sweep/data_opt_nfp4_phase_one_aspect_7.0_iota_1.05_mean_energy_SAA_surface_0.25_tmax_0.0001_bobyqa_mmode_1_iota_None.pickle
"""


# tracing params
s_label = 0.25 # 0.25 or full
n_particles = 10000
h_fdiff = 5e-3 # finite difference
h_fdiff_y = 0.01
h_fdiff_qs = 1e-4 # finite difference quasisymmetry
helicity_m = 1 # quasisymmetry M
helicity_n = -1 # quasisymmetry N

# tracing accuracy params
tracing_tol=1e-8
interpolant_degree=3
interpolant_level=8
bri_mpol=16
bri_ntor=16


# load the file
data_file = sys.argv[1]
indata = pickle.load(open(data_file,"rb"))
vmec_input = indata['vmec_input']
if debug:
  vmec_input=vmec_input[3:] # remove the ../
x0 = indata['xopt']
max_mode=indata['max_mode']
aspect_target = indata['aspect_target']
major_radius = indata['major_radius']
target_volavgB = indata['target_volavgB']
tmax = indata['tmax']

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
tracer.x0 = np.copy(x0)
tracer.surf.x = np.copy(x0)
dim_x = len(x0)

# quasisymmetry objective
qsrr = QuasisymmetryRatioResidual(tracer.vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=helicity_m, helicity_n=helicity_n)  # (M, N) you want in |B|


# load the gradients
grad_energy =  indata[f'post_process_s_{s_label}']['grad_energy'] 
grad_qs =  indata[f'post_process_s_{s_label}']['grad_qs'] 


"""
Now linesearch along -grad(energy)
"""

# fixed sample
tracer.sync_seeds()
if s_label == "full":
  stz_inits,vpar_inits = tracer.sample_volume(n_particles)
else:
  stz_inits,vpar_inits = tracer.sample_surface(n_particles,s_label)

def call_c_times(x):
  """
  Compute quasisymmetry and mean-energy objective.
  """
  c_times = tracer.compute_confinement_times(x,stz_inits,vpar_inits,tmax)
  if rank == 0:
    print(np.mean(3.5*np.exp(-2*c_times/tmax)))
    sys.stdout.flush()
  return c_times

if rank == 0:
  print("")
  print("linesearching the gradient direction")

# TODO: remove
## linesearch step sizes
#T_ls = np.array([0.0,1e-4,2e-4,5e-4,1e-3,2e-3,3e-3,5e-3,7e-3,8e-3,9e-3,1e-2,2e-2])
## perform the linesearch along -grad(energy)
#X_ls = x0 - np.array([ti*grad_energy for ti in T_ls])
#c_times_ls   = np.array([call_c_times(e) for e in X_ls])
## compute values
#energy_ls = np.mean(3.5*np.exp(-2*c_times_ls/tmax),axis=1)
#loss_ls = np.mean(c_times_ls < tmax,axis=1)
#
#if rank == 0:
#  print("")
#  print('energy linesearch',energy_ls)
#  print('loss fraction linesearch', loss_ls)
#  sys.stdout.flush()




"""
Compute the jacobian of quasisymmetry residuals
"""

def compute_values(x):
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
  print('dim_x',len(x0))
  sys.stdout.flush()

# jacobian
Ep   = x0 + h_fdiff_qs*np.eye(dim_x)
Fp = np.array([compute_values(e) for e in Ep])
F0 = compute_values(x0)
jac = (Fp - F0).T/h_fdiff_qs

"""
Rescale the variables
"""

# build the Gauss-Newton hessian approximation
Hess = jac.T @ jac
jit = 1e-6*np.eye(dim_x) # jitter
L = np.linalg.cholesky(Hess + jit)

if rank == 0:
  print('')
  print('Hessian eigenvalues')
  print(np.linalg.eigvals(Hess))
  sys.stdout.flush()

# rescale the variables y = L.T @ x
def x_to_y(x):
  """maps to new variables"""
  return L.T @ x
def y_to_x(y):
  """maps back to old variables"""
  return np.linalg.solve(L.T,y)

# map x0 to y0
y0 = x_to_y(x0)


def call_c_times(y):
  """
  Compute quasisymmetry and mean-energy objective.
  Using the rescaled variables y.
  """
  x = y_to_x(y)
  c_times = tracer.compute_confinement_times(x,stz_inits,vpar_inits,tmax)
  if rank == 0:
    print("")
    print("loss frac",np.mean(c_times < tmax))
  return c_times

"""
Finite difference the objective under the rescaling
"""
## determine the finite difference step size
#d = np.random.randn(dim_x)
#d = d/np.linalg.norm(d)
#yhat = x_to_y(x0 + h_fdiff*d)
#h_fdiff_y = np.linalg.norm(yhat - y0)/3
#if rank == 0:
#  print("")
#  print("finite difference step size in y",h_fdiff_y)

# gradients
Ep = y0 + h_fdiff_y*np.eye(dim_x)
Fp = np.array([call_c_times(e) for e in Ep])
Fp = np.mean(3.5*np.exp(-2*Fp/tmax),axis=1)
F0 = call_c_times(y0)
F0 = np.mean(3.5*np.exp(-2*F0/tmax))
grad_energy_y = (Fp - F0).T/h_fdiff_y


"""
Linesearch the confinement times under the change of variables
"""
# linesearch step sizes
T_ls = h_fdiff_y*np.array([0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0])
# perform the linesearch along -grad(energy_y)
X_ls = y0 - np.array([ti*grad_energy_y for ti in T_ls])
c_times_ls   = np.array([call_c_times(e) for e in X_ls])
# compute values
energy_ls = np.mean(3.5*np.exp(-2*c_times_ls/tmax),axis=1)
loss_ls = np.mean(c_times_ls < tmax,axis=1)

if rank == 0:
  print("")
  print('energy linesearch',energy_ls)
  print('loss fraction linesearch', loss_ls)
  sys.stdout.flush()


