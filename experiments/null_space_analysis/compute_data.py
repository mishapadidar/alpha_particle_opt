import numpy as np
from mpi4py import MPI
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
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
Compute the following information around a point
- the gradient of the energy objective
- the gradient of the quasisymmetry objective
- linesearch the energy objective along the QS direction
"""


# tracing params
s_label = 0.25 # 0.25 or full
n_particles = 10000
h_fdiff = 1e-2 # finite difference
h_fdiff_qs = 1e-5 # finite difference quasisymmetry
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

if rank == 0:
    print('aspect',tracer.surf.aspect_ratio())
    print('iota',tracer.vmec.mean_iota())
    print('major radius',tracer.surf.get('rc(0,0)'))
    print('toroidal flux',tracer.vmec.indata.phiedge)
    print('qs total',qsrr.total())


"""
Compute the gradient of quasisymmetry objective
"""
# print
if rank == 0:
  print("")
  print('computing gradient of QS with finite difference')
  print('dim_x',len(x0))

def compute_quasisymmetry(x):
  tracer.surf.x = np.copy(x)
  return qsrr.total()

# Quasisymmetry gradient
Ep   = x0 + h_fdiff_qs*np.eye(dim_x)
qs_plus = np.array([compute_quasisymmetry(e) for e in Ep])
qs0 = compute_quasisymmetry(x0)
grad_qs = (qs_plus - qs0)/h_fdiff_qs

if rank == 0:
  print('qs total',qs0)
  print('norm qs grad',np.linalg.norm(grad_qs))


"""
Compute the gradient of the energy objective
"""
# print
if rank == 0:
  print('computing gradient of energy with finite difference')
  print('dim_x',len(x0))

# fixed sample
tracer.sync_seeds()
if s_label == "full":
  stz_inits,vpar_inits = tracer.sample_volume(n_particles)
else:
  stz_inits,vpar_inits = tracer.sample_surface(n_particles,s_label)

# forward difference the confinement times
Ep   = x0 + h_fdiff*np.eye(dim_x)
c_times_plus   = np.array([tracer.compute_confinement_times(e,stz_inits,vpar_inits,tmax) for e in Ep])
c_times0 = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax) 
# compute energy
energy0 = np.mean(3.5*np.exp(-2*c_times0/tmax))
energy_plus = np.mean(3.5*np.exp(-2*c_times_plus/tmax),axis=1)
# gradient
grad_energy = (energy_plus - energy0)/h_fdiff

if rank == 0:
  print('energy',energy0)
  print('norm energy grad',np.linalg.norm(grad_energy))
  print('qs total',qs0)
  print('norm qs grad',np.linalg.norm(grad_qs))

# save data
outdata = {}
outdata['tmax'] = tmax
outdata['n_particles'] = n_particles
outdata['h_fdiff'] = h_fdiff
outdata['s_label'] = s_label
outdata['helicity_m'] = helicity_m
outdata['helicity_n'] = helicity_n
outdata['s_label'] = s_label
outdata['x0'] = x0
outdata['Xp'] = Ep
outdata['c_times_plus'] = c_times_plus
outdata['c_times0'] = c_times0
outdata['energy0'] = energy0
outdata['energy_plus'] = energy_plus
outdata['qs0'] = qs0
outdata['qs_plus'] = qs_plus
outdata['grad_energy'] =grad_energy
outdata['grad_qs'] = grad_qs

# dump data
indata[f'post_process_s_{s_label}'] = outdata
pickle.dump(indata,open(data_file,"wb"))


"""
Now linesearch along -grad(qs)
"""
def objectives(x):
  """
  Compute quasisymmetry and mean-energy objective.
  """
  c_times = tracer.compute_confinement_times(x,stz_inits,vpar_inits,tmax)
  qs = qsrr.total() # quasisymmetry
  return np.append(c_times,qs)

# linesearch step sizes
T_ls = np.array([1e-5,1e-4,5e-4,1e-3,5e-3,1e-2,5e-2])
# perform the linesearch along -grad(qs)
X_ls = x0 - np.array([ti*grad_qs for ti in T_ls])
F_ls   = np.array([objectives(e) for e in X_ls])
# split the arrays
c_times_ls = F_ls[:,:-1]
qs_ls = F_ls[:,-1]
energy_ls = np.mean(3.5*np.exp(-2*c_times_ls/tmax),axis=1)

if rank == 0:
  print("")
  print('energy linesearch',energy_ls)
  print('qs linesearch',qs_ls)

# save the data
outdata['T_ls'] = T_ls
outdata['X_ls'] = X_ls
outdata['qs_ls'] = qs_ls
outdata['energy_ls'] = energy_ls
outdata['c_times_ls'] = c_times_ls
# dump data
indata[f'post_process_s_{s_label}'] = outdata
pickle.dump(indata,open(data_file,"wb"))


