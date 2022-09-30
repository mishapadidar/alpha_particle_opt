import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.tracing import SurfaceClassifier
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
import sys
sys.path.append("../../utils")
from constants import *
from grids import symlog_grid
sys.path.append("../../trace")
from trace_boozer import trace_boozer

"""
Compute particle losses with MC tracing
"""
np.random.seed(0)


n_particles = 2**3
tmax = 1e-2
# starting point
vmec_input="../../vmec_input_files/input.nfp2_QA"

# load vmec and mpi
n_partitions = 1
comm = MpiPartition(n_partitions)
vmec = Vmec(vmec_input, mpi=comm,keep_all_files=False,verbose=False)

# define parameters
max_mode = 1
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)") # fix the Major radius

# variables
x0 = surf.x # nominal starting point
dim_x = len(surf.x) # dimension

# bounds
vpar_lb = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(-1)
vpar_ub = np.sqrt(FUSION_ALPHA_SPEED_SQUARED)*(1)   

# use fixed particle locations
s_label = 0.2 # fixed surface
root_n_particles = int(np.cbrt(n_particles))
thetas = np.linspace(0, 2*np.pi, root_n_particles)
zetas = np.linspace(0,2*np.pi/surf.nfp, root_n_particles)
vpars = symlog_grid(vpar_lb,vpar_ub,root_n_particles)
[thetas,zetas,vpars] = np.meshgrid(thetas, zetas,vpars)
stz_inits = np.zeros((n_particles, 3))
stz_inits[:, 0] = s_label
stz_inits[:, 1] = thetas.flatten()
stz_inits[:, 2] = zetas.flatten()
vpar_inits = vpars.flatten()

# set up the objective
def objective(x):
  surf.x = np.copy(x)
  vmec.run()
  exit_states,exit_times = trace_boozer(vmec,stz_inits,vpar_inits,tmax=tmax)
  # minimize negative confinement time
  res = tmax-np.mean(exit_times)
  print('E[tau]:',res,'P(loss):',len(exit_times)/n_particles)
  return res

# optimize
from pdfo import pdfo
res = pdfo(objective, x0, method='cobyla', options={'maxfev': 200, 'ftarget': 0.0,'rhobeg':1e-4,'rhoend':1e-9})
print(res)

