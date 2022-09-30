import numpy as np
from mpi4py import MPI
import sys
sys.path.append("../../utils")
sys.path.append("../../trace")
from trace_boozer import TraceBoozer

"""
Compute particle losses with MC tracing
"""


n_particles = 10**3
tmax = 1e-2
vmec_input="../../vmec_input_files/input.nfp2_QA"
n_partitions = 1
max_mode = 1

# build a tracer object
tracer = TraceBoozer(vmec_input,n_partitions=n_partitions,max_mode=max_mode)
tracer.sync_seeds(0)
x0 = tracer.x0
dim_x = tracer.dim_x
ntheta=nzeta=nvpar = int(np.cbrt(n_particles))
stz_inits,vpar_inits = tracer.surface_grid(0.4,ntheta,nzeta,nvpar)

# set up the objective
def objective(x):
  c_times = tracer.compute_confinement_times(x,stz_inits,vpar_inits,tmax)
  # minimize negative confinement time
  res = tmax-np.mean(c_times)
  loss_frac = np.mean(res<tmax)
  print('obj:',res,'P(loss):',loss_frac)
  return res

# optimize
from pdfo import pdfo
res = pdfo(objective, x0, method='cobyla', options={'maxfev': 200, 'ftarget': 0.0,'rhobeg':1e-1,'rhoend':1e-6})
print(res)

