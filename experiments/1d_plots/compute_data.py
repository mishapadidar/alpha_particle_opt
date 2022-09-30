import numpy as np
from mpi4py import MPI
import pickle
import sys
sys.path.append("../../utils")
sys.path.append("../../trace")
from trace_boozer import TraceBoozer


# load the problem
n_particles = 10**3
tmax = 1e-2
vmec_input="../../vmec_input_files/input.nfp2_QA"
n_partitions = 1
max_mode = 1
tracer = TraceBoozer(vmec_input,n_partitions=n_partitions,max_mode=max_mode)
tracer.sync_seeds(0)
ntheta=nzeta=nvpar = int(np.cbrt(n_particles))
stz_inits,vpar_inits = tracer.surface_grid(0.4,ntheta,nzeta,nvpar)

# set up the objective
def objective(x):
  # return confinement times (n_particles,)
  c_times = tracer.compute_confinement_times(x,stz_inits,vpar_inits,tmax)
  return c_times

# get the starting piont
x0 = tracer.x0
dim_x = tracer.dim_x

# discretization parameters
n_directions = dim_x
n_points_per = 10 # total points per direction

# make the discretization
max_pert = 0.5
ub = max_pert
lb = -max_pert
n1 = int(n_points_per/2)
T1 = np.linspace(lb,ub,n1)
min_log,max_log = -5,-2
n2 = int((n_points_per - n1)/2)
T2 = np.logspace(min_log,max_log,n2)
T2 = np.hstack((-T2,T2))
T = np.sort(np.unique(np.hstack((T1,T2))))

# use an orthogonal frame
Q = np.eye(dim_x)

# storage
X = np.zeros((n_directions,n_points_per,dim_x))
FX = np.zeros((n_directions,n_points_per,n_particles))

for ii in range(n_directions):
    print(f"direction {ii}/{dim_x}")
    sys.stdout.flush()
    # eval point
    Y = x0 + Q[ii]*np.reshape(T,(-1,1))
    fY = np.array([objective(y) for y in Y])
    # save it
    X[ii] = np.copy(Y)
    FX[ii] = np.copy(fY)

# dump a pickle file
outfile = "1dplot_data.pickle"
outdata = {}
outdata['X'] = X
outdata['FX'] = FX
outdata['n_directions'] = n_directions
outdata['n_points_per'] = n_points_per
outdata['Q'] = Q
outdata['T'] = T
pickle.dump(outdata,open(outfile,"wb"))
