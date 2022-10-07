import numpy as np
from mpi4py import MPI
import pickle
import sys
debug = False
if debug:
  sys.path.append("../../utils")
  sys.path.append("../../trace")
  sys.path.append("../../../SIMPLE/build/")
else:
  sys.path.append("../../../utils")
  sys.path.append("../../../trace")
  sys.path.append("../../../../SIMPLE/build/")
#from trace_boozer import TraceBoozer
from trace_simple import TraceSimple


# load the problem
if debug:
  vmec_input="../../vmec_input_files/input.nfp2_QA_high_res"
else:
  vmec_input="../../../vmec_input_files/input.nfp2_QA_high_res"

major_radius = 5
tmax = 1e-2
n_partitions = 1
max_mode = 1
tracer = TraceSimple(vmec_input,n_partitions=n_partitions,max_mode=max_mode,major_radius=major_radius)
tracer.sync_seeds(0)

# particle locations
ntheta=nzeta=10
nvpar = 10
#s_label = 0.4
s_label = float(sys.argv[1])
stz_inits,vpar_inits = tracer.surface_grid(s_label,ntheta,nzeta,nvpar)
n_particles = len(stz_inits)


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
n_points_per = 50 # total points per direction

# make the discretization
max_pert = 0.5*major_radius
ub = max_pert
lb = -max_pert
n1 = int(n_points_per/2)
T1 = np.linspace(lb,ub,n1)
min_log,max_log = -5,-1
n2 = int((n_points_per - n1)/2)
T2 = np.logspace(min_log,max_log,n2)
T2 = np.hstack((-T2,T2))
T = np.sort(np.unique(np.hstack((T1,T2))))
# just in case np.unique drops points.
n_points_per = len(T)

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
  outfile = f"1dplot_data_s_{s_label}.pickle"
  outdata = {}
  outdata['X'] = X
  outdata['FX'] = FX
  outdata['n_directions'] = n_directions
  outdata['n_points_per'] = n_points_per
  outdata['Q'] = Q
  outdata['T'] = T
  outdata['max_mode'] = max_mode
  outdata['major_radius'] = major_radius
  outdata['vmec_input'] = vmec_input
  outdata['tmax'] = tmax
  outdata['s_label'] = s_label
  pickle.dump(outdata,open(outfile,"wb"))
