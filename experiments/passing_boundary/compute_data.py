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
from constants import *


# load the problem
if debug:
  vmec_input="../../vmec_input_files/input.nfp2_QA_high_res"
else:
  vmec_input="../../../vmec_input_files/input.nfp2_QA_high_res"

major_radius = 5
tmax = 1e-2
n_partitions = 1
max_mode = 1
contr_pp = -1e-16
tracer = TraceSimple(vmec_input,
                    n_partitions=n_partitions,
                    max_mode=max_mode,
                    major_radius=major_radius,
                    contr_pp=contr_pp)
tracer.sync_seeds(0)

# particle locations
s_label = float(sys.argv[1]) # surface label
phi_label = float(sys.argv[2]) # toroidal angle
ntheta = 50
nvpar  = 50

# build a mesh
thetas = np.linspace(0, 1.0, ntheta)
vpars = np.linspace(-np.sqrt(FUSION_ALPHA_SPEED_SQUARED),np.sqrt(FUSION_ALPHA_SPEED_SQUARED),nvpar)
[thetas,vpars] = np.meshgrid(thetas, vpars)
stp_inits = np.zeros((ntheta*nvpar, 3))
stp_inits[:, 0] = s_label
stp_inits[:, 1] = thetas.flatten()
stp_inits[:, 2] = phi_label
vpar_inits = vpars.flatten()
n_particles = len(stp_inits)


# set up the objective
def objective(x):
  # return confinement times (n_particles,)
  c_times = tracer.compute_confinement_times(x,stp_inits,vpar_inits,tmax)
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
min_log,max_log = -3,0
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
  outfile = f"plot_data_s_{s_label}_phi_{phi_label}.pickle"
  outdata = {}
  outdata['X'] = X
  outdata['FX'] = FX
  outdata['n_directions'] = n_directions
  outdata['n_points_per'] = n_points_per
  outdata['Q'] = Q
  outdata['T'] = T
  outdata['stp_inits'] = stp_inits
  outdata['vpar_inits'] = vpar_inits
  outdata['max_mode'] = max_mode
  outdata['major_radius'] = major_radius
  outdata['vmec_input'] = vmec_input
  outdata['tmax'] = tmax
  outdata['s_label'] = s_label
  outdata['phi_label'] = phi_label
  pickle.dump(outdata,open(outfile,"wb"))
