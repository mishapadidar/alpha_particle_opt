import numpy as np
from mpi4py import MPI
import pickle
import sys
debug = False
if debug:
  sys.path.append("../../utils")
  sys.path.append("../../trace")
  sys.path.append("../../sample")
  sys.path.append("../../../SIMPLE/build/")
else:
  sys.path.append("../../../utils")
  sys.path.append("../../../trace")
  sys.path.append("../../../sample")
  sys.path.append("../../../../SIMPLE/build/")
from trace_simple import TraceSimple
from radial_density import RadialDensity
from constants import V_MAX

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# configuration parameters
major_radius = 5
tmax = 1e-1 
n_partitions = 1
max_mode = 1
vmec_input="../../vmec_input_files/input.nfp2_QA_high_res"

if not debug:
  vmec_input="../" + vmec_input

# read inputs
sampling_level = sys.argv[1] # "full" or a float surface label
n_particles = int(sys.argv[2]) # number of particles

if sampling_level == "full":
  # SAA sampling over (s,theta,phi,vpar)
  s_inits = np.zeros(n_particles)
  theta_inits = np.zeros(n_particles)
  phi_inits = np.zeros(n_particles)
  vpar_inits = np.zeros(n_particles)
  if rank == 0:
    sampler = RadialDensity(1000)
    s_inits = sampler.sample(n_particles)
    # randomly sample theta,phi,vpar
    theta_inits = np.random.uniform(0,1,n_particles)
    phi_inits = np.random.uniform(0,1,n_particles)
    vpar_inits = np.random.uniform(-V_MAX,V_MAX,n_particles)
  # broadcast the points
  comm.Bcast(s_inits,root=0)
  comm.Bcast(theta_inits,root=0)
  comm.Bcast(phi_inits,root=0)
  comm.Bcast(vpar_inits,root=0)
  # stack the samples
  stp_inits = np.vstack((s_inits,theta_inits,phi_inits)).T

else:
  # SAA sampling over (theta,phi,vpar) for a fixed surface
  s_inits = float(sampling_level)*np.ones(n_particles)
  theta_inits = np.zeros(n_particles)
  phi_inits = np.zeros(n_particles)
  vpar_inits = np.zeros(n_particles)
  if rank == 0:
    # randomly sample theta,phi,vpar
    theta_inits = np.random.uniform(0,1,n_particles)
    phi_inits = np.random.uniform(0,1,n_particles)
    vpar_inits = np.random.uniform(-V_MAX,V_MAX,n_particles)
  # broadcast the points
  comm.Bcast(theta_inits,root=0)
  comm.Bcast(phi_inits,root=0)
  comm.Bcast(vpar_inits,root=0)
  # stack the samples
  stp_inits = np.vstack((s_inits,theta_inits,phi_inits)).T

# double check
assert n_particles == len(stp_inits), "n_particles does not equal length of points"

# build a tracer object
tracer = TraceSimple(vmec_input,n_partitions=n_partitions,max_mode=max_mode,major_radius=major_radius)
tracer.sync_seeds()


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
n_points_per = 25 # total points per direction

# make the discretization
max_pert = 0.1
ub = max_pert
lb = -max_pert
n1 = int(n_points_per/2)
T1 = np.linspace(lb,ub,n1)
min_log,max_log = -4,-1
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
  outfile = f"./data_surface_{sampling_level}_tmax_{tmax}.pickle"
  outdata = {}
  outdata['X'] = X
  outdata['FX'] = FX
  outdata['n_directions'] = n_directions
  outdata['n_points_per'] = n_points_per
  outdata['Q'] = Q
  outdata['T'] = T
  outdata['n_particles'] = n_particles
  outdata['stp_inits'] = stp_inits
  outdata['vpar_inits'] = vpar_inits
  outdata['max_mode'] = max_mode
  outdata['major_radius'] = major_radius
  outdata['vmec_input'] = vmec_input
  outdata['tmax'] = tmax
  outdata['sampling_level'] = sampling_level
  pickle.dump(outdata,open(outfile,"wb"))
