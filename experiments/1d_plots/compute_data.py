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

"""
Compute data for 1d plots

ex.
  mpiexec -n 1 python3 compute_data.py SAA 0.5 10 10 10 10
"""


# tracing parameters
tmax = 1e-3
n_timesteps = 100000
# configuration parmaeters
n_partitions = 1
max_mode = 1
major_radius = 5
#vmec_input="../../vmec_input_files/input.nfp2_QA_cold_high_res"
vmec_input="../../vmec_input_files/input.nfp2_QA_high_res"

if not debug:
  vmec_input="../" + vmec_input

# read inputs
sampling_type = sys.argv[1] # SAA or grid
sampling_level = sys.argv[2] # "full" or a float surface label
ns = int(sys.argv[3])  # number of surface samples
ntheta = int(sys.argv[4]) # num theta samples
nphi = int(sys.argv[5]) # num phi samples
nvpar = int(sys.argv[6]) # num vpar samples
assert sampling_type in ['SAA' or "grid"]

# number of tracers
n_particles = ns*ntheta*nphi*nvpar

# build a tracer object
tracer = TraceSimple(vmec_input,n_partitions=n_partitions,max_mode=max_mode,major_radius=major_radius)
tracer.sync_seeds()
x0 = tracer.x0
dim_x = tracer.dim_x

if sampling_type == "grid" and sampling_level == "full":
  # grid over (s,theta,phi,vpar)
  stp_inits,vpar_inits = tracer.flux_grid(ns,ntheta,nzeta,nvpar)
elif sampling_type == "grid":
  # grid over (theta,phi,vpar) for a fixed surface label
  s_label = float(sampling_level)
  stp_inits,vpar_inits = tracer.surface_grid(s_label,ntheta,nzeta,nvpar)
elif sampling_type == "SAA" and sampling_level == "full":
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

elif sampling_type == "SAA":
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
# sync seeds again
tracer.sync_seeds()


# set up the objective
def objective(x):
  # return confinement times (n_particles,)
  c_times = tracer.compute_confinement_times(x,stp_inits,vpar_inits,tmax,n_timesteps)
  return c_times

# get the starting piont
x0 = tracer.x0
dim_x = tracer.dim_x

# discretization parameters
n_directions = dim_x
n_points_per = 200 # total points per direction

# make the discretization
max_pert = 0.1
ub = max_pert
lb = -max_pert
n1 = int(n_points_per/2)
T1 = np.linspace(lb,ub,n1)
min_log,max_log = -5,-2
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
  outfile = f"./data_surface_{sampling_type}_{sampling_level}_tmax_{tmax}.pickle"
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
  outdata['n_timesteps'] = n_timesteps
  outdata['sampling_type'] = sampling_type
  outdata['sampling_level'] = sampling_level
  pickle.dump(outdata,open(outfile,"wb"))
