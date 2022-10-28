import numpy as np
from mpi4py import MPI
import sys
import pickle
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
from finite_difference import finite_difference

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# resampling parameters
n_particles = 50000
sampling_level = "full" # full, or float surface label
# manually override vmec input
vmec_input = "../vmec_input_files/input.nfp2_QA_cold_high_res"

# load the point
infile = sys.argv[1] 
indata = pickle.load(open(infile,"rb"))
xopt = indata['xopt']
#vmec_input = indata['vmec_input']
major_radius = indata['major_radius']
max_mode = indata['max_mode']
objective_type = indata['objective_type'] 
tmax = indata['tmax']

# build a tracer object
n_partitions = 1
tracer = TraceSimple(vmec_input,n_partitions=n_partitions,max_mode=max_mode,major_radius=major_radius)
x0 = tracer.x0
tracer.sync_seeds()
dim_x = tracer.dim_x

# sample some points
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
# sync seeds again
tracer.sync_seeds()

# compute the confinement times
c_times = tracer.compute_confinement_times(xopt,stp_inits,vpar_inits,tmax)

# dump a pickle file with the trace times
if rank == 0:
  out_of_sample = {}
  out_of_sample['x'] = xopt
  out_of_sample['stp_inits'] = stp_inits
  out_of_sample['vpar_inits'] = vpar_inits
  out_of_sample['c_times'] = c_times
  out_of_sample['tmax'] = tmax
  out_of_sample['n_particles'] = n_particles
  # append the new information to the indata
  indata['out_of_sample'] = out_of_sample
  pickle.dump(indata,open(infile,"wb"))
  
