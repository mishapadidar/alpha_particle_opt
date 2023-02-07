import numpy as np
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
import pickle
import time
import sys
sys.path.append("../../trace")
sys.path.append("../../utils")
sys.path.append("../../sample")
from trace_boozer import TraceBoozer
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

"""
Make data for the timing plots

usage:
  mpiexec -n 1 python3 make_timing_data.py

We rescale all configurations to the same minor radius and same volavgB.
Scale the device so that the major radius is 
  R = aspect*target_minor
where aspect is the current aspect ratio and target_minor is the desired
minor radius.
"""
# config params
vmec_input = './configs/input.new_QH'
target_minor_radius = 1.7
target_volavgB = 5.0

# tracing parameters
n_particles = 2000
tmax_list = [1e-4,1e-3,1e-2,1e-1]
tracing_tol = 1e-8
interpolant_degree = 3
interpolant_level = 8
bri_mpol = 16
bri_ntor = 16
n_partitions = 1


# get the aspect ratio for rescaling the device
mpi = MpiPartition(n_partitions)
vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
surf = vmec.boundary
aspect_ratio = surf.aspect_ratio()
major_radius = target_minor_radius*aspect_ratio

# build a tracer object
tracer = TraceBoozer(vmec_input,
                      n_partitions=n_partitions,
                      max_mode=-1,
                      aspect_target=aspect_ratio,
                      major_radius=major_radius,
                      target_volavgB=target_volavgB,
                      tracing_tol=tracing_tol,
                      interpolant_degree=interpolant_degree,
                      interpolant_level=interpolant_level,
                      bri_mpol=bri_mpol,
                      bri_ntor=bri_ntor)
tracer.sync_seeds()
x0 = tracer.x0

# initialize particles
stz_inits,vpar_inits = tracer.sample_volume(n_particles)

# get the startup time
t0 = time.time()
field,bri = tracer.compute_boozer_field(x0)
tracer.compute_confinement_times(x0,np.atleast_2d(stz_inits[0]),[vpar_inits[0]],1e-6)
startup_time = time.time() - t0

# print some stuff
major_rad = tracer.surf.get("rc(0,0)")
aspect = tracer.surf.aspect_ratio()
minor_rad = major_rad/aspect
if rank == 0:
  print("")
  print('minor radius',minor_rad)
  print('volavgB',tracer.vmec.wout.volavgB)
  print('toroidal flux',tracer.vmec.indata.phiedge)

# storage
c_times_list = np.zeros((len(tmax_list),n_particles))
trace_timings = np.zeros(len(tmax_list))

for ii,tmax in enumerate(tmax_list):
  # trace particles
  t0 = time.time()
  c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
  tf = time.time() - t0

  # store the data
  c_times_list[ii] = np.copy(c_times)
  trace_timings[ii] = tf

  # save the data
  outfile = "./timing_data.pickle"
  outdata = {}
  outdata['tmax_list'] = tmax_list
  outdata['c_times_list'] = c_times_list
  outdata['trace_timings'] = trace_timings
  outdata['startup_time'] = startup_time
  outdata['vmec_input'] = vmec_input
  outdata['target_minor_radius'] =target_minor_radius
  outdata['target_volavgB'] = target_volavgB
  outdata['n_particles'] = n_particles
  outdata['tracing_tol'] = tracing_tol
  outdata['interpolant_degree'] = interpolant_degree
  outdata['interpolant_level'] =  interpolant_level
  outdata['bri_mpol'] = bri_mpol
  outdata['bri_ntor'] = bri_ntor
  pickle.dump(outdata,open(outfile,"wb"))
