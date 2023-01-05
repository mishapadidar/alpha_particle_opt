import numpy as np
from mpi4py import MPI
import sys
import pickle
import time
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

"""
Time the objective computation.

Run 
  mpiexec -n 1 python3 compute_data.py 1000
"""

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# tracing params
s_label = 0.25 
tmax_list = [1e-4,1e-3,1e-2,1e-1]
infile = "../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep/data_opt_nfp4_phase_one_aspect_7.0_iota_0.89_mean_energy_SAA_surface_0.25_tmax_0.01_bobyqa_mmode_4_iota_None.pickle"
# tracing accuracy params
tracing_tol=1e-8
interpolant_degree=3
interpolant_level=8
bri_mpol=16
bri_ntor=16

# number of particles
n_particles = int(sys.argv[1])

# load the config
if debug:
  infile = infile[3:]
indata = pickle.load(open(infile,"rb"))
vmec_input = indata['vmec_input']
if debug:
  vmec_input = vmec_input[3:]
aspect_target = indata['aspect_target']
major_radius = indata['major_radius']
max_mode = indata['max_mode']
target_volavgB = indata['target_volavgB']

# tracer
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
x0 = tracer.x0
tracer.sync_seeds(0)

# tracer locations
stz_inits,vpar_inits = tracer.sample_surface(n_particles,s_label)

# compute the field initialization time
t0 = time.time()
tracer.compute_boozer_field(x0)
tracer.compute_confinement_times(x0,stz_inits[:1],vpar_inits[:1],1e-6) # to ensure the bri is initialized.
startup_time = time.time() - t0

if rank == 0:
  print(f'start up time: {startup_time}')

# storage
c_times = np.zeros((len(tmax_list),n_particles))
timings = []

# trace particles
for ii, tmax in enumerate(tmax_list):
  t0 = time.time()
  c_times[ii] = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
  duration = time.time() - t0
  timings.append(duration)
  if rank == 0:
    print(f'tmax: {tmax}, time per particle: {duration/n_particles}')
    
# dump data
if rank == 0:
  outfilename = f"./timing_data_nprocs_{size}_nparticles_{n_particles}.pickle"
  outdata = {}
  outdata['c_times'] = c_times
  outdata['timings'] = timings
  outdata['tracing_tol'] = tracing_tol
  outdata['interpolant_degree'] = interpolant_degree
  outdata['interpolant_level'] = interpolant_level
  outdata['bri_mpol'] = bri_mpol
  outdata['bri_ntor'] = bri_ntor
  outdata['target_volavgB'] = target_volavgB
  outdata['max_mode'] = max_mode
  outdata['major_radius'] = major_radius
  outdata['aspect_target'] = aspect_target
  outdata['vmec_input'] = vmec_input
  outdata['n_particles'] = n_particles
  outdata['tmax_list'] = tmax_list
  outdata['s_label'] = s_label
  outdata['stz_inits'] = stz_inits
  outdata['vpar_inits'] = vpar_inits
  outdata['startup_time'] = startup_time
  outdata['comm_size'] = size
  # outdata[''] = 
  print('dumping data to ',outfilename)
  pickle.dump(outdata,open(outfilename,"wb"))

