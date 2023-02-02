import numpy as np
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
import pickle
import sys
sys.path.append("../../trace")
sys.path.append("../../utils")
sys.path.append("../../sample")
from trace_boozer import TraceBoozer
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

"""
Print out some relevant statistics regarding the configuration.

usage:
  mpiexec -n 1 python3 configs/data_file.pickle
where data_file.pickle is replaced by the file of interest.
"""

infile = sys.argv[1]
n_partitions = 1
if "input" in infile:
  using_vmec_input = True
  vmec_input = sys.argv[1]
else:
  indata = pickle.load(open(infile,"rb"))
  vmec_input = indata['vmec_input']
  vmec_input = vmec_input[3:] # remove the first ../
  x0 = indata['xopt'] 
  max_mode = indata['max_mode']

#aspect_target = indata['aspect_target']
#major_radius = indata['major_radius']
#target_volavgB = indata['target_volavgB']
#tracing_tol = indata['tracing_tol']
#interpolant_degree = indata['interpolant_degree'] 
#interpolant_level = indata['interpolant_level'] 
#bri_mpol = indata['bri_mpol'] 
#bri_ntor = indata['bri_ntor'] 

# tracing parameters
n_particles = 10000
tmax = 0.01
tracing_tol = 1e-8
interpolant_degree = 3
interpolant_level = 8
bri_mpol = 16
bri_ntor = 16

"""
Rescale all configurations to the same minor radius and same volavgB.
Scale the device so that the major radius is 
  R = aspect*target_minor
where aspect is the current aspect ratio and target_minor is the desired
minor radius.
"""
mpi = MpiPartition(n_partitions)
vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
if using_vmec_input:
  mpol = vmec.indata.mpol
  ntor = vmec.indata.ntor
  assert mpol == ntor, "uh oh, mpol is not ntor"
  max_mode = mpol
else:
  mpol = ntor = max_mode
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=mpol,
                 nmin=-ntor, nmax=ntor, fixed=False)
aspect_ratio = surf.aspect_ratio()

## TODO: remove
#aspect_ratio = 7.0

# ensure devices are scaled the same
minor_radius = 1.7
major_radius = minor_radius*aspect_ratio
target_volavgB = 5.0

"""
Get the device properties
"""

# build a tracer object
tracer = TraceBoozer(vmec_input,
                      n_partitions=n_partitions,
                      max_mode=max_mode,
                      aspect_target=aspect_ratio,
                      major_radius=major_radius,
                      target_volavgB=target_volavgB,
                      tracing_tol=tracing_tol,
                      interpolant_degree=interpolant_degree,
                      interpolant_level=interpolant_level,
                      bri_mpol=bri_mpol,
                      bri_ntor=bri_ntor)
tracer.sync_seeds()
if using_vmec_input:
  x0 = tracer.x0
else:
  tracer.surf.x = np.copy(x0)


# compute the boozer field
field,bri = tracer.compute_boozer_field(x0)

if rank == 0:
  print("processing", infile)


# compute mirror ratio
modB = tracer.compute_modB(field,bri,ns=32,ntheta=32,nphi=32)
Bmax = np.max(modB)
Bmin = np.min(modB)
mirror = Bmax/Bmin
if rank == 0:
  print("")
  print('Bmax',Bmax)
  print('Bmin',Bmin)
  print('mirror ratio',mirror)

# print aspect ratio and iota
aspect = tracer.surf.aspect_ratio()
iota = tracer.vmec.mean_iota()
R = tracer.surf.get('rc(0,0)')
if rank == 0:
  print('aspect',aspect)
  print('major radius',R)
  print('minor radius',R/aspect)
  print('iota',iota)

if rank == 0:
  print("")
  print('compute volume losses')
stz_inits,vpar_inits = tracer.sample_volume(n_particles)
c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
std_err = np.std(c_times)/np.sqrt(len(c_times))
mu = np.mean(c_times)
nrg = np.mean(3.5*np.exp(-2*c_times/tmax))
if rank == 0:
  print('volume losses')
  print('mean confinement time',mu)
  print('standard err',std_err)
  print('energy loss',nrg)
  lf = np.mean(c_times < tmax)
  print('loss fraction',lf)
  print('standard err',np.sqrt(lf*(1-lf)/n_particles))

if rank == 0:
  print("")
  print('compute surface losses')
stz_inits,vpar_inits = tracer.sample_surface(n_particles,0.25)
c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
std_err = np.std(c_times)/np.sqrt(len(c_times))
mu = np.mean(c_times)
nrg = np.mean(3.5*np.exp(-2*c_times/tmax))
if rank == 0:
  print('surface losses')
  print('mean confinement time',mu)
  print('standard err',std_err)
  print('energy loss',nrg)
  lf = np.mean(c_times < tmax)
  print('loss fraction',lf)
  print('standard err',np.sqrt(lf*(1-lf)/n_particles))
