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
  mpiexec -n 1 python3 print_configuration_info.py
"""


# scaling params
target_minor_radius = 1.7 #meters
target_B00_on_axis = 5.7 # Tesla

# tracing parameters
n_particles = 10000
tmax = 0.01
tracing_tol = 1e-8
interpolant_degree = 3
interpolant_level = 8
bri_mpol = 8
bri_ntor = 8
n_partitions = 1

filelist = ["nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_0.89",
          "nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_1.043"]

for ii,infile in enumerate(filelist):

  # load the vmec input
  vmec_input = "./configs/input." + infile

  mpi = MpiPartition(n_partitions)
  vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
  surf = vmec.boundary
  
  # get the aspect ratio for rescaling the device
  aspect_ratio = surf.aspect_ratio()
  major_radius = target_minor_radius*aspect_ratio
  
  
  # build a tracer object
  tracer = TraceBoozer(vmec_input,
                        n_partitions=n_partitions,
                        max_mode=-1,
                        aspect_target=aspect_ratio,
                        major_radius=major_radius,
                        target_volavgB=1.0, # dummy value
                        tracing_tol=tracing_tol,
                        interpolant_degree=interpolant_degree,
                        interpolant_level=interpolant_level,
                        bri_mpol=bri_mpol,
                        bri_ntor=bri_ntor)
  tracer.sync_seeds()
  x0 = tracer.x0

  # compute the boozer field
  field,bri = tracer.compute_boozer_field(x0)
  
  # now scale the toroidal flux by B(0,0)[s=0]
  if rank == 0:
    # b/c only rank 0 does the boozXform
    bmnc0 = bri.booz.bx.bmnc_b[0]
    B00 = 1.5*bmnc0[1] - 0.5*bmnc0[2]
    B00 = np.array([B00])
  else:
    B00 = np.array([0.0])
  comm.Barrier()
  comm.Bcast(B00,root=0)
  B00 = B00[0] # unpack the array
  # scale the toroidal flux
  tracer.vmec.indata.phiedge *= target_B00_on_axis/B00

  # re-compute the boozer field
  tracer.vmec.need_to_run_code = True
  tracer.vmec.run()
  tracer.field = None # so the boozXform recomputes
  field,bri = tracer.compute_boozer_field(x0)

  # now get B00 just to make sure it was set right
  if rank == 0:
    # b/c only rank 0 does the boozXform
    bmnc0 = bri.booz.bx.bmnc_b[0]
    B00 = 1.5*bmnc0[1] - 0.5*bmnc0[2]
    B00 = np.array([B00])
  else:
    B00 = np.array([0.0])
  comm.Barrier()
  comm.Bcast(B00,root=0)
  B00 = B00[0] # unpack the array

  if rank == 0:
    print("")
    print("processing", infile)
    print("axis B00",B00)
    print('volavgB',tracer.vmec.wout.volavgB)
    print('toroidal flux',tracer.vmec.indata.phiedge)

  # compute mirror ratio
  modB = tracer.compute_modB(field,bri,ns=32,ntheta=32,nphi=32)
  Bmax = np.max(modB)
  Bmin = np.min(modB)
  mirror = Bmax/Bmin
  if rank == 0:
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
  mu = np.mean(c_times)
  nrg = np.mean(3.5*np.exp(-2*c_times/tmax))
  if rank == 0:
    print('volume losses')
    print('mean confinement time',mu)
    print('energy loss',nrg)
    lf = np.mean(c_times < tmax)
    print('loss fraction',lf)
  
  if rank == 0:
    print("")
    print('compute surface losses')
  stz_inits,vpar_inits = tracer.sample_surface(n_particles,0.25)
  c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
  mu = np.mean(c_times)
  nrg = np.mean(3.5*np.exp(-2*c_times/tmax))
  if rank == 0:
    print('surface losses')
    print('mean confinement time',mu)
    print('energy loss',nrg)
    lf = np.mean(c_times < tmax)
    print('loss fraction',lf)
