import numpy as np
import pickle
import sys
import time
import glob
from mpi4py import MPI
sys.path.append("../../utils")
sys.path.append("../../trace")
sys.path.append("../../sample")
from trace_boozer import TraceBoozer
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

tmax = 1e-2
n_particles = 10000

#filelist = glob.glob("data_nfp2_QA_cold_high_res_max_mode_*_quasisymmetry_opt.pickle")
filelist = glob.glob("data_nfp4_QH_cold_high_res_max_mode_*_quasisymmetry_opt.pickle")

for infile in filelist:
  if rank == 0:
    print("")
    print("processing",infile)

  indata = pickle.load(open(infile,"rb"))
  max_mode = indata['max_mode']
  vmec_input = indata['vmec_input']
  aspect_target = indata['aspect_target']
  major_radius = indata['major_radius']
  minor_radius = major_radius/aspect_target
  target_volavgB = indata['target_volavgB']
  xopt = indata['xopt']

  tracer = TraceBoozer(vmec_input,
                      n_partitions=1,
                      max_mode=max_mode,
                      aspect_target=aspect_target,
                      major_radius=major_radius,
                      target_volavgB=target_volavgB,
                      tracing_tol=1e-8,
                      interpolant_degree=3,
                      interpolant_level=10,
                      bri_mpol=16,
                      bri_ntor=16)
  tracer.sync_seeds(0)
  x0 = np.copy(xopt)
  
  # compute the mirror ratio
  field,bri = tracer.compute_boozer_field(x0)
  modB = tracer.compute_modB(field,bri,ns=64,ntheta=64,nphi=64)
  Bmax = np.max(modB)
  Bmin = np.min(modB)
  mirror_ratio = Bmax/Bmin
  if rank == 0:
    print('mirror ratio',mirror_ratio)
    print('Bmax',Bmax)
  
  # tracing points
  stz_inits,vpar_inits = tracer.sample_volume(n_particles)
  
  t0  = time.time()
  if rank == 0:
    print('tracing')

  c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)

  if rank == 0:
    print('time',time.time() - t0)
    print('mean',np.mean(c_times))
    print('loss fraction',np.mean(c_times < tmax))
  

  # dump a pickle file
  if rank == 0:
    indata['c_times'] = c_times
    indata['mirror_ratio'] = mirror_ratio
    indata['Bmax'] = Bmax
    indata['Bmin'] = Bmin
    indata['tmax'] = tmax
    pickle.dump(indata,open(infile,"wb"))
