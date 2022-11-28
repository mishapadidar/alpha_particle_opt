import numpy as np
import pickle
import sys
import time
sys.path.append("../../utils")
sys.path.append("../../trace")
sys.path.append("../../sample")
from trace_boozer import TraceBoozer

minor_radius = 1.7
major_radius = 8.0*1.7
target_volavgB = 5.0
tmax = 1e-2
n_particles = 1000

filelist = glob.glob("data_nfp2_QA_cold_high_res_max_mode_*_quasisymmetry_opt.pickle")
#filelist = glob.glob("data_nfp4_QH_cold_high_res_max_mode_*_quasisymmetry_opt.pickle")

for infile in file_list 
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
                      minor_radius=minor_radius,
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
  print('mirror ratio',np.max(modB)/np.min(modB))
  
  # tracing points
  stz_inits,vpar_inits = tracer.sample_volume(n_particles)
  
  t0  = time.time()
  print('tracing')
  c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
  print('time',time.time() - t0)
  print('mean',np.mean(c_times))
  print('loss fraction',np.mean(c_times < tmax))
  

  # dump a pickle file
  indata['c_times'] = c_times
  #pickle.dump(indata,open(infilename,"wb"))
