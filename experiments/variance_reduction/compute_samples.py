import numpy as np
from mpi4py import MPI
import sys
import pickle
import time
sys.path.append("../../utils")
sys.path.append("../../trace")
sys.path.append("../../sample")
sys.path.append("../../opt")
#from trace_boozer import TraceBoozer
from test_boozer import TraceBoozer
from eval_wrapper import EvalWrapper
from radial_density import RadialDensity
from constants import V_MAX

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

#vmec_input = '../vmec_input_files/input.nfp2_QA_cold_high_res'
vmec_input = '../../vmec_input_files/input.nfp4_QH_warm_start_high_res'
tmax = 1e-2
n_particles = 1000
max_mode = 1
minor_radius = 1.7
major_radius = 2.0*1.7
target_volavgB = 5.0
tracing_tol = 1e-8
interpolant_degree = 3
interpolant_level  = 8
bri_mpol = 16
bri_ntor = 16
tracer = TraceBoozer(vmec_input,
                    n_partitions=1,
                    max_mode=max_mode,
                    minor_radius=minor_radius,
                    major_radius=major_radius,
                    target_volavgB=target_volavgB,
                    tracing_tol=tracing_tol,
                    interpolant_degree=interpolant_degree,
                    interpolant_level=interpolant_level,
                    bri_mpol=bri_mpol,
                    bri_ntor=bri_ntor)
tracer.sync_seeds()
x0 = tracer.x0
dim_x = tracer.dim_x

# compute the field
field,bri = tracer.compute_boozer_field(x0)

# compute modB and mu_crit
modB = tracer.compute_modB(field,bri)
mu_crit = tracer.compute_mu_crit(field,bri)

# compute the moments of mu
stz_inits,vpar_inits = tracer.sample_volume(10000) 
mu = tracer.compute_mu(field,bri,stz_inits,vpar_inits)
mu_mean = np.mean(mu)
mu_std = np.std(mu)

# compute the two strata probabilities
prob_mu_strata1 = np.mean(mu<mu_crit)
prob_mu_strata2 = np.mean(mu>=mu_crit)

# get points for tracing
stz_inits,vpar_inits = tracer.sample_volume(n_particles) 
mu = tracer.compute_mu(field,bri,stz_inits,vpar_inits)

# compute the moments of s
sampler = RadialDensity(1000)
s_mean = sampler.mean()
s_std = sampler.std()
s_pdf = sampler._pdf(stz_inits[:,0])

print('tracing')
t0  = time.time()
c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax,field,bri)
print('time',time.time() - t0)

if rank == 0:
  outdata = {}
  outdata['tmax'] = tmax
  outdata['n_particles'] = n_particles
  outdata['mu_crit'] = mu_crit
  outdata['c_times'] = c_times
  outdata['stz_inits'] = stz_inits
  outdata['vpar_inits'] = vpar_inits
  outdata['mu'] = mu
  outdata['modB'] = modB
  outdata['s_pdf'] = s_pdf
  outdata['s_mean'] = s_mean
  outdata['s_std'] = s_std
  outdata['mu_mean'] = mu_mean
  outdata['mu_std'] = mu_std
  outdata['prob_mu_strata1'] = prob_mu_strata1
  outdata['prob_mu_strata2'] = prob_mu_strata2
  outfilename = "data_test_boozer.pickle"
  pickle.dump(outdata,open(outfilename,"wb"))
  
