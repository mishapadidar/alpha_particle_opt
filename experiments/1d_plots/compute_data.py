import numpy as np
from mpi4py import MPI
import pickle
import sys
from scipy.integrate import simpson
debug = False
if debug:
  sys.path.append("../../utils")
  sys.path.append("../../trace")
  sys.path.append("../../sample")
else:
  sys.path.append("../../../utils")
  sys.path.append("../../../trace")
  sys.path.append("../../../sample")
from gauss_quadrature import gauss_quadrature_nodes_coeffs
from trace_boozer import TraceBoozer
from radial_density import RadialDensity
from constants import V_MAX

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""
Compute data for 1d plots

ex.
  mpiexec -n 1 python3 compute_data.py SAA 0.5 0.001 1 10 10 10
"""


# tracing parameters
#tmax = 1e-4
tracing_tol=1e-8
interpolant_degree=3
interpolant_level=8
bri_mpol=16
bri_ntor=16
s_min = 0.0
s_max = 1.0
# configuration parmaeters
vmec_input="../../vmec_input_files/input.nfp4_QH_warm_start_high_res"
n_partitions = 1
max_mode = 1
minor_radius = 1.7
aspect_target=7.0
major_radius=minor_radius*aspect_target
target_volavgB=5.0

# read inputs
#objective_type = sys.argv[1] # mean_energy, mean_time
sampling_type = sys.argv[1] # SAA, random, gauss, simpson
sampling_level = sys.argv[2] # "full" or a float surface label
tmax = float(sys.argv[3])
ns = int(sys.argv[4])  # number of surface samples
ntheta = int(sys.argv[5]) # num theta samples
nzeta = int(sys.argv[6]) # num phi samples
nvpar = int(sys.argv[7]) # num vpar samples
assert sampling_type in ['random','SAA','gauss','simpson']

if not debug:
  vmec_input="../" + vmec_input

# number of tracers
if sampling_level == "full":
  n_particles = ns*ntheta*nzeta*nvpar
else:
  n_particles = ntheta*nzeta*nvpar

# build a tracer object
tracer = TraceBoozer(vmec_input,
                      n_partitions=n_partitions,
                      max_mode=max_mode,
                      major_radius=major_radius,
                      aspect_target=aspect_target,
                      target_volavgB=target_volavgB,
                      tracing_tol=tracing_tol,
                      interpolant_degree=interpolant_degree,
                      interpolant_level=interpolant_level,
                      bri_mpol=bri_mpol,
                      bri_ntor=bri_ntor)
tracer.sync_seeds()
x0 = tracer.x0
dim_x = tracer.dim_x

"""
Generate points for tracing
"""

def get_random_points(sampling_level):
  """
  Return a set of random points to trace.
  """
  # sync seeds
  tracer.sync_seeds()
  if sampling_level == "full":
    # volume sampling
    stz_inits,vpar_inits = tracer.sample_volume(n_particles)
  else:
    # surface sampling
    s_label = float(sampling_level)
    stz_inits,vpar_inits = tracer.sample_surface(n_particles,s_label)
  return stz_inits,vpar_inits

# make a sampler for computing probabilities
radial_sampler = RadialDensity(1000)

if sampling_type == "gauss" and sampling_level == "full":
  # gauss quadrature
  s_lin,s_weights = gauss_quadrature_nodes_coeffs(ns,s_min,s_max)
  theta_lin,theta_weights = gauss_quadrature_nodes_coeffs(ntheta,0,2*np.pi)
  zeta_lin,zeta_weights = gauss_quadrature_nodes_coeffs(nzeta,0,2*np.pi/tracer.surf.nfp)
  vpar_lin,vpar_weights = gauss_quadrature_nodes_coeffs(nvpar,-V_MAX,V_MAX)
  [surfaces,thetas,zetas,vpars] = np.meshgrid(s_lin,theta_lin,zeta_lin,vpar_lin,indexing='ij')
  [w1,w2,w3,w4] = np.meshgrid(s_weights,theta_weights,zeta_weights,vpar_weights,indexing='ij')
  quad_weights = w1*w2*w3*w4

  # build a mesh
  stz_inits = np.zeros((ns*ntheta*nzeta*nvpar, 3))
  stz_inits[:, 0] = surfaces.flatten()
  stz_inits[:, 1] = thetas.flatten()
  stz_inits[:, 2] = zetas.flatten()
  vpar_inits = vpars.flatten()
  # radial likelihood
  likelihood = radial_sampler._pdf(stz_inits[:,0])
  likelihood *= (1/(2*np.pi))*(tracer.surf.nfp/(2*np.pi))*(1/(2*V_MAX))

elif sampling_type == "gauss":
  # grid over (theta,phi,vpar) for a fixed surface label
  s_label = float(sampling_level)

  # gauss quadrature
  theta_lin,theta_weights = gauss_quadrature_nodes_coeffs(ntheta,0,2*np.pi)
  zeta_lin,zeta_weights = gauss_quadrature_nodes_coeffs(nzeta,0,2*np.pi/tracer.surf.nfp)
  vpar_lin,vpar_weights = gauss_quadrature_nodes_coeffs(nvpar,-V_MAX,V_MAX)
  [thetas,zetas,vpars] = np.meshgrid(theta_lin,zeta_lin,vpar_lin,indexing='ij')
  [w1,w2,w3] = np.meshgrid(theta_weights,zeta_weights,vpar_weights,indexing='ij')
  quad_weights = w1*w2*w3

  # build a mesh
  stz_inits = np.zeros((ntheta*nzeta*nvpar, 3))
  stz_inits[:, 0] = s_label
  stz_inits[:, 1] = thetas.flatten()
  stz_inits[:, 2] = zetas.flatten()
  vpar_inits = vpars.flatten()
  # constant likelihood
  likelihood = np.ones(len(vpar_inits))
  likelihood *= (1/(2*np.pi))*(tracer.surf.nfp/(2*np.pi))*(1/(2*V_MAX))

if sampling_type == "simpson" and sampling_level == "full":
  # simpson
  s_lin = np.linspace(s_min,s_max, ns)
  theta_lin = np.linspace(0, 2*np.pi, ntheta)
  zeta_lin = np.linspace(0,2*np.pi/tracer.surf.nfp, nzeta)
  vpar_lin = np.linspace(-V_MAX,V_MAX,nvpar)
  [surfaces,thetas,zetas,vpars] = np.meshgrid(s_lin,theta_lin,zeta_lin,vpar_lin)

  # build a mesh
  stz_inits = np.zeros((ns*ntheta*nzeta*nvpar, 3))
  stz_inits[:, 0] = surfaces.flatten()
  stz_inits[:, 1] = thetas.flatten()
  stz_inits[:, 2] = zetas.flatten()
  vpar_inits = vpars.flatten()
  # radial likelihood
  likelihood = radial_sampler._pdf(stz_inits[:,0])
  likelihood *= (1/(2*np.pi))*(tracer.surf.nfp/(2*np.pi))*(1/(2*V_MAX))
  # no quad weights
  quad_weights = []

elif sampling_type == "simpson":
  # grid over (theta,phi,vpar) for a fixed surface label
  s_label = float(sampling_level)

  # simpson
  theta_lin = np.linspace(0, 2*np.pi, ntheta)
  zeta_lin = np.linspace(0,2*np.pi/tracer.surf.nfp, nzeta)
  vpar_lin = np.linspace(-V_MAX,V_MAX,nvpar)
  [thetas,zetas,vpars] = np.meshgrid(theta_lin,zeta_lin,vpar_lin)

  # build a mesh
  stz_inits = np.zeros((ntheta*nzeta*nvpar, 3))
  stz_inits[:, 0] = s_label
  stz_inits[:, 1] = thetas.flatten()
  stz_inits[:, 2] = zetas.flatten()
  vpar_inits = vpars.flatten()
  # constant likelihood
  likelihood = np.ones(len(vpar_inits))
  likelihood *= (1/(2*np.pi))*(tracer.surf.nfp/(2*np.pi))*(1/(2*V_MAX))
  # no quad weights
  quad_weights = []

elif sampling_type == "SAA":
  stz_inits,vpar_inits = get_random_points(sampling_level)


# safety check
if sampling_type != "random":
  assert n_particles == len(vpar_inits)

"""
Define the optimization objective
"""

def objective(x):
  """
  Two objectives for minimization:
  
  1. expected energy retatined
    f = E[3.5*np.exp(-2*c_times/tmax)]
  2. expected confinement time
    f = tmax - E[c_times]

  x: array,vmec configuration variables
  tmax: float, max trace time
  """
  if sampling_type == "random":
    stzs,vpars = get_random_points(sampling_level)
  else:
    stzs = np.copy(stz_inits)
    vpars = np.copy(vpar_inits)

  c_times = tracer.compute_confinement_times(x,stzs,vpars,tmax)


  if np.any(~np.isfinite(c_times)):
    # vmec failed here; return worst possible value
    c_times = np.zeros(len(vpars))

  return c_times

#if objective_type == "mean_energy": 
#  # minimize energy retained by particle
#  feat = 3.5*np.exp(-2*c_times/tmax)
#elif objective_type == "mean_time": 
#  # expected confinement time
#  feat = tmax-c_times


## now perform the averaging/quadrature
#if sampling_type in ["SAA", "random"]:
#  # sample average
#  res = np.mean(feat)
#  loss_frac = np.mean(c_times<tmax)
#elif sampling_type == "gauss" and sampling_level == "full":
#  # gauss quadrature
#  int0 = feat*likelihood
#  int0 = int0.reshape((ns,ntheta,nzeta,nvpar))
#  int0 = int0*quad_weights
#  res = np.sum(int0)

#  # loss frac for printing
#  loss_frac = c_times<tmax
#  int0 = loss_frac*likelihood
#  int0 = int0.reshape((ns,ntheta,nzeta,nvpar))
#  int0 = int0*quad_weights
#  loss_frac = np.sum(int0)

#elif sampling_type == "gauss":
#  # gauss quadrature
#  int0 = feat*likelihood
#  int0 = int0.reshape((ntheta,nzeta,nvpar))
#  int0 = int0*quad_weights
#  res = np.sum(int0)

#  # loss fraction for printing
#  loss_frac = c_times<tmax
#  int0 = loss_frac*likelihood
#  int0 = int0.reshape((ntheta,nzeta,nvpar))
#  int0 = int0*quad_weights
#  loss_frac = np.sum(int0)

#elif sampling_type == "simpson" and sampling_level == "full":
#  # loss frac for printing
#  loss_frac = c_times<tmax
#  int0 = loss_frac*likelihood
#  int0 = int0.reshape((ns,ntheta,nzeta,nvpar))
#  int0 = int0*quad_weights
#  loss_frac = np.sum(int0)

#  # simpson
#  int0 = feat*likelihood
#  int0 = int0.reshape((ns,ntheta,nzeta,nvpar))
#  int1 = simpson(int0,vpar_lin,axis=-1)
#  int2 = simpson(int1,zeta_lin,axis=-1)
#  int3 = simpson(int2,theta_lin,axis=-1)
#  res = simpson(int3,s_lin,axis=-1)
#elif sampling_type == "simpson":
#  # loss fraction for printing
#  loss_frac = c_times<tmax
#  int0 = loss_frac*likelihood
#  int0 = int0.reshape((ntheta,nzeta,nvpar))
#  int0 = int0*quad_weights
#  loss_frac = np.sum(int0)

#  # simpson
#  int0 = feat*likelihood
#  int0 = int0.reshape((ntheta,nzeta,nvpar))
#  int1 = simpson(int0,vpar_lin,axis=-1)
#  int2 = simpson(int1,zeta_lin,axis=-1)
#  res = simpson(int2,theta_lin,axis=-1)

#if rank == 0:
#  print('obj:',res,'P(loss):',loss_frac)
#sys.stdout.flush()

#return res



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
#Q = np.eye(dim_x)
np.random.seed(0) # make sure we get the same directions
Q = np.random.randn(dim_x,dim_x)
Q,_ = np.linalg.qr(Q)
Q = Q.T

# storage
X = np.zeros((n_directions,n_points_per,dim_x))
FX = np.zeros((n_directions,n_points_per,n_particles))

# sync seeds again
tracer.sync_seeds()

for ii in range(n_directions):
  if rank == 0:
    print(f"direction {ii}/{dim_x}")
    sys.stdout.flush()
  # eval point
  Y = x0 + Q[ii]*np.reshape(T,(-1,1))
  fY = np.array([objective(y) for y in Y])
  # save it
  X[ii] = np.copy(Y)
  FX[ii] = np.copy(fY)

  # dump a pickle file
  if rank == 0:
    outfile = f"./data_surface_{sampling_type}_{sampling_level}_tmax_{tmax}_ns_{ns}.pickle"
    outdata = {}
    outdata['X'] = X
    outdata['FX'] = FX
    outdata['n_directions'] = n_directions
    outdata['n_points_per'] = n_points_per
    outdata['Q'] = Q
    outdata['T'] = T

    if sampling_type != "random":
      outdata['stz_inits'] = stz_inits
      outdata['vpar_inits'] = vpar_inits
    
    if sampling_type == 'simpson':
      if sampling_level == "full":
        outdata['s_lin'] = s_lin
      outdata['theta_lin'] = theta_lin
      outdata['zeta_lin'] = zeta_lin
      outdata['vpar_lin'] = vpar_lin
    if sampling_type in ['simpson', 'gauss']:
      outdata['likelihood'] = likelihood
      outdata['quad_weights'] = quad_weights

    outdata['ns'] = ns
    outdata['ntheta'] = ntheta
    outdata['nzeta'] = nzeta
    outdata['nvpar'] = nvpar
    outdata['max_mode'] = max_mode
    outdata['target_volavgB'] = target_volavgB
    outdata['major_radius'] = major_radius
    outdata['vmec_input'] = vmec_input
    outdata['tmax'] = tmax
    outdata['sampling_type'] = sampling_type
    outdata['sampling_level'] = sampling_level
    pickle.dump(outdata,open(outfile,"wb"))
