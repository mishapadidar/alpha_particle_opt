import numpy as np
from mpi4py import MPI
import sys
import pickle
from pdfo import pdfo,NonlinearConstraint as pdfo_nlc
sys.path.append("../../utils")
sys.path.append("../../trace")
sys.path.append("../../sample")
from trace_boozer import TraceBoozer
from radial_density import RadialDensity
from constants import V_MAX

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# configuration parmaeters
max_mode = 1
n_partitions = 1
minor_radius = 1.7
aspect_target = 8.0
major_radius = aspect_target*minor_radius
target_volavgB = 5.0
vmec_label = "nfp2_QA_cold_high_res"
# optimizer params
maxfev = 5000
ftarget = 0.0
rhobeg = 1.5
rhoend = 1e-6
# trace boozer params
tracing_tol = 1e-8
interpolant_degree = 3
interpolant_level  = 8
bri_mpol = 16
bri_ntor = 16

# vmec file
if vmec_label == "nfp2_QA_cold_high_res":
  vmec_input="../../vmec_input_files/input.nfp2_QA_cold_high_res"
  outfile = "./input.nfp2_QA_cold_high_res_mirror_feasible"
elif vmec_label == "nfp2_QA_high_res":
  vmec_input="../../vmec_input_files/input.nfp2_QA_high_res"
  outfile = "./input.nfp2_QA_high_res_mirror_feasible"
elif vmec_label == "nfp4_QH_warm_high_res":
  vmec_input="../../vmec_input_files/input.nfp4_QH_warm_start_high_res"
  outfile = "./input.nfp4_QH_warm_start_high_res_mirror_feasible"

# make tracer object
tracer = TraceBoozer(vmec_input,
                      n_partitions=n_partitions,
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

# penalize the mirror ratio
ns_B=ntheta_B=nzeta_B=32
len_B_field_out = ns_B*ntheta_B*nzeta_B
def B_field(x):
  """
  Compute modB on a grid
  """
  field,bri = tracer.compute_boozer_field(x)
  if field is None:
    return np.zeros(len_B_field_out)
  modB = tracer.compute_modB(field,bri,ns=ns_B,ntheta=ntheta_B,nphi=nzeta_B)
  if rank == 0:
    print("B interval:",np.min(modB),np.max(modB))
    print("Mirror Ratio:",np.max(modB)/np.min(modB))
  return modB
B_mean = 5.0
eps_B = 0.35/2.35
B_ub = B_mean*(1 + eps_B)*np.ones(len_B_field_out)
B_lb = B_mean*(1 - eps_B)*np.ones(len_B_field_out)

def penalty_objective(x):
  """
  Penalize violation of the mirror ratio
  """
  modB = B_field(x)
  ret = np.sum(np.maximum(modB-B_ub,0.0)**2)
  ret += np.sum(np.maximum(B_lb-modB,0.0)**2)
  ret = ret/2/len(modB)
  if rank == 0:
    print("obj:",ret)
  return ret

# aspect constraint
def aspect_ratio(x):
  """
  Compute the aspect ratio
  """
  # update the surface
  tracer.surf.x = np.copy(x)

  # evaluate the objectives
  try:
    asp = tracer.surf.aspect_ratio()
  except:
    asp = np.inf

  # catch partial failures
  if np.isnan(asp):
    asp = np.inf

  if rank == 0:
    print("aspect",asp)
  return asp
aspect_constraint = pdfo_nlc(aspect_ratio,-np.inf,aspect_target)
constraints = [aspect_constraint]

# optimize
res = pdfo(penalty_objective, x0, method='cobyla',constraints=constraints,options={'maxfev': maxfev, 'ftarget': ftarget,'rhobeg':rhobeg,'rhoend':rhoend})
xopt = np.copy(res.x)

# save result
if rank == 0:
  tracer.vmec.write_input(outfile)
  #outdata = {}
  #outdata['xopt'] = xopt
  #outdata['major_radius'] = major_radius
  #outdata['minor_radius'] =  minor_radius
  #outdata['target_volavgB'] = target_volavgB
  #outdata['vmec_input'] = vmec_input
  #outdata['max_mode'] = max_mode
  #outfile = f"./{vmec_label}_mirror_feasible.pickle"
  #pickle.dump(outdata,open(outfile,"wb"))
