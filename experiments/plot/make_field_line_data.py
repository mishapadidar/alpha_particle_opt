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

"""
Computes data for a contour plot of the field strength in boozer coordinates.

usage:
  mpiexec -n 1 python3 configs/data_file.pickle
where data_file.pickle is replaced by the file of interest.
"""

infile = sys.argv[1]
indata = pickle.load(open(infile,"rb"))
vmec_input = indata['vmec_input']
vmec_input = vmec_input[3:] # remove the first ../

# build a tracer object
n_partitions=1
xopt = indata['xopt'] 
max_mode = indata['max_mode']
major_radius = indata['major_radius']
aspect_target = indata['aspect_target']
target_volavgB = indata['target_volavgB']
tracing_tol = indata['tracing_tol']
interpolant_degree = indata['interpolant_degree'] 
interpolant_level = indata['interpolant_level'] 
#bri_mpol = indata['bri_mpol'] 
#bri_ntor = indata['bri_ntor'] 
bri_mpol = 32
bri_ntor = 32
tracer = TraceBoozer(vmec_input,
                      n_partitions=n_partitions,
                      max_mode=max_mode,
                      aspect_target=aspect_target,
                      major_radius=major_radius,
                      target_volavgB=target_volavgB,
                      tracing_tol=tracing_tol,
                      interpolant_degree=interpolant_degree,
                      interpolant_level=interpolant_level,
                      bri_mpol=bri_mpol,
                      bri_ntor=bri_ntor)
tracer.sync_seeds()
tracer.surf.x = np.copy(xopt)

# compute the boozer field
field,bri = tracer.compute_boozer_field(xopt)

s_list = [0.05,0.25,0.5,1.0]
# theta is [0,pi] with stellsym
ntheta = 128
nzeta = 128
thetas = np.linspace(0, 2*np.pi, ntheta)
zetas = np.linspace(0,2*np.pi/tracer.surf.nfp, nzeta)
[thetas,zetas] = np.meshgrid(thetas, zetas)
# storage
modB_list = np.zeros((len(s_list),ntheta*nzeta))
for ii,s_label in enumerate(s_list):
  # get a list of points
  stz_inits = np.zeros((ntheta*nzeta, 3))
  stz_inits[:, 0] = s_label
  stz_inits[:, 1] = thetas.flatten()
  stz_inits[:, 2] = zetas.flatten()
  # evaluate the field on the mesh
  field.set_points(stz_inits)
  modB = field.modB().flatten()
  # append to modB_list
  modB_list[ii] = np.copy(modB)

  # to reshape modB into mesh 
  #modB_mesh = np.reshape(modB,((nzeta,ntheta)))

# dump the data
outdata = {}
outdata['s_list'] = s_list
outdata['ntheta'] = ntheta
outdata['nzeta'] = nzeta
outdata['theta_mesh'] = thetas
outdata['zeta_mesh'] = zetas
outdata['modB_list'] = modB_list
indata['field_line_data'] = outdata
pickle.dump(indata,open(infile,"wb"))

