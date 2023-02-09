
import numpy as np
import sys
sys.path.append("../../sample")
sys.path.append("../../trace")
sys.path.append("../../utils")
from angle_density import compute_det_jac_dcart_dbooz
from trace_boozer import TraceBoozer

"""
Run with mpiexec 
"""
#vmec_input = "../vmec_input_files/input.nfp4_QH_warm_start_high_res"
vmec_input = "./configs/input.nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_0.89"
max_mode = 1
aspect_target = 7.0
major_radius = 1.7*aspect_target
target_volavgB = 5.0

# buld a tracer
tracer = TraceBoozer(vmec_input,
                    n_partitions=1,
                    max_mode=max_mode,
                    major_radius=major_radius,
                    aspect_target=aspect_target,
                    target_volavgB=target_volavgB,
                    tracing_tol=1e-8,
                    interpolant_degree=3,
                    interpolant_level=8,
                    bri_mpol=16,
                    bri_ntor=16)

x0 = tracer.x0

# compute the boozer field
field,bri = tracer.compute_boozer_field(x0)

# generate point in Boozer space
s_label = 0.25
ntheta=nzeta=256
nfp = tracer.surf.nfp
thetas = np.linspace(0, 2*np.pi, ntheta)
zetas = np.linspace(0,2*np.pi/nfp, nzeta)
# build a mesh
[thetas,zetas] = np.meshgrid(thetas, zetas)
stz_grid = np.zeros((ntheta*nzeta, 3))
stz_grid[:, 0] = s_label
stz_grid[:, 1] = thetas.flatten()
stz_grid[:, 2] = zetas.flatten()

# compute the determinant of the jacobian
detjac = compute_det_jac_dcart_dbooz(field,stz_grid)

# dump a pickle file
import pickle
outdata = {}
outdata['vmec_input'] = vmec_input
outdata['thetas'] = thetas
outdata['zetas'] = zetas
outdata['stz_grid'] = stz_grid
outdata['detjac'] = detjac
outdata['s_label'] = s_label
outdata['ntheta'] = ntheta
outdata['nzeta'] = nzeta
outdata['nfp'] = nfp
outfilename = "./angle_density_data.pickle"
pickle.dump(outdata,open(outfilename,"wb"))
  
