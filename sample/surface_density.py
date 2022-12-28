import numpy as np
import sys
sys.path.append("../trace")
from trace_boozer import TraceBoozer




vmec_input = "../vmec_input_files/input.nfp4_QH_warm_start_high_res"
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
ns = ntheta=nphi=16
stz_grid,_ = tracer.flux_grid(ns,ntheta,nphi,1)
n_points = len(stz_grid)
field.set_points(stz_grid)
print(n_points)

# convert points to cylindrical
# TODO: check the shape
R = field.R()
Z = field.Z()
print(R.shape,Z.shape)
# assume phi = zeta
Phi = stz_grid[:,-1]

# dcylindrical/dboozer
# TODO: check the shape
R_derivs = field.R_derivs()
Z_derivs = field.Z_derivs()
# assume phi = zeta, so deriv = 1
Phi_derivs = np.zeros((n_points,3))
Phi_derivs[:,-1] = 1.0
print(R_derivs.shape,Z_derivs.shape)
quit()
quit()
# TODO: compute dcartesian/dcylindrical
JD = np.vstack((np.cos(phi)*D[:,0] + np.sin(phi)*D[:,1],
               (-np.sin(phi)*D[:,0] + np.cos(phi)*D[:,1])/r,
               D[:,2])).T

# TODO: chain rule the derivatives

# TODO: now compute the determinants of the jacobians

# plot the density
# TODO: plot the density as a color, and use R and Z as contours.


