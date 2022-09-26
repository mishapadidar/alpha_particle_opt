#!/usr/bin/env python

import numpy as np
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field.tracing import trace_particles_boozer, MinToroidalFluxStoppingCriterion, \
    MaxToroidalFluxStoppingCriterion, ToroidalTransitStoppingCriterion, \
    compute_resonances, LevelsetStoppingCriterion
from simsopt.mhd import Vmec
import sys
sys.path.append("../utils")
import constants
sys.path.append("../stella")
from bfield import make_surface_classifier

"""
Run with mpiexec
"""



# Compute VMEC equilibrium
vmec_input = "../stella/input.new_QA_scaling"
vmec = Vmec(vmec_input)

# Construct radial interpolant of magnetic field
order = 3
bri = BoozerRadialInterpolant(vmec, order,mpol=4, ntor=4, enforce_vacuum=True)

# Construct 3D interpolation
nfp = vmec.wout.nfp
degree = 3
srange = (0, 1, 15)
thetarange = (0, np.pi, 15)
zetarange = (0, 2*np.pi/nfp, 15)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

# Evaluate error in interpolation
#print('Error in |B| interpolation', field.estimate_error_modB(1000), flush=True)


# initialize the points for tracing
thetas = np.linspace(0, 2*np.pi, 2)
s = np.linspace(0.05, 0.95, 2)
[s, thetas] = np.meshgrid(s, thetas)
n_particles = len(s.flatten())
print("n_particles",n_particles)
stz_inits = np.zeros((n_particles, 3))
stz_inits[:, 0] = s.flatten()
stz_inits[:, 1] = thetas.flatten()
stz_inits[:, 2] = np.zeros_like(s.flatten())
# TODO:initialize vpar isotropically
vpar_ub = (constants.FUSION_ALPHA_SPEED_SQUARED)
vpar_lb = -(constants.FUSION_ALPHA_SPEED_SQUARED)
vpar_inits = np.linspace(vpar_lb,vpar_ub,n_particles)

# stopping criteria
ntheta=nphi=32
classifier = make_surface_classifier(vmec_input=vmec_input, rng="full torus",ntheta=ntheta,nphi=nphi)
stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)]

# tracing parameters
tmax = 1e-6
trace_tol = 1e-8

# now trace
print("tracing")
gc_tys, gc_zeta_hits = trace_particles_boozer(
     field, 
     stz_inits, 
     vpar_inits,
     tmax=tmax,
     mass=constants.ALPHA_PARTICLE_MASS ,
     charge=constants.ALPHA_PARTICLE_CHARGE ,
     Ekin=constants.FUSION_ALPHA_PARTICLE_ENERGY,
     tol=trace_tol,
     mode='gc_vac',
     stopping_criteria=stopping_criteria,
     forget_exact_path=True)

print(gc_zeta_hits)


#field.set_points(stz_inits)
#modB_inits = field.modB()
#mu_inits = (Ekin - mass*0.5*vpar**2)/modB_inits  # m vperp^2 /(2B)
#
#Nparticles = len(gc_tys)
#for i in range(Nparticles):
#    vpar = gc_tys[i][:, 4]
#    points = np.zeros((len(gc_tys[i][:, 0]), 3))
#    points[:, 0] = gc_tys[i][:, 1]
#    points[:, 1] = gc_tys[i][:, 2]
#    points[:, 2] = gc_tys[i][:, 3]
#    field.set_points(points)
#    modB = np.squeeze(field.modB())
#    E = 0.5*mass*vpar**2 + mu_inits[i]*modB
#    E = (E - Ekin)/Ekin
#    print('Relative error in energy: ', np.max(np.abs(E)))
