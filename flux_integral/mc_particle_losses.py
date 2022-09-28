import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.field.tracing import SurfaceClassifier
import sys
sys.path.append("../field")
from biot_savart_field import load_field
sys.path.append("../utils")
from uniform_sampler import UniformSampler
sys.path.append("../trace")
from trace_cartesian import compute_loss_times

"""
Compute particle losses with MC tracing
"""

#np.random.seed(0)

n_particles = 10000
tmax = 1e-6
n_skip = np.inf
vmec_input="../vmec_input_files/input.new_QA_scaling"
bs_path="../field/bs.new_QA_scaling"

# sura=face
nphi=ntheta=32
surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="field period",nphi=nphi, ntheta=ntheta)

# load the surface classifier
nphi=ntheta=128
sclass = SurfaceRZFourier.from_vmec_input(vmec_input, range="full torus",nphi=nphi, ntheta=ntheta)
classifier = SurfaceClassifier(sclass, h=0.1, p=2)

# load the bfield
bs = load_field(bs_path,surf.nfp)

# build a sampler
sampler = UniformSampler(surf,classifier)

# get the initial particles
X = sampler.sample(n_particles)

# trace the particles
exit_states,exit_times = compute_loss_times(X,bs,classifier,tmax)

