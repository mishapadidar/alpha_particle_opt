import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
import sys
from guiding_center_eqns_cartesian import *
sys.path.append("../stella")
from bfield import load_field,compute_rz_bounds,compute_plasma_volume,make_surface_classifier
sys.path.append("../utils")
from coords import cyl_to_cart,cart_to_cyl
from concentricSurfaceClassifier import concentricSurfaceClassifier
from constants import *
from grids import loglin_grid
import vtkClass

"""
Compute particle losses with MC tracing
"""

#np.random.seed(0)

n_particles = 100000
tmax = 1e-5
n_skip = np.inf
vmec_input="../stella/input.new_QA_scaling"
bs_path="../stella/bs.new_QA_scaling"

# load the surface classifier
nphi=ntheta=512
classifier = make_surface_classifier(vmec_input=vmec_input, rng="full torus",ntheta=ntheta,nphi=nphi)

#ntheta=nphi=2048
#surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="full torus", nphi=nphi, ntheta=ntheta)

# load the bfield
ntheta=nphi=64
bs = load_field(vmec_input,bs_path,ntheta=ntheta,nphi=nphi)
def bfield(xyz):
  # add zero to shut simsopt up
  bs.set_points(xyz + np.zeros(np.shape(xyz)))
  return bs.B()
def gradAbsB(xyz):
  # add zero to shut simsopt up
  bs.set_points(xyz + np.zeros(np.shape(xyz)))
  return bs.GradAbsB()
GC = GuidingCenter(bfield,gradAbsB)



"""
Sample from the initial distribution
"""
# set the bounds
rmin,rmax,zmin,zmax = compute_rz_bounds(vmec_input,ntheta=ntheta,nphi=nphi)
lb_vpar = -1.0*np.sqrt(FUSION_ALPHA_SPEED_SQUARED)
ub_vpar = 1.0*np.sqrt(FUSION_ALPHA_SPEED_SQUARED)
lb_r,ub_r = rmin,rmax
lb_phi,ub_phi = 0,2*np.pi
lb_z,ub_z = zmin,zmax
## full plasma volume
#spatial_vol = compute_plasma_volume(vmec_input = "../stella/input.new_QA_scaling",nphi=nphi,ntheta=ntheta)
## compute the initial state volume
#vpar_vol = ub_vpar - lb_vpar
## probability constant
#prob_const = 1/spatial_vol/vpar_vol

# do rejection sampling
X = np.zeros((0,4)) # [r,phi,z, vpar]
while len(X) < n_particles:
  # sample r using the inverse transform r = F^{-1}(U)
  # where the CDF inverse if F^{-1}(u) = sqrt(2u/D + r0_lb^2) 
  # D = 2/(r0_ub^2 - r0_lb^2)
  U = np.random.uniform(0,1)
  D = 2.0/(ub_r**2 - lb_r**2)
  r = np.sqrt(2*U/D + lb_r**2)
  # sample uniformly from phi
  phi = np.random.uniform(lb_phi,ub_phi)
  # sample uniformly from z
  z = np.random.uniform(lb_z,ub_z)
  cyl  = np.array([r,phi,z])
  xyz = cyl_to_cart(np.atleast_2d(cyl))
  # check if particle is in plasma
  if classifier.evaluate(np.atleast_2d(xyz)) > 0:
    vpar = np.random.uniform(lb_vpar,ub_vpar,1)
    point = np.append(cyl,vpar)
    X =np.vstack((X,point)) # [r,phi,z, vpar]


# storage
exit_times = np.zeros(0)
exit_states = np.zeros((0,4))

"""
Trace particles with simsopt. Write each particle trajectory to its own vtk file.
We use the color atribute to encode time.
"""
from simsopt.field.tracing import trace_particles, LevelsetStoppingCriterion,ToroidalTransitStoppingCriterion

xyz_inits = cyl_to_cart(X[:,:-1])
stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)]
loss_count = 0
for ii in range(n_particles):
  #print("tracing particle ",ii)

  # get the particle
  xyz = xyz_inits[ii].reshape((1,-1))
  vpar = [X[ii,-1]]

  # trace
  res_tys, res_phi_hits= trace_particles(bs, xyz, vpar, tmax=tmax, mass=ALPHA_PARTICLE_MASS,
               charge=ALPHA_PARTICLE_CHARGE, Ekin=FUSION_ALPHA_PARTICLE_ENERGY, 
               tol=1e-10, stopping_criteria=stopping_criteria, mode='gc_vac',
                forget_exact_path=False)

  # get the final state at end of trace
  txyz = res_tys[0] # trajectory [t,x,y,z,vpar]
  final_xyz = np.atleast_2d(txyz[-1,1:4])

  # loss fraction
  if len(res_phi_hits[0])>0:
    # exit time.
    tau = res_phi_hits[0].flatten()[0]
    # exit state
    state = res_phi_hits[0].flatten()[2:]
    
    # print some stuff
    print("lost particle ",ii)
    print(res_phi_hits)
    print('exit time',tau)
    loss_count += 1
    print('loss fraction', loss_count/(ii+1))
   
    # save it
    exit_times = np.append(exit_times,tau)
    exit_states = np.vstack((exit_states,state))

print("")
print('loss fraction', loss_count/n_particles)
print('exit times')
print(exit_times)

## vtk writer
#vtk_writer = vtkClass.VTK_XML_Serial_Unstructured()
#vtkfilename = f"./plot_data/exit_points.vtu"
#vtk_writer.snapshot(vtkfilename, 
#    exit_states[:,0],exit_states[:,1],exit_states[:,2],
#    x_force=normals[:,0],y_force=normals[:,1],z_force=normals[:,2],
#    x_jump=v_gc[:,0],y_jump=v_gc[:,1],z_jump=v_gc[:,2])
