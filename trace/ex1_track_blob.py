#from stella import *    
import numpy as np
import sys
from guiding_center_eqns import *
from simsopt.field.tracing import trace_particles, LevelsetStoppingCriterion,ToroidalTransitStoppingCriterion
sys.path.append("../utils")
sys.path.append("../stella")
from bfield import load_field,compute_rz_bounds,compute_plasma_volume,make_surface_classifier
import vtkClass

"""
Track a blob of probability mass through space.

The blob is initialized with a uniform distribution in 
Cartesian space over a small block in r,phi,z,vpar space.

We can trace with either simsopt or with the euler or midpoint
method. If tracing with euler or midpoint we can choose
to turn the drift terms off or on in the guiding center equations.

Depending on if we trace with simsopt or one of our methods the 
vtk files look a bit different. With our method the pvd file will
play a movie that moves forward in time. When tracing with
simsopt each frame of the pvd will be an individual particles 
trajectory. In either case, make a glyph of the pvd with spheres
to visualize.
"""
n_particles = 100
tmax = 1e-3
dt = 1e-7
n_skip = 10
trace_simsopt = False
method = 'midpoint' # euler or midpoint
include_drifts = True

# seed
#np.random.seed(0)

# load the bfield
ntheta=nphi=32
vmec_input="../stella/input.new_QA_scaling"
bs_path="../stella/bs.new_QA_scaling"
bs = load_field(vmec_input,bs_path,ntheta=ntheta,nphi=nphi)
def bfield(xyz):
  # add zero to shut simsopt up
  bs.set_points(xyz + np.zeros(np.shape(xyz)))
  return bs.B()
def gradAbsB(xyz):
  # add zero to shut simsopt up
  bs.set_points(xyz + np.zeros(np.shape(xyz)))
  return bs.GradAbsB()

"""
define the initial distribution
"""
# load constants
PROTON_MASS = 1.67262192369e-27  # kg
NEUTRON_MASS = 1.67492749804e-27  # kg
ONE_EV = 1.602176634e-19  # J
ALPHA_PARTICLE_MASS = 2*PROTON_MASS + 2*NEUTRON_MASS
FUSION_ALPHA_PARTICLE_ENERGY = 3.52e6 *ONE_EV # Ekin
FUSION_ALPHA_SPEED_SQUARED = 2*FUSION_ALPHA_PARTICLE_ENERGY/ALPHA_PARTICLE_MASS
ELEMENTARY_CHARGE = 1.602176634e-19  # C
ALPHA_PARTICLE_CHARGE = 2*ELEMENTARY_CHARGE
# set the bounds
rmin,rmax,zmin,zmax = compute_rz_bounds(vmec_input,ntheta=ntheta,nphi=nphi)
delta_r = rmax - rmin 
# expand the r,z bounds to fully enclose the plasma
rmax = rmax + 0.3*delta_r
rmin = max(rmin - 0.3*delta_r,0.0)
delta_z = zmax - zmin 
zmax = zmax + 0.3*delta_z
zmin = zmin - 0.3*delta_z
# set the initial density bounds for vpar
vpar0_lb = -1.0*np.sqrt(FUSION_ALPHA_SPEED_SQUARED)
vpar0_ub = 1.0*np.sqrt(FUSION_ALPHA_SPEED_SQUARED)
# set the vpar grid bounds
vparmin = vpar0_lb
vparmax = vpar0_ub
# set the initial density bounds for r,phi,z
n_r = n_phi=n_z=n_vpar = 64
r0_lb = (rmin+rmax)/2 - 2*(rmax-rmin)/n_r
r0_ub = (rmin+rmax)/2 + 2*(rmax-rmin)/n_r
phi0_lb = np.pi/2 - np.pi/8
phi0_ub = np.pi/2 + np.pi/8
z0_lb = (zmin+zmax)/2 - 2*(zmax-zmin)/n_z
z0_ub = (zmin+zmax)/2 + 2*(zmax-zmin)/n_z
# compute the initial state volume
vpar_vol = vpar0_ub - vpar0_lb
spatial_vol = 0.5*(phi0_ub-phi0_lb)*(z0_ub-z0_lb)*(r0_ub**2 - r0_lb**2)
prob0 = 1/spatial_vol/vpar_vol

# sample phi,z,vpar uniformly at random
# sample r using the inverse transform r = F^{-1}(U)
# where the CDF inverse if F^{-1}(u) = sqrt(2u/D + r0_lb^2) 
# D = 2/(r0_ub^2 - r0_lb^2)
U = np.random.uniform(0,1,n_particles)
D = 2.0/(r0_ub**2 - r0_lb**2)
R = np.sqrt(2*U/D + r0_lb**2)
#R = np.random.uniform(r0_lb,r0_ub,n_particles)
Phi = np.random.uniform(phi0_lb,phi0_ub,n_particles)
Z = np.random.uniform(z0_lb,z0_ub,n_particles)
Vpar = np.random.uniform(vpar0_lb,vpar0_ub,n_particles)
X = np.vstack((R,Phi,Z,Vpar)).T

# vtk writer
vtk_writer = vtkClass.VTK_XML_Serial_Unstructured()

# make a guiding center object
GC = GuidingCenter(bfield,gradAbsB,include_drifts=include_drifts)

if trace_simsopt:
  """
  Trace particles with simsopt. Write each particle trajectory to its own vtk file.
  We use the color atribute to encode time.
  """
  # load the surface classifier
  classifier = make_surface_classifier(vmec_input=vmec_input, ntheta=ntheta,nphi=nphi)
  xyz_inits = GC.cyl_to_cart(X[:,:-1])
  stopping_criteria=[LevelsetStoppingCriterion(classifier.dist),ToroidalTransitStoppingCriterion(100, True)]
  for ii in range(n_particles):
    print("tracing particle ",ii)
    xyz = xyz_inits[ii].reshape((1,-1))
    vpar = [Vpar[ii]]
    res_tys, _ = trace_particles(bs, xyz, vpar, tmax=tmax, mass=ALPHA_PARTICLE_MASS,
                 charge=ALPHA_PARTICLE_CHARGE, Ekin=FUSION_ALPHA_PARTICLE_ENERGY, 
                 tol=1e-09, stopping_criteria=stopping_criteria, mode='gc_vac',
                  forget_exact_path=False)
    txyz = res_tys[0]
    vtkfilename = f"./plot_data/trace_simsopt_particle_{ii:05d}.vtu"
    vtk_writer.snapshot(vtkfilename, txyz[:,1],txyz[:,2],txyz[:,3],colors=txyz[:,0])
  # make the pvd
  pvd_filename = f"./plot_data/trajectories_simsopt.pvd"
else:
  """
  Trace all particle simultaneously. Each vtk file contains a time slice.
  """

  times = np.arange(0,tmax,dt)
  for n_step,t in enumerate(times):
    # make a paraview file
    if n_step % n_skip == 0:
      print("t = ",t)
      vtkfilename = f"./plot_data/trace_{method}_{n_step:09d}.vtu"
      xyz = GC.cyl_to_cart(X[:,:-1])
      vtk_writer.snapshot(vtkfilename, xyz[:,0],xyz[:,1],xyz[:,2])
    # take a step
    if method == 'euler':
      g =  GC.GC_rhs(X)
      Xtp1 = np.copy(X + dt*g)
    elif method == 'midpoint':
      g =  GC.GC_rhs(X)
      Xstar = np.copy(X + dt*g/2)
      g =  GC.GC_rhs(Xstar)
      Xtp1 = np.copy(X + dt*g)
    X = np.copy(Xtp1)

  # make the pvd
  pvd_filename = f"./plot_data/trajectories_{method}.pvd"
vtk_writer.writePVD(pvd_filename)
  
