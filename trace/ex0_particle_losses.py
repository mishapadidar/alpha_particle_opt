#from stella import *    
import numpy as np
import sys
from guiding_center_eqns import *
from simsopt.field.tracing import trace_particles, LevelsetStoppingCriterion,ToroidalTransitStoppingCriterion
sys.path.append("../utils")
from constants import *
sys.path.append("../stella")
from bfield import load_field,compute_rz_bounds,compute_plasma_volume,make_surface_classifier
import vtkClass

"""
Compute the loss fraction of particles initialized within the plasma.

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
n_particles = 1000
tmax = 1e-6
dt = 1e-9
n_skip = np.inf
trace_simsopt = True
method = 'midpoint' # euler or midpoint
include_drifts = True

# seed
np.random.seed(0)

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

# make a guiding center object
GC = GuidingCenter(bfield,gradAbsB,include_drifts=include_drifts)

# load the surface classifier
classifier = make_surface_classifier(vmec_input=vmec_input, rng="full torus",ntheta=ntheta,nphi=nphi)

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
# full plasma volume
spatial_vol = compute_plasma_volume(vmec_input = "../stella/input.new_QA_scaling",nphi=nphi,ntheta=ntheta)
# compute the initial state volume
vpar_vol = ub_vpar - lb_vpar
# probability constant
prob_const = 1/spatial_vol/vpar_vol

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
  xyz = GC.cyl_to_cart(np.atleast_2d(cyl))
  # check if particle is in plasma
  if classifier.evaluate(np.atleast_2d(xyz)) > 0:
    vpar = np.random.uniform(lb_vpar,ub_vpar,1)
    point = np.append(cyl,vpar)
    X =np.vstack((X,point)) # [r,phi,z, vpar]
print(X)

# vtk writer
vtk_writer = vtkClass.VTK_XML_Serial_Unstructured()

if trace_simsopt:
  """
  Trace particles with simsopt. Write each particle trajectory to its own vtk file.
  We use the color atribute to encode time.
  """
  xyz_inits = GC.cyl_to_cart(X[:,:-1])
  stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)]
  loss_count = 0
  for ii in range(n_particles):
    print("tracing particle ",ii)
    xyz = xyz_inits[ii].reshape((1,-1))
    vpar = [X[ii,-1]]

    # method 1
    res_tys, res_phi_hits= trace_particles(bs, xyz, vpar, tmax=tmax, mass=ALPHA_PARTICLE_MASS,
                 charge=ALPHA_PARTICLE_CHARGE, Ekin=FUSION_ALPHA_PARTICLE_ENERGY, 
                 tol=1e-10, stopping_criteria=stopping_criteria, mode='gc_vac',
                  forget_exact_path=False)
    txyz = res_tys[0] # trajectory [t,x,y,z,vpar]
    final_xyz = np.atleast_2d(txyz[-1,1:4])

    # loss fraction
    if len(res_phi_hits[0])>0:
      print(res_phi_hits)
      loss_count += 1

    ## method 2: just check the endpoints
    #res_tys, res_phi_hits= trace_particles(bs, xyz, vpar, tmax=tmax, mass=ALPHA_PARTICLE_MASS,
    #             charge=ALPHA_PARTICLE_CHARGE, Ekin=FUSION_ALPHA_PARTICLE_ENERGY, 
    #             tol=1e-09, mode='gc_vac',
    #              forget_exact_path=False)
    #txyz = res_tys[0] # trajectory [t,x,y,z,vpar]
    #final_xyz = np.atleast_2d(txyz[-1,1:4])
    #if classifier.evaluate(final_xyz)[0].item() <0:
    #  loss_count += 1

    vtkfilename = f"./plot_data/trace_simsopt_particle_{ii:05d}.vtu"
    #vtk_writer.snapshot(vtkfilename, txyz[:,1],txyz[:,2],txyz[:,3],colors=txyz[:,0])

  print("loss fraction:", loss_count/n_particles)
  # make the pvd
  pvd_filename = f"./plot_data/trajectories_simsopt.pvd"
else:
  """
  Trace all particle simultaneously. Each vtk file contains a time slice.
  """

  # method 1: 
  # time spent in plasma
  ejected = np.zeros(len(X),dtype=bool)

  # trace the particles
  times = np.arange(0,tmax,dt)
  for n_step,t in enumerate(times):
    print("t = ",t)
    # make a paraview file
    if n_step % n_skip == 0:
      #print("t = ",t)
      vtkfilename = f"./plot_data/trace_{method}_{n_step:09d}.vtu"
      xyz = X[:,:-1]
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

    # check the level set stopping criteria
    xyz = GC.cyl_to_cart(X[:,:-1])
    #in_plasma = classifier.evaluate(xyz).flatten()
    in_plasma= np.array([classifier.evaluate(x.reshape((1,-1)))[0].item() for x in xyz])
    just_ejected = in_plasma < 0

    # update ejected particles
    ejected[~ejected] = np.copy(just_ejected)

    # update the time in plasma for particles that have left
    #tau_in_plasma[ejected] = np.minimum(tau_in_plasma[ejected],t)

    # only trace particles still in the plasma
    X = X[~just_ejected]
 
    if len(X) == 0:
      break

  print('loss fraction:',1-len(X)/n_particles) 


  # method 2: only check endpoints of trajectories
  #times = np.arange(0,tmax,dt)
  #for n_step,t in enumerate(times):
  #  # make a paraview file
  #  if n_step % n_skip == 0:
  #    #print("t = ",t)
  #    vtkfilename = f"./plot_data/trace_{method}_{n_step:09d}.vtu"
  #    xyz = GC.cyl_to_cart(X[:,:-1])
  #    vtk_writer.snapshot(vtkfilename, xyz[:,0],xyz[:,1],xyz[:,2])
  #  # take a step
  #  if method == 'euler':
  #    g =  GC.GC_rhs(X)
  #    Xtp1 = np.copy(X + dt*g)
  #  elif method == 'midpoint':
  #    g =  GC.GC_rhs(X)
  #    Xstar = np.copy(X + dt*g/2)
  #    g =  GC.GC_rhs(Xstar)
  #    Xtp1 = np.copy(X + dt*g)
  #  X = np.copy(Xtp1)

  # make the pvd
  #pvd_filename = f"./plot_data/trajectories_{method}.pvd"

  ## check the particle losses
  #xyz = GC.cyl_to_cart(X[:,:-1])
  #c= np.array([classifier.evaluate(x.reshape((1,-1)))[0].item() for x in xyz])
  #print("loss fraction: ", np.mean(c < 0.0))

#vtk_writer.writePVD(pvd_filename)
  
