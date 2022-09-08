import numpy as np
import sys
sys.path.append('../utils')
import vtkClass 

def trace_particles(X,GC,tmax,dt,classifier=None,
  method='midpoint',n_skip=1,direction='forward'):
  """
  Trace a particle backwards in time until they hit the boundary or until the 
  max time.

  X: (N,4) array, initial particle positions in cartesian coords
  GC: guiding center object
  tmax: float, max time
  dt: float,timestep size
  classifier: surface classifier object
  method: 'euler' or 'midpoint'
  n_skip: number of files to skip when writing files, 
          use np.inf to never drop a file
  direction: 'foward' or 'backward' to trace forward or backward in time.
  """ 
  # get the time direction
  if direction == 'backward':
    time_dir = -1.0
  else:
    time_dir = 1.0

  # time spent in plasma
  tau_in_plasma = tmax*np.ones(len(X))
  ejected = np.zeros(len(X),dtype=bool)

  vtk_writer = vtkClass.VTK_XML_Serial_Unstructured()
 
  # trace the particles
  times = np.arange(0,tmax,dt)
  for n_step,t in enumerate(times):
    # make a paraview file
    if n_step % n_skip == 0:
      #print("t = ",t)
      vtkfilename = f"./plot_data/trace_{method}_{n_step:09d}.vtu"
      xyz = X[:,:-1]
      vtk_writer.snapshot(vtkfilename, xyz[:,0],xyz[:,1],xyz[:,2])

    # take a step
    if method == 'euler':
      g =  GC.GC_rhs(X)
      Xtp1 = np.copy(X + time_dir*dt*g)
    elif method == 'midpoint':
      g =  GC.GC_rhs(X)
      Xstar = np.copy(X + time_dir*dt*g/2)
      g =  GC.GC_rhs(Xstar)
      Xtp1 = np.copy(X + time_dir*dt*g)

    X = np.copy(Xtp1)

    if classifier is not None:
      # check the level set stopping criteria
      in_plasma= np.array([classifier.evaluate(x.reshape((1,-1)))[0].item() for x in X[:,:-1]])
      just_ejected = in_plasma < 0

      # update ejected particles
      ejected[~ejected] = np.copy(just_ejected)

      # update the time in plasma for particles that have left
      tau_in_plasma[ejected] = np.minimum(tau_in_plasma[ejected],t)

      # only trace particles still in the plasma
      X = X[~just_ejected]
 
      if len(X) == 0:
        return tau_in_plasma
    
  if classifier is not None:
    return tau_in_plasma
  else:
    return X


if __name__=="__main__":
  from guiding_center_eqns_cartesian import *
  from constants import FUSION_ALPHA_SPEED_SQUARED
  sys.path.append("../stella")
  from bfield import load_field,compute_rz_bounds,compute_plasma_volume,make_surface_classifier
  from grids import loglin_grid

  tmax = 1e-5
  dt = 1e-8
  n_skip = 10
  method = 'midpoint' # euler or midpoint
  include_drifts = True

  x0 = np.array([11.58893179261604,0., -0.7127779268497292])
  #vpar_disc = np.linspace(-np.sqrt(FUSION_ALPHA_SPEED_SQUARED),np.sqrt(FUSION_ALPHA_SPEED_SQUARED),100)
  #vpar_disc = np.logspace(-1,np.log10(np.sqrt(FUSION_ALPHA_SPEED_SQUARED)),10)
  vpar_disc = loglin_grid(-np.sqrt(FUSION_ALPHA_SPEED_SQUARED),np.sqrt(FUSION_ALPHA_SPEED_SQUARED),30)
  # stack the values
  y = np.tile(x0,len(vpar_disc)).reshape((-1,3))
  X = np.hstack((y,vpar_disc.reshape((-1,1)) ))

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
  
  # build a guiding center object
  GC = GuidingCenter(bfield,gradAbsB,include_drifts=include_drifts)

  # trace
  X = trace_particles(X,GC,tmax,dt,method=method,n_skip=n_skip,direction='backward')

