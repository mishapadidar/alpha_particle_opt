import numpy as np
import sys
sys.path.append('../utils')
import vtkClass 

def trace_particles(X,GC,tmax,dt,classifier=None,method='midpoint',n_skip=1,direction='forward'):
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

    # check the level set stopping criteria
    in_plasma = classifier.evaluate(X[:,:-1]).flatten()
    just_ejected = in_plasma < 0

    # update ejected particles
    ejected[~ejected] = np.copy(just_ejected)

    # update the time in plasma for particles that have left
    tau_in_plasma[ejected] = np.minimum(tau_in_plasma[ejected],t)

    # only trace particles still in the plasma
    X = X[~just_ejected]
 
    if len(X) == 0:
      return tau_in_plasma
    
  return tau_in_plasma
