import numpy as np


def cart_to_tor(x,y,z,R0=1.):
  """
  Convert cartesian to toroidal coordinates.
  Inverse is undefined if r==0

  input: 
    cartesian coordinates x,y,z
    major radius R0
  return: r,theta,phi (theta is poloidal angle)

  """ 
  sqrtxyR = np.sqrt(x**2 + y**2) - R0
  r = np.sqrt(sqrtxyR**2 + z**2)
  theta = np.arctan2(z,sqrtxyR)
  phi = np.arctan2(y,x) 
  return r,theta,phi

def tor_to_cart(r,theta,phi,R0=1.):
  """
  Convert toroidal to cartesian coords.
  input: 
    toroidal coords r,theta,phi
    major radius R0
  return: 
    cartesian x,y,z
  """ 
  assert r <=R0
  x = (R0 + r*np.cos(theta))*np.cos(phi)
  y = (R0 + r*np.cos(theta))*np.sin(phi)
  z = r*np.sin(theta)
  return x,y,z

def plot_tor_to_cart():
  """
  Plot the torus that results from meshing r,theta,phi and 
  mapping back to cartesian space.
  """
  from mpl_toolkits.mplot3d import axes3d
  import matplotlib.pyplot as plt

  # choose a minor radius
  r = 0.5

  # choose a major radius
  R0 = 1.0
  dtheta = 0.2
  Theta,Phi = np.meshgrid(np.arange(0.0, 2*np.pi+dtheta,dtheta),
                        np.arange(0.0, 2*np.pi+dtheta,dtheta),indexing='ij')
  
  X = np.zeros_like(Theta)
  Y = np.zeros_like(Theta)
  Z = np.zeros_like(Theta)
  for ii,theta in enumerate(Theta):
    for jj,phi in enumerate(Phi):
      theta,phi = Theta[ii,jj],Phi[ii,jj]
      x,y,z = tor_to_cart(r,theta,phi,R0)
      X[ii,jj] = x
      Y[ii,jj] = y
      Z[ii,jj] = z
  
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
  plt.show()
  return 


def verify_coordinate_change():
  """
  Map from toroidal to cartesian then back to toroidal to check
  if our coordinate change works
  """
  from mpl_toolkits.mplot3d import axes3d
  import matplotlib.pyplot as plt

  # choose a major radius
  R0 = 1.0
  dtheta = 0.17
  # Inverse is undefined for r=0
  R,Theta,Phi = np.meshgrid(np.arange(0.1,R0,0.1),np.arange(0.0, 2*np.pi+dtheta,dtheta),
                        np.arange(0.0, 2*np.pi+dtheta,dtheta),indexing='ij')
  
  for ii,r in enumerate(R):
    for jj,theta in enumerate(Theta):
      for kk,phi in enumerate(Phi):
        r,theta,phi = R[ii,jj,kk],Theta[ii,jj,kk],Phi[ii,jj,kk]
        x,y,z = tor_to_cart(r,theta,phi,R0)
        rh,th,ph = cart_to_tor(x,y,z,R0)
        assert np.isclose(rh,r,atol=1e-10)
        assert np.isclose(th,theta,atol=1e-10)
        assert np.isclose(ph,phi,atol=1e-10)
verify_coordinate_change() 

