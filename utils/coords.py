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

def cart_to_cyl(xyz):
  """ cartesian to cylindrical coordinates
  input:
  xyz: (N,3) array of points (x,y,z)

  return
  rphiz: (N,3) array of point (r,phi,z)
  """
  x = xyz[:,0]
  y = xyz[:,1]
  z = xyz[:,2]
  return np.vstack((np.sqrt(x**2+y**2),np.mod(np.arctan2(y,x),2*np.pi),z)).T

def cyl_to_cart(r_phi_z):
  """ cylindrical to cartesian coordinates 
  input:
  r_phi_z: (N,3) array of points (r,phi,z)

  return
  xyz: (N,3) array of point (x,y,z)
  """
  r = r_phi_z[:,0]
  phi = r_phi_z[:,1]
  z = r_phi_z[:,2]
  return np.vstack((r*np.cos(phi),r*np.sin(phi),z)).T

def jac_cart_to_cyl(r_phi_z,D):
  """
  Compute the jacobian vector product of the jacobian of the coordinate 
  transformation from cartesian to cylindrical with a set of vectors.
  The Jacobian 
    J = [[cos(phi), sin(phi),0]
         [-sin(phi),cos(phi),0]/r
         [0,0,1]]
  is evaluated at the points in r_phi_z, then the product is computed against
  the vectors in D.
 
  input:
  r_phi_z: (N,3) array of points in cylindrical coordinates, (r,phi,z)
  D: (N,3) array of vectors in xyz coordinates to compute the directional derivatives.
  return 
  JD: (N,3) array of jacobian vector products, J @ D
  """
  #J = np.array([[np.cos(phi),np.sin(phi),0.],[-np.sin(phi)/r,np.cos(phi)/r,0.0],[0,0,1.0]])
  r = r_phi_z[:,0]
  phi = r_phi_z[:,1]
  JD = np.vstack((np.cos(phi)*D[:,0] + np.sin(phi)*D[:,1],
                 (-np.sin(phi)*D[:,0] + np.cos(phi)*D[:,1])/r,
                 D[:,2])).T
  return JD

def jac_cart_to_cyl(r_phi_z,D):
  """
  Compute the jacobian vector product of the jacobian of the coordinate 
  transformation from cartesian to cylindrical with a set of vectors.
  The Jacobian 
    J = [[cos(phi), sin(phi),0]
         [-sin(phi),cos(phi),0]/r
         [0,0,1]]
  is evaluated at the points in r_phi_z, then the product is computed against
  the vectors in D.
 
  input:
  r_phi_z: (N,3) array of points in cylindrical coordinates, (r,phi,z)
  D: (N,3) array of vectors in xyz coordinates to compute the directional derivatives.
  return 
  JD: (N,3) array of jacobian vector products, J @ D
  """
  #J = np.array([[np.cos(phi),np.sin(phi),0.],[-np.sin(phi)/r,np.cos(phi)/r,0.0],[0,0,1.0]])
  r = r_phi_z[:,0]
  phi = r_phi_z[:,1]
  JD = np.vstack((np.cos(phi)*D[:,0] + np.sin(phi)*D[:,1],
                 (-np.sin(phi)*D[:,0] + np.cos(phi)*D[:,1])/r,
                 D[:,2])).T
  return JD
