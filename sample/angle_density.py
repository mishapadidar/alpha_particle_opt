import numpy as np


def compute_det_jac_dcart_dbooz(field,stz_grid):
  """
  Compute the abs determinant jacobian of cart wrt booz
    |D(X,Y,Z)/D(s,theta,zeta)|

  input: BoozerMagneticField object
  stz_grid: (n_points,3) array of (s,theta,zeta)
  """
  n_points = len(stz_grid)
  field.set_points(stz_grid)

  # convert points to cylindrical
  R = field.R() # (n_points,1)
  Z = field.Z() # (n_points,1)
  # phi = zeta - nu
  Phi = stz_grid[:,-1].reshape((-1,1)) - field.nu_ref().reshape((-1,1))
  
  # dcylindrical/dboozer
  R_derivs = field.R_derivs() # (n_points,3)
  Z_derivs = field.Z_derivs() # (n_points,3)
  Phi_derivs = -field.nu_derivs()
  Phi_derivs[:,-1] += 1.0
  
  # compute dcartesian/dBoozer
  # X = R*cos(Phi)
  dX_dR = np.cos(Phi) # (n_points,1)
  dX_dPhi = -R*np.sin(Phi) # (n_points,1)
  # (Dcyl/Dbooz).T @ dX/dcyl 
  X_derivs = R_derivs*dX_dR + Phi_derivs*dX_dPhi  # (n_points,3)
  # Y = R*sin(Phi)
  dY_dR = np.sin(Phi) # (n_points,1)
  dY_dPhi = R*np.cos(Phi) # (n_points,1)
  # (Dcyl/Dbooz).T @ dY/dcyl 
  Y_derivs = R_derivs*dY_dR + Phi_derivs*dY_dPhi  # (n_points,3)
  
  # build the (3,3) jacobians
  dcart_dbooz = np.zeros((n_points,3,3))
  dcart_dbooz[:,0,:]  = np.copy(X_derivs)
  dcart_dbooz[:,1,:]  = np.copy(Y_derivs)
  dcart_dbooz[:,2,:]  = np.copy(Z_derivs)
  
  # density \propto |determinant(jacobian)|
  jac = np.abs(np.linalg.det(dcart_dbooz))
  return jac
  
  
  
