import numpy as np
import sys
sys.path.append("../trace")
from trace_boozer import TraceBoozer


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
  Phi = stz_grid[:,-1].reshape((-1,1)) # phi = zeta
  
  # dcylindrical/dboozer
  R_derivs = field.R_derivs() # (n_points,3)
  Z_derivs = field.Z_derivs() # (n_points,3)
  # dphi/dzeta = 1
  Phi_derivs = np.zeros((n_points,3))
  Phi_derivs[:,-1] = 1.0
  
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
  
  
  

if __name__ == "__main__":
  """
  Run with mpiexec 
  """
  vmec_input = "../vmec_input_files/input.nfp4_QH_warm_start_high_res"
  max_mode = 1
  aspect_target = 7.0
  major_radius = 1.7*aspect_target
  target_volavgB = 5.0
  
  # buld a tracer
  tracer = TraceBoozer(vmec_input,
                      n_partitions=1,
                      max_mode=max_mode,
                      major_radius=major_radius,
                      aspect_target=aspect_target,
                      target_volavgB=target_volavgB,
                      tracing_tol=1e-8,
                      interpolant_degree=3,
                      interpolant_level=8,
                      bri_mpol=16,
                      bri_ntor=16)
  
  x0 = tracer.x0
  
  # compute the boozer field
  field,bri = tracer.compute_boozer_field(x0)
  
  # generate point in Boozer space
  s_label = 0.25
  ntheta=nzeta=128
  nfp = tracer.surf.nfp
  thetas = np.linspace(0, 2*np.pi, ntheta)
  zetas = np.linspace(0,2*np.pi/nfp, nzeta)
  # build a mesh
  [thetas,zetas] = np.meshgrid(thetas, zetas)
  stz_grid = np.zeros((ntheta*nzeta, 3))
  stz_grid[:, 0] = s_label
  stz_grid[:, 1] = thetas.flatten()
  stz_grid[:, 2] = zetas.flatten()

  # compute the determinant of the jacobian
  detjac = compute_det_jac_dcart_dbooz(field,stz_grid)

  # dump a pickle file
  import pickle
  outdata = {}
  outdata['vmec_input'] = vmec_input
  outdata['thetas'] = thetas
  outdata['zetas'] = zetas
  outdata['stz_grid'] = stz_grid
  outdata['detjac'] = detjac
  outdata['s_label'] = s_label
  outdata['ntheta'] = ntheta
  outdata['nzeta'] = nzeta
  outdata['nfp'] = nfp
  outfilename = "./angle_density_data.pickle"
  pickle.dump(outdata,open(outfilename,"wb"))
  
