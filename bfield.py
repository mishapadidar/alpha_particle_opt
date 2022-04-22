import numpy as np

def bfield1(r,theta,phi,R0):
  """
  A purely toroidal B-field. Returns
  B-field in cartesian coordinates.
  input: r,theta,phi
  return: [B_x, B_y, B_z]
  """
  if r>=0.4*R0 and r <= 1*R0:
    x = -(R0 + r*np.cos(theta))*np.sin(phi)
    y = (R0 + r*np.cos(theta))*np.cos(phi)
    z = 0.0
    return x,y,z
  else:
    return 0.0,0.0,0.0

def write_Bfield_vtk(path='./data/bfield.vtu',R0=1.0,n_r=10,n_theta=10,n_phi=10):
  """
  Write a vtk file with points and B-field values.
  """
  from coords import tor_to_cart
  R,Theta,Phi = np.meshgrid(np.linspace(0.2, 0.8, n_r),
                        np.linspace(0.0,2*np.pi, n_theta),
                        np.linspace(0.0,2*np.pi, n_phi),indexing='ij')
  X = np.zeros(0)
  Y = np.zeros(0)
  Z = np.zeros(0)
  Bx = np.zeros(0)
  By = np.zeros(0)
  Bz = np.zeros(0)
  for ii in range(n_r):
    for jj in range(n_theta):
      for kk in range(n_phi): 
        r,theta,phi = R[ii,jj,kk],Theta[ii,jj,kk],Phi[ii,jj,kk]
        x,y,z = tor_to_cart(r,theta,phi,R0)
        bx,by,bz = bfield1(r,theta,phi,R0)
        X = np.append(X,x)
        Y = np.append(Y,y)
        Z = np.append(Z,z)
        Bx = np.append(Bx,bx)
        By = np.append(By,by)
        Bz = np.append(Bz,bz)

  from vtkClass import VTK_XML_Serial_Unstructured 
  vtk_writer =VTK_XML_Serial_Unstructured()
  colors   = np.sqrt(Bx**2 + By**2 + Bz**2)
  vtk_writer.snapshot(path, X,Y,Z,[],[],[],Bx,By,Bz,[],colors)	#show velocity in vtu 

def write_Bsurface_vtk(path='./data/surface',R0=1.0,n_r=10,n_theta=10,n_phi=10):
  """
  Plot the value of |B| over a torus.
  """
  from coords import tor_to_cart
  R,Theta,Phi = np.meshgrid(np.linspace(0.2, 0.8, n_r),
                        np.linspace(0.0,2*np.pi, n_theta),
                        np.linspace(0.0,2*np.pi, n_phi),indexing='ij')
  X = np.zeros_like(R)
  Y = np.zeros_like(R)
  Z = np.zeros_like(R)
  B = np.zeros_like(R)
  for ii in range(n_r):
    for jj in range(n_theta):
      for kk in range(n_phi): 
        r,theta,phi = R[ii,jj,kk],Theta[ii,jj,kk],Phi[ii,jj,kk]
        x,y,z = tor_to_cart(r,theta,phi,R0)
        bx,by,bz = bfield1(r,theta,phi,R0)
        X[ii,jj,kk] = x
        Y[ii,jj,kk] = y
        Z[ii,jj,kk] = z
        B[ii,jj,kk] = np.sqrt(bx**2 + by**2 + bz**2)

  from pyevtk.hl import gridToVTK 
  gridToVTK(path, X,Y,Z,pointData = {'B':B})

if __name__=="__main__":
  write_Bfield_vtk()
  write_Bsurface_vtk()
 


