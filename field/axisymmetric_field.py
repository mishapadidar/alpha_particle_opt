import numpy as np

class AxisymmetricField():
  """
  An Axisymmetric magnetic field. 
  We can use `B = grad(psi) \cross \grad(phi) + G*\grad(\phi)` where `G` is a constant,
  `\psi = (R-R0)^2 + Z^2` and `R,phi,Z` are cylindrical coordinates. 

  input:
    R0: float, major radius
    B0: float, bfield strength when R = R0, Z = 0. B0 relates to G via B0 = |G/R0|.

  return:
    magnetic field object.
  """

  def __init__(self, R0=1.0, B0=1.0):
    self.R0 = R0
    self.B0 = B0
    self.G  = np.abs(R0*B0)

  def B(self,xyz):
    """
    Evaluate the B field.

    B simplies to 
      B = -(z/R)*grad(R) + 2*((R-R0)/R)*e_z + G*grad(phi)
    where 
      grad(R) = (x/R,y/R,0)
      grad(phi) = (-y/R^2, x/R^2,0)
      e_z = (0,0,1)

    input
    xyz: 2d array of shape (N,3) Cartesian points.

    return 
    B: 2d array of shape (N,3) Bfield vectors in Cartesian coordinates.
    """ 

    R = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
    ret = np.vstack((-xyz[:,0]*xyz[:,2]/R/R - self.G*xyz[:,1]/R/R,
                     -xyz[:,1]*xyz[:,2]/R/R + self.G*xyz[:,0]/R/R, 
                     2*(R-self.R0)/R)).T
    return  ret

  def AbsB(self,xyz):
    b = self.B(xyz)
    ret =  np.sqrt(b[:,0]**2 + b[:,1]**2 + b[:,2]**2)

    #R = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
    #two = np.sqrt(xyz[:,2]**2 + 4*(R-self.R0)**2 + self.G**2)/R
    #diff = two - ret
    #print(diff)
    #print(np.max(np.abs(diff),axis=0))
    #quit()
    return ret
    

  def GradAbsB(self,xyz):
    """
    Evalute the gradient of |B|.

    |B| satisfies the equality


    input
    xyz: 2d array of shape (N,3) Cartesian points.

    return 
    1d array of shape (N,3) of grad(|B|) values.
    """
    R = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
    absB = self.AbsB(xyz)
    gradR = np.vstack((
                xyz[:,0]/R,
                xyz[:,1]/R,
                np.zeros(len(absB))
                ))
    # gradR terms
    ret = ((-absB/R + 4*(R-self.R0)/R/R/absB)*gradR).T
    # gradZ term
    last = xyz[:,2]/R/R/absB
    # divide by R^2
    ret[:,2] += last
    
    return ret
  
  def verify(self,nr=10, nphi=10, nz=10, rmin=1.0, rmax=2.0, zmin=-0.5, zmax=0.5):
      """
      Test the gradAbsB computation against finite difference.
      """
      assert rmin > 0, "axisymmetric field doesnt exist at rmin=0"
      rs = np.linspace(rmin, rmax, nr, endpoint=True)
      phis = np.linspace(0, 2*np.pi, nphi, endpoint=True)
      zs = np.linspace(zmin, zmax, nz, endpoint=True)

      R, Phi, Z = np.meshgrid(rs, phis, zs)
      X = R * np.cos(Phi)
      Y = R * np.sin(Phi)
      Z = Z
      xyz = np.vstack((X.flatten(),Y.flatten(),Z.flatten())).T
     
      # compute gradAbsB
      grad = self.GradAbsB(xyz)
      # forward difference
      h = 1e-6
      fp = np.array([self.AbsB(xx + h*np.eye(3)).flatten() for xx in xyz])
      fm = np.array([self.AbsB(xx - h*np.eye(3)).flatten() for xx in xyz])
      f0 = self.AbsB(xyz)
      grad_fd = (fp-fm)/2/h
      diff = grad - grad_fd
      print(diff)
      print(np.max(np.abs(diff),axis=0))

  def to_vtk(self, filename, nr=10, nphi=10, nz=10, rmin=1.0, rmax=2.0, zmin=-0.5, zmax=0.5):
      """Export the field evaluated on a regular grid for visualisation with e.g. Paraview."""
      assert rmin > 0, "axisymmetric field doesnt exist at rmin=0"
      from pyevtk.hl import gridToVTK
      rs = np.linspace(rmin, rmax, nr, endpoint=True)
      phis = np.linspace(0, 2*np.pi, nphi, endpoint=True)
      zs = np.linspace(zmin, zmax, nz, endpoint=True)

      R, Phi, Z = np.meshgrid(rs, phis, zs)
      X = R * np.cos(Phi)
      Y = R * np.sin(Phi)
      Z = Z

      xyz = np.vstack((X.flatten(),Y.flatten(),Z.flatten())).T

      vals = self.B(xyz).reshape((R.shape[0], R.shape[1], R.shape[2], 3))
      contig = np.ascontiguousarray
      gridToVTK(filename, X, Y, Z, pointData={"B": (contig(vals[..., 0]), contig(vals[..., 1]), contig(vals[..., 2]))})

if __name__=="__main__":
  R0 = 10.0
  B0 = 1.0
  bs = AxisymmetricField(R0,B0)
  bs.to_vtk('magnetic_field_axisymmetry',rmin=2.0,rmax=2*R0,nphi=50,zmin=-R0,zmax=R0,nr = 50,nz=50)
  #bs.verify(rmin=0.1,rmax=2*R0,nphi=50,zmin=-R0,zmax=R0,nr = 50,nz=50)
