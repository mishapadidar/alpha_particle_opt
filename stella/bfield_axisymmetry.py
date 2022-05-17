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
      B = (2*Z + G)*e_phi/R + 2*(R-R0)*e_z/R
    where 
      e_phi = (-sin(phi),cos(phi),0) = (-y/R,x/R,0)
      e_z = (0,0,1)

    input
    xyz: 2d array of shape (N,3) Cartesian points.

    return 
    B: 2d array of shape (N,3) Bfield vectors in Cartesian coordinates.
    """ 

    R = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
    a = (2*xyz[:,2] + self.G)/R
    ret = np.vstack((-a*xyz[:,1]/R,a*xyz[:,0]/R, 2*(R-R0)/R)).T
    return  ret

  def AbsB(self,xyz):
    b = self.B(xyz)
    return np.sqrt(b[:,0]**2 + b[:,1]**2 + b[:,2]**2)

  def GradAbsB(self,xyz):
    """
    Evalute the gradient of |B|.

    |B| satisfies the equality

      |B|^2 = 1/R^2 *[G^2 + 4*Z*G + 4*psi]

    so then grad(|B|) can be expressed as
      2*|B|*grad(|B|) = (-2/R^3)(grad(R))*[G^2 + 4*Z*G + 4*psi] + (4/R^2)*[G*e_z + grad(psi)]
                      = (-2/R)(grad(R))|B|^2 + (4/R^2)*[G*e_z + grad(psi)]
    simplifying gives us
      grad(|B|) = -1/R*grad(R)|B| + (2/R^2)*[G*e_z + grad(psi)]/|B|
    where 
      e_z = (0,0,1), 
      grad(R) = (x/R,y/R,0) = (cos(phi),sin(phi),0)
    and 
      grad(psi) = 2*((R-R0)cos(phi),(R-R0)sin(phi),z)
                = 2*((R-R0)x/R,(R-R0)y/R,z)

    input
    xyz: 2d array of shape (N,3) Cartesian points.

    return 
    1d array of shape (N,3) of grad(|B|) values.
    """
    R = np.sqrt(xyz[:,0]**2 + xyz[:,1]**2)
    psi = (R - self.R0)**2 + xyz[:,2]**2
    absB = np.sqrt(self.G**2 + 4*xyz[:,2]*self.G + 4*psi)/R
    # gradR and e_z term
    first = np.vstack((
                -xyz[:,0]*absB,
                -xyz[:,1]*absB,
                2*self.G/absB
                )).T
    # grad(psi) term
    last = np.vstack((
           2*2*(R-R0)*xyz[:,0]/R/absB,
           2*2*(R-R0)*xyz[:,1]/R/absB,
           2*2*xyz[:,2]/absB,
           )).T
    # divide by R^2
    ret = (first + last).T/R/R
    
    return ret.T
  
  def verify(self,nr=10, nphi=10, nz=10, rmin=1.0, rmax=2.0, zmin=-0.5, zmax=0.5):
      """
      Test the gradAbsB computation against finite difference.
      """
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
  bs.to_vtk('magnetic_field_axisymmetry')
  bs.verify(rmin=0.1,rmax=R0,zmin=0.0,zmax=R0)
