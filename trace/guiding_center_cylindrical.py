import numpy as np

class GuidingCenter:
  """
  A class for computing the guiding center equations in cylindrical
  coordinates.
  """

  def __init__(self,B,gradAbsB,include_drifts=True):
    self.B = B
    self.gradAbsB = gradAbsB
    self.include_drifts = include_drifts
    self.PROTON_MASS = 1.67262192369e-27  # kg
    self.NEUTRON_MASS = 1.67492749804e-27  # kg
    self.ELEMENTARY_CHARGE = 1.602176634e-19  # C
    self.ONE_EV = 1.602176634e-19  # J
    self.ALPHA_PARTICLE_MASS = 2*self.PROTON_MASS + 2*self.NEUTRON_MASS
    self.ALPHA_PARTICLE_CHARGE = 2*self.ELEMENTARY_CHARGE
    self.FUSION_ALPHA_PARTICLE_ENERGY = 3.52e6 * self.ONE_EV # Ekin
    self.FUSION_ALPHA_SPEED_SQUARED = 2*self.FUSION_ALPHA_PARTICLE_ENERGY/self.ALPHA_PARTICLE_MASS
  
  def GC_rhs(self,X):
    """ 
    Right hand side of the vacuum guiding center equations in 
    cylindrical coordinates.
  
    input:
    X: (N,4) array of points in cylindrical coords (r,phi,z,vpar)
    
    return
    (N,4) array, time derivatives [dr/dt,dphi/dt,dz/dt,dvpar/dt]
    """
    # extract values
    r_phi_z = np.copy(X[:,:-1])
    vpar = np.copy(X[:,-1])
  
    # convert to xyz
    xyz = self.cyl_to_cart(r_phi_z)
  
    # compute B on grid
    Bb = self.B(xyz) # shape (N,3)
  
    # compute Bg on grid
    Bg = self.gradAbsB(xyz) # shape (N,3)
  
    # field values
    B = np.linalg.norm(Bb,axis=1) # shape (N,)
    b = (Bb.T/B).T # shape (N,3)
  
    vperp_squared = self.FUSION_ALPHA_SPEED_SQUARED - vpar**2 # shape (N,)
    c = self.ALPHA_PARTICLE_MASS /self.ALPHA_PARTICLE_CHARGE/B/B/B # shape (N,)
    mu = vperp_squared/2/B
  
    # compute d/dt (x,y,z); shape (N,3)
    if self.include_drifts:
      dot_xyz = ((b.T)*vpar + c*(vperp_squared/2 + vpar**2) * np.cross(Bb,Bg).T).T
    else:
      dot_xyz = ((b.T)*vpar).T
  
    # compute d/dt (r,phi,z); shape (N,3)
    dot_rphiz =self.jac_cart_to_cyl(r_phi_z,dot_xyz)
  
    # compute d/dt (vpar); shape (N,)
    dot_vpar = -mu*np.sum(b * Bg,axis=1)
  
    # compile into a vector; shape (N,4)
    dot_state = np.hstack((dot_rphiz,np.reshape(dot_vpar,(-1,1))))
    return dot_state

  def jac_cart_to_cyl(self,r_phi_z,D):
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

  def cyl_to_cart(self,r_phi_z):
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
