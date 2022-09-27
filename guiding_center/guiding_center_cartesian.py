import numpy as np

class GuidingCenter:
  """
  A class for computing the guiding center equations in cartesian
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
    cartesian coordinates.
  
    input:
    X: (N,4) array of points in cartesian coords (x,y,z,vpar)
    
    return
    (N,4) array, time derivatives [dx/dt,dy/dt,dz/dt,dvpar/dt]
    """
    # extract values
    xyz = np.copy(X[:,:-1])
    vpar = np.copy(X[:,-1])
  
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
  
    # compute d/dt (vpar); shape (N,)
    dot_vpar = -mu*np.sum(b * Bg,axis=1)
  
    # compile into a vector; shape (N,4)
    dot_state = np.hstack((dot_xyz,np.reshape(dot_vpar,(-1,1))))
    return dot_state

