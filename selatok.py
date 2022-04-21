import numpy as np


class SELATOK:
  """
  A semi-lagrangian method for solving the guiding center advection equation
  on a cylinder 
    u_t + G @ grad(u) = 0
  where u is a function u(r,theta,z,vpar,t) and G is a function G(r,theta,z,vpar)
  that represents the rhs of the vacuum guiding center equations 
  in three spatial dimensions. We assume that the magnetic field B is
  axis-symmetric and time-independent, B = B(r,theta).
  
  We assume stellarator symmetry, so that G is periodic across field periods.
  Thus we use periodic boundary conditions on each field period for z.
  """
  
  def __init__(self,u0,dr,dtheta,dz,dt,tmax,nfp,Rmajor,Rminor):
    """
    """
    # initial distribution function u(r,z,vpar,0)
    self.u0 = u0
    # B field function B(r,z)
    self.B = B
    # plasma boundary function p(z)
    #self.pbndry = pbndry
    self.nfp = nfp # number field periods
    self.Rmajor = Rmajor # major radius
    self.Rminor = Rminor # minor radius
    # step/mesh sizes
    self.dr = dr
    self.dtheta = dtheta
    self.dz = dz
    self.dt = dt
    # mesh bounds
    self.rmax = Rmajor + dr
    self.thetamax = 2*np.pi
    self.zmax = 2*np.pi/nfp
    self.tmin = 0.0
    self.tmax = tmax

    # mesh
    self.rmesh = np.arange(0.0,self.rmax,self.dr)
    self.thetamesh = np.arange(0.0,self.thetamax,self.dtheta)
    
    # mass # TODO: double check
    self.mass = 1.67262192369e-27  # proton mass kg
    self.ONE_EV = 1.602176634e-19 # one EV

  def GC_rhs(self,r,z,vpar):
    """ 
    Right hand side of the guiding center equations.
    Implements vacuum and non-vacuum equations.

    For now we simulate the simsopt gc_vac eqns.
    """
    #TODO: implement gc_vac eqns
    Bb = self.B(r,z)
    B = np.linalg.norm(Bb)
    b = B/B
    raise NotImplementedError

  def startup(self):
    """
    Evaluate u0 over the mesh.

    return: 2d array of evaluations U_{ij} = u0(r[i],theta[j])
    """
    R,T = np.meshgrid(self.rmesh,self.thetamesh)
    U = np.zeros_like(R)
    for ii,x in enumerate(R):
      for jj,y in enumerate(T):
        U[ii,jj] = self.u0(R[ii,jj],T[ii,jj])
    return U

  def backtrack(self):
    """
    Backwards integrate along the characteristic from time
    t_np1 to time t_n. Return the points (x,y,z,vpar) at the 
    origin of the characteristic.
    """
    raise NotImplementedError

  def interpolate(self):
    """
    Interpolate the value of u(x,y,z,vpar) from grid points.
    """
    raise NotImplementedError

  def solve(self):
    """
    Perform the pde solve.
    """
    raise NotImplementedError

  def compute_loss_fraction(self,pbndry):
    """
    Compute the loss fraction by integrating the probability density
    over the plasma boundary.

    input: plasma boundary should be a radial function, pndry(z,theta).
    return: probability mass of particles within pbndry.
    """
    raise NotImplementedError
