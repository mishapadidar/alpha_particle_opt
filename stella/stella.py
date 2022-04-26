import numpy as np
from scipy.interpolate import RegularGridInterpolator


class STELLA:
  """
  A semi-lagrangian method for solving the guiding center advection equation
  on a torus
    u_t + G @ grad(u) = 0
  where u is a function u(r,theta,phi,vpar,t) and G is a function G(r,theta,phi,vpar)
  that represents the rhs of the vacuum guiding center equations 
  in three spatial dimensions. 
  
  We assume stellarator symmetry, so that G is periodic across field periods.
  """
  
  def __init__(self,u0,B,Bgrad,R0,rmax,dr,dtheta,dphi,vparmax,vparmin,dvpar,dt,tmax,nfp):
    """
    """
    # initial distribution function u(r,z,vpar,0)
    self.u0 = u0
    # B field function B(r,z)
    self.B = B
    self.Bgrad = Bgrad
    self.nfp = nfp # number field periods
    # toroidal mesh sizes
    self.dr = dr
    self.dtheta = dtheta
    self.dphi = dphi
    self.dvpar = dvpar
    self.dt = dt
    # mesh bounds
    self.R0 = R0  # torus major radius
    self.rmax = rmax
    self.thetamax = 2*np.pi
    self.phimax = 2*np.pi/nfp
    self.vparmax = vparmax
    self.vparmin = vparmin
    self.tmin = 0.0
    self.tmax = tmax
 
    # mass # TODO: double check with ML
    self.mass = 1.67262192369e-27  # proton mass kg
    self.ONE_EV = 1.602176634e-19 # one EV
    self.charge = 1.0

  def GC_rhs(self,r,z,vpar):
    """ 
    Right hand side of the guiding center equations.
    Implements vacuum and non-vacuum equations.

    For now we simulate the simsopt gc_vac eqns.
    """
    #TODO: implement gc_vac eqns in toroidal coords
    #TODO: vectorize computations
    Bb = self.B(r,z)
    B = np.linalg.norm(Bb)
    b = B/B
    raise NotImplementedError

  def startup(self):
    """
    Evaluate u0 over the mesh.

    return: 2d array of evaluations U_{ij} = u0(r[i],theta[j])
    """
    # mesh spacing
    rmesh = np.arange(0.0,self.rmax,self.dr)
    thetamesh = np.arange(0.0,self.thetamax,self.dtheta)
    phimesh = np.arange(0.0,self.phimax,self.dphi)
    vparmesh = np.arange(0.0,self.vparmax,self.dvpar)
    n_r = len(rmesh)
    n_theta = len(thetamesh)
    n_phi = len(phimesh)
    n_vpar = len(vparmesh)

    # build the grid
    r_grid,theta_grid,phi_grid,vpar_grid = np.meshgrid(rmesh,thetamesh,
                             phimesh,vparmesh,indexing='ij',sparse=True)
    ## evaluate 
    #U_grid = np.zeros_like(r_grid)
    #for ii in range(n_r):
    #  for jj in range(n_theta):
    #    for kk in range(n_phi):
    #      for ll in range(n_vpar):
    #        U_grid[ii,jj,kk,ll] = self.u0(r_grid[ii,jj,kk,ll],theta_grid[ii,jj,kk,ll],
    #                         phi_grid[ii,jj,kk,ll],vpar_grid[ii,jj,kk,ll])

    # reshape the grid to points
    X = np.vstack((np.ravel(self.r_grid),np.ravel(self.r_grid),
            np.ravel(self.r_grid),np.ravel(self.r_grid))).T

    # compute initial density along the mesh, and reshape it
    U_grid = np.array([self.u0(*xx) for xx in X])
    U_grid = np.reshape(U_grid,np.shape(r_grid))

    # store the value of the density
    self.U_grid = np.copy(U_grid)

    # save the grids for interpolation
    self.r_grid = r_grid
    self.theta_grid = theta_grid
    self.phi_grid = phi_grid
    self.vpar_grid = vpar_grid
    return 

  def backstep(self):
    """
    Backwards integrate along the characteristic from time
    t_np1 to time t_n to find the deparature for each grid point. 
    Return the points (r,theta,phi,vpar) at the foot of the characteristic.
 
    return 
    X: (...,4) array of departuare points
    """
    
    # reshape the grid to points
    X = np.vstack((np.ravel(self.r_grid),np.ravel(self.r_grid),
            np.ravel(self.r_grid),np.ravel(self.r_grid))).T
    # backwards integrate the points
    G =  np.array([self.GC_rhs(*xx) for xx in X])
    # TODO: upgrade back integration
    Xtm1 = X - self.dt*G
    return Xtm1

  def interpolate(self,X):
    """
    Interpolate the value of u(r,theta,phi,vpar) from grid points to 
    the set of points X.

    input:
    X: (N,4) array, points at which to interpolate the value of u(x,y,z,vpar)

    return: 
    U: (N,) array, interpolated values of u(state) for state in X.
    """
    interpolator = RegularGridInterpolator((self.r_grid,self.theta_grid,
                   self.phi_grid,self.vpar_grid), self.U_grid)
    UX = interpolator(X)
    return UX

  def update_U_grid(self,UX):
    """
    Take a forward timestep in the advection equation
    in u to compute the value of u at the grid points at
    time t+1 from the interpolated points. 

    For our problem, U does not change along characteristics.
    So the value of U at the grid points at time t+1 is the
    value of U at the interpolated points. 
    """
    # reshape the list of evals into a grid shape
    self.U_grid = np.copy(np.reshape(UX,np.shape(self.U_grid)))
    return 
    

  def solve(self):
    """
    Perform the pde solve.
    """
    # build the mesh and evalute U
    self.startup()

    # TODO: double check
    # only backwards integrate once b/c time independent
    X = self.backstep()
    
    times = np.arange(self.tmin,self.tmax,self.dt)
    for tt in times:

      # intepolate the values of X
      UX = self.interpolate(X)
      # forward step: set U(grid,t+1) from UX
      self.update_U_grid(UX)
    
      # TODO: set up a vtk writer for the marginal of U_grid
    return 

  def compute_loss_fraction(self,pbndry):
    """
    Compute the loss fraction by integrating the probability density
    over the plasma boundary.

    input: plasma boundary should be a radial function, pndry(z,theta).
    return: probability mass of particles within pbndry.
    """
    raise NotImplementedError
