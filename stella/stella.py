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

  Potential problems
  - We are fixing the probability density to zero, for r > rmax. This would destroy 
    probability mass for particles that exist then reenter the plasma. Hence, we should
    set rmax >> the plasma minor radius, to any particle that exists and then 
    reenters the plasma, remains on the grid the for its entire trajectory.
  - the coordinate system does not uniquely represent points, i.e. r=0, theta in [0,2pi]
    all define the same point in Cartesian coordinates. This may be problemtic because we 
    may be defining the density to different values at the same point in Cartesian space.
    This may also artificially create or destroy probability mass.
  - Our method does not conserve mass. For this we should use a mass-corrector method.
  - The boundary of the plasma may be problematic because it represents a discontinuity in
    the density function. So we should have high resolution grid/interpolation there. 
    Furthermore, there may be problems with the fractal like structure generated by the ODE.
  """
  
  def __init__(self,u0,B,Bgrad,R0,rmax,dr,dtheta,dphi,vparmax,vparmin,
    dvpar,dt,tmax,nfp,integration_method):
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

    # for backwards integration
    assert integration_method in ['euler','midpoint','rk4']
    self.integration_method = integration_method
 
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
    # TODO:
    # These equations should be able to handle a negative value of r
    # and theta, phi not within [0, 2pi]
    # TODO: implement gc_vac eqns in toroidal coords
    # TODO: vectorize computations
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
    # TODO: vectorize u0
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
    t_np1 to time t_n to find the deparature points. 
    Return the departure points (r,theta,phi,vpar).

    We have implemented three choices of methods: Euler's method, the
    Explicit Midpoint method, and the 4th order Runge Kutta. 
 
    return 
    X: (...,4) array of departuare points
    """
    
    # reshape the grid to points
    X = np.vstack((np.ravel(self.r_grid),np.ravel(self.theta_grid),
            np.ravel(self.phi_grid),np.ravel(self.vpar_grid))).T
    # backwards integrate the points
    if self.integration_method == "euler":
      # TODO: verify correctness
      G =  np.array([self.GC_rhs(*xx) for xx in X])
      Xtm1 = X - self.dt*G
    elif self.integration_method == "midpoint":
      # TODO: verify correctness
      G =  np.array([self.GC_rhs(*xx) for xx in X])
      Xstar = np.copy(X - self.dt*G/2)
      G =  np.array([self.GC_rhs(*xx) for xx in Xstar])
      Xtm1 = X - self.dt*G/2

    elif self.integration_method == "rk4":
      # TODO: verify correctness
      G = np.array([self.GC_rhs(*xx) for xx in X])
      k1 = np.copy(dt*G)
      G = np.array([self.GC_rhs(*xx) for xx in X+k1/2])
      k2 = np.copy(dt*G)
      G = np.array([self.GC_rhs(*xx) for xx in X+k2/2])
      k3 = np.copy(dt*G)
      G = np.array([self.GC_rhs(*xx) for xx in X+k3])
      k4 = np.copy(dt*G)
      Xtm1 = np.copy(X -(k1+2*k2+2*k3+k4)/6)
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
    # TODO: should we use sparse arrays for U_grid to avoid computation
    # with zeros?
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

  def apply_boundary_conds(self,X):
    """
    Correct the departure points and apply the boundary conditions.
    Some departure points may have r < 0 or theta,phi not within [0,2pi], 
    of vpar not within [-vpar,vpar]. 
    To ensure that the departure points have the correct interpolated value
    of the density, we correct [r,theta,phi,vpar] so that the point lies within
    the grid. 
    If r is negative, we set r = abs(r) and increase theta by 2pi.
    Since theta,phi are periodic we simply apply the periodic boundary conditions
    to map them back to the domain [0,2pi].
    We apply periodic boundary conditions to vpar. 

    Note there is still a multiplicity in how the toroidal axis is defined, i.e.
    for a fixed phi and vpar, and r=0, any value of theta defines the same point 
    in cartesian coordinates.
    """
    # TODO: set dirichlet boundary conditions at rmax, so that u = 0 if r > r_max
    # correct r
    neg_r = X[:,0]<0
    X[:,0] = np.abs(X[:,0]) # set r > 0
    X[:,1][neg_r] += (2*np.pi) # correct theta
    # correct theta
    X[:,1] %= (2*np.pi) # put theta in [-2pi,2pi]
    X[:,1][X[:,1]<0] += (2*np.pi) # correct negative thetas
    # correct phi
    X[:,2] %= self.phimax # phi in [-2pi/nfp,2pi/nfp]
    X[:,2][X[:,2]<0] += self.phimax # correct negative phis
    # correct vpar
    vpardiff = self.vparmax - self.vparmin
    idx_up =  X[:,3] > self.vparmax
    X[:,3][idx_up] = self.vparmin + (X[:,3][idx_up]-vparmax)%vpardiff
    idx_down =  X[:,3] < self.vparmin
    X[:,3][idx_up] = self.vparmax - (vparmin-X[:,3][idx_down])%vpardiff

    return np.copy(X)
    

  def solve(self):
    """
    Perform the pde solve.
    """
    # build the mesh and evalute U
    self.startup()

    # only backwards integrate once b/c time independent
    X = self.backstep()

    # apply the boundary conditions to correct X
    X = self.apply_boundary_conds(X)
    
    times = np.arange(self.tmin,self.tmax,self.dt)
    for tt in times:

      # intepolate the values of X
      UX = self.interpolate(X)
      # forward step: set U(grid,t+1) from UX
      self.update_U_grid(UX)
    
      # TODO: set up a vtk writer for the marginal of U_grid
    return 

  def compute_spatial_marginal(self):
    """ 
    Compute the marginal density over the spatial variables.
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
