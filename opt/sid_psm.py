import numpy as np
from scipy.linalg import null_space

class SIDPSM():
  """
  The SID-PSM direct search method for incorporating linear
  models in the direct search procedure. See 
  "Incorporating minimum Frobenius norm models in direct search" - Custudio 2009
  "Using sampling and simplex derivatives in pattern search methods" - Custudio 2007
  """

  def __init__(self,obj,x0,delta=0.01,eta=1e-5,max_eval=100,delta_max=1.0,delta_min=1e-6,theta=1e-3):
    """
    input:
    obj: function handle for minimization
    x0: starting point, 1d array
    delta: initial trust region size
    eta: sufficient decrease condition parameter in (0,1)
    max_eval: maximum number of evaluations, a stopping criteria
    delta_max: maximum trust region size, typically a few orders of magnitude larger
               than the initial size
    delta_min: smallest trust region size, a stopping criteria
    theta: threshold to determine validity of interpolation set, in (0,1)
    """

    self.obj = obj
    self.x0 = x0
    self.dim_x = len(x0)
    self.delta0 = delta
    self.eta = eta
    self.max_eval = max_eval
    self.delta_min = delta_min
    self.delta_max = delta_max
    self.theta = theta
    self.gamma_inc = 2.0

    # storage
    self.X = np.zeros((0,self.dim_x))
    self.fX = np.zeros(0)

    # for ill-conditioning
    self.n_fails = 0
    self.max_fails = 3

    assert (max_eval > self.dim_x + 1), "Need more evals"
    assert delta > 0, "Need larger TR radius"
    assert (delta_min > 0 and delta_min < delta), "Invalid delta_min"
    assert (0 < eta and eta < 1), "Invalid eta"

  def solve(self):

    x0 = np.copy(self.x0)
    f0 = self.obj(self.x0)
    delta_k = self.delta0

    # positive spannin set
    I = np.eye(self.dim_x)
    e = np.ones(self.dim_x)
    pos_basis = np.vstack((e,-e,I,-I))

    # initial sampling
    _X  = x0 + delta_k*np.eye(self.dim_x)
    _fX = np.array([self.obj(xx) for xx in _X])

    # storage
    self.X  = np.copy(_X)
    self.X  = np.vstack((self.X,x0))
    self.fX = np.copy(_fX)
    self.fX = np.copy(np.append(self.fX,f0))

    ## build the interpolation set
    #idx_best = np.argmin(self.fX)
    #x_k = np.copy(self.X[idx_best])
    #f_k = np.copy(self.fX[idx_best])
    #self.n_evals = len(self.fX)
    #_X,_fX = self.get_model_points(x_k,f_k,delta_k)
    #print(self.n_evals)

    # TODO: remove
    # select the best point to be the center
    idx_best = np.argmin(self.fX)
    idx_other = [ii for ii in range(len(self.fX)) if ii != idx_best]
    x_k = np.copy(self.X[idx_best])
    f_k = np.copy(self.fX[idx_best])
    _X  = np.copy(self.X[idx_other])
    _fX = np.copy(self.fX[idx_other])
    self.n_evals = len(self.fX)
    # only keep points with finite value
    idx_keep = np.isfinite(_fX)
    _fX = _fX[idx_keep]
    _X = _X[idx_keep]

    # determine if we start with poll or not
    if len(_X) < self.dim_x:
      poll = True
    else:
      poll = False

    while self.n_evals < self.max_eval and delta_k > self.delta_min:

      # poll step
      if poll:
        # collect all points in the expanded TR
        _X,_fX = self.get_model_points(x_k,f_k,2*delta_k)
        if len(_X) < self.dim_x:
          """
          Sort mesh points geometrically. We assume the objective can be modeled
          accurately by a linear model f(x) = f(x_k) + g @ (x-x_k). 
          Hence this linear model g should satisfy 
            sign(f(x) - f_k) = sign( g @ (x-x_k))
          Given evaluations _X and _fX, we can find a polytope which g must live in:
            K = {y | sign( (_X - x_k ) @ y) = sign(_fX - f_k) }
          Since ideally we want to take steps in the negative gradient direction we 
          should sort mesh points based on their similarity to the negative gradient.
          We evaluate similarity of a mesh point y to the negative gradient by counting
            sim(y) = sum[ sign( (_X - x_k ) @ -y) = sign(_fX - f_k)]
          """
          val = np.sum(np.sign(-pos_basis @ (_X-x_k).T) == np.sign(_fX - f_k),axis=1)
        else:
          # sort the polling points according to the linear model on the expanded TR
          m_k = self.make_model(x_k,f_k,_X,_fX)
          val = pos_basis @ m_k

        # TODO: poll step should be aware of the simulation failures
        # to improve the sampling scheme similar to nelder-mead.
        #

        # sort the postive spanning set
        idx_sort = np.argsort(val)
        pos_basis_k = np.copy(pos_basis[idx_sort])
        # evaluate the new points
        success = False
        c = 0
        for p_k in pos_basis_k: 
          # evaluate point on mesh
          x_plus = x_k + delta_k*p_k
          f_plus = self.obj(x_plus)
          # save eval
          self.X  = np.copy(np.vstack((self.X,x_plus)))
          self.fX = np.copy(np.append(self.fX,f_plus))
          self.n_evals +=1
          c += 1
          # TODO:
          # we should be selecting our point to satisfy a sufficient decrease condition
          # otherwise convergence will be slow!
          # 
          # break if decrease is found
          if f_plus < f_k:            
            success=True
            break

        if success:
          x_kp1 = np.copy(x_plus)
          f_kp1 = f_plus
          # expand TR
          delta_kp1 = min(self.gamma_inc*delta_k,self.delta_max)
        else:
          x_kp1 = np.copy(x_k)
          f_kp1 = f_k
          # shrink TR
          delta_kp1 = delta_k/self.gamma_inc

      # search step
      else:
        # form the linear model 
        m_k = self.make_model(x_k,f_k,_X,_fX)
        # solve the TR subproblem
        y_plus = self.solve_subproblem(m_k,delta_k)
        # shift back to original domain
        x_plus = np.copy(x_k + y_plus)
        # evaluate model and objective
        m_plus = f_k + m_k @ y_plus
        f_plus = self.obj(x_plus)
        self.n_evals += 1
      
        # save eval
        self.X  = np.copy(np.vstack((self.X,x_plus)))
        self.fX = np.copy(np.append(self.fX,f_plus))

        # do ratio rest
        rho = (f_k - f_plus)/(f_k - m_plus)
        # choose next iterate
        if rho >= self.eta:
          x_kp1 = np.copy(x_plus)
          f_kp1 = f_plus
        else:
          x_kp1 = np.copy(x_k)
          f_kp1 = f_k
        # shrink/expand TR
        if rho >= self.eta and np.linalg.norm(x_k - x_plus) >= 0.75*delta_k:
          delta_kp1 = min(self.gamma_inc*delta_k,self.delta_max)
        elif rho >= self.eta:
          delta_kp1 = delta_k
        else:
          delta_kp1 = delta_k/self.gamma_inc

      # get a new model
      _X,_fX = self.get_model_points(x_kp1,f_kp1,delta_kp1)
      if len(_X) < self.dim_x:
        poll = True
      else:
        poll = False

      # prepare for next iteration
      x_k = np.copy(x_kp1)
      f_k = f_kp1
      delta_k = delta_kp1

    result = {}
    result['x']   = x_k
    result['f']   = f_k
    result['X']   = np.copy(self.X)
    result['fX']  = np.copy(self.fX)
    return result
    
  def make_model(self,x0,f0,_X,_fX):
    """
    Build the linear model from interpolation 
    points.

    input:
    x0: 1d array, (dim_x,), Trust region center
    f0: float, function value at x0
    _X: 2d array, (dim_x,dim_x), interpolation points
        points are rows.
    _fX: 2d array, (dim_x,), interpolation values
 
    output:
    m: 1d array, linear model such that
     f(x) ~ f0 + m @ (x-x0)
    """
    # shift the points
    _Y = np.copy(_X - x0)
    # shift the function values
    _fY = _fX - f0
    # interpolate
    try:
      m = np.linalg.solve(_Y,_fY)
    except:
      # jitter if unstable
      _Y = self.jitter(_Y)
      self.n_fails +=1 
      # increase theta to prevent more fails
      self.theta = 10*self.theta
      if self.n_fails >= self.max_fails:
        print("Exiting: Too many failed solves")
        print("Try increasing theta")
        result = {}
        result['x']   = x0
        result['f']   = f0
        result['X']   = np.copy(self.X)
        result['fX']  = np.copy(self.fX)
        return x0
      m = np.linalg.solve(_Y,_fY)
    return np.copy(m)

  def solve_subproblem(self,m,delta):
    """
    Solve the Trust region subproblem
      min_x  m @ x
      s.t. ||x|| <= delta
    Problem is solved around the origin.
    So solution should be shifted back around
    trust region center.
    
    m: 1d array, (dim_x,), linear model
    delta: float, trust region radius
    """
    return np.copy(-delta*m/np.linalg.norm(m))

  def get_model_points(self,x0,f0,delta):
    """
    A model selection and improvement routine. 

    This function attempts to return a set of dim_x sufficiently affinely 
    independent points. It does not accept points at which the objective
    value is np.inf into the set.

    Find a set of affinely independent points, within the eval history.
    If no such set exists, propose new evaluation points
    to complete the set.
  
    This method is essentially the AffPoints method from Stefan Wild's
    ORBIT algorithm.
  
    This algorithm ensures, by
    the analysis in Stefan's 2013 paper, Global convergence of radial 
    basis function trust-region algorithms for derivative-free optimization,
    that our model is fully linear.
  
    First we seek a set of sufficiently affinely independent points within a radius
    delta of x0. sufficiently affinely independent is determined by a tolerance 
    theta. If such a set of points exists, then our model will be fully linear on 
    a radius delta. 
    If we still do not have a complete set we 
    evaluate model improvement points.
    """
    # storage for new displacements
    _Y = np.zeros((0,self.dim_x))
    _fY = np.zeros(0)
  
    # rows are basis for null(_Y)
    _Z = np.eye(self.dim_x)     
  
    # default
    linear = False

    # find points within distance delta
    idx = np.linalg.norm(self.X-x0,axis=1) <= delta
    # use shifted points
    Yt   = self.X[idx] - x0
    fYt  = self.fX[idx] - f0
    for jj,dj in enumerate(Yt):
      fj = fYt[jj]
      # skip points with unbounded objective
      if not np.isfinite(fj):
        continue
      # check |proj_Z(dj/delta)| >= theta
      if np.linalg.norm(_Z.T @ (_Z @ dj/delta)) >= self.theta:
        _Y  = np.copy(np.vstack((_Y,dj)))
        _fY = np.copy(np.append(_fY,fj))
        # find new null space
        _Z = null_space(_Y).T
      if len(_Y) == self.dim_x:
        linear = True
        break
      else:
        linear = False
  
    # now propose new points
    if linear == False:
      # evaluate f(Z)
      _fZ = np.array([self.obj(x0 + delta*zz) for zz in _Z]) - f0
      for jj,zz in enumerate(_Z):
        fj = _fZ[jj]
        # dont add points with unbounded value to TR list
        if np.isfinite(fj):
          _Y = np.vstack((_Y,delta*zz))
          _fY = np.append(_fY,fj)
      # save the new evals
      self.n_evals += len(_fZ)
      self.X  = np.copy(np.vstack((self.X, x0 + delta*_Z)))
      self.fX = np.copy(np.append(self.fX, f0 + _fZ))

    # return interpolation points
    _X = np.copy(x0 + _Y)
    _fX = np.copy(f0 + _fY)
  
    return _X,_fX


  def jitter(self,A,jit=1e-10):
    """
    Add a "jitter" to the matrix
  
    input
    A: (n,n) matrix

    return 
    (n,n) matrix, A + jit*np.eye(n)
    """
    return np.copy(A + jit*np.eye(len(A)))
    
  
