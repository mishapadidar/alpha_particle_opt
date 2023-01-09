import numpy as np
from scipy.linalg import null_space

"""
Functions for building quadratic interpolation models
"""

def quadratic_features(X):
    """
    Form the set of dim_x*(dim_x+1)/2 quadratic features
    of a symmetric quadratic model.
      feat = [x1^2, ..., xn^2, x1*x2,...,x_{n-1}x_n]
    A quadratic model m(x) = beta @ feat is equivalent to 
    m(x) = x @ B @ x where
        B = np.zeros((dim_x,dim_x))
        B[np.triu_indices(dim_x)] = beta
        D = np.diag(np.diag(B))
        B = B - D # remove the diagonal
        B = (B + B.T) # symmetrize the matrix
        B = B + D # add the diagonal back in
    X: (N,dim_x) array of points
    return
    feat: (N,dim_x*(dim_x+1)/2) array of quadratic features. 
    """
    N,dim_x = np.shape(X)
    n_feat = int(dim_x*(dim_x+1)/2)
    feat = np.zeros((N,n_feat))
    for ii,x in enumerate(X):
      outer = np.outer(x,x)
      x_feat = outer[np.triu_indices(dim_x)]
      feat[ii,:] = np.copy(x_feat)
    return np.copy(feat)


def minimumNormHessianModel(x0,f0,X,FX):
  """
  Given an interpolation set, compute the coefficients
  of the minimum norm hessian quadratic interpolation model
    m(x) = f0 + alpha @ (x-x0) + beta @ quadratic_features(x-x0)
  
  Let Y be the row vectors for the interpolation
  set, centered around zero. Let Z be an orthogonal basis for 
  the null space of Y.T, and Q@R = Y. Let NY.T be the quadratic
  feature vectors, as rows.
  
  alpha and beta satisfy equtions (3.5) and (3.6) of Stefan
  Wild's MNH paper:
  Then there exist a unique alpha, beta, w which satisfy
    Z.T @ NY.T @ NY @ Z @ w = Z.T @ fY
    R @ alpha = Q.T @ (fY - NY.T @ NY @ Z@ w)
    beta = NY @ Z @ w
  where fY are the function values. lambda = Z @ w are the lagrange
  multipliers, and the array shapes are
    w: |len(Y)| - dim_x - 1 
    alpha: (dim_x,)
    beta: (dim_x*(dim_x+1)/2,)

  input:
    x0: center point of the quadratic model, array (dim_x,)
    f0: f(x0), float
    X: interpolation points, array (N,dim_x)
    FX: interpolation values f(X), array (N,)
  return:
    alpha: gradient approximation, array (dim_x,)
    beta: Hessian approximation, array (dim_x*(dim_x+1)/2,) array
  """
  # shift the points
  _Y = np.copy(X - x0)
  # shift the function values
  _fY = FX - f0
  # form quadratic features
  _NYT = quadratic_features(_Y) # row vectors
  # compute Null(_Y.T)
  _Z = null_space(_Y.T) # col vectors
  # QR factor _NYT @ Z
  Q,R = np.linalg.qr(_NYT.T @ _Z)
  # solve for lagrange multipliers
  w = np.linalg.solve(R.T @ R, _Z.T @ _fY)
  # solve for beta
  beta = _NYT.T @ _Z @ w
  # QR factor _Y
  Q,R = np.linalg.qr(_Y)
  # solve for alpha
  alpha = np.linalg.solve(R,Q.T @ (_fY - _NYT @ beta))
  return alpha,beta
