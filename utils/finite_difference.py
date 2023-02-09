import numpy as np

def forward_difference(f,x0,h=1e-6,return_evals=False):
  """
  Compute the jacobian of f with 
  forward difference
  """
  F0 = f(x0)
  Ep = x0 + h*np.eye(len(x0))
  Fp = np.array([f(e) for e in Ep])
  jac = (Fp - F0)/h
  if return_evals:
    return jac.T, F0, Ep, Fp
  else:
    return jac.T

def central_difference(f,x0,h=1e-6,return_evals=False):
  """Compute the jacobian of f with 
  central difference
  """
  h2   = h/2.0
  dim  = len(x0)
  Ep   = x0 + h2*np.eye(dim)
  Fp   = np.array([f(e) for e in Ep])
  Em   = x0 - h2*np.eye(dim)
  Fm   = np.array([f(e) for e in Em])
  jac = (Fp - Fm)/(h)
  if return_evals:
    return jac.T, F0, Em, Fm, Ep, Fp
  else:
    return jac.T

def finite_difference_hessian(f,x0,h=1e-6):
  """
  Second Order Central difference of a scalar
  valued function f at x0.
  """
  dim = len(x0)
  h2 = h/2
  f0 = f(x0)
  hess = np.zeros((dim,dim))
  E = h2*np.eye(dim)
  for ii in range(dim):
    for jj in range(ii,dim):
      if ii == jj:
        # compute the diagonal
        pij = f(x0 + 2*E[ii])
        pij -= 2*f0
        pij += f(x0 - 2*E[ii])
        hess[ii,jj] = pij/h/h
      else:
        # compute the upper and lower triangles
        pij = f(x0 + E[ii] + E[jj])
        pij -= f(x0 + E[ii] - E[jj])
        pij -= f(x0 - E[ii] + E[jj])
        pij += f(x0 - E[ii] - E[jj])
        hess[ii,jj] = pij/h/h
        hess[jj,ii] = pij/h/h # symmetrize hessian

  return hess

if __name__=="__main__":
    dim = 3
    A = np.random.randn(dim)
    func = lambda x: A @ x
    x0 = np.random.randn(dim)
    print(A - finite_difference(func,x0,h=1e-5))

    dim = 3
    A = np.random.randn(dim,dim)
    func = lambda x: A @ x
    x0 = np.random.randn(dim)
    print("")
    print(A - finite_difference(func,x0,h=1e-5))

    dim = 3
    A = np.random.randn(dim,dim)
    func = lambda x: x @ A @ x
    x0 = np.random.randn(dim)
    print("")
    print(A + A.T - finite_difference_hessian(func,x0,h=1e-2))

    dim = 3
    c = np.random.randn(dim)
    func = lambda x: np.cos(c @ x)
    x0 = np.random.randn(dim)
    print("")
    print(-np.cos(c @ x0)*np.outer(c,c) - finite_difference_hessian(func,x0,h=1e-4))
