import numpy as np


def NoiseTolerantGradientDescent(func,grad,x0,eps_f=0.0,alpha0 = 1.0,gamma=0.5,max_iter=10000,gtol=1e-3,c_1=1e-4,verbose=False):
  """
  Gradient descent with sim failure safe and noise tolerant armijo linesearch.
  We assume that the function has noise that is bounded by eps_f and that the
  gradients are noise-free.
  Optimization will stop if any of the stopping criteria are met.

  func: objective function handle, for minimization
  grad: gradient handle
  x0: feasible starting point
  eps_f: a bound on the noise level in f
  gamma: linesearch decrease parameter
  max_iter: maximimum number of iterations
  gtol: projected gradient tolerance
  c_1: Armijo parameters for linesearch.
           must satisfy 0 < c_1 < c_2 < 1
  """
  assert (0 < c_1 and c_1< 1), "unsuitable linesearch parameters"

  # inital guess
  x_k = np.copy(x0)
  dim = len(x_k)

  # minimum step size
  alpha_min = 1e-18
  # initial step size
  alpha_k = alpha0

  # compute gradient
  g_k    = np.copy(grad(x_k))
  # compute function value
  f_k    = np.copy(func(x_k))

  # stop when gradient is flat (within tolerance)
  nn = 0
  stop = False
  while stop==False:
    if verbose:
      print(f_k,np.linalg.norm(g_k))
    # increase alpha to counter backtracking
    alpha_k = alpha_k/gamma

    # compute search direction
    p_k = - g_k

    # compute step
    x_kp1 = x_k + alpha_k*p_k
    f_kp1 = func(x_kp1);

    # linsearch with Armijo condition
    armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1 - x_k) - eps_f
    while armijo==False:
      # reduce our step size
      alpha_k = gamma*alpha_k;
      # take step
      x_kp1 = np.copy(x_k + alpha_k*p_k)
      # f_kp1
      f_kp1 = func(x_kp1);
      if np.isfinite(f_kp1) == False:
        armijo = False
      else:
        # compute the armijo condition
        armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1 - x_k) - eps_f

      # break if alpha is too small
      if alpha_k <= alpha_min:
        if verbose:
          print('Exiting: alpha too small.')
        return x_k

    # gradient
    g_kp1 = np.copy(grad(x_kp1))

    # reset for next iteration
    x_k  = np.copy(x_kp1)
    f_k  = f_kp1;
    g_k  = np.copy(g_kp1);

    # update iteration counter
    nn += 1

    # stopping criteria
    if np.linalg.norm(g_k) <= gtol:
      if verbose:
        print('Exiting: gtol reached')
      stop = True
    elif nn >= max_iter:
      if verbose:
        print('Exiting: max_iter reached')
      stop = True

  return x_k


