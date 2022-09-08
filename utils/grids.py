import numpy as np

def chebyshev_grid(a,b,n):
  """
  Create a one dimensional grid using chebyshev nodes,
  see https://en.wikipedia.org/wiki/Chebyshev_nodes
  for a definition.

  input: 
  a,b: float, interval bounds
  n: int, number of grid points

  return:
  x: 1d array, entries are chebyshev nodes.
  """
  k = np.arange(1,n+1,1)
  arg = np.pi*(2*k-1)/2/n
  ret = 0.5*(a+b) + 0.5*(b-a)*np.cos(arg)
  return np.copy(ret[::-1])

def loglin_grid(a,b,n):
  """
  Make a grid with log and linear spaced points

  a,b: endpoints
  n: number of points, must be even

  return: grid of log linear spaced points
  
  WARNING: the grid may have less than n points if the log and linear
    grids overlap.
  """
  assert a<b,"a must be strictly less than b"
  assert n % 2 == 0, "n must be even"

  n2 = int(n/2)
  x1 = np.linspace(a,b,n2)
  x2 = symlog_grid(a,b,n2)
  # np.unique may drop some points
  x = np.sort(np.unique(np.append(x1,x2)))

  return x

def symlog_grid(a,b,n):
  """
  Create a logspaced grid over potentially negative numbers.
  
  a,b: endpoints
  n: number of points on grid
  
  return: logspaced grid. if a is negative or 0 we create
    a symlog grid where the threshold is determined by n. 
  """
  assert a < b, "a must be strictly less than b"
  assert type(n) == int, "n must be of type int"
  assert n > 0, "n must be positive"


  if (a>0) & (b>0):
    la = np.log10(abs(a))
    lb = np.log10(abs(b))
    return np.logspace(la,lb,n)
  elif (a<0) & (b<0):
    la = np.log10(abs(a))
    lb = np.log10(abs(b))
    return -np.logspace(la,lb,n)
  elif (a==0) & (b>0):
    lb = np.log10(abs(b))
    return np.logspace(lb-n+1,lb,n)
  elif (a<0) & (b==0):
    la = np.log10(abs(a))
    return -np.logspace(la,la-n+1,n)
  elif (a<0) & (b>0):
    la = np.log10(abs(a))
    lb = np.log10(abs(b))
    # split the number of points evenly amongst [a,0] and [0,b]
    # we give an extra point to [0,b] if n is odd
    na = int(np.floor(n/2))
    nb = int(np.ceil(n/2))
    xa = symlog_grid(a,0,na)
    xb = symlog_grid(0,b,nb)
    return np.append(xa,xb)


if __name__=="__main__":

  import matplotlib.pyplot as plt
  
  # tests
  x = symlog_grid(0.1,1000,3)
  print(x)
  # tests
  x = symlog_grid(-0.1,1000,3)
  print(x)
  # tests
  x = symlog_grid(0,100,3)
  print(x)
  # tests
  x = symlog_grid(-10,0.0,4)
  print(x)
  # tests
  x = symlog_grid(-10,-0.001,5)
  print(x)
  # tests
  x = loglin_grid(-10,102.6,12)
  print(x)
