
import numpy as np
from scipy.integrate import quad as sp_quad
import sys
sys.path.append("../utils")
from grids import loglin_grid

class RadialDensity:
  """
  Sample from the radial density
    p(s,theta,phi) = (1-s^5)^2 (1-s)^(-2/3) exp(-19.94 (1-s)^(-1/3))
  where s in (0,1).

  Sampling is done via the inverse transform method through a tabulated CDF.
  """

  def __init__(self,n_tabs=10000):
    """
    n_tabs: number of points to tagbulate the CDF at.
            Use at least 1000.
    """
    # support
    self.lb = 0.0
    self.ub = 1.0
    # tabulate the cdf
    self.n_tabs = n_tabs
    self.s_vals,self.cdf_vals = self.tabulate_cdf(n_tabs)

  def _pdf(self,s):
    const = 0.21636861430315135e5
    a = (1-s**5)**2
    b = (1-s)**(-2/3)
    c = (12.0**(-1/3))*((1-s)**(-1/3))
    d = np.exp(-19.94*c)
    return const*a*b*d
  
  def _cdf(self,s):
    if s <= self.lb:
      return 0.0
    elif s >= self.ub:
      return 1.0
    else:
      return sp_quad(self._pdf,self.lb,s)[0]

  def tabulate_cdf(self,n_tabs):
    """
    Tabulate CDF values.
    We space the x values out so that the spacing in CDF space
    is approximately uniform. In other words we try to select 
    {x_i} so that F(x_{i+1}) - F(x_i) = 1/n_tabs where n_tabs is 
    the number of points we are tabulating the cdf at. The advantage
    of this is improved inverse sampling accuracy.

    Notice that F(x_{i+1})  = 1/n_tabs + F(x_i). Also the CDF value
    can be computed by 
      F(x_{i+1})  = integral_0^x_{i+1} pdf(x) dx 
                  = F(x_i) + integral_{x_i}^{x_{i+1}} pdf(x) dx 
                  ~ F(x_i) + pdf(x_i)*(x_{i+1} - x_i) 
    where the last approximation is a Reimann sum approximation.
    Substituting the two equations gives us the selection rule
      x_{i+1} = x_i + (1/n_tabs)/pdf(x_i)
    This is as accurate as the Reimann sum approximation to the integral.

    n_tabs: integer, number of points to use in the tabulation. 
            Recommend at least 1000
    """
    if n_tabs < 500
      # just use a linspace
      dx = (self.ub-self.lb)/n_tabs
      s_vals = np.arange(self.lb,self.ub+dx/2, dx)
    else:
      # adaptive spacing
      dy = 1.0/n_tabs    
      s_vals = np.zeros(n_tabs)
      for ii in range(1,n_tabs):
        s_vals[ii] = s_vals[ii-1] + dy/self._pdf(s_vals[ii-1])

    # evaluate the CDF
    cdf_vals = np.array([self._cdf(s) for s in s_vals])


    return s_vals,cdf_vals

  def sample(self,n_points):
    """
    Inverse transform sampling using tabulated CDF values.
    """
    # generate uniform random variables
    unif = np.random.uniform(0,1,n_points)
    # find the nearest cdf value
    snap = np.array([np.argmin(np.abs(self.cdf_vals - u)) for u in unif])
    # return the points
    return self.s_vals[snap]


if __name__=="__main__":
  import matplotlib.pyplot as plt
  n_tabs = 100
  sampler = RadialDensity(n_tabs)

  # plot the pdf
  x = sampler.s_vals
  y = sampler._pdf(x)
  plt.plot(x,y,'-o',markersize=5)
  #plt.yscale('log')
  plt.title("probability density")
  plt.xlabel("radial coordinate")
  plt.show()

  # plot the tabulated cdf values
  x = sampler.s_vals
  y = sampler.cdf_vals
  plt.plot(x,y,'-o',markersize=5)
  plt.yscale('log')
  plt.title("cumulative density")
  plt.xlabel("radial coordinate")
  plt.show()

  # plot the change in the cdf values from the tabulation
  # should be roughly constant b/c of our step size rule
  x = sampler.s_vals[:-1]
  y = n_tabs*(sampler.cdf_vals[1:]- sampler.cdf_vals[:-1])
  plt.plot(x,y,'-o',markersize=5)
  plt.title("normalized change in cumulative density")
  plt.xlabel("radial coordinate")
  plt.ylabel("change in CDF times number of samples")
  plt.ylim(0.7,1.3)
  plt.show()

  
