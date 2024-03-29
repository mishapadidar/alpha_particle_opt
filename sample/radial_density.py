
import numpy as np
from scipy.integrate import quad as sp_quad
from scipy.interpolate import interp1d as sp_interp

class RadialDensity:
  """
  Sample from the radial density
    p(s,theta,phi) = (1-s^5)^2 (1-s)^(-2/3) exp(-19.94 (1-s)^(-1/3))
  where s in (0,1).

  Sampling is done via the inverse transform method through an interpolant
  of the inverse cdf.
  """

  def __init__(self,n_points=1000):
    """
    n_points: number of points used to interpolate the inverse cdf.
    """
    # support
    self.lb = 0.0
    self.ub = 1.0
    # compute the inverse cdf
    self.n_points = n_points
    self._cdf_inv = self.build_inverse_cdf(n_points)

  def _pdf(self,s):
    if isinstance(s,float):
      if s == 1.0:
        ret = 0.0
      else:
        const = 0.21636861430315135e5
        a = (1-s**5)**2
        b = (1-s)**(-2/3)
        c = (12.0**(-1/3))*((1-s)**(-1/3))
        d = np.exp(-19.94*c)
        ret = const*a*b*d
    else:
      # treat the corner case, s==1
      idx_0 = (s == 1.0)
      const = 0.21636861430315135e5
      a = (1-s[~idx_0]**5)**2
      b = (1-s[~idx_0])**(-2/3)
      c = (12.0**(-1/3))*((1-s[~idx_0])**(-1/3))
      d = np.exp(-19.94*c)
      ret = np.zeros(len(s))
      ret[~idx_0] = const*a*b*d
    return ret

  def raw_moment(self,k=1):
    f = lambda s: (s**k)*self._pdf(s)
    return sp_quad(f,self.lb,self.ub)[0]

  def conditional_raw_moment(self,a,b,k=1):
    """
    Compute conditional raw moments.
      E[X^k | a <= X <= b]
    The conditional pdf is 
      p(x|a <= x <= b) = p(x)/(F(b) - F(a)) if x in [a,b]
    """
    f = lambda s: (s**k)*self._pdf(s)
    exp = sp_quad(f,a,b)[0]
    prob =self._cdf(b) - self._cdf(a)
    return exp/prob

  def mean(self):
    return self.raw_moment(k=1)

  def variance(self):
    return self.raw_moment(k=2) - self.mean()**2

  def std(self):
    return np.sqrt(self.variance())
  
  def _cdf(self,s):
    if s <= self.lb:
      return 0.0
    elif s >= self.ub:
      return 1.0
    else:
      return sp_quad(self._pdf,self.lb,s)[0]

  def build_inverse_cdf(self,n_points=1000):
    """
    Build an interpolant of the inverse cdf.
      X = F^{-1}(u)
    n_points: number of points used in interpolation.
    """
    #s_vals = np.linspace(self.lb,self.ub,n_points)

    # adaptive spacing
    dy = 1.0/n_points
    n_pad = int(0.1*n_points)
    s_vals = np.zeros(n_points+n_pad)
    for ii in range(1,n_points):
      s_vals[ii] = s_vals[ii-1] + dy/self._pdf(s_vals[ii-1])
    # mix with linspace
    diff = (self.ub - np.max(s_vals))/n_pad/2
    s_vals[n_points:] = np.linspace(np.max(s_vals)+diff,self.ub,n_pad)

    # evaluate the CDF
    cdf_vals = np.array([self._cdf(s) for s in s_vals])
    # interpolate x = F^-1(u)
    interpolant = sp_interp(cdf_vals,s_vals,kind='cubic')
    return interpolant

  def sample(self,n_points):
    """
    Inverse transform sampling using tabulated CDF values.
    """
    # generate uniform random variables
    unif = np.random.uniform(0,1,n_points)

    # evaluate the inverse CDF
    return self._cdf_inv(unif)


if __name__=="__main__":
  import matplotlib.pyplot as plt
  n_points = 1000
  sampler = RadialDensity(n_points)

  print('mean',sampler.mean())
  print('std',sampler.std())

  # plot the pdf
  x = np.linspace(0,1,1000)
  y = sampler._pdf(x)
  plt.plot(x,y)
  # plot a histogram of samples
  n_samples = 200000
  s = sampler.sample(n_samples)
  plt.hist(s,bins=100,density=True)
  plt.title("probability density")
  plt.xlabel("radial coordinate")
  plt.show()

  # plot the inverse CDF
  x = np.linspace(0,1,10000)
  y = [sampler._cdf(xi) for xi in x] 
  plt.plot(y,x,'-o',markersize=3,label='inverse cdf')
  y = np.linspace(0,1,100000)
  plt.plot(y,sampler._cdf_inv(y),label='interpolant')
  plt.legend(loc='upper left')
  plt.show()

  
