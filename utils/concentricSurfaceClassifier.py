from simsopt.geo.surface import signed_distance_from_surface
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
import numpy as np
from coords import cart_to_cyl,cyl_to_cart

class concentricSurfaceClassifier:
  """
  Construct a classifier to determine whether a point is
  within a concentric surface of a given surface. 
  
  We use the field period symmetry to improve computation 
  efficiency. We build a classifier for a single field period, 
  and map points to that field period for evaluation.
  """

  def __init__(self,vmec_input,ntheta=64,nphi=64,eps=0.0):
    """
    vmec_input: a vmec file that describes the surface
    nphi,ntheta: toroidal and poloidal discretizations. Use high values > 256 for good accuracy.
    eps: a distance from the initial surface. If positive, then a particle outside the surface
      within a distance eps will be classified as 1. If negative, then particles must be interior
      to the surface by a distance eps to be classified as 1.
    """
    # only discretize a field period
    self.surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="field period", nphi=nphi, ntheta=ntheta)
    self.eps = eps
    self.ntheta = ntheta
    self.nphi = nphi

  def signed_dist(self,xyz):
    """
    Compute the signed distance function. returns positive
    numbers from particles inside the surface, 0 for those on
    the boundary and -1 for those outside the surface.
    
    We use the field period symmetry to improve computation 
    efficiency. 
    """
    # map to the field period: phi to [0,2pi/nfp]
    rphiz = cart_to_cyl(xyz)
    rphiz[:,1] = np.mod(rphiz[:,1],2*np.pi/self.surf.nfp)
    # convert back to xyz
    xyz = cyl_to_cart(rphiz)
    # evaluate the distance
    dist = signed_distance_from_surface(xyz,self.surf)
    return dist


  def __call__(self,xyz,return_type=float):
    """
    Determine if the signed distance function to the surface.
    is at least -eps.

    Particles inside the surface volume have positive distances 
    and particles outside have negative distances.

  
    xyz: 2d np array of cartesian points, shape (N,3)
    return_type: bool or float, array type to return

    return: 1d array, shape (N,) of values 0 or 1. 
      1 if signed_distance_function(xyz,surf) >= -eps and
      0 otherwise.
    """
    assert return_type in [bool,float], "invalid return_type"
    dist = self.signed_dist(xyz)
    res = (dist >= -self.eps)
    if return_type == float:
      return res.astype(float)
    else:
      return res


if __name__=="__main__":

  vmec_input="../stella/input.new_QA_scaling"

  # build a classifier 
  ntheta=nphi=512
  eps = 0.0
  classifier = concentricSurfaceClassifier(vmec_input, nphi=nphi, ntheta=ntheta,eps=eps)

  # generate a set of points on the surface 
  ntheta=nphi=2048
  surf = SurfaceRZFourier.from_vmec_input(vmec_input, range="full torus", nphi=nphi, ntheta=ntheta)
  X = surf.gamma().reshape((-1,3))
  # evaluate the points
  dist = classifier.signed_dist(X)
  # determine the error; they should evaluate to zero.
  print(np.max(np.abs(dist)))


















