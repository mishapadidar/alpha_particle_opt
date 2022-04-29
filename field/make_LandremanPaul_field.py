import numpy as np
from scipy.optimize import minimize
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.curveobjectives import CurveLength, CurveCurveDistance
from simsopt.geo.plot import plot


def make_LandremanPaul_field():
  """
  Compute the Landreman Paul coils and BiotSavart field, then
  save it.
  """
  nphi = 32
  ntheta = 32
  
  filename = "./input.LandremanPaul2021_QA"
  s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
  
  # Number of unique coil shapes:
  ncoils = 4
  
  # Major radius for the initial circular coils:
  R0 = 1.0
  
  # Minor radius for the initial circular coils:
  R1 = 0.5
  
  # Number of Fourier modes describing each Cartesian component of each coil:
  order = 5
  
  base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
  base_currents = [Current(1e5) for i in range(ncoils)]
  
  base_currents[0].fix_all()
  
  coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
  
  curves = [c.curve for c in coils]
  curves_to_vtk(curves, "curves_init")
  s.to_vtk("surf_init")
  
  bs = BiotSavart(coils)
  bs.set_points(s.gamma().reshape((-1, 3)))
  
  B_dot_n = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
  print('Initial max B dot n:', np.max(B_dot_n))
  
  # Weight on the curve lengths in the objective function:
  ALPHA = 1e-7
  # Threshhold for the coil-to-coil distance penalty in the objective function:
  MIN_DIST = 0.1
  # Weight on the coil-to-coil distance penalty term in the objective function:
  BETA = 10
  
  Jf = SquaredFlux(s, bs)
  Jls = [CurveLength(c) for c in base_curves]
  Jdist = CurveCurveDistance(curves, MIN_DIST)
  # Scale and add terms to form the total objective function:
  objective = Jf + ALPHA * sum(Jls) + BETA * Jdist
  
  print(objective.dof_names)
  
  def fun(dofs):
    objective.x = dofs
    return objective.J(), objective.dJ()
  
  res = minimize(fun, objective.x, jac=True, method='L-BFGS-B',
                 options={'maxiter': 400, 'iprint': 5}, tol=1e-15)
  
  B_dot_n = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
  print('Final max B dot n:', np.max(B_dot_n))
  
  outpath="bs.LandremanPaul2021_QA"
  # run some computation / optimization
  np.savetxt(outpath,bs.x)
  return bs


def load_LandremanPaul_field():
  """
  Load the BiotSavart field for the LandremanPaul coils.
  """
  nphi = 32
  ntheta = 32
  
  filename = "./input.LandremanPaul2021_QA"
  s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
  
  # Number of unique coil shapes:
  ncoils = 4
  
  # Major radius for the initial circular coils:
  R0 = 1.0
  
  # Minor radius for the initial circular coils:
  R1 = 0.5
  
  # Number of Fourier modes describing each Cartesian component of each coil:
  order = 5
  
  base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
  base_currents = [Current(1e5) for i in range(ncoils)]
  
  base_currents[0].fix_all()
  
  coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
  
  curves = [c.curve for c in coils]
  curves_to_vtk(curves, "curves_init")
  s.to_vtk("surf_init")
  
  bs = BiotSavart(coils)

  # load the biot savart field
  inpath="bs.LandremanPaul2021_QA"
  bs.x = np.loadtxt(inpath)
  return bs

def make_interpolated_field():
  """
  Make an interpolated field from the BiotSavart field. Interpolated fields 
  can be evaluated in cylindrical coordinates (r,phi,z).
  """
  from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule

  # load the surface to get appropriate ranges
  nphi = 32
  ntheta = 32
  filename = "./input.LandremanPaul2021_QA"
  s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
  nfp = 2

  # load the field
  bs = load_LandremanPaul_field()

  degree =2
  n = 16
  rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
  zs = s.gamma()[:, :, 2]
  rrange = (np.min(rs), np.max(rs), n)
  phirange = (0, 2*np.pi/nfp, n*2)
  zrange = (0, np.max(zs), n//2)
  bsh = InterpolatedField(bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True)
  return bsh

def load_LandremanPaul_bounds():
  # load the surface to get appropriate ranges
  nphi = 64
  ntheta = 64
  filename = "./input.LandremanPaul2021_QA"
  s = SurfaceRZFourier.from_vmec_input(filename, range="field period", nphi=nphi, ntheta=ntheta)
  nfp = 2

  # load the field
  bs = load_LandremanPaul_field()

  rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
  zs = s.gamma()[:, :, 2]
  return np.min(rs),np.max(rs), np.min(zs),np.max(zs)

def compute_plasma_volume():
  # load the surface to get appropriate ranges
  nphi = 64
  ntheta = 64
  filename = "./input.LandremanPaul2021_QA"
  s = SurfaceRZFourier.from_vmec_input(filename, range="field period", nphi=nphi, ntheta=ntheta)
  nfp = 2
  return s.volume()

def make_surface_classifier():
  from simsopt.field.tracing import SurfaceClassifier
  # load the surface to get appropriate ranges
  nphi = 64
  ntheta = 64
  filename = "./input.LandremanPaul2021_QA"
  s = SurfaceRZFourier.from_vmec_input(filename, range="field period", nphi=nphi, ntheta=ntheta)
  nfp = 2
  sc_particle = SurfaceClassifier(s, h=0.1, p=2)
  return sc_particle

if __name__=="__main__":
  # make the field and save it
  make_LandremanPaul_field()
  # load it
  field = load_LandremanPaul_field()
  # eval it
  field.set_points(np.array([[0.5, 0.5, 0.1], [0.1, 0.1, -0.3]]))
  print(field.B())
  # make an interpolated field in (r,phi,z)
  bsh = make_interpolated_field()
  # eval it
  bsh.set_points(np.array([[0.5, np.pi, 0.1], [0.1, np.pi/2, -0.3]]))
  print(bsh.B())
  rmin,rmax,zmin,zmax = load_LandremanPaul_bounds()
  print(rmin,rmax,zmin,zmax)
  vol = compute_plasma_volume()
  print(vol)
  classifier = make_surface_classifier()

