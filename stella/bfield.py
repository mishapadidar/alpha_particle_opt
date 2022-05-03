import numpy as np
from scipy.optimize import minimize
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.curveobjectives import CurveLength, CurveCurveDistance
from simsopt.geo.plot import plot


def write_field(filename = "./input.LandremanPaul2021_QA",outpath="bs.LandremanPaul2021_QA",nphi=64,ntheta=64):
  """
  Compute the Landreman Paul coils and BiotSavart field, then
  save it.
  """
  s = SurfaceRZFourier.from_vmec_input(filename, range="field period", nphi=nphi, ntheta=ntheta)
  
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
  #curves_to_vtk(curves, "curves_init")
  #s.to_vtk("surf_init")
  
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
  
  def fun(dofs):
    objective.x = dofs
    return objective.J(), objective.dJ()
  
  res = minimize(fun, objective.x, jac=True, method='L-BFGS-B',
                 options={'maxiter': 400, 'iprint': 5}, tol=1e-15)
  
  B_dot_n = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)
  print('Final max B dot n:', np.max(B_dot_n))
  
  # run some computation / optimization
  np.savetxt(outpath,bs.x)
  return bs


def load_field(filename = "./input.LandremanPaul2021_QA",bs_path="./bs.LandremanPaul2021_QA",nphi=64,ntheta=64):
  """
  Load the BiotSavart field for the LandremanPaul coils.
  """
  
  s = SurfaceRZFourier.from_vmec_input(filename, range="field period", nphi=nphi, ntheta=ntheta)
  
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
  
  #curves = [c.curve for c in coils]
  #curves_to_vtk(curves, "curves_init")
  #s.to_vtk("surf_init")
  
  bs = BiotSavart(coils)

  # load the biot savart field
  bs.x = np.loadtxt(bs_path)
  return bs

def compute_rz_bounds(filename = "./input.LandremanPaul2021_QA",bs_path="./bs.LandremanPaul2021_QA",nphi=64,ntheta=64):
  # load the surface to get appropriate ranges
  s = SurfaceRZFourier.from_vmec_input(filename, range="field period", nphi=nphi, ntheta=ntheta)
  rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
  zs = s.gamma()[:, :, 2]
  return np.min(rs),np.max(rs), np.min(zs),np.max(zs)

def compute_plasma_volume(filename = "./input.LandremanPaul2021_QA",nphi=64,ntheta=64):
  s = SurfaceRZFourier.from_vmec_input(filename, range="field period", nphi=nphi, ntheta=ntheta)
  return s.volume()

def make_surface_classifier(filename = "./input.LandremanPaul2021_QA",nphi=64,ntheta=64):
  from simsopt.field.tracing import SurfaceClassifier
  s = SurfaceRZFourier.from_vmec_input(filename, range="field period", nphi=nphi, ntheta=ntheta)
  sc_particle = SurfaceClassifier(s, h=0.1, p=2)
  return sc_particle

if __name__=="__main__":
  # make the field and save it
  write_field()
  # load it
  field = load_field()
  # eval it
  field.set_points(np.array([[0.5, 0.5, 0.1], [0.1, 0.1, -0.3]]))
  print(field.B())
  rmin,rmax,zmin,zmax = compute_rz_bounds()
  print(rmin,rmax,zmin,zmax)
  vol = compute_plasma_volume()
  print(vol)
  classifier = make_surface_classifier()

