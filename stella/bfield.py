import numpy as np
from scipy.optimize import minimize
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.geo.curve import curves_to_vtk, create_equally_spaced_curves
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.curveobjectives import CurveLength, CurveCurveDistance
from simsopt.geo.plot import plot


def write_field(vmec_input = "./input.new_QA_scaling",bs_path="bs.new_QA_scaling",nphi=64,ntheta=64):
  """
  Compute the Landreman Paul coils and BiotSavart field, then
  save it.
  """
  s = SurfaceRZFourier.from_vmec_input(vmec_input, range="field period", nphi=nphi, ntheta=ntheta)
  
  # Number of unique coil shapes:
  ncoils = 4
  
  # Major radius for the initial circular coils:
  R0 = 10.0
  
  # Minor radius for the initial circular coils:
  R1 = 5
  
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
  ALPHA = 1e-9
  # Threshhold for the coil-to-coil distance penalty in the objective function:
  MIN_DIST = 0.1
  # Weight on the coil-to-coil distance penalty term in the objective function:
  BETA = 1
  
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

  # unfix currents before saving
  bs.coils[0].current.unfix_all()
  print(len(bs.x))
  
  # run some computation / optimization
  np.savetxt(bs_path,bs.x)
  return bs


def load_field(vmec_input = "./input.new_QA_scaling",bs_path="./bs.new_QA_scaling",nphi=64,ntheta=64):
  """
  Load the BiotSavart field for the LandremanPaul coils.
  """
  
  s = SurfaceRZFourier.from_vmec_input(vmec_input, range="field period", nphi=nphi, ntheta=ntheta)
  
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
  
  #base_currents[0].fix_all()
  
  coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
  
  #curves = [c.curve for c in coils]
  #curves_to_vtk(curves, "curves_init")
  #s.to_vtk("surf_init")
  
  bs = BiotSavart(coils)

  # load the biot savart field
  bs.x = np.loadtxt(bs_path)
  return bs

def compute_rz_bounds(vmec_input = "./input.new_QA_scaling",nphi=64,ntheta=64):
  # load the surface to get appropriate ranges
  s = SurfaceRZFourier.from_vmec_input(vmec_input, range="field period", nphi=nphi, ntheta=ntheta)
  rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
  zs = s.gamma()[:, :, 2]
  return np.min(rs),np.max(rs), np.min(zs),np.max(zs)

def compute_plasma_volume(vmec_input = "./input.new_QA_scaling",nphi=64,ntheta=64):
  s = SurfaceRZFourier.from_vmec_input(vmec_input, range="field period", nphi=nphi, ntheta=ntheta)
  return s.volume()

def make_surface_classifier(vmec_input = "./input.new_QA_scaling",nphi=64,ntheta=64):
  from simsopt.field.tracing import SurfaceClassifier
  s = SurfaceRZFourier.from_vmec_input(vmec_input, range="field period", nphi=nphi, ntheta=ntheta)
  sc_particle = SurfaceClassifier(s, h=0.1, p=2)
  return sc_particle

def rescale_coil_currents(vmec_input = "./input.new_QA_scaling",bs_path="./bs.new_QA_scaling",nphi=64, ntheta=64,target_tflux=5.048228826746e1,overwrite=False):
  """
  Rescale coil currents based on the target toroidal flux.

  vmec_input: a vmec input file
  bs_path: path to coils corresponding to the vmec input
  nphi,ntheta: toroidal and poloidal discretizations of plasma
  target_toroidal_flux: typically the toroidal flux for the vmec_input, or another toroidal
    flux that you want to scale the currents based on
  overwrite: bool, overwrite the bs_path
  """
  from simsopt.geo.surfaceobjectives import ToroidalFlux
  s = SurfaceRZFourier.from_vmec_input(vmec_input, range="field period", nphi=nphi, ntheta=ntheta)
  bs = load_field(vmec_input = vmec_input,bs_path=bs_path,nphi=nphi,ntheta=ntheta)

  ncoils = 4

  tflux = ToroidalFlux(s,bs)
  tfluxJ = tflux.J()
  print('initial tflux',tfluxJ)

  # scale the coil currents
  scale_factor = target_tflux/tfluxJ

  #print('scale factor',scale_factor)
  # set the coils currents
  for coil in bs.coils[:ncoils]:
    coil.current.unfix_all()
    #print(coil.current.x)
    coil.current.x *=scale_factor
    #print(coil.current.x)
    #print("")

  tflux = ToroidalFlux(s,bs)
  tfluxJ = tflux.J()
  print('new tflux',tfluxJ)


  ## print coil B normal
  #bs.set_points(s.gamma().reshape((-1, 3)))
  #print(bs.B())
  #B_dot_n = np.sum(bs.B().reshape((nphi,ntheta, 3)) * s.unitnormal(), axis=2)
  #print('max B dot n:', np.max(B_dot_n))

  if overwrite:
    np.savetxt(bs_path,bs.x)
  return bs

if __name__=="__main__":

  # make the field and save it
  write_field(vmec_input="input.new_QA_scaling", bs_path="bs.new_QA_scaling",ntheta=128,nphi=128)

  # scale the coil currents to reactor strength
  print('')
  rescale_coil_currents(ntheta=128,nphi=128,overwrite=True)

  # load it
  bs = load_field()

  ## eval it
  #nphi=ntheta=128
  #s = SurfaceRZFourier.from_vmec_input("input.new_QA_scaling", range="field period",ntheta=ntheta,nphi=nphi)
  #bs.set_points(s.gamma().reshape((-1, 3)))
  #B_dot_n = np.sum(bs.B().reshape((nphi,ntheta, 3)) * s.unitnormal(), axis=2)
  #print('')
  #print(bs.B())
  #print('max B dot n:', np.max(B_dot_n))

  # get bounds
  rmin,rmax,zmin,zmax = compute_rz_bounds()
  print('')
  print('r and z bounds')
  print(rmin,rmax,zmin,zmax)

  # plot it
  bs.to_vtk('magnetic_field',nphi=32,rmin=rmin,rmax=rmax,zmin=zmin,zmax=zmax)

  # get volume
  vol = compute_plasma_volume()
  print('')
  print('volume',vol)

  # make a surface classifier
  classifier = make_surface_classifier()

