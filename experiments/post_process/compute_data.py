import numpy as np
from mpi4py import MPI
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt._core import Optimizable
import sys
import pickle
debug = False
if debug:
  sys.path.append("../../utils")
  sys.path.append("../../trace")
  sys.path.append("../../sample")
else:
  sys.path.append("../../../utils")
  sys.path.append("../../../trace")
  sys.path.append("../../../sample")
from trace_boozer import TraceBoozer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

"""
Compute the following information around a point
- the gradient of the energy objective
- the gradient of the quasisymmetry objective
- the gradient of the aspect ratio 
- the gradient of the B-field constraint
- linesearch the energy objective along the QS direction

Run 
  mpiexec -n 1 python3 compute_data.py ../min_energy_loss/data_phase_one_tmax_0.0001_SAA_sweep/data_opt_nfp4_phase_one_aspect_7.0_iota_1.05_mean_energy_SAA_surface_0.25_tmax_0.0001_bobyqa_mmode_1_iota_None.pickle
"""

# TODO: should we rescale our configs to a standard size?
# TODO: and also rescale the B-field?


# tracing params
s_label = 0.25 # 0.25 or full
n_particles = 10000 
h_fdiff_x = 5e-3 # finite difference
h_fdiff_qs = 1e-4 # finite difference quasisymmetry
h_fdiff_y = 1e-2 # finite difference in scaled space
helicity_m = 1 # quasisymmetry M
helicity_n = -1 # quasisymmetry N

# tracing accuracy params
tracing_tol=1e-8
interpolant_degree=3
interpolant_level=8
bri_mpol=16
bri_ntor=16


# load the file
data_file = sys.argv[1]
indata = pickle.load(open(data_file,"rb"))
vmec_input = indata['vmec_input']
if debug:
  vmec_input=vmec_input[3:] # remove the ../
x0 = indata['xopt']
max_mode=indata['max_mode']
aspect_target = indata['aspect_target']
major_radius = indata['major_radius']
target_volavgB = indata['target_volavgB']
tmax = indata['tmax']

tracer = TraceBoozer(vmec_input,
                    n_partitions=1,
                    max_mode=max_mode,
                    major_radius=major_radius,
                    aspect_target=aspect_target,
                    target_volavgB=target_volavgB,
                    tracing_tol=tracing_tol,
                    interpolant_degree=interpolant_degree,
                    interpolant_level=interpolant_level,
                    bri_mpol=bri_mpol,
                    bri_ntor=bri_ntor)
tracer.x0 = np.copy(x0)
tracer.surf.x = np.copy(x0)
dim_x = len(x0)

# quasisymmetry objective
qsrr = QuasisymmetryRatioResidual(tracer.vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=helicity_m, helicity_n=helicity_n)  # (M, N) you want in |B|

"""
Compute the Gauss-Newton Hessian
"""
def qs_residuals(x):
  """
  Compute the QS residuals
  """
  tracer.surf.x = np.copy(x)
  try:
    qs = qsrr.residuals() # quasisymmetry
  except:
    qs = np.inf
  ret = qs
  if rank == 0:
    print(ret)
    sys.stdout.flush()
  return ret

# print
if rank == 0:
  print("")
  print('computing jacobian of QS with finite difference')
  print('dim_x',dim_x)
  sys.stdout.flush()

# jacobian
Ep = x0 + h_fdiff_qs*np.eye(dim_x)
Fp = np.array([qs_residuals(e) for e in Ep])
F0 = qs_residuals(x0)
jac = (Fp - F0).T/h_fdiff_qs

"""
Rescale the variables
"""

# build the Gauss-Newton hessian approximation
Hess = jac.T @ jac
jit = 1e-6*np.eye(dim_x) # jitter
L = np.linalg.cholesky(Hess + jit)

if rank == 0:
  print('')
  print('Hessian eigenvalues')
  print(np.linalg.eigvals(Hess))
  sys.stdout.flush()

# rescale the variables y = L.T @ x
def to_scaled(x):
  """maps to new variables"""
  return L.T @ x
def from_scaled(y):
  """maps back to old variables"""
  return np.linalg.solve(L.T,y)

# map x0 to y0
y0 = to_scaled(x0)

class MirrorCon:
    """
    Constraints on |B|:
    |B| <= B_mean(1+eps_B)
    |B| >= B_mean(1-eps_B)
    where B_mean is, say, 5 Tesla.
    """
    def __init__(self, B_mean,mirror_target):
        eps_B = (mirror_target - 1.0)/(mirror_target + 1.0)
        self.B_ub = B_mean*(1 + eps_B)
        self.B_lb = B_mean*(1 - eps_B)
        self.ns=8 # maxB should be on boundary
        self.ntheta=16
        self.nzeta=16
        self.default = np.zeros(self.ns*self.ntheta*self.nzeta)
        # grid points for modB
        stz_grid,_ = tracer.flux_grid(self.ns,self.ntheta,self.nzeta,1)
        self.stz_grid = stz_grid

    def B_field(self,x):
      """
      Compute modB on a grid
      """
      field,bri = tracer.compute_boozer_field(x)
      if field is None:
        return self.default
      field.set_points(self.stz_grid)
      modB = field.modB().flatten()
      #modB = tracer.compute_modB(field,bri,ns=self.ns,ntheta=self.ntheta,nphi=self.nzeta)
      return modB

    def B_shifted(self,x):
        """B-field shifted for the constraint c(x) <= 0"""
        # compute the B-field
        modB = self.B_field(x)
        # modB - B_ub <= 0
        c_ub = modB - self.B_ub
        # B_lb - modB <= 0
        c_lb = self.B_lb - modB
        ret = np.append(c_ub,c_lb)
        return ret

    def resid(self,x):
        """penalty residuals"""
        ret = np.maximum(self.B_shifted(x),0.0)
        return ret

    def total(self,x):
        """
        Sum of squares objectve.
        """
        ret = np.sum(self.resid(x)**2)
        return ret

## bounds on the mirror ratio
#class MirrorCon(Optimizable):
#    """
#    Constraints on |B|:
#    |B| <= B_mean(1+eps_B)
#    |B| >= B_mean(1-eps_B)
#    where B_mean is, say, 5 Tesla.
#    """
#    def __init__(self, v, B_mean,mirror_target):
#        self.v = v
#        Optimizable.__init__(self, depends_on=[v])
#        eps_B = (mirror_target - 1.0)/(mirror_target + 1.0)
#        self.B_ub = B_mean*(1 + eps_B)
#        self.B_lb = B_mean*(1 - eps_B)
#
#    def resid(self):
#        """
#        Constraint penalty. 
#
#        return the residuals
#          [np.max(|B| - B_ub,0.0),np.max(B_lb - |B|,0.0)]
#        """
#        # get modB from quasisymmetry function
#        data = qsrr.compute()
#        modB = data.modB
#        #print(np.min(modB),np.max(modB))
#        # modB <= B_ub
#        c_ub = np.maximum(modB - self.B_ub,0.0)
#        # modB >= B_lb
#        c_lb = np.maximum(self.B_lb - modB,0.0)
#        ret = np.append(c_ub,c_lb)
#        return ret
#
#    def total(self):
#        """
#        Sum of squares objectve.
#        """
#        resid = self.resid()
#        return np.sum(resid**2)
#
#    def B_minmax(self):
#        data = qs.compute()
#        modB = data.modB
#        return np.min(modB),np.max(modB)
#
#    def mirror_ratio(self):
#        Bmin,Bmax =  self.B_minmax()
#        return Bmax/Bmin

# mirror ratio constraint
B_mean = 5.0
mirror_target = 1.35
#mirror = MirrorCon(tracer.vmec,B_mean,mirror_target)
mirror = MirrorCon(B_mean,mirror_target)

def compute_values(y):
  """
  Compute the values
  [ QS sum of squares, mirror constraint residuals, aspect ratio]
  """
  x = from_scaled(y)
  tracer.surf.x = np.copy(x)
  try:
    qs = qsrr.total() # quasisymmetry
  except:
    qs = np.inf
  ret = np.array([qs,tracer.surf.aspect_ratio(),*mirror.B_shifted(x)])
  if rank == 0:
    print(ret)
    sys.stdout.flush()
  return ret

"""
Compute the gradient of quasisymmetry objective
"""
# print
if rank == 0:
  print("")
  print('computing gradient of QS with finite difference')
  print('dim_x',dim_x)
  sys.stdout.flush()

# gradients
Ep   = y0 + h_fdiff_qs*np.eye(dim_x)
Fp = np.array([compute_values(e) for e in Ep])
F0 = compute_values(y0)
jac = (Fp - F0).T/h_fdiff_qs
qs0     = F0[0]
aspect0 = F0[1]
mirror0 = F0[2:]
qs_plus     = Fp[:,0]
aspect_plus = Fp[:,1]
mirror_plus = Fp[:,2:]
grad_qs     = jac[0]
grad_aspect = jac[1]
grad_mirror = jac[2:]


if rank == 0:
  print('qs total',qs0)
  print('norm qs grad',np.linalg.norm(grad_qs))
  print('mirror total',np.sum(np.maximum(mirror0,0.0)**2))
  print('aspect',aspect0)
  print('norm aspect grad',np.linalg.norm(grad_aspect))
  sys.stdout.flush()

# save data
if rank == 0:
  outdata = {}
  outdata['qs0'] = qs0
  outdata['qs_plus'] = qs_plus
  outdata['grad_qs'] = grad_qs
  outdata['mirror0'] = mirror0
  outdata['mirror_plus'] = mirror_plus
  outdata['grad_mirror'] = grad_mirror
  outdata['aspect0'] = aspect0
  outdata['aspect_plus'] = aspect_plus
  outdata['grad_aspect'] = grad_aspect
  outdata['h_fdiff_qs'] = h_fdiff_qs
  outdata['helicity_m'] = helicity_m
  outdata['helicity_n'] = helicity_n
  outdata['mirror_stz_grid'] = mirror.stz_grid
  outdata['mirror_ns'] = mirror.ns
  outdata['mirror_ntheta'] = mirror.ntheta
  outdata['mirror_nzeta'] = mirror.nzeta
  indata[f'post_process_s_{s_label}'] = outdata
  pickle.dump(indata,open(data_file,"wb"))

"""
Compute the gradient of the energy objective
"""
# print
if rank == 0:
  print('computing gradient of energy with finite difference')
  print('dim_x',dim_x)

# fixed sample
tracer.sync_seeds()
if s_label == "full":
  stz_inits,vpar_inits = tracer.sample_volume(n_particles)
else:
  stz_inits,vpar_inits = tracer.sample_surface(n_particles,s_label)

def confinement_times(y):
  """
  Shortcut for computing confinement times from scaled space
  """
  x = from_scaled(y)
  c_times = tracer.compute_confinement_times(x,stz_inits,vpar_inits,tmax) 
  return c_times

# central difference the confinement times
h2 = h_fdiff_y/2
Ep   = y0 + h2*np.eye(dim_x)
Em   = y0 - h2*np.eye(dim_x)
c_times_plus   = np.array([confinement_times(e) for e in Ep])
c_times_minus   = np.array([confinement_times(e)  for e in Em])
c_times0 = confinement_times(y0) 
# compute energy
energy0 = np.mean(3.5*np.exp(-2*c_times0/tmax))
energy_plus = np.mean(3.5*np.exp(-2*c_times_plus/tmax),axis=1)
energy_minus = np.mean(3.5*np.exp(-2*c_times_minus/tmax),axis=1)
# gradient; central difference
grad_energy = (energy_plus - energy_minus)/h_fdiff_y

if rank == 0:
  print('energy',energy0)
  print('norm energy grad',np.linalg.norm(grad_energy))
  print('qs total',qs0)
  print('norm qs grad',np.linalg.norm(grad_qs))

# save data
if rank == 0:
  outdata['tmax'] = tmax
  outdata['n_particles'] = n_particles
  outdata['h_fdiff_x'] = h_fdiff_x
  outdata['h_fdiff_y'] = h_fdiff_y
  outdata['s_label'] = s_label
  outdata['x0'] = x0
  outdata['y0'] = y0
  outdata['L'] = L
  outdata['Yp'] = Ep
  outdata['Ym'] = Em
  outdata['c_times_plus'] = c_times_plus
  outdata['c_times_minus'] = c_times_minus
  outdata['c_times0'] = c_times0
  outdata['energy0'] = energy0
  outdata['energy_plus'] = energy_plus
  outdata['energy_minus'] = energy_minus
  outdata['grad_energy'] =grad_energy
  
  # dump data
  indata[f'post_process_s_{s_label}'] = outdata
  pickle.dump(indata,open(data_file,"wb"))


"""
Now linesearch along -grad(qs)
"""
def objectives(y):
  """
  Compute quasisymmetry and mean-energy objective.
  """
  x = from_scaled(y)
  c_times = tracer.compute_confinement_times(x,stz_inits,vpar_inits,tmax)
  try:
    qs = qsrr.total() # quasisymmetry
  except:
    qs = np.inf
  asp = tracer.surf.aspect_ratio()
  mc = mirror.B_shifted(x)
  return np.array([*c_times,qs,asp,*mc])

# linesearch step sizes
#T_ls = np.array([0.0,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,3e-3,5e-3,7e-3,8e-3,9e-3,1e-2,2e-2,4e-2,5e-2])
T_ls = h_fdiff_y*np.array([0.0,0.05,0.1,0.5,1.0,2.0,5.0,10.0,20.0,50.0])
# perform the linesearch along -grad(qs)
X_ls = y0 - np.array([ti*grad_qs for ti in T_ls])
F_ls   = np.array([objectives(e) for e in X_ls])
# split the arrays
c_times_ls = F_ls[:,:n_particles]
qs_ls = F_ls[:,n_particles]
asp_ls = F_ls[:,n_particles+1]
mirror_ls = F_ls[:,n_particles+2:]
energy_ls = np.mean(3.5*np.exp(-2*c_times_ls/tmax),axis=1)

if rank == 0:
  print("")
  print('energy linesearch',energy_ls)
  print('qs linesearch',qs_ls)

if rank == 0:
  # save the data
  outdata['T_qsls'] = T_ls
  outdata['Y_qsls'] = X_ls
  outdata['qs_qsls'] = qs_ls
  outdata['aspect_qsls'] = asp_ls
  outdata['mirror_qsls'] = mirror_ls
  outdata['energy_qsls'] = energy_ls
  outdata['c_times_qsls'] = c_times_ls
  # dump data
  indata[f'post_process_s_{s_label}'] = outdata
  pickle.dump(indata,open(data_file,"wb"))
  

"""
Linesearch along active mirror constraint gradients
"""
ctol=1e-3
idx_active_mirror = np.where(mirror0 >= 0.0 - ctol)[0]
active_mirror_jac = grad_mirror[idx_active_mirror]
n_active_mirror = np.shape(idx_active_mirror)[0]


# save the linesearch directions
if rank == 0:
  outdata['ctol'] = ctol
  outdata['idx_active_mirror'] = idx_active_mirror
  outdata['active_mirror_jac'] = active_mirror_jac

if n_active_mirror <= 6:
  for ii in range(n_active_mirror):

    # perform the linesearch along grad(constraint)
    direction = active_mirror_jac[ii]
    X_ls = y0 + np.array([ti*direction for ti in T_ls])
    F_ls   = np.array([objectives(e) for e in X_ls])

    # current constraint index 
    idx_mirror_con = idx_active_mirror[ii]

    # split the arrays
    c_times_ls = F_ls[:,:n_particles]
    qs_ls = F_ls[:,n_particles]
    asp_ls = F_ls[:,n_particles+1]
    mirror_ls = F_ls[:,n_particles+2:]
    mirror_ls = mirror_ls[:,idx_mirror_con] # only keep current constraint
    energy_ls = np.mean(3.5*np.exp(-2*c_times_ls/tmax),axis=1)
    

    if rank == 0:
      print("")
      print('energy linesearch',energy_ls)
      print('mirror con',idx_mirror_con, mirror_ls)
    
    if rank == 0:
      # save the data
      outdata[f'T_mirrorls']  = T_ls
      outdata[f'idx_mirror_con_mirrorls_{ii}'] = idx_mirror_con # constraint index
      outdata[f'mirror_mirrorls_{ii}'] = np.copy(mirror_ls) # constraint value
      outdata[f'Y_mirrorls_{ii}']  = np.copy(X_ls)
      outdata[f'qs_mirrorls_{ii}'] = np.copy(qs_ls)
      outdata[f'aspect_mirrorls_{ii}'] = np.copy(asp_ls)
      outdata[f'energy_mirrorls_{ii}'] = np.copy(energy_ls)
      outdata[f'c_times_mirrorls_{ii}'] = np.copy(c_times_ls)
      # dump data
      indata[f'post_process_s_{s_label}'] = outdata
      pickle.dump(indata,open(data_file,"wb"))
    
