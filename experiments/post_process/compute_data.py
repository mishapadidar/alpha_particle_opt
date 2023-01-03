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
h_fdiff = 1e-2 # finite difference
h_fdiff_qs = 1e-4 # finite difference quasisymmetry
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

def compute_values(x):
  """
  Compute the values
  [ QS sum of squares, mirror constraint residuals, aspect ratio]
  """
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
  print('dim_x',len(x0))
  sys.stdout.flush()

# gradients
Ep   = x0 + h_fdiff_qs*np.eye(dim_x)
Fp = np.array([compute_values(e) for e in Ep])
F0 = compute_values(x0)
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
  print('dim_x',len(x0))

# fixed sample
tracer.sync_seeds()
if s_label == "full":
  stz_inits,vpar_inits = tracer.sample_volume(n_particles)
else:
  stz_inits,vpar_inits = tracer.sample_surface(n_particles,s_label)

# forward difference the confinement times
Ep   = x0 + h_fdiff*np.eye(dim_x)
c_times_plus   = np.array([tracer.compute_confinement_times(e,stz_inits,vpar_inits,tmax) for e in Ep])
c_times0 = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax) 
# compute energy
energy0 = np.mean(3.5*np.exp(-2*c_times0/tmax))
energy_plus = np.mean(3.5*np.exp(-2*c_times_plus/tmax),axis=1)
# gradient
grad_energy = (energy_plus - energy0)/h_fdiff

if rank == 0:
  print('energy',energy0)
  print('norm energy grad',np.linalg.norm(grad_energy))
  print('qs total',qs0)
  print('norm qs grad',np.linalg.norm(grad_qs))

# save data
if rank == 0:
  outdata['tmax'] = tmax
  outdata['n_particles'] = n_particles
  outdata['h_fdiff'] = h_fdiff
  outdata['s_label'] = s_label
  outdata['x0'] = x0
  outdata['Xp'] = Ep
  outdata['c_times_plus'] = c_times_plus
  outdata['c_times0'] = c_times0
  outdata['energy0'] = energy0
  outdata['energy_plus'] = energy_plus
  outdata['grad_energy'] =grad_energy
  
  # dump data
  indata[f'post_process_s_{s_label}'] = outdata
  pickle.dump(indata,open(data_file,"wb"))


"""
Now linesearch along -grad(qs)
"""
def objectives(x):
  """
  Compute quasisymmetry and mean-energy objective.
  """
  c_times = tracer.compute_confinement_times(x,stz_inits,vpar_inits,tmax)
  try:
    qs = qsrr.total() # quasisymmetry
  except:
    qs = np.inf
  asp = tracer.surf.aspect_ratio()
  mc = mirror.B_shifted(x)
  return np.array([*c_times,qs,asp,*mc])

# linesearch step sizes
T_ls = np.array([0.0,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3,2e-3,3e-3,5e-3,7e-3,8e-3,9e-3,1e-2,2e-2,4e-2,5e-2])
# perform the linesearch along -grad(qs)
X_ls = x0 - np.array([ti*grad_qs for ti in T_ls])
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
  outdata['T_ls'] = T_ls
  outdata['X_ls'] = X_ls
  outdata['qs_ls'] = qs_ls
  outdata['aspect_ls'] = asp_ls
  outdata['mirror_ls'] = mirror_ls
  outdata['energy_ls'] = energy_ls
  outdata['c_times_ls'] = c_times_ls
  # dump data
  indata[f'post_process_s_{s_label}'] = outdata
  pickle.dump(indata,open(data_file,"wb"))
  

# TODO: linesearch along aspect and mirror constraint gradients
  
