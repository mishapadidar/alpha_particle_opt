#!/usr/bin/env python

import sys
import numpy as np
import pickle
#from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
from simsopt._core import Optimizable
#from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual,vmec_compute_geometry
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve


"""
Phase one optimization to find configuration with
- specified aspect ratio
- specified iota
- bounded |B|
"""

debug=False
if debug:
    # for 4 field period
    vmec_label = "nfp4_QH_cold_high_res"
    aspect_target = 7.0
    iota_target = -1.043
    
    ## for 2 field period
    ##vmec_label = "nfp2_QA_cold_high_res"
    #aspect_target = 6.0
    #iota_target = 0.42
    
    # mirror ratio
    mirror_target = 1.35
    vmec_input = "../../vmec_input_files/" +"input." + vmec_label
else:
    # read inputs
    vmec_label = sys.argv[1] 
    mirror_target = float(sys.argv[2])
    aspect_target = float(sys.argv[3])
    iota_target = float(sys.argv[4])
    vmec_input = "../../../vmec_input_files/" +"input." + vmec_label


largest_mode = 1

mpi = MpiPartition()
vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
surf = vmec.boundary

if mpi.proc0_world:
  print("vmec_input:",vmec_input)
  print("mirror_target:", mirror_target)
  print("aspect_target:",aspect_target)
  print("iota_target:",iota_target)

# Only used for the computation of modB
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|

# bounds on the mirror ratio
class MirrorCon(Optimizable):
    """
    Constraints on |B|:
    |B| <= B_mean(1+eps_B)
    |B| >= B_mean(1-eps_B)
    where B_mean is, say, 5 Tesla.
    """
    def __init__(self, v, B_mean,mirror_target):
        self.v = v
        Optimizable.__init__(self, depends_on=[v])
        eps_B = (mirror_target - 1.0)/(mirror_target + 1.0)
        self.B_ub = B_mean*(1 + eps_B)
        self.B_lb = B_mean*(1 - eps_B)

    def J(self):
        """
        Constraint penalty. 

        return the residuals
          [np.max(|B| - B_ub,0.0),np.max(B_lb - |B|,0.0)]
        """
        # get modB from quasisymmetry function
        data = qs.compute()
        modB = data.modB
        #print(np.min(modB),np.max(modB))
        # modB <= B_ub
        c_ub = np.maximum(modB - self.B_ub,0.0)
        # modB >= B_lb
        c_lb = np.maximum(self.B_lb - modB,0.0)
        ret = np.append(c_ub,c_lb)
        return ret

    def B_minmax(self):
        data = qs.compute()
        modB = data.modB
        return np.min(modB),np.max(modB)

    def mirror_ratio(self):
        Bmin,Bmax =  self.B_minmax()
        return Bmax/Bmin

# target minor radius
minor_radius = surf.get('rc(0,0)')/aspect_target
# set B_mean based off of the toroidal flux Psi = pi*a^2 * B_mean
B_mean = vmec.indata.phiedge/np.pi/minor_radius/minor_radius
mirror = MirrorCon(vmec,B_mean,mirror_target)

prob = LeastSquaresProblem.from_tuples([(surf.aspect_ratio, aspect_target, 1),
                                        (vmec.mean_iota, iota_target, 1),
                                        (mirror.J, 0.0, 1)])
## phase one without mirror constraint
#prob = LeastSquaresProblem.from_tuples([(surf.aspect_ratio, aspect_target, 1),
#                                        (vmec.mean_iota, iota_target, 1)])

if mpi.proc0_world:
    print("Quasisymmetry objective before optimization:", qs.total())
    print("Total objective before optimization:", prob.objective())
    print('volavgB',vmec.wout.volavgB)
    print('mirror ratio',mirror.mirror_ratio())


for step in range(largest_mode):
    max_mode = step + 1

    # VMEC's mpol & ntor will be 3, 4, 5, 6:
    #vmec.indata.mpol = 3 + step
    #vmec.indata.ntor = vmec.indata.mpol

    if mpi.proc0_world:
        print("")
        print("Beginning optimization with max_mode =", max_mode, \
              ", vmec mpol=ntor=", vmec.indata.mpol, \
              ". Previous vmec iteration = ", vmec.iter)

    # Define parameter space:
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)") # Major radius

    # Carry out the optimization for this step:
    least_squares_mpi_solve(prob, mpi, grad=True)
    #max_nfev = 1
    #least_squares_mpi_solve(prob, mpi, grad=True,max_nfev=max_nfev)

    # get xopt
    xopt = np.copy(prob.x)
    surf.x = np.copy(xopt)

    if mpi.proc0_world:
        print("Done optimization with max_mode =", max_mode, \
              ". Final vmec iteration = ", vmec.iter)

        print("Final aspect ratio:", vmec.aspect())
        print("Quasisymmetry objective after optimization:", qs.total())
        print("Total objective after optimization:", prob.objective())
        print('major radius',surf.get("rc(0,0)"))
        print('aspect',surf.aspect_ratio())
        print('volavgB',vmec.wout.volavgB)
        print('iota',vmec.mean_iota())
        Bmin,Bmax = mirror.B_minmax()
        print("Bmin,Bmax",Bmin,Bmax)
        print("mirror ratio",Bmax/Bmin)
    
    # write the data to a file
    tail = f"_phase_one_mirror_{mirror_target}_aspect_{aspect_target}_iota_{iota_target}"
    outfilename = "input." + vmec_label + tail    
    vmec.write_input(outfilename)

    if mpi.proc0_world:
      outdata = {}
      outdata['xopt'] = np.copy(xopt)
      outdata['vmec_input'] = vmec_input
      outdata['max_mode'] = max_mode
      outdata['aspect_target'] = aspect_target
      outdata['iota_target'] = iota_target
      outdata['mirror_target'] = mirror_target
      outfilename = "data_" + vmec_label + tail + ".pickle"
      pickle.dump(outdata,open(outfilename,"wb"))

if mpi.proc0_world:
    print("============================================")