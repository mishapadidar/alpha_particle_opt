#!/usr/bin/env python

import os
import numpy as np
import pickle
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
from simsopt.mhd.vmec_diagnostics import QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve


"""
Optimize a VMEC equilibrium for quasi-axis symmetry (M=1, N=0)
throughout the volume.
"""

largest_mode = 5

# for QA optimization
QA = True
aspect_target = 6.0
iota_target = 0.42
vmec_label = "nfp2_QA_cold_high_res_aspect_6_iota_0.42"
vmec_input = "./input.nfp2_QA_cold_high_res_max_mode_1_aspect_6_iota_0.42"

## for QH optimization
#QA = False
#aspect_target = 7.0
#iota_target = -1.0437569
#vmec_label = "input.nfp4_QH_cold_high_res_aspect_7_iota_-1.043"
#vmec_input = "./input.nfp4_QH_cold_high_res_max_mode_1_aspect_7_iota_-1.043"


mpi = MpiPartition()
vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
surf = vmec.boundary

if QA:
    # quasi-axis
    qs = QuasisymmetryRatioResidual(vmec,
                                    np.arange(0, 1.01, 0.1),  # Radii to target
                                    helicity_m=1, helicity_n=0)  # (M, N) you want in |B|
else:
    # quasi-helical
    qs = QuasisymmetryRatioResidual(vmec,
                                    np.arange(0, 1.01, 0.1),  # Radii to target
                                    helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|

# Define objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, aspect_target, 1),
                                        (vmec.mean_iota, iota_target, 1),
                                        (qs.residuals, 0, 1)])

if mpi.proc0_world:
    print("Quasisymmetry objective before optimization:", qs.total())
    print("Total objective before optimization:", prob.objective())
    print('volavgB',vmec.wout.volavgB)


for step in range(largest_mode):
    max_mode = step + 1

    ## VMEC's mpol & ntor will be 3, 4, 5, 6:
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
    

    # write the data to a file
    outfilename = "input." + vmec_label + f"_max_mode_{max_mode}_quasisymmetry_opt"
    vmec.write_input(outfilename)

    if mpi.proc0_world:
      outdata = {}
      outdata['xopt'] = np.copy(xopt)
      outdata['vmec_input'] = vmec_input
      outdata['max_mode'] = max_mode
      outdata['aspect_target'] = aspect_target
      outdata['iota_target'] = iota_target
      outfilename = "data_" + vmec_label + f"_max_mode_{max_mode}_quasisymmetry_opt.pickle"
      pickle.dump(outdata,open(outfilename,"wb"))

if mpi.proc0_world:
    print("============================================")
