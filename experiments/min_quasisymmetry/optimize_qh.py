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
Optimize a VMEC equilibrium for quasi-helical symmetry (M=1, N=1)
throughout the volume.
"""

mpi = MpiPartition()
largest_mode = 5
target_volavgB = 5.7
aspect_target = 8.0
major_radius = 1.7*aspect_target
# input file extension
vmec_label = "nfp4_QH_cold_high_res"

vmec_input = "../../vmec_input_files/" +"input." + vmec_label
vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
surf = vmec.boundary

# rescale the major radius
factor = major_radius/surf.get("rc(0,0)")
surf.x = surf.x*factor

# rescale the B field
target_avg_minor_rad = major_radius/aspect_target # target avg minor radius
vmec.indata.phiedge = np.pi*(target_avg_minor_rad**2)*target_volavgB
vmec.need_to_run_code = True


# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|

# Define objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, aspect_target, 1),
                                        (qs.residuals, 0, 1)])

if mpi.proc0_world:
    print("Quasisymmetry objective before optimization:", qs.total())
    print("Total objective before optimization:", prob.objective())
    print('volavgB',vmec.wout.volavgB)


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
    #least_squares_mpi_solve(prob, mpi, grad=True,max_nfev=1)

    if mpi.proc0_world:
        print("Done optimization with max_mode =", max_mode, \
              ". Final vmec iteration = ", vmec.iter)

        print("Final aspect ratio:", vmec.aspect())
        print("Quasisymmetry objective after optimization:", qs.total())
        print("Total objective after optimization:", prob.objective())
    
    xopt = np.copy(prob.x)

    # write the data to a file
    outfilename = "input." + vmec_label + f"_max_mode_{max_mode}_quasisymmetry_opt"
    vmec.write_input(outfilename)

    if mpi.proc0_world:
      outdata = {}
      outdata['xopt'] = np.copy(xopt)
      outdata['vmec_input'] = vmec_input
      outdata['max_mode'] = max_mode
      outdata['major_radius'] = major_radius
      outdata['aspect_target'] = aspect_target
      outdata['target_volavgB'] = target_volavgB
      outfilename = "data_" + vmec_label + f"_max_mode_{max_mode}_quasisymmetry_opt.pickle"
      pickle.dump(outdata,open(outfilename,"wb"))

if mpi.proc0_world:
    print("============================================")
