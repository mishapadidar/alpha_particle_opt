import numpy as np
from mpi4py import MPI
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
import pickle
import sys
sys.path.append("../../trace")
sys.path.append("../../utils")
sys.path.append("../../sample")
from trace_boozer import TraceBoozer

infile = sys.argv[1]
indata = pickle.load(open(infile,"rb"))
vmec_input = indata['vmec_input']
vmec_input = vmec_input[3:] # remove the first ../

# build a tracer object
n_partitions=1
xopt = indata['xopt'] 
max_mode = indata['max_mode']
major_radius = indata['major_radius']
aspect_target = indata['aspect_target']
target_volavgB = indata['target_volavgB']
tracing_tol = indata['tracing_tol']
interpolant_degree = indata['interpolant_degree'] 
interpolant_level = indata['interpolant_level'] 
bri_mpol = indata['bri_mpol'] 
bri_ntor = indata['bri_ntor'] 
tracer = TraceBoozer(vmec_input,
                      n_partitions=n_partitions,
                      max_mode=max_mode,
                      aspect_target=aspect_target,
                      major_radius=major_radius,
                      target_volavgB=target_volavgB,
                      tracing_tol=tracing_tol,
                      interpolant_degree=interpolant_degree,
                      interpolant_level=interpolant_level,
                      bri_mpol=bri_mpol,
                      bri_ntor=bri_ntor)
tracer.sync_seeds()
tracer.surf.x = np.copy(xopt)
tracer.vmec.run()
