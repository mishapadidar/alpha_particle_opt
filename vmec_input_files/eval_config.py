from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec

vmec_input = "input.torus"
comm = MpiPartition(1)
vmec = Vmec(vmec_input, mpi=comm,keep_all_files=False,verbose=False)
vmec.run()
