from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec
import sys

vmec_input = sys.argv[1]
comm = MpiPartition(1)
vmec = Vmec(vmec_input, mpi=comm,keep_all_files=False,verbose=False)
vmec.run()
