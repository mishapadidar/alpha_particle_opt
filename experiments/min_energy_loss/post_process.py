import numpy as np
from mpi4py import MPI
import sys
import pickle
import glob
sys.path.append("../../utils")
sys.path.append("../../trace")
sys.path.append("../../sample")
from trace_boozer import TraceBoozer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# files to process
searchfor = "./data/*.pickle"

# tracing params
tmax = 1e-2
n_particles = 1000
s_label_list = [0.25,"full"]

# tracing accuracy params
tracing_tol=1e-8
interpolant_degree=3
interpolant_level=8
bri_mpol=16
bri_ntor=16


# find a set of files
filelist = glob.glob(searchfor)

for data_file in filelist:
   
    # load the file
    data = pickle.load(open(data_file,"rb"))
    vmec_input = data['vmec_input']
    vmec_input = vmec_input[3:] # remove the first ../
    x0 = data['xopt']
    max_mode=data['max_mode']
    aspect_target = data['aspect_target']
    major_radius = data['major_radius']
    target_volavgB = data['target_volavgB']
    
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
    
    if rank == 0:
        print('aspect',tracer.surf.aspect_ratio())
        print('iota',tracer.vmec.mean_iota())
        print('major radius',tracer.surf.get('rc(0,0)'))
        print('toroidal flux',tracer.vmec.indata.phiedge)
    
    
    # tracing points
    for s_label in s_label_list:

        # sample
        tracer.sync_seeds()
        if s_label == "full":
            stz_inits,vpar_inits = tracer.sample_volume(n_particles)
        else:
            stz_inits,vpar_inits = tracer.sample_surface(n_particles,s_label)

        # trace
        if rank == 0:
            print('tracing')
        c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)

        # print
        std_err = np.std(c_times)/np.sqrt(len(c_times))
        mu = np.mean(c_times)
        nrg = np.mean(3.5*np.exp(-2*c_times/tmax))
        if rank == 0:
            print(f'sampled from {s_label}')
            print('mean',mu,'std_err',std_err)
            print('energy',nrg)
            print('loss fraction',np.mean(c_times < tmax))
        
        # save data
        data[f'c_times_out_of_sample_{s_label}'] = np.copy(c_times)
  
    # dump data
    pickle.dump(data,open(data_file,"wb"))
