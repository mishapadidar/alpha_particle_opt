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
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

"""
Make data for the loss profile plots.

usage:
  mpiexec -n 1 python3 make_loss_profile_data.py

We rescale all configurations to the same minor radius and same volavgB.
Scale the device so that the major radius is 
  R = aspect*target_minor
where aspect is the current aspect ratio and target_minor is the desired
minor radius.
"""
# scaling params
target_minor_radius = 1.7
target_B00_on_axis = 5.7
#target_volavgB = 5.0

# tracing parameters
n_particles = 5000
tmax = 0.01
tracing_tol = 1e-8
interpolant_degree = 3
interpolant_level = 8
bri_mpol = 32
bri_ntor = 32
n_partitions = 1

filelist = [\
            ('nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_0.89', "A"),
            ('nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_1.043', "B"),
            #('20210721-02-180_new_QA_aScaling_t2e-1_s0.25_newB00', 'LP QA'),
            ('new_QA', 'LP QA'),
            #('20210721-02-181_new_QA_magwell_aScaling_t2e-1_s0.25_newB00', 'LP QA+well'),
            #('20210721-02-182_new_QH_aScaling_t2e-1_s0.25_newB00', 'LP QH'),
            ('new_QH', 'LP QH'),
            #('20210721-02-183_new_QH_magwell_aScaling_t2e-1_s0.25_newB00', 'LP QH+well'),
            #('20210721-02-184_li383_aScaling_t2e-1_s0.25_newB00', 'NCSX'),
            #('li383_1.4m', 'NCSX'),
            ('li383_1.4m_aScaling', 'NCSX'),
            #('ARIES-CS_aScaling_t2e-1_s0.25_newB00', 'ARIES-CS'),
            ('n3are_R7.75B5.7', 'ARIES-CS'),
            #('20210721-02-186_GarabedianQAS_aScaling_t2e-1_s0.25_newB00', 'NYU'),
            ('GarabedianQAS2', 'NYU'),
            #('20210721-02-187_cfqs_aScaling_t2e-1_s0.25_newB00', 'CFQS'),
            ('cfqs_2b40', 'CFQS'),
            #('20210721-02-188_Henneberg_aScaling_t2e-1_s0.25_newB00', 'IPP QA'),
            ('st_a34_i32v22_beta_35_scaledAUG', 'IPP QA'),
            #('20210721-02-189_Nuhrenberg_aScaling_t2e-1_s0.25_newB00', 'IPP QH'),
            ('NuhrenbergZille_1988_QHS', 'IPP QH'),
            #('20210721-02-190_HSX_aScaling_t2e-1_s0.25_newB00', 'HSX'),
            ('HSX_QHS_vacuum_ns201', 'HSX'),
            #('20210721-02-191_aten_aScaling_t2e-1_s0.25_newB00', 'Wistell-A'),
            ('aten', 'Wistell-A'),
            #('20210721-02-192_w7x-d23p4_tm_aScaling_t2e-1_s0.25_newB00', 'W7-X')]
            ('d23p4_tm', 'W7-X')]
#           ('20210721-02-148_w7x_aScaling_t2e-1_s0.3_newB00', 'W7-X')]
#           ('20210721-02-140_ncsx_aScaling_t2e-1_s0.3_newB00', 'NCSX'),

n_configs = len(filelist)

# storage arrays
c_times_surface = -np.inf*np.ones((n_configs,n_particles))
c_times_vol = -np.inf*np.ones((n_configs,n_particles))

# for saving data
outfile = "./loss_profile_data.pickle"
outdata = {}
outdata['filelist'] = filelist
outdata['target_minor_radius'] =target_minor_radius
#outdata['target_volavgB'] = target_volavgB
outdata['target_B00_on_axis'] = target_B00_on_axis
outdata['n_particles'] = n_particles
outdata['tmax'] = tmax
outdata['tracing_tol'] = tracing_tol
outdata['interpolant_degree'] = interpolant_degree
outdata['interpolant_level'] =  interpolant_level
outdata['bri_mpol'] = bri_mpol
outdata['bri_ntor'] = bri_ntor

for ii,(infile,config_name) in enumerate(filelist):

  # load the vmec input
  vmec_input = "./configs/input." + infile

  mpi = MpiPartition(n_partitions)
  vmec = Vmec(vmec_input, mpi=mpi,keep_all_files=False,verbose=False)
  surf = vmec.boundary

  ## set vmec tolerances
  #ns_array = np.zeros(100,dtype=int)
  #ns_array[0] = 16
  #ns_array[1] = 50
  #vmec.indata.ns_array = ns_array
  #niter_array = np.zeros(100,dtype=int)
  #niter_array[0] = 16
  #niter_array[1] = 50
  #vmec.indata.niter_array = niter_array
  #ftol_array = np.zeros(100,dtype=int)
  #ftol_array[0] = 600
  #ftol_array[1] = 5000
  #vmec.indata.ftol_array = ftol_array
  
  # get the aspect ratio for rescaling the device
  aspect_ratio = surf.aspect_ratio()
  major_radius = target_minor_radius*aspect_ratio
  
  
  # build a tracer object
  tracer = TraceBoozer(vmec_input,
                        n_partitions=n_partitions,
                        max_mode=-1,
                        aspect_target=aspect_ratio,
                        major_radius=major_radius,
                        target_volavgB=1.0, # dummy value
                        tracing_tol=tracing_tol,
                        interpolant_degree=interpolant_degree,
                        interpolant_level=interpolant_level,
                        bri_mpol=bri_mpol,
                        bri_ntor=bri_ntor)
  tracer.sync_seeds()
  x0 = tracer.x0

  # compute the boozer field
  field,bri = tracer.compute_boozer_field(x0)

  if field is None:
    # boozXform failed
    if rank == 0:
      print("boozXform failed for ",config_name)
    continue
  
  # now scale the toroidal flux by B(0,0)[s=0]
  if rank == 0:
    # b/c only rank 0 does the boozXform
    bmnc0 = bri.booz.bx.bmnc_b[0]
    B00 = 1.5*bmnc0[1] - 0.5*bmnc0[2]
    B00 = np.array([B00])
  else:
    B00 = np.array([0.0])
  comm.Barrier()
  comm.Bcast(B00,root=0)
  B00 = B00[0] # unpack the array
  # scale the toroidal flux
  tracer.vmec.indata.phiedge *= target_B00_on_axis/B00

  # re-compute the boozer field
  tracer.vmec.need_to_run_code = True
  tracer.vmec.run()
  tracer.field = None # so the boozXform recomputes
  field,bri = tracer.compute_boozer_field(x0)

  # now get B00 just to make sure it was set right
  if rank == 0:
    # b/c only rank 0 does the boozXform
    bmnc0 = bri.booz.bx.bmnc_b[0]
    B00 = 1.5*bmnc0[1] - 0.5*bmnc0[2]
    B00 = np.array([B00])
  else:
    B00 = np.array([0.0])
  comm.Barrier()
  comm.Bcast(B00,root=0)
  B00 = B00[0] # unpack the array

  # also get the minor radius
  major_rad = tracer.surf.get("rc(0,0)")
  aspect = tracer.surf.aspect_ratio()
  minor_rad = major_rad/aspect

  
  if rank == 0:
    print("")
    print("processing", infile)
    print('minor radius',minor_rad)
    print("axis B00",B00)
    print('volavgB',tracer.vmec.wout.volavgB)
    print('toroidal flux',tracer.vmec.indata.phiedge)
  stz_inits,vpar_inits = tracer.sample_volume(n_particles)
  c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
  if rank == 0:
    lf = np.mean(c_times < tmax)
    print('volume loss fraction:',lf)

  # store the data
  c_times_vol[ii] = np.copy(c_times)
  
  stz_inits,vpar_inits = tracer.sample_surface(n_particles,0.25)
  c_times = tracer.compute_confinement_times(x0,stz_inits,vpar_inits,tmax)
  if rank == 0:
    lf = np.mean(c_times < tmax)
    print('surface loss fraction:',lf)
    print("")

  # store the data
  c_times_surface[ii] = np.copy(c_times)

  # save the data
  outdata['c_times_surface'] = c_times_surface
  outdata['c_times_vol'] = c_times_vol
  pickle.dump(outdata,open(outfile,"wb"))
