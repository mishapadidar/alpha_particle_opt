
filelist=(
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_surf_0.25/data_opt_nfp4_phase_one_aspect_7.0_iota_-1.043_mean_energy_SAA_surface_0.25_tmax_0.01_bobyqa_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_surf_0.25/data_opt_nfp4_phase_one_aspect_7.0_iota_0.71_mean_energy_SAA_surface_0.25_tmax_0.01_bobyqa_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_surf_0.25/data_opt_nfp4_phase_one_aspect_7.0_iota_0.89_mean_energy_SAA_surface_0.25_tmax_0.01_bobyqa_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_surf_0.25/data_opt_nfp4_phase_one_aspect_7.0_iota_0.97_mean_energy_SAA_surface_0.25_tmax_0.01_bobyqa_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_surf_0.25/data_opt_nfp4_phase_one_aspect_7.0_iota_1.05_mean_energy_SAA_surface_0.25_tmax_0.01_bobyqa_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_surf_full/data_opt_nfp4_phase_one_aspect_7.0_iota_-1.043_mean_energy_SAA_surface_full_tmax_0.01_bobyqa_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_surf_full/data_opt_nfp4_phase_one_aspect_7.0_iota_0.53_mean_energy_SAA_surface_full_tmax_0.01_bobyqa_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_surf_full/data_opt_nfp4_phase_one_aspect_7.0_iota_0.89_mean_energy_SAA_surface_full_tmax_0.01_bobyqa_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_surf_full/data_opt_nfp4_phase_one_aspect_7.0_iota_0.97_mean_energy_SAA_surface_full_tmax_0.01_bobyqa_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_surf_full/data_opt_nfp4_phase_one_aspect_7.0_iota_1.05_mean_energy_SAA_surface_full_tmax_0.01_bobyqa_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_ebfgs/data_opt_nfp4_phase_one_aspect_7.0_iota_0.71_mean_energy_SAA_surface_0.25_tmax_0.01_ebfgs_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_ebfgs/data_opt_nfp4_phase_one_aspect_7.0_iota_0.89_mean_energy_SAA_surface_0.25_tmax_0.01_ebfgs_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_ebfgs/data_opt_nfp4_phase_one_aspect_7.0_iota_0.97_mean_energy_SAA_surface_0.25_tmax_0.01_ebfgs_mmode_3_iota_None.pickle'
'../../min_energy_loss/data_phase_one_tmax_0.01_SAA_sweep_resolved_ebfgs/data_opt_nfp4_phase_one_aspect_7.0_iota_1.05_mean_energy_SAA_surface_0.25_tmax_0.01_ebfgs_mmode_3_iota_None.pickle'
)
NODES=4
CORES=12
for idx in ${!filelist[@]}
do
  datafile=${filelist[idx]}

  # make a dir
  dir="_batch_${idx}"
  mkdir $dir

  # copy the compute data
  cp "./compute_data.py" "${dir}/compute_data.py"

  # write the run file
  RUN="${dir}/run.sh"
  if [ ! -f "${RUN}" ]; then
    printf '%s\n' "sbatch --requeue submit.sub" >> ${RUN}
    chmod +x ${RUN}
  fi

  # write the submit file
  SUB="${dir}/submit.sub"
  if [ -f "${SUB}" ]; then
    rm "${SUB}"
  fi
  printf '%s\n' "#!/bin/bash" >> ${SUB}
  printf '%s\n' "#SBATCH -J pp_${idx} # Job name" >> ${SUB}
  printf '%s\n' "#SBATCH -o ./job_%j.out    # Name of stdout output file(%j expands to jobId)" >> ${SUB}
  printf '%s\n' "#SBATCH -e ./job_%j.err    # Name of stderr output file(%j expands to jobId)" >> ${SUB}
  printf '%s\n' "#SBATCH -N ${NODES}       # Total number of nodes requested" >> ${SUB}
  printf '%s\n' "#SBATCH -n $[$NODES*$CORES]  # Total number of cores requested" >> ${SUB}
  printf '%s\n' "#SBATCH --ntasks-per-node ${CORES}    # Total number of cores requested" >> ${SUB}
  printf '%s\n' "#SBATCH --get-user-env     # Tells sbatch to retrieve the users login environment" >> ${SUB}
  printf '%s\n' "#SBATCH -t 96:00:00        # Time limit (hh:mm:ss)" >> ${SUB}
  printf '%s\n' "#SBATCH --mem-per-cpu=4000   # Memory required per allocated CPU" >> ${SUB}
  #printf '%s\n' "#SBATCH --partition=default_partition  # Which partition/queue it should run on" >> ${SUB}
  #printf '%s\n' "#SBATCH --partition=bindel  # Which partition/queue it should run on" >> ${SUB}
  printf '%s\n' "#SBATCH --exclude=g2-cpu-[01-11],g2-cpu-[97-99],g2-compute-[94-97]" >> ${SUB}
  printf '%s\n' "mpiexec -n $[$NODES*$CORES] python3 compute_data.py ${datafile}" >> ${SUB}
  
  ## submit
  cd "./${dir}"
  "./run.sh"
  cd ..

done
