
TMAX='0.001'

SAMPLINGTYPE='grid' # grid, random, SAA
#SAMPLINGTYPE='SAA' # grid, random, SAA

SURFS=('0.3' '0.5' 'full') 

OBJECTIVE='mean_energy'
#OBJECTIVE='mean_time'

METHOD="pdfo" # pdfo, nelder, snobfit, diff_evol, sidpsm
MAXMODE=1

#VMEC="nfp2_QA_cold_high_res"
VMEC="nfp2_QA_cold_high_res_mirror_feas"
#VMEC="nfp4_QH_warm_high_res"
#VMEC="nfp4_QH_cold_high_res"

WARM=("None" "None" "None") # None or filename
#WARM=("../data/data_opt_nfp4_QH_cold_high_res_mean_energy_grid_surface_0.3_tmax_0.001_pdfo_mmode_2.pickle" 
#      "../data/data_opt_nfp4_QH_cold_high_res_mean_energy_grid_surface_0.5_tmax_0.001_pdfo_mmode_2.pickle" 
#      "../data/data_opt_nfp4_QH_cold_high_res_mean_energy_grid_surface_0.7_tmax_0.001_pdfo_mmode_2.pickle" 
#      "../data/data_opt_nfp4_QH_cold_high_res_mean_energy_grid_surface_full_tmax_0.001_pdfo_mmode_2.pickle" 
#      )

NS=10
NTHETA=6
NPHI=6
NVPAR=10

NODES=1
CORES=12
for idx in ${!SURFS[@]}
do
  surf=${SURFS[idx]}
  warm=${WARM[idx]}

  # make a dir
  dir="_batch_${VMEC}_${OBJECTIVE}_${SAMPLINGTYPE}_surf_${surf}_tmax_${TMAX}_${METHOD}_mmode_${MAXMODE}"
  mkdir $dir

  # copy the compute data
  cp "./optimize.py" "${dir}/optimize.py"

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
  printf '%s\n' "#SBATCH -J ${METHOD}_${surf} # Job name" >> ${SUB}
  printf '%s\n' "#SBATCH -o ./job_%j.out    # Name of stdout output file(%j expands to jobId)" >> ${SUB}
  printf '%s\n' "#SBATCH -e ./job_%j.err    # Name of stderr output file(%j expands to jobId)" >> ${SUB}
  printf '%s\n' "#SBATCH -N ${NODES}       # Total number of nodes requested" >> ${SUB}
  printf '%s\n' "#SBATCH -n ${CORES}       # Total number of cores requested" >> ${SUB}
  printf '%s\n' "#SBATCH --ntasks-per-node ${CORES}    # Total number of cores requested" >> ${SUB}
  printf '%s\n' "#SBATCH --get-user-env     # Tells sbatch to retrieve the users login environment" >> ${SUB}
  printf '%s\n' "#SBATCH -t 96:00:00        # Time limit (hh:mm:ss)" >> ${SUB}
  printf '%s\n' "#SBATCH --mem-per-cpu=4000   # Memory required per allocated CPU" >> ${SUB}
  #printf '%s\n' "#SBATCH --partition=default_partition  # Which partition/queue it should run on" >> ${SUB}
  #printf '%s\n' "#SBATCH --partition=bindel  # Which partition/queue it should run on" >> ${SUB}
  printf '%s\n' "#SBATCH --exclude=g2-cpu-[01-11],g2-cpu-[29-30],g2-cpu-[97-99],g2-compute-[94-97],luxlab-cpu-02" >> ${SUB}
  #printf '%s\n' "#SBATCH --exclusive" >> ${SUB}
  printf '%s\n' "mpiexec -n $[$NODES*$CORES] python3 optimize.py ${SAMPLINGTYPE} ${surf} ${OBJECTIVE} ${METHOD} ${MAXMODE} ${VMEC} ${warm} ${TMAX} ${NS} ${NTHETA} ${NPHI} ${NVPAR}" >> ${SUB}
  
  ## submit
  cd "./${dir}"
  "./run.sh"
  cd ..

done
