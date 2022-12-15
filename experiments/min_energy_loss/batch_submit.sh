
#TMAX='0.01'
#TMAX='0.001'
TMAX='0.0001'

SAMPLINGTYPE='grid' # grid, random, SAA
#SAMPLINGTYPE='SAA' # grid, random, SAA

#surf='0.25' # 0.25, 'full'
surf='full' # 0.25, 'full'
NODES=1
CORES=4

NS=9
NTHETA=7
NPHI=7
NVPAR=9

OBJECTIVE='mean_energy'  # mean_energy, mean_time
METHOD="bobyqa" # bobyqa, cobyla

MINMODE=1
MAXMODE=5

## 2 field period
#aspect=6.0
#iota='None'
#VMEC=(
#"nfp2_phase_one_aspect_6.0_iota_0.28"
#"nfp2_phase_one_aspect_6.0_iota_0.42"
#"nfp2_phase_one_aspect_6.0_iota_0.53"
#"nfp2_phase_one_aspect_6.0_iota_0.71"
#)
#WARM=("None" "None" "None" "None") # None or filename
##WARM=(
##"../data/data_opt_nfp2_phase_one_mean_energy_grid_surface_0.25_tmax_0.01_bobyqa_mmode_2_iota_None.pickle"
##"../data/data_opt_nfp2_phase_one_mean_energy_grid_surface_0.5_tmax_0.01_bobyqa_mmode_2_iota_None.pickle"
##"../data/data_opt_nfp2_phase_one_mean_energy_grid_surface_full_tmax_0.01_bobyqa_mmode_2_iota_None.pickle"
##      )

# 4 field period
aspect=7.0
iota='None'
VMEC=(
"nfp4_phase_one_aspect_7.0_iota_0.28"
"nfp4_phase_one_aspect_7.0_iota_0.42"
"nfp4_phase_one_aspect_7.0_iota_0.53"
"nfp4_phase_one_aspect_7.0_iota_0.71"
"nfp4_phase_one_aspect_7.0_iota_0.89"
"nfp4_phase_one_aspect_7.0_iota_0.97"
"nfp4_phase_one_aspect_7.0_iota_1.05"
"nfp4_phase_one_aspect_7.0_iota_1.29"
"nfp4_phase_one_aspect_7.0_iota_1.44"
)
WARM=("None" "None" "None" "None" "None" "None" "None" "None" "None") # None or filename
#WARM=(
#"../data/data_opt_nfp4_phase_one_mean_energy_SAA_surface_0.25_tmax_0.01_bobyqa_mmode_2_iota_None.pickle"
#"../data/data_opt_nfp4_phase_one_mean_energy_SAA_surface_0.5_tmax_0.01_bobyqa_mmode_2_iota_None.pickle"
#"../data/data_opt_nfp4_phase_one_mean_energy_SAA_surface_full_tmax_0.01_bobyqa_mmode_2_iota_None.pickle"
#      )



for idx in ${!VMEC[@]}
do
  warm=${WARM[idx]}
  vmec=${VMEC[idx]}

  # make a dir
  dir="_batch_${vmec}_${OBJECTIVE}_${SAMPLINGTYPE}_surf_${surf}_tmax_${TMAX}_${METHOD}_mmode_${MINMODE}_${MAXMODE}_iota_${iota}"
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
  printf '%s\n' "#SBATCH -n $[$NODES*$CORES]       # Total number of cores requested" >> ${SUB}
  printf '%s\n' "#SBATCH --ntasks-per-node ${CORES}    # Total number of cores requested" >> ${SUB}
  printf '%s\n' "#SBATCH --get-user-env     # Tells sbatch to retrieve the users login environment" >> ${SUB}
  printf '%s\n' "#SBATCH -t 96:00:00        # Time limit (hh:mm:ss)" >> ${SUB}
  printf '%s\n' "#SBATCH --mem-per-cpu=4000   # Memory required per allocated CPU" >> ${SUB}
  #printf '%s\n' "#SBATCH --partition=default_partition  # Which partition/queue it should run on" >> ${SUB}
  #printf '%s\n' "#SBATCH --partition=bindel  # Which partition/queue it should run on" >> ${SUB}
  printf '%s\n' "#SBATCH --exclude=g2-cpu-[01-11],g2-cpu-[29-30],g2-cpu-[97-99],g2-compute-[94-97],luxlab-cpu-02" >> ${SUB}
  printf '%s\n' "mpiexec -n $[$NODES*$CORES] python3 optimize.py ${SAMPLINGTYPE} ${surf} ${OBJECTIVE} ${METHOD} ${MINMODE} ${MAXMODE} ${vmec} ${warm} ${TMAX} ${iota} ${aspect} ${NS} ${NTHETA} ${NPHI} ${NVPAR}" >> ${SUB}
  
  ## submit
  cd "./${dir}"
  "./run.sh"
  cd ..

done
