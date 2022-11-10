
SAMPLINGTYPE='grid' # grid, random
SURFS=('0.3' '0.5' '0.7' 'full') 
OBJECTIVE='mean_energy'
#OBJECTIVE='mean_time'
METHOD="sidpsm" # pdfo, nelder, snobfit, diff_evol, sidpsm
MAXMODE=3
VMEC="nfp2_QA_cold_high_res"
NS=7
NTHETA=7
NPHI=7
NVPAR=10

NODES=1
CORES=2
for idx in ${!SURFS[@]}
do
  surf=${SURFS[idx]}

  # make a dir
  dir="_safe_batch_${VMEC}_${OBJECTIVE}_${SAMPLINGTYPE}_surf_${surf}_${METHOD}_mmode_${MAXMODE}"
  mkdir $dir

  # copy the compute data
  cp "./safe_optimize.py" "${dir}/safe_optimize.py"
  cp "./safe_eval.py" "${dir}/safe_eval.py"

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
  printf '%s\n' "python3 safe_optimize.py ${SAMPLINGTYPE} ${surf} ${OBJECTIVE} ${METHOD} ${MAXMODE} ${VMEC} ${NS} ${NTHETA} ${NPHI} ${NVPAR}" >> ${SUB}
  
  ## submit
  cd "./${dir}"
  "./run.sh"
  cd ..

done