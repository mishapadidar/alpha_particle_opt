
N_PARTICLES=(100 500 1000 2000 4000 8000 10000)

# nodes/cores to run with
NODES=6
CORES=7

for idx in ${!N_PARTICLES[@]}
do
  n_particles=${N_PARTICLES[idx]}

  # make a dir
  n_procs=$[$NODES*$CORES]
  dir="_batch_nprocs_${n_procs}_nparticles_${n_particles}"
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
  printf '%s\n' "#SBATCH -J timing_${n_procs} # Job name" >> ${SUB}
  printf '%s\n' "#SBATCH -o ./job_%j.out    # Name of stdout output file(%j expands to jobId)" >> ${SUB}
  printf '%s\n' "#SBATCH -e ./job_%j.err    # Name of stderr output file(%j expands to jobId)" >> ${SUB}
  printf '%s\n' "#SBATCH -N ${NODES}       # Total number of nodes requested" >> ${SUB}
  printf '%s\n' "#SBATCH -n ${n_procs}  # Total number of cores requested" >> ${SUB}
  printf '%s\n' "#SBATCH --ntasks-per-node ${CORES}    # Total number of cores requested" >> ${SUB}
  printf '%s\n' "#SBATCH --get-user-env     # Tells sbatch to retrieve the users login environment" >> ${SUB}
  printf '%s\n' "#SBATCH -t 96:00:00        # Time limit (hh:mm:ss)" >> ${SUB}
  printf '%s\n' "#SBATCH --mem-per-cpu=4000   # Memory required per allocated CPU" >> ${SUB}
  #printf '%s\n' "#SBATCH --partition=default_partition  # Which partition/queue it should run on" >> ${SUB}
  printf '%s\n' "#SBATCH --partition=bindel  # Which partition/queue it should run on" >> ${SUB}
  printf '%s\n' "#SBATCH --exclude=g2-cpu-[01-11],g2-cpu-[97-99],g2-compute-[94-97]" >> ${SUB}
  printf '%s\n' "mpiexec -n ${n_procs} python3 compute_data.py ${n_particles}" >> ${SUB}
  
  ## submit
  cd "./${dir}"
  "./run.sh"
  cd ..

done
