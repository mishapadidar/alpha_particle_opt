
# 5 field period
vmec="nfp5_cold_high_res"
aspect=5.0

## 4 field period
#vmec="nfp4_QH_cold_high_res"
#aspect=7.0

## for 2 field period
#vmec="nfp2_QA_cold_high_res"
#aspect=6.0

IOTAS=('0.28' '0.42' '0.53' '0.71' '0.89' '0.97' '1.05' '1.29' '1.44')
mirror=1.35

NODES=1
CORES=9
for idx in ${!IOTAS[@]}
do
  iota=${IOTAS[idx]}

  # make a dir
  dir="_batch_${vmec}_mirror_${mirror}_aspect_${aspect}_iota_${iota}"
  mkdir $dir

  # copy the compute data
  cp "./phase_one.py" "${dir}/phase_one.py"

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
  printf '%s\n' "#SBATCH -J p1_${aspect}_${iota} # Job name" >> ${SUB}
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
  printf '%s\n' "mpiexec -n $[$NODES*$CORES] python3 phase_one.py ${vmec} ${mirror} ${aspect} ${iota}" >> ${SUB}
  
  ## submit
  cd "./${dir}"
  "./run.sh"
  cd ..

done
