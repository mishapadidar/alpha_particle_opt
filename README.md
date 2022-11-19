# alpha_particle_opt

Optimal number of cores to use on G2 for particle tracing.
If in doubt, using less cores is often better than using more. 

|tmax | nparticles  | cores | eval time [sec] | 
|---- | ----------- | ----  | --------------- |
|1e-2 | 500         | 12    | 90              |
|1e-2 | 1000        | 48    | 82              |
|1e-2 | 2000        | 48    | 101             |
|1e-2 | 5000        | 48    | 155             |
|1e-3 | 500         | 4     | 23              |
|1e-3 | 1000        | 8     | 38              |
|1e-3 | 2000        | 8     | 40              |
|1e-3 | 10000       | 36    | 70              |
|1e-4 | 1000        | 2     | 10              |
|1e-4 | 2000        | 4     | 11              |
|1e-4 | 4000        | 4     | 13              |
