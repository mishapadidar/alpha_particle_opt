
## Lab Notebook

### TODO:
- debug MPI errors
- move MPI sampling routine to inside the tracer object
- move mean energy and mean confinement time objectives to inside 
  the tracer object
- set up iterative increase of nparticles in optimization
- set up surface sampling using SIMPLE
- figure out how to set particles with SAA b/c the distribution changes with boundary
- write a direct search method
- Look at optimization results
- set up and verify Matts classification of particles
- look at sensitivity to constraint and distribution parameters.

- look into variance reduction
  - hill climbing by just optimizing over a single surface
  - determine if we can trace particles based on the passing-trapped boundary for hill climbing. 
    - Plot the passing-trapped boundary along poloidal slices
    - make 1d plots of particles which are far from the pt-boundary
    - if the plot is smoother, that is a good sign!
  - determine if outer surfaces are more noisy than inner surfaces.
  - determine if B field can be used for classifying particles
    - look at criteria based on modB.
  - look into control variates and importance sampling
  - choose problems to directly handle randomness (NOMAD, Snobfit, Turbo)

## Notes
- Lots of noise at tmax = 1e-1. 
- We can probably optimize up to tmax = 1e-3 with 10,000 particles over all surfaces.
- Energy objective is much smoother than confinement time. 
