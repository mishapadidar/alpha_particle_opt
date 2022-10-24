
## Lab Notebook

### TODO:
- determine sample sizes required to dampen noise
- set up a direct search method
- optimize energy and confinement time objectives with SAA
- look at out of sample performance.
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
