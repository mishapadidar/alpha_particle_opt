
## Lab Notebook

### TODO:
- Examine 1d plots at high resolution and noise plots
  - look at results by surface
  - look at three objectives: loss fraction, mean energy, mean time.
  - look at effect of sample size
  - look for deterministic noise
  - determine number of particles needed for each tmax
- optimize energy and confinement time objectives
  - dont exceed tmax = 1e-3
  - use more particles than 10,000 for tmax >= 1e-4 
  - increase ntimsteps
  - dont optimize losses on full plasma, use a subset of surfaces.
  - use a solver which can handle deterministic noise (TuRBO, differential evolution)
- Think of a way to regularize or smooth objective
  - are there any approximations which are smoother?
  - can we use multifidelity?
  - can we use backwards tracing at all?
- set up post optimization diagnostics
  - determine finite difference step size through ECNoise algorithm.
  - Make 1d plots at optima to visually measure noise at solution.
    - use this to determine if we need to run optimization again with more samples.
  - check optimality through the norm gradient at minimum.
  - look at out of sample performance.
  - look at quasisymmetry objective.
  - look at plots from Matts plotting script.
- write a direct search method
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
