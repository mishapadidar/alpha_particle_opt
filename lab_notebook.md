
# Lab Notebook

## Questions for Matt
- have Matt look over your input files
- show matt your radius rescaling
- what kind of confinement can we get with major radius 5? Should i rescale to 10?
- Can I constraint minor radius rather than aspect?

## TODO:
- examine optimization results 
  - analyze effect of sampling strategy 
    - compare grid vs random points
  - analyze effect of simulation failures
    - look at pdfo vs nelder 
    - look at plot of fX over time
  - look at plot of fX over time to understand
    - analyze effect of local minima
    - effect of noise
  - analyze effect of starting point
    - look at warm start vs cold start

- optimize again
  - Increase dimensionality maxmode=3
  - try optimizing directly with tmax=1e-4

- Implement SAA option for sampling routine
- Implement a global or stochastic opt method.

- correct sampling probability with jacobian.
  - correct this is in the grid objective too
  - simsopt BoozerMagneticField class or BoozerRadialInterpolant have methods
    dRds, dRdtheta etc for derivatives of cylindrical w.r.t. Boozer coords.
- set up variance reduction
  - importance sampling based on |B|: use |B| method from simsopt BoozerMagneticField class
  - set up and verify Matts classification of particles
  - backwards tracing, can do this in Boozer using simsopt methods.
  - control variates based on surfaces or shorter tracing or input variables.
- look at sensitivity to constraint and distribution parameters.

# Notes
- Noise has a substantial effect on convergence. So use SAA, or set up a stochastic solver.
