
# Lab Notebook

## Questions for Matt




## TODO:
- implement OpenMP tracing for parallelism.
- use garabedian coordinates. fix the minor radius to 1.7, delta00 is minor radius. fix the major radius to aspect x minor radius, delta10 is major radius
major radius is m=1,n=0
- phiedge = pi * a^2 * B to fix the field strength. use B = 5. a is minor radius, aspect is major/minor. verify that field strength is 5, use vmec.wout.volavgB.

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

- Make ctimesopt at end of optimization truly random.
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
- Noise is intrinsically about 20%, which we reduced to 2% with 10000 samples.
- Noise has a substantial effect on convergence. So use SAA, or set up a stochastic solver.
- COBYLA gets wrecked by simulation failures, nelder mead does better.
