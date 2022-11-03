
## Lab Notebook

### TODO:
- examine optimization results to determine if we should
  - use grid vs random points # look at grid vs random
  - use global optimization or run with more particles # look at plot of fX over time
  - try other methods to avoid sim failures # check nelder mead vs pdfo
  - increase dimensionality # look at best performance for higher dim
  - increase to major radius 10
  - use a different starting point
- correct sampling probability with jacobian.
  - correct this is in the grid objective too
  - simsopt BoozerMagneticField class or BoozerRadialInterpolant have methods
    dRds, dRdtheta etc for derivatives of cylindrical w.r.t. Boozer coords.
- set up another optimization routine
  - coordinate descent
  - BFGS with central difference
  - for bad noise do stomads, BO, TURBO, SPSA, STARS
  - fix routine for finding a bounding box
- set up variance reduction
  - importance sampling based on |B|: use |B| method from simsopt BoozerMagneticField class
  - set up and verify Matts classification of particles
  - backwards tracing, can do this in Boozer using simsopt methods.
  - control variates based on surfaces or shorter tracing or input variables.
- look at sensitivity to constraint and distribution parameters.
