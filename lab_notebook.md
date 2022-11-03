
## Lab Notebook

### TODO:
- examine optimization results to determine if we should
  - increase to major radius 10
  - run with more particles
  - increase dimensionality
  - use a different starting point
  - use global optimization
  - use grid vs random points
- optimize again
- correct sampling probability with jacobian.
  - correct this is in the grid objective too
  - simsopt BoozerMagneticField class or BoozerRadialInterpolant have methods
    dRds, dRdtheta etc for derivatives of cylindrical w.r.t. Boozer coords.
- set up another optimization routine
  - coordinate descent
  - BFGS with central difference
  - for bad noise do stomads, BO, TURBO, SPSA
  - fix routine for finding a bounding box
- set up variance reduction
  - importance sampling based on |B|: use |B| method from simsopt BoozerMagneticField class
  - set up and verify Matts classification of particles
  - backwards tracing, can do this in Boozer using simsopt methods.
  - control variates based on surfaces or shorter tracing or input variables.
- look at sensitivity to constraint and distribution parameters.
