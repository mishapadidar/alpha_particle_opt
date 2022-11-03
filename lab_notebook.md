
## Lab Notebook

### TODO:
- make post processing script batch submitable
- post process optimization results
  - determine if we should increase to major radius 10
  - determine if we need to run with more particles
  - compute variance of confinement times at optima
- increase variable dimension and reoptimize.
- generalize bounding box routine to handle constraint on acceptance probability
- set up another local optimization routine
  - coordinate descent
  - BFGS with central difference
  - for bad noise do BO, TURBO
- set up variance reduction
  - importance sampling based on |B|: use |B| method from simsopt BoozerMagneticField class
  - set up and verify Matts classification of particles
  - backwards tracing, can do this in Boozer using simsopt methods.
  - control variates based on surfaces or shorter tracing or input variables.
- correct sampling probability with jacobian.
  - perhaps we can just copy the routine from simple.
  - simsopt BoozerMagneticField class or BoozerRadialInterpolant have methods
    dRds, dRdtheta etc for derivatives of cylindrical w.r.t. Boozer coords.
- look at sensitivity to constraint and distribution parameters.
