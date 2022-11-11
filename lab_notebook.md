
# Lab Notebook

## Questions for Matt
- classification of particles based on mucrit does not work...
- moved back to RZFourier
- I find the aspect ratio is staying around 5 in the configs. Why? 
- what do you think of the configs?

## TODO:
- Implement distribution based sampling routines
  - stop tracing using grid! Build a grid to be uniform in the radial CDF.
  - Implement SAA option for sampling routine
- set up a warm start routine
- set up reduced tracing via mucrit tracing
- optimize again
  - optimize with more variables
  - from other starting points

- Upgrade SID-PSM 
  - implement constraint capabilities via an l1-penalty method
  - use MNH method
  - use sim fails in direct search procedure.
- Implement stochastic optimization
  - local TuRBO, with multifidelity, constraint handling, and hidden constraint handling.
  - StoMADS

- correct sampling probability with jacobian.
  - correct this is in the grid objective too
  - simsopt BoozerMagneticField class or BoozerRadialInterpolant have methods
    dRds, dRdtheta etc for derivatives of cylindrical w.r.t. Boozer coords.
- set up variance reduction
  - set up and verify Matts classification of particles
  - importance sampling based on |B|: use |B| method from simsopt BoozerMagneticField class
  - backwards tracing, can do this in Boozer using simsopt methods.
  - control variates based on surfaces or shorter tracing or input variables.
- look at sensitivity to constraint and distribution parameters.

# Notes
- Noise is intrinsically about 20%, which we reduced to 2% with 10000 samples.
- Noise has a substantial effect on convergence. So use SAA, or set up a stochastic solver.
- COBYLA gets wrecked by simulation failures, nelder mead does better.
