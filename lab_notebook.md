
# Lab Notebook

## Discuss with Matt
- in what ways is QS a restrictive condition? What other performance must be sacrificed?
- what is the mean collision time of particles?


## TODO:

- make objective timing plot

- make 1d plots of objective, and objective approximation plots
  - currently generating data for tmax=1e-3
  - plot data for tmax=1e-3
- make plot of iota0 vs losses for tmax=0.01
  - currently optimizing losses for nfp5 for tmax=0.01
  - plot data for tmax=0.01

- select multiple sols and post process results
  - establish (local) optimality
  - determine the constraint activity
  - compute sensitivity to constraint parameters
  - find a local QS sol by starting the optimization here
    - if the QS sol is different, then we are in a local minima
  - make line between bad sol and nearby QS sol.
  - determine if QS gradient is in hessian null space at the bad sol.
  

## Long Term:
- set up tracing in vmec coords
  - finish writing the guiding center eqns
  - test the guiding center eqns
  - [x] generalize the stopping criteria
  - test the particle tracing

- set up a trace vmec module.
  - fix the comms in the guiding center tracing
  - set up concurrent evaluations for vmec tracing
  - develop gradient evals.

- set up gradient based optimization
- Build module to compute probabilities of theta,zeta.
- set up composite optimization with concurrent evaluations
