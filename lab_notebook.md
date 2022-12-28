
# Lab Notebook

## Discuss with Matt
- in what ways is QS a restrictive condition? What other performance must be sacrificed?


## TODO:
- are tmax=0.1 configurations quasisymmetric?
  - just look at the results for the tmax=0.1 configs

- set up constraint gradients in post processor.

- post process results
  - make sure they are minima
  - look at the linesearch along the QS grad direction.

- make plot of iota0 vs losses for tmax=0.01
  - [x] generate nfp5 phase-one points with mmode=1,2 in phase-one.
  - [x] generate nfp2 phase-one points with mmode=1,2 in phase-one.
  - optimize losses for nfp4 to 4 modes
    - currently running super high res 4 modes
  - optimize losses for nfp5
    - currently running t=0.01 optimization to 4 modes.

- make 1d plots of objective, and objective approximation plots
  - [x] write scripts to generate data
  - generate data for tmax=1e-4
  - generate data for tmax=1e-3
  - plot results in notebook and decide on plot format

- select 1 sol with large losses
  - establish (local) optimality
  - determine the constraint activity
  - compute sensitivity to constraint parameters
  - find a local QS sol by starting the optimization here
    - if the QS sol is different, then we are in a local minima
  - make line between bad sol and nearby QS sol.
  - determine if QS gradient is in hessian null space at the bad sol.

- select 1 sol with small losses
  - establish (local) optimality
  - determine the constraint activity
  - compute sensitivity to constraint parameters
  - check if the QS gradient is in the null space of the hessian at the good sol

- make objective timing plot
  

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
