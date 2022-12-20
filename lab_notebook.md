
# Lab Notebook


## TODO:
- make plot of iota0 vs losses for tmax=0.01
  - generate nfp5 phase-one points with mmode=1,2 in phase-one.
  - optimize losses for nfp4 and nfp5
  - warm start from sol where tmax=0.001, mmode=2
- make 1d plots of objective.
  - [x] write scripts to generate data for tmax=1e-4
  - plot results in notebook and decide on plot format
  - generate data for tmax=1e-2
  

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
