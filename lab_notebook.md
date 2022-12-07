
# Lab Notebook


## TODO:
- optimize QS from phase one to show that good sols exist
  - test the particle losses
  - save QS results
  - save phase one results
- optimize to 4 modes tmax=1e-4
  - use phase one starting points

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

- post process results
  - out-of-sample performance
  - establish that our sols are minima.
  - comparison differences with quasisymmetry solutions.
    - compare to solutions optimized for quasisymmetry.
    - are these minima of confinement time.
    - plot E[confinement time] over trajectory of quasisymmetry optimization.
      to justify multimodality of E[confinement time]
  - physics analysis (Matt)
  - sensitivity to constraint and distribution parameters.
