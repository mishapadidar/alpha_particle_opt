
# Lab Notebook

## Questions for Matt
- look at results
  - does QA need iota constraint?
  - Why is |B| so ridiculous? Even in quasisymmetry optimization.
  - In construction do we control |B| or toroidal flux?
  - solutions are not QS.
- Optimization summary
  - highly constrained problem
  - gradient based opt could be used for t<=1e-3 or surface based opt
  - need concurrent evals.
- How do we set up Trace VMEC?

## TODO:
- Build module to compute probabilities of theta,zeta.
- do optimization with cobyla

- set up tracing in vmec coords
- set up gradient based optimization
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
