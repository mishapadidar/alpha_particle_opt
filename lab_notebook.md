
# Lab Notebook

## TODO:
- incorporate pdf values in optimization over grid.
- Build sampling module sampling theta,zeta uniformly in space.
- do quasisymmetry optimization with aspect and mirror constraints.

- look over optimization results; maybe start opt with 4 modes

- develop a deterministic (biased) integration method for computing objective.
  - develop a gaussian quadrature on s.
  - develop a model of the confinement time.
    - leverage periodicity in theta,zeta
    - leverage mucrit boundary or 0.25v^2
    - leverage s,vpar scatter plot of confinement times.
    - try to model discontinuity.
  - integrate the model plus integrate the difference.

- use composite optimization

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
