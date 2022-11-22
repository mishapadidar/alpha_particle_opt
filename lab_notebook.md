
# Lab Notebook

## TODO:
- develop a deterministic (biased) integration method for computing objective.
  - develop a gaussian quadrature on s.
  - develop a model of the confinement time.
    - leverage periodicity in theta,zeta
    - leverage mucrit boundary or 0.25v^2
    - leverage s,vpar scatter plot of confinement times.
    - try to model discontinuity.
  - integrate the model plus integrate the difference.
- Build sampling module sampling theta,zeta uniformly in space.
- post process results
  - physics analysis (Matt)
  - out-of-sample performance
  - sensitivity to constraint and distribution parameters.
- Upgrade SID-PSM 
  - implement constraint capabilities via an l1-penalty method
  - use MNH method
  - use sim fails in direct search procedure.
