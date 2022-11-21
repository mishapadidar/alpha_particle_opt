
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
- add constraint Bmax/Bmin >= 1.35 (value used for w7x)
- use more parallelism for optimization
- Build sampling module sampling theta,zeta uniformly in space.
- Upgrade SID-PSM 
  - implement constraint capabilities via an l1-penalty method
  - use MNH method
  - use sim fails in direct search procedure.
- look at sensitivity to constraint and distribution parameters.

# Notes
- Noise is intrinsically about 20%, which we reduced to 2% with 10000 samples.
- Noise has a substantial effect on convergence. So use SAA, or set up a stochastic solver.
- COBYLA gets wrecked by simulation failures, nelder mead does better.
