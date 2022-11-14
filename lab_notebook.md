
# Lab Notebook

## TODO:
- add constraint Bmax/Bmin >= 1.35 (value used for w7x)
- send new results to Matt

- Build sampling module for 
  - sampling theta,zeta uniformly in angle
  - computing moments of modB.
- test variance reduction
  - importance sampling (uniform in s,mu; linear in s,mu)
  - antithetic variables.
  - [x] stratified sampling based on s, mu
  - [x] control variate using s,mu.

- Implement local TuRBO, with multifidelity, constraint handling, and hidden constraint handling.
- Upgrade SID-PSM 
  - implement constraint capabilities via an l1-penalty method
  - use MNH method
  - use sim fails in direct search procedure.
- look at sensitivity to constraint and distribution parameters.

# Notes
- Noise is intrinsically about 20%, which we reduced to 2% with 10000 samples.
- Noise has a substantial effect on convergence. So use SAA, or set up a stochastic solver.
- COBYLA gets wrecked by simulation failures, nelder mead does better.
