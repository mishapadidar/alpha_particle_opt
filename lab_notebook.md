
## Lab Notebook

### TODO:
- [ ] Parallelize concurrent evaluations
  - Concurrent worker groups keep returning an identical set of evaluations. I think
    the boozer radial interpolant is somehow getting shared across worker groups. Talk
    to David about boxing this out.
- [ ] make 1D plots
  - [ ] analyze effect on sensitivity of adding more points to objective
- [ ] Write up a deterministic optimization formulation
  - [ ] choose a few objectives
  - [ ] choose constraints
  - [ ] choose variable set size
- [ ] solve a deterministic optimization problem
  - [ ] choose a good solver based on 1d plot smoothness, sim failures, parallelism, objective structure and constraints.
  - [ ] Increase the VMEC resolution, use higher res input files.
- [ ] do an out-of-sample analysis.
- [ ] analyze physics metrics like QS, aspect, poincare plots, etc.
- [ ] do a perturbation analysis.
- [ ] analyze constraint tradeoffs locally.


### Extensions
- [ ] gradients
- [ ] collisionallity
- [ ] pressure (non-vacuum)
- [ ] standard, non-guiding center, tracing
