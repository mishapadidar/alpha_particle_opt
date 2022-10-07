
## Lab Notebook

### Questions for Matt:
- pdf questions
  - We need the ability to sample from the pdf in boozer coordinates. How can we get the jacobian? Or a change of
    coordinates function?
  - We need to decide on a distribution, and have motivation for it in the paper. Can you choose one?
  - Should the pdf be uniform over vpar or over the angle formed by vpar, vperp?
  - Should the pdf be uniform spatially or have some radial decay?
- Objective questions
  - Can you point to any good functions for the energy deposition rates?
  - What statistic of the energy do you want optimized? Lets choose a few. 
- SIMPLE questions
  - what are the parameters in ```init_params``` in ```example_pyimple.py```
  - does simple have sampling capabilities?
  - simle sometimes returns -1 for trace time, why is that?

### TODO:
- [ ] Parallelize concurrent evaluations
  - Concurrent worker groups keep returning an identical set of evaluations. I think
    the boozer radial interpolant is somehow getting shared across worker groups. Talk
    to David about boxing this out.
- [ ] make 1D plots
  - [ ] look at 1d plots surface by surface of some summary statistics
  - [ ] for the surfaces with nasty sensitivity, 
    - [ ] look at the effect of adding more points on the 1d plots.
    - [ ] choose a dimension, and look at the violin plots of confinement time 
          along the dimension for a large number of MC points. This should show 
          that the density of confinement time is not sensitive, so long as we take
          enough points.
- [ ] look at sensitivity to initial particle states.
