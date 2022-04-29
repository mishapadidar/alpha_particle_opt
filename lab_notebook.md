
## Lab Notebook

### Questions 
- Should the `Bb` in the vparallel equation be a lowercase `b`?
- Does it matter that i havent fully converted my gc equations to cylindrical? 

### TODO
- vectorize `u0` computation and `GH_rhs` to get faster startup and backstep.
- remove `r_grid,phi_grid,...` from self. just save the list of points instead.
- write volume integration method to compute probability mass over grid. Verify that probability sums to 1.
- find appropriate width for vpar interval. 
- Correct rk4 method to allow for larger timesteps.
- write method to compute marginal over `x,y,z`.
- Implement a bfield test case for stella. 
    - use biotsavart field and GradAbsB feature. Wrap the x,y,z computations
      to compute in cylindrical.
- Run a Tokomak test case with analytic solution.

### Sela improvements
- B field only depends on x,y,z and not vpar. So there is redundancy in computing B over the x,y,z,vpar grid that can be removed.
- only compute phi over a half field period and use reflective boundary conditions.
- build a vtk writer for visualization.
- Look into mass conservative methods.
- Use VMEC coordinates (Zernike polynomials?).
- Talk to Andrew or Alex Vlad about hyperbolic pdes, or spectral solvers.
- Use higher order interpolation (cascade interpolation, cubic by Ritchie)
- High resolution mesh around plasma boundary.
- use adaptive grid for vpar.
- Implement adjoint
- Implement GPU or MPI parallelism

