
## Lab Notebook

### Questions 
- Should the `Bb` in the vparallel equation be a lowercase `b`?
- Does it matter that i havent fully converted my gc equations to cylindrical? 

### TODO
- remove `r_grid,phi_grid,...` from self. just save the list of points instead.
- Correct rk4 method to allow for larger timesteps.
- vectorize `u0` computation and `GH_rhs` to get faster startup and backstep.
- write method to compute marginal over `x,y,z`.
- write volume integration method to compute probability mass over grid.
- Implement a bfield test case for stella. 
    - use biotsavart field and GradAbsB feature. Wrap the x,y,z computations
      to compute in cylindrical.
- Run a Tokomak test case with analytic solution.

### Sela improvements
- vectorize computations
- build a vtk writer for visualization.
- Look into mass conservative methods.
- Use VMEC coordinates (Zernike polynomials?).
- Talk to Andrew or Alex Vlad about hyperbolic pdes, or spectral solvers.
- Use higher order interpolation (cascade interpolation, cubic by Ritchie)
- High resolution mesh around plasma boundary.
- Implement adjoint
- Implement GPU or MPI parallelism

