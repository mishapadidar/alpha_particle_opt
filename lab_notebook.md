
## Lab Notebook

### Questions 
- Should the `Bb` in the vparallel equation be a lowercase `b`?
- Does it matter that i havent fully converted my gc equations to cylindrical? 

### TODO
- vectorize `u0` computation 
- set up scipy Nd integrator to compute integral over volume.
- verify that u0 integrates to 1.
- vectorize `GH_rhs` to get faster startup and backstep.
- vectorize midpoint method integrator.
- validate that divergence of `GC_rhs` is actually zero and the Louiville actually holds. Otherwise we 
  need to add terms to our pde.
- find appropriate width for vpar interval. 
- Correct rk4 method to allow for larger timesteps.
- Run the biotsavart test case, compute the probability mass, and verify against particle tracing.
- Run a Tokomak test case with analytic solution.
- set up scipy integrator to compute marginal over `x,y,z`.

### Sela improvements
- B field only depends on x,y,z and not vpar. So there is redundancy in computing B over the x,y,z,vpar grid that can be removed.
- only compute phi over a half field period and use reflective boundary conditions.
- build a vtk writer for visualization of `u` over the mesh.
- Look into mass conservative methods.
- Use VMEC coordinates (Zernike polynomials?).
- Talk to Andrew or Alex Vlad about hyperbolic pdes, or spectral solvers.
- Use higher order interpolation (cascade interpolation, cubic by Ritchie)
- High resolution mesh around plasma boundary.
- use adaptive grid for vpar.
- Implement adjoint
- Implement GPU or MPI parallelism

