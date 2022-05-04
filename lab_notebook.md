
## Lab Notebook

### TODO
- [x] vectorize `u0` computation 
- [x] verify that u0 integrates to 1.
- [x] vectorize `GH_rhs` to get faster startup and backstep.
- [x] vectorize midpoint method integrator.
- Initialize a small parcel of probability density and track it through the pde. Verify against particle tracing.
- set up scipy Nd integrator to compute integral over volume.
- Run the biotsavart test case, compute the probability mass, and verify against particle tracing.
- find appropriate width for vpar interval. 
- Correct rk4 method to allow for larger timesteps.
- Run a Tokomak test case with analytic solution.
- set up scipy integrator to compute marginal over `x,y,z`.

### Tokomak example
- initialize distribution which depends on r,z,vpar only through energy v^2/2, canonical angular momentum p\_phi, and mu
- we should be able to find the orbits analytically. equation 68 from matts intro to quasisymmetry shows how we can relate the toroidal flux to vpar.
- some characteristics may leave the domain, so the distribution should be set to zero on those.
- need a current density.

### Low energy example
- particles initialized with low energy, but vpar=v should stay on a flux surface.
- there should be zero losses out to a gyro radius around the plasma.
- For LP QA config run particle tracing starting particles on the boundary. All the losses should be here.
- Do particle traacing and compare losses with pde.


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

