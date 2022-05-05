
## Lab Notebook

### TODO
- Set up a reimann sum integrator for marginal and probability mass; might work better for the discontinuous problem
- Track a parcel of particles through probability space and compare to guiding center solution.
- Compare particle losses to particle tracing losses for LP QA problem.
- Find appropriate width for vpar interval. 
- [x] vectorize `u0` computation 
- [x] verify that u0 integrates to 1.
- [x] vectorize `GH_rhs` to get faster startup and backstep.
- [x] vectorize midpoint method integrator.
- [x] set up scipy Nd integrator to compute integral over volume.
- [x] set up scipy integrator to compute marginal over `x,y,z`.

### Sela improvements
- B field only depends on x,y,z and not vpar. So there is redundancy in computing B over the x,y,z,vpar grid that can be removed.
- Correct rk4 method to allow for larger timesteps.
- Use higher order interpolation (cascade interpolation, cubic by Ritchie)
- only compute phi over a half field period and use reflective boundary conditions.
- Look into mass conservative methods.
- Use VMEC coordinates (Zernike polynomials?).
- High resolution mesh around plasma boundary.
- use adaptive grid for vpar.
- Implement adjoint
- Implement GPU or MPI parallelism
- [x] build a vtk writer for visualization of `u` over the mesh.

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
