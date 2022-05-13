
## Lab Notebook


### Problems
- Cubic interpolation is not positivity preserving, we need to justify that it is not problematic as they did
  in the GYSELA 4D paper.
- Cubic interpolation is not a great choice for interpolation of the discontinuous function with only two heights
  that arrises from the initial conditions.

### TODO
- Run particle tracing on reactor scale device to verify that the coils and device get low particle losses. Visualize the trajectories. 
- Derive and implement a second order time splitting routine. 
- Implement cubic interpolation that leverages the time splitting to do lower dimension interpolation. Periodic directions can be interpolated with spectral accuracy using trig polynomials.
- Test cubic interpolation with a rotating cubic example (see cascade interpolation paper).
- Implement MPI parallelism
- Test 1: Run a driftless QA example. Multiply the drift terms in the guiding center equations by zero. Then particles 
  should only move along field lines, and should spread out over flux surfaces, but not move across flux surfaces.
  To avoid problems with discontinuities we could initialize our density to be continuous but still with compact.
  support.
- Test 2: Run a truly axisymmetric driftless computation. In true axisymmetry we just need a B field that satisfies 
  `div(B) = 0`. We can use `B = grad(psi) \cross \grad(phi) + G*\grad(\phi)` where `G` is a constant,
  `\psi = (R-R0)^2 + Z^2` and `R,phi,Z` are cylindrical coordinates. We can also parametrize the B field to 
   get other options, see the toroidal coordinates page of the Fusion wiki. In this examples particles should 
   simply spread out over flux surfaces since the drift terms will be set to zero. 
   They should not move across contours of psi.
- Test 3: Solve the PDE when the initial density has compact support over a small blob on the interior of 
  the plasma. Then compute the guiding center trajectories with particles starting from the same distribution and
  plot the video alongside the density in paraview.

### Completed
- [x] Change grid spacing: periodic directions are fine as uniform mesh, but other directions should use a better grid such as chebyshev.
- [x] Scale Bfield and device to reactor scale.
    - Device should be scaled up so that major radius is a few meters, say 5 or 10. To do this just multiply all of       the fourier coefficients of the boundary by the scale factor. This should improve confinement substantially.
    - The field strength should be scaled so that it is roughly 5 or 6 Tesla. For instance |B| inside the surface 
      should average to roughly 5 tesla. We can achieve this by multiplying the coil current by a scale factor.
      We could also build a toroidal flux objective from a surface and the coils, compute the .J quantity, and then
      compare to the desired target. The scale factor .J/target should be used to scale the coil currents.
    - Matt has sent a reactor-scaled QA configuration. Now we just need to scale the coil currents based on the
      target toroidal flux he sent.
- [x] Find appropriate width for vpar interval.
    > vpar can always be bound by [-vtotal,vtotal] because of conservation of energy. 
- [x] vectorize `u0` computation 
- [x] verify that u0 integrates to 1.
- [x] vectorize `GH_rhs` to get faster startup and backstep.
- [x] vectorize midpoint method integrator.
- [x] set up scipy Nd integrator to compute integral over volume.
- [x] set up scipy integrator to compute marginal over `x,y,z`.

### Sela improvements
- Look into mass conservative methods, such as mass conservative operator splitting (Durran).
- Look into positivity preserving interpolation that corrects over/undershoots.
- Only compute phi over a half field period and use reflective boundary conditions.
- Use VMEC or Boozer coordinates with a cartesian grid around the magnetic axis.
- High resolution mesh around plasma boundary.
- Implement a higher order timestepper.
- Implement adjoint
- [x] B field only depends on x,y,z and not vpar. So there is redundancy in computing B over the x,y,z,vpar grid that can be removed.
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
