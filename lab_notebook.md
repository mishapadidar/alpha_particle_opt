
## Lab Notebook


### TODO
- Implement strang splitting... this will allow us to reduce our interpolation to one 3-dimensional and one 1-dimensional while giving us second order time accuracy.
- Implement cubic interpolation that leverages the time splitting to do lower dimension interpolation. Remember that periodic directions can be interpolated with spectral accuracy using trig polynomials. For cubic interpolation we can use the eqtools package (https://eqtools.readthedocs.io/en/latest/eqtools.html#eqtools.trispline.Spline) or we can implement the somewhat cheaper cubic interpolation by Ritchie.
- Update overleaf document with Stella algorithm details.
- Run Test 2, the axisymmetry test.
- Implement MPI parallelism
- Switch to boozer/VMEC coordinates.


### Verification Tests
- Test 1: Run a driftless QA example. Multiply the drift terms in the guiding center equations by zero. Then particles 
  should only move along field lines, and should spread out over flux surfaces, but not move across flux surfaces.
    > We mostly see the expecteed result. The problem is that particles are still flowing out of the plasma because
      they cannot navigate through the strange stellarator geometry. I think this could definitely be fixed by
      substantially increasing the grid discretization, using higher interpolation. Both of these implementations
      would likely require us to parallelize our implementation. A more computationally practical approach would
      be to use boozer coordinates, which naturally handle the geometry.
- Test 2: Run a truly axisymmetric driftless computation. In true axisymmetry we just need a B field that satisfies 
  `div(B) = 0`. We can use `B = grad(psi) \cross \grad(phi) + G*\grad(\phi)` where `G` is a constant,
  `\psi = (R-R0)^2 + Z^2` and `R,phi,Z` are cylindrical coordinates. We can also parametrize the B field to 
   get other options, see the toroidal coordinates page of the Fusion wiki. In this examples particles should 
   simply spread out over flux surfaces since the drift terms will be set to zero. 
   They should not move across contours of psi.
- Test 3: Solve the PDE when the initial density has compact support over a small blob on the interior of 
  the plasma. Then compute the guiding center trajectories with particles starting from the same distribution and
  plot the video alongside the density in paraview.
- Test 4: Tokamak with analytic solution. Initialize distribution which depends on `r,z,vpar` only through energy `v^2/2`, canonical angular momentum `p\_phi`, and `mu`. We should be able to find the orbits analytically. Equation 68 from matts intro to quasisymmetry shows how we can relate the toroidal flux to `vpar`. Some characteristics may leave the domain, so the distribution should be set to zero on those. Need a current density.


### Sela improvements
- Implement Ritchie's efficient cubic interpolation, find reference from Durran's book. Test cubic interpolation with a rotating cubic example (see cascade interpolation paper).
- Look into mass conservative methods, such as mass conservative operator splitting (Durran).
- Look into positivity preserving interpolation that corrects over/undershoots or justify that the 
  overshoots and undershoots are not problematic like in the GYSELA-4D paper.
- Only compute phi over a half field period and use reflective boundary conditions.
- Use VMEC or Boozer coordinates with a cartesian grid around the magnetic axis.
- High resolution mesh around plasma boundary.
- Implement a higher order timestepper.
- Implement adjoint
- [x] B field only depends on x,y,z and not vpar. So there is redundancy in computing B over the x,y,z,vpar grid that can be removed.
- [x] build a vtk writer for visualization of `u` over the mesh.


### Completed
- [x] Run particle tracing on reactor scale device to verify that the coils and device get low particle losses. Visualize the trajectories. 
- [x] Increase initial coil radius to actually enclose the plasma so that we get a good bfield.
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
- [x] Write a README for this stella and trace.
