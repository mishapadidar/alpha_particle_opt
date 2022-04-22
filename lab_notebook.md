
## Lab Notebook

### Questions
- How do we compute curl(vpar b) in true guiding center equations?
- How many timesteps of the ODE solver do we take in the charactersitic backtracking?

### TODO
- Validate Toroidal guiding center equations
    - Trace particles in Cartesian and toroidal coordinates.
- Develop Sela
- Test Sela
    - Use a purely toroidal B-Field that is non-zero inside some torus. 
    - Try more complicated B-Fields within a toroidal boundary shape. 
    - Use a tokomak B-field. Write the plasma boundary using the toroidal parameterization with 
      elongation, triangulation etc. Define a simple B-Field inside this boundary.
    - Use VMEC to build a full surface and B-Field.

### Selatok improvements
- Look into mass conservative methods through mass-consistent splitting.
- Use timestep splitting splitting or RK2 to improve efficiency of steps.
- Use higher order interpolation (cascade interpolation?)
- Implement adjoint
- Implement GPU or MPI parallelism

### Long Term improvements
- Implement adjoint
- Use VMEC coordinates to make system compatible with VMEC
