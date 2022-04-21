
## Lab Notebook

### Questions
> How do we compute curl(vpar b) in true guiding center equations?
> How many timesteps of the ODE solver do we take in the charactersitic backtracking?

### Build example problems
> Define a B-Field using Fourier coefficients. B should be independent of z,
  and should have constant curvature.
> Write a script to compute the QFM to get the plasma boundary.

### Selatok improvements
> Implement adjoint
> Use (strang) splitting to improve efficiency of steps.
> Use higher order interpolation (cascade interpolation?)
> Implement a RK2 integrator.
> Implement GPU or MPI parallelism

### Long Term improvements
> Implement adjoint
> Use VMEC coordinates to make system compatible with VMEC
