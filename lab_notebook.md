
## Lab Notebook

### Questions 
- Should the `Bb` in the vparallel equation be a lowercase `b`?
- Does it matter that i havent fully converted my gc equations to cylindrical? 

### TODO
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

