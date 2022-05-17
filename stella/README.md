
### Guiding Center probability calculations with STELLA

The file `stella.py` contains the semi-lagrangian pde solver.

`input.new_QA_scaling` is an input file to the LP QA configuration
that is scaled to reactor size. Simply running `python3 bfield.py` will
generate coils for this configuration as well as 
a biotsavart field. It will also rescale the coil currents to reactor 
level toroidal flux, and plot the surface and initial coils.

`bfield.py` contains functions to generate a bfield for the scaled LP QA
configuration. It also contains runctions to generate a surface classifier
and compute the plasma volume.

`bfield_axisymmetry.py` contains an axisymmetric bfield class. This is used
in `ex3_track_blob_axisymmetry.py`.

Running the examples should populate `plot_data` with vtk files for paraview.
To visualize these files run `python3 write_pvd.py` to generate a `.pvd` file.

`stella.py` contains the semi-lagrangian solver.

`ex0_uniform_u0.py` places a uniform distribution over the plasma volume of the 
scaled LP QA configuration and simulates it with STELLA.

`ex1_track_blob.py` initialize a small packet of probability inside the plasma
then simulates the evolution with STELLA. To visualize the `.pvd` correctly, 
use the threshold function in paraview to filter out all probability values near 
zero... a good lower bound is `0.001` or `0.0001`.

`ex_2_track_blob_toroidal.py` initializes a small packet of probability in a 
purely Toroidal magnetic field. With then visualize the DRIFTLESS computation 
with STELLA. We solve the driftless computation because the field is not 
divergence free (vacuum). The expectd behavior is for particles to follow field
lines. To visualize the `.pvd` correctly, 
use the threshold function in paraview to filter out all probability values near 
zero... a good lower bound is `1.0`.

