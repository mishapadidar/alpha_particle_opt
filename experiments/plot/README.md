
## Plotting scripts

The pickle files of optimized configurations should be copied to the `./configs` directory.
Solutions may need post processing before plotting. See instructions below.


`plot_configurations` Matt's plotting script for plotting general properties of the configurations.

`plot_loss_profile` plots the loss profiles. Requires running `make_loss_profile_data.py` first to generate the data.

`plot_field_strength_contour.py` plots the field strength in boozer coordinates. Requires running `make_field_strength_contour_data.py` first to generate the data to plot the field strength in boozer coordinates.

`print_configuration_data.py` can be used to print out data about the configurations. This can be run on SLURM using `./run.sh`, and by specying the file in `submit.sub`

`make_paraview_data.py` generates data for making 3d plots of the configurations in paraview.

