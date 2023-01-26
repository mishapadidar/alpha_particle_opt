
## Plotting scripts

The pickle files of optimized configurations should be copied to the `./configs` directory.
Solutions may need post processing before plotting. See instructions below.


`plot_configurations` plots the B field contours and the 3d configuration.

`plot_loss_profile` plots the loss profiles.

`plot_field_strength_contour.ipynb` plots the field strength in boozer coordinates. Requires running `make_field_strength_contour_data.py` first to generate the data to plot the field strength in boozer coordinates.

`print_configuration_data.py` can be used to print out data about the configurations. This can be run on SLURM using `./run.sh`, and by specying the file in `submit.sub`

