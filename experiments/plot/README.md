
## Plotting scripts

The pickle files of optimized configurations should be copied to the `./configs` directory.
Solutions may need post processing before plotting. See instructions below.

Figure 1,
`make_density_data.py` generate the probability density data.
`plot_densities.py` the plot of the probability densities for the paper.

Figure 2
`make_timing_data.py` generate the timing data.
`plot_timing_data.py` the plot of showing wall-clock-time of particle tracing.

Figure 3
`collisional_tracing` contains everything needed to make Figure 3, the validation of the energy model.

Figure 5
`make_paraview_data.py` generates data for making 3d plots of the configurations in paraview.
`plot_cross_sections.py` plots the cross sectional slices of the stellarators.

Table 1
`print_configuration_info.py` can be used to print out data about the configurations. This can be run on SLURM using `./run.sh`, and by specying the file in `submit.sub`


Figure 6
`make_loss_profile_data.py` generates the loss profiles via particle tracing.
`plot_loss_profile` plots the loss profiles. Requires running `make_loss_profile_data.py` first to generate the data.

Figure 7
`make_field_strength_contour_data.py` generates the data to plot the field strength in boozer coordinates.
`plot_field_strength_contour.py` plots the field strength in boozer coordinates. 


Other
`plot_configurations` Matt's plotting script for plotting general properties of the configurations.


