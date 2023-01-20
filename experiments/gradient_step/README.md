

This script was used to determine if rescaling the decision variables was effective.

In the experiment we 
1. compute the gradient of the energy loss, and linesearch the energy loss along its negative gradient direction
2. rescale the decision variables using the QS rescaling, recompute the gradient of the energy loss in the 
   new variables, then linesearch the energy loss along its negative gradient direction

The idea was to show that the finite difference gradient would be more accurate under the new variables.
This would be emphasized by the linesearch, which would show substantial decrease in the new coordinates and
minor decrease in the original coordinates.
