import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier


vmec_input = "input.torus"
surf = SurfaceRZFourier().from_vmec_input(vmec_input)
surf.to_vtk(vmec_input)
