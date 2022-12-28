
from radial_density import RadialDensity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 16})

n_points = 1000
sampler = RadialDensity(n_points)

# plot the pdf
x = np.linspace(0,1,1000)
y = sampler._pdf(x)
plt.plot(x,y,linewidth=3)
plt.title("probability density $p(s)$")
plt.xlabel("Normalized toroidal flux $s$")
plt.tight_layout()
plt.show()
