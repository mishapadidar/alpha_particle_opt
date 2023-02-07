
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
from scipy.integrate import simpson
import sys
sys.path.append("../../sample")
from radial_density import RadialDensity
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 22})
#from matplotlib.colors import ListedColormap
#cmap = ListedColormap(sns.color_palette("colorblind",256))
#colors = cmap(np.linspace(0,1,n_configs))


# load the angle density data
indata = pickle.load(open("timing_data.pickle","rb"))
# load the relevant keys
for key in list(indata.keys()):
    s  = f"{key} = indata['{key}']"
    print(key)
    exec(s)



fig,ax_both = plt.figure(figsize=(10,6))

# plot
plt.bar(tmax_list,trace_timings,width=tmax_list)

# grid
plt.grid(axis='both',zorder=10)

# axis labels
plt.xlabel('$t_{\max}$')
plt.ylabel('Wall-clock-time per particle [sec]')

# scales
plt.yscale('log')
plt.xscale('log')


# darken the border
fig.patch.set_edgecolor('black')  
fig.patch.set_linewidth('2')  

plt.tight_layout()

filename = "timing_plot.pdf"
#plt.savefig(filename, format="pdf", bbox_inches="tight")
plt.show()
