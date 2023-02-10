
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle

plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 23})
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

print("")
print('startup time', startup_time)
print('mean confinement time', np.mean(c_times_list[-1]))
print('loss fraction', np.mean(c_times_list[-1] < tmax_list[-1]))

# average the timings over the number of particles
trace_timings = trace_timings/n_particles

# convert tmax to milliseconds
tmax_list = 1000*np.array(tmax_list)


fig,ax= plt.subplots(figsize=(10,8))

# plot
ax.bar(tmax_list,trace_timings,width=tmax_list)

# grid
ax.grid(axis='both',zorder=10)

# axis labels
plt.xlabel('$t_{\max}$ [ms]')
plt.ylabel('Wall-clock-time per particle [sec]')

# xticks
#ax.set_xticks([1e-4,1e-3,1e-2,1e-1]

# scales
plt.yscale('log')
plt.xscale('log')

## darken the border
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(2)  

plt.tight_layout()

filename = "timing_plot.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
#plt.show()
