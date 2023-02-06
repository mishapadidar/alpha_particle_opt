import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import seaborn as sns
import glob

plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 22})
#import matplotlib.style as style
#style.use('tableau-colorblind10')

# colorblind colors
colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']
markers = ['s','o','^','x']
linestyles=['solid'
            ,'solid'
            ,'dotted'
            ,'dotted'
            ,'dotted'
            ,'dotted'
            ,'dotted'
            ,'dashed'
            ,'dashed'
            ,'dashed'
            ,'dashed'
            ,'dashed'
            ,'dashdot'
            ,'dashdot'
            ,'dashdot'
            ]

# load the data
infile = "./loss_profile_data.pickle"
indata = pickle.load(open(infile,"rb"))
# load the relevant keys
for key in list(indata.keys()):
    s  = f"{key} = indata['{key}']"
    exec(s)
n_configs = len(filelist)

# skip LP-QH and IPP-QA
skip = [3,8]


# compute the loss profiles
times = np.logspace(-5,np.log10(tmax),1000)
lp_vol = np.array([np.mean(c_times_vol< t,axis=1) for t in times])
lp_surf = np.array([np.mean(c_times_surface< t,axis=1) for t in times])

# get the configuration names
config_names = [ff[1] for ff in filelist]

# make a figure
fig, ax_both = plt.subplots(figsize=(14,6),ncols=2)
ax1,ax2 = ax_both

# choose colors
#from matplotlib.pyplot import cm
#colors = cm.jet(np.linspace(0, 1, n_configs))
from matplotlib.colors import ListedColormap
cmap = ListedColormap(sns.color_palette("colorblind",256))
colors = cmap(np.linspace(0,1,n_configs))

# plot the data
for ii in range(n_configs):
    if ii in skip:
        continue
    ax1.plot(times,lp_vol[:,ii],linewidth=3,linestyle=linestyles[ii],color=colors[ii],label=config_names[ii])
    ax2.plot(times,lp_surf[:,ii],linewidth=3,linestyle=linestyles[ii],color=colors[ii])
    print(config_names[ii],'volume losses',lp_vol[-1,ii],'surface losses',lp_surf[-1,ii])

# legend
#ax1.legend(ncols=3,fontsize=16,frameon=False)
#ax2.legend(ncols=3,fontsize=16,frameon=False)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels,bbox_to_anchor=(0.092, 0.93,0.90,0.93),loc=3,
          fancybox=True, shadow=False, ncol=7,fontsize=17,labelspacing=0.5,
           columnspacing=1.3)

for ax in ax_both:
    # darken the border
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth('2')  
    # log space 
    ax.set_yscale('log')
    ax.set_xscale('log')
    # limits
    ax.set_ylim([1e-3,0.3])
    ax.set_xlim([1e-5,1e-2])
    # ticks
    ax.set_xticks([1e-5,1e-4,1e-3,1e-2])
    ax.set_yticks([1e-3,1e-2,1e-1])
    # labels
    ax.set_xlabel('Time [sec]')
ax1.set_ylabel("Fraction of alpha particles lost",fontsize=18)

plt.tight_layout()


filename = "loss_profile.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
#plt.show()
