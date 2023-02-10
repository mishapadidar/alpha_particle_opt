import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import seaborn as sns
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 18})


filelist = ["quasisymmetry_data_config_A_mmode_3_tmax_0.01_n_0_step_GD.pickle"
            ,"quasisymmetry_data_config_A_mmode_3_tmax_0.01_n_1_step_GD.pickle"
            ,"quasisymmetry_data_config_A_mmode_3_tmax_0.01_n_-1_step_GD.pickle"
            ]

n_files = len(filelist)
step_sizes_list  = []
c_times_step_list = []
qs_step_list = []
energy_step_list = []
std_errs_energy_list = []
std_errs_loss_list = []
mn_list = []
qs0_list = []
energy0_list = []
loss_frac_list = []
for infile in filelist:
    
    indata = pickle.load(open(infile,"rb"))
    # load the keys quick and dirty
    for key in list(indata.keys()):
        s  = f"{key} = indata['{key}']"
        exec(s)
    
    # convert qs0 from residual to value
    qs0 = np.sum(qs0**2)

    # append x0 into the point list
    step_sizes = np.append(step_sizes,0.0)
    c_times_step = np.vstack((c_times_step,c_times_x0))
    qs_step = np.append(qs_step,qs0)
    energy_step = np.append(energy_step,energy_x0)
    
    # sort the arrays
    idx_sort = np.argsort(step_sizes)
    step_sizes = step_sizes[idx_sort]
    c_times_step = c_times_step[idx_sort]
    qs_step = qs_step[idx_sort]
    energy_step = energy_step[idx_sort]
    
    # filter out big steps
    idx_keep = (step_sizes <= 3e-4) & (step_sizes >= -3e-4)
    step_sizes = step_sizes[idx_keep]
    c_times_step = c_times_step[idx_keep]
    qs_step = qs_step[idx_keep]
    energy_step = energy_step[idx_keep]

    # remove step sizes < 0
    idx_keep = (step_sizes >= 0.0)
    step_sizes = step_sizes[idx_keep]
    c_times_step = c_times_step[idx_keep]
    qs_step = qs_step[idx_keep]
    energy_step = energy_step[idx_keep]

    # compute standard deviations of the energy
    feat = 3.5*np.exp(-2*c_times_step/tmax)
    stds = np.std(feat,axis=-1)
    std_errs_energy = 1.96*stds/np.sqrt(n_particles)

    # compute the standard error of the loss frac
    feat = c_times_step < tmax
    stds = np.std(feat,axis=-1)
    std_errs_loss = 1.96*stds/np.sqrt(n_particles)

    # compute loss fractions
    loss_frac = np.mean(c_times_step < tmax,axis=1)

    # now move the data into an array
    step_sizes_list.append(step_sizes)
    c_times_step_list.append(c_times_step)
    qs_step_list.append(qs_step)
    energy_step_list.append(energy_step)
    std_errs_energy_list.append(std_errs_energy)
    std_errs_loss_list.append(std_errs_loss)
    mn_list.append((helicity_m,helicity_n))
    qs0_list.append(qs0)
    energy0_list.append(energy_x0)
    loss_frac_list.append(loss_frac)


# colorblind colors
colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']


"""
#####################################
Plot the energy and quasisymmetry
#####################################
"""

## plot the energy 
fig,ax_both = plt.subplots(figsize=(9,6),ncols=2)
ax1,ax2 = ax_both
# darken the border
for ax in ax_both:
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(2)  

for ii in range(n_files):
    # load data for plot
    step_sizes = step_sizes_list[ii]
    energy_step = energy_step_list[ii]
    std_errs_energy = std_errs_energy_list[ii]
    helicity_m = mn_list[ii][0]
    helicity_n = mn_list[ii][1]
    energy0 = energy0_list[ii]
    qs0 = qs0_list[ii]
    qs_step = qs_step_list[ii]

    # normalize qs
    qs_step = qs_step_list[ii]/qs0

    # plot energy
    ax1.plot(step_sizes,energy_step,label="$Q_{%d,%d}$"%(helicity_m,helicity_n),marker='s',markersize=10,linewidth=3,color=colors[ii])
    ax1.fill_between(step_sizes,energy_step + std_errs_energy, energy_step - std_errs_energy,alpha=0.2,color=colors[ii])

    # plot QS
    ax2.plot(step_sizes,qs_step,label="$Q_{%d,%d}$"%(helicity_m,helicity_n),marker='s',markersize=10,linewidth=3)

# axis labels
ax1.set_ylabel("$\mathcal{J}_{1/4}$")
ax1.set_xlabel("$\\alpha$")
ax2.set_xlabel("$\\alpha$")
ax2.set_ylabel("$Q_{m,n}(w(\\alpha))/Q_{m,n}(w_A)$")

# axis scale
ax1.set_xscale('symlog',linthresh=1e-5)
ax2.set_xscale('symlog',linthresh=1e-5)

# grid
ax1.grid()
ax2.grid()

# legend
#ax1.legend(loc='upper left')
#ax2.legend(loc='lower left')
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels,bbox_to_anchor=(0.29, 0.95,0.90,0.93),loc=3,
          fancybox=True, shadow=False, ncol=3,fontsize=17,labelspacing=0.5,
           columnspacing=1.6)

plt.tight_layout()

filename = "energy_and_qs.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
#plt.show()



"""
#############################
Plot a pareto curve
#############################
"""

# plot the pareto front
fig,ax = plt.subplots(figsize=(7,6))
# darken the border
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(2)  

for ii in range(n_files):
    # get data
    step_sizes = step_sizes_list[ii]
    energy_step = energy_step_list[ii]
    std_errs = std_errs_energy_list[ii]
    helicity_m = mn_list[ii][0]
    helicity_n = mn_list[ii][1]
    energy0 = energy0_list[ii]
    qs0 = qs0_list[ii]
    qs_step = qs_step_list[ii]
    loss_step = loss_frac_list[ii]

    # normalize
    qs_step = qs_step/qs0

    # downselect points
    loss_step = np.delete(loss_step,[1,2])
    qs_step = np.delete(qs_step,[1,2])
    energy_step = np.delete(energy_step,[1,2])
    std_errs = np.delete(std_errs,[1,2])

    plt.plot(qs_step,energy_step,label="$Q_{%d,%d}$"%(helicity_m,helicity_n),marker='s',markersize=10,linewidth=3,color=colors[ii])
    plt.fill_between(qs_step,energy_step + std_errs, energy_step - std_errs,alpha=0.2,color=colors[ii])

    # plot our starting point
    plt.scatter([1.0],energy0,marker='s',s=120,color='k',zorder=100)

plt.ylabel("$\mathcal{J}_{1/4}$")
plt.xlabel("$Q_{m,n}(w)/Q_{m,n}(\mathrm{w}_{A})$")
plt.legend(loc='lower left')
plt.grid()
plt.tight_layout()

filename = "pareto_energy.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
#plt.show()


"""
##################################
Plot a loss fraction pareto curve
##################################
"""

# plot the pareto front
fig,ax = plt.subplots(figsize=(7,6))
# darken the border
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(2)  

for ii in range(n_files):
    # get data
    step_sizes = step_sizes_list[ii]
    std_errs_loss = std_errs_loss_list[ii]
    helicity_m = mn_list[ii][0]
    helicity_n = mn_list[ii][1]
    qs0 = qs0_list[ii]
    qs_step = qs_step_list[ii]
    loss_step = loss_frac_list[ii]

    # normalize
    qs_step = qs_step/qs0

    # downselect points
    loss_step = np.delete(loss_step,[1,2])
    qs_step = np.delete(qs_step,[1,2])
    std_errs_loss = np.delete(std_errs_loss,[1,2])

    plt.plot(qs_step,loss_step,label="$Q_{%d,%d}$"%(helicity_m,helicity_n),marker='s',markersize=10,linewidth=3,color=colors[ii])
    plt.fill_between(qs_step,loss_step + std_errs_loss, loss_step - std_errs_loss,alpha=0.2,color=colors[ii])

plt.ylabel("Fraction of alpha particles lost")
plt.xlabel("$Q_{m,n}(w)/Q_{m,n}(\mathrm{w}_{A})$")
plt.legend(loc='lower left')
plt.grid()
plt.tight_layout()

filename = "pareto_loss_frac.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
#plt.show()
