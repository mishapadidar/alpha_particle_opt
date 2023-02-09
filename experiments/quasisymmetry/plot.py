import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import seaborn as sns
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 18})


#infile = "quasisymmetry_data_config_A_mmode_3_tmax_0.01_n_0_step_GD.pickle"
#infile = "quasisymmetry_data_config_A_mmode_3_tmax_0.01_n_1_step_GD.pickle"
#infile = "quasisymmetry_data_config_A_mmode_3_tmax_0.01_n_-1_step_GD.pickle"
filelist = ["quasisymmetry_data_config_A_mmode_3_tmax_0.01_n_0_step_GD.pickle"
            ,"quasisymmetry_data_config_A_mmode_3_tmax_0.01_n_1_step_GD.pickle"
            ,"quasisymmetry_data_config_A_mmode_3_tmax_0.01_n_-1_step_GD.pickle"
            ]

n_files = len(filelist)
step_sizes_list  = []
c_times_step_list = []
qs_step_list = []
energy_step_list = []
std_errs_list = []
mn_list = []
qs0_list = []
energy0_list = []
for infile in filelist:
    
    indata = pickle.load(open(infile,"rb"))
    # load the keys quick and dirty
    for key in list(indata.keys()):
        s  = f"{key} = indata['{key}']"
        print(key)
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

    # compute standard deviations of the energy
    feat = 3.5*np.exp(-2*c_times_step/tmax)
    stds = np.std(feat,axis=-1)
    std_errs = 1.96*stds/np.sqrt(n_particles)

    # now move the data into an array
    step_sizes_list.append(step_sizes)
    c_times_step_list.append(c_times_step)
    qs_step_list.append(qs_step)
    energy_step_list.append(energy_step)
    std_errs_list.append(std_errs)
    mn_list.append((helicity_m,helicity_n))
    qs0_list.append(qs0)
    energy0_list.append(energy_x0)


"""
#############################
Plot the energy
#############################
"""

# plot the energy 
fig = plt.figure(figsize=(8,6))
for ii in range(n_files):
    step_sizes = step_sizes_list[ii]
    energy_step = energy_step_list[ii]
    std_errs = std_errs_list[ii]
    helicity_m = mn_list[ii][0]
    helicity_n = mn_list[ii][1]
    energy0 = energy0_list[ii]
    qs0 = qs0_list[ii]
    qs_step = qs_step_list[ii]
    plt.plot(step_sizes,energy_step,label="$Q_{%d,%d}$"%(helicity_m,helicity_n),marker='s',markersize=10,linewidth=3)
    plt.fill_between(step_sizes,energy_step + std_errs, energy_step - std_errs,alpha=0.5)
    plt.scatter([0.0],[energy0],s=50,color='k')
plt.ylabel("$\mathcal{J}_{1/4}$")
plt.xlabel("$\\alpha$")
plt.legend(loc='upper left')
plt.grid()
plt.xscale('symlog',linthresh=1e-5)
plt.show()


"""
#############################
Now plot the quasisymmetry metrics
#############################
"""

fig = plt.figure(figsize=(8,6))
for ii in range(n_files):
    step_sizes = step_sizes_list[ii]
    energy_step = energy_step_list[ii]
    helicity_m = mn_list[ii][0]
    helicity_n = mn_list[ii][1]
    energy0 = energy0_list[ii]
    qs0 = qs0_list[ii]
    qs_step = qs_step_list[ii]/qs0
    plt.plot(step_sizes,qs_step,label="$Q_{%d,%d}$"%(helicity_m,helicity_n),marker='s',markersize=10,linewidth=3)
    #plt.fill_between(step_sizes,energy_step + std_errs, energy_step - std_errs,alpha=0.5)
    plt.scatter([0.0],[1.0],s=20,color='k')
plt.xlabel("$\\alpha$")
plt.ylabel("$Q_{m,n}$")
plt.legend(loc='upper left')
plt.grid()
plt.xscale('symlog',linthresh=1e-5)
plt.show()


"""
#############################
Plot a pareto curve
#############################
"""

# plot the pareto front
fig = plt.figure(figsize=(6,6))
for ii in range(n_files):
    # get data
    step_sizes = step_sizes_list[ii]
    energy_step = energy_step_list[ii]
    std_errs = std_errs_list[ii]
    helicity_m = mn_list[ii][0]
    helicity_n = mn_list[ii][1]
    energy0 = energy0_list[ii]
    qs0 = qs0_list[ii]
    qs_step = qs_step_list[ii]

    # normalize
    qs_step = qs_step/qs0

    # TODO: dont normalize energy0
    #energy_step = energy_step/energy0

    ## downselect points
    #qs_step = qs_step[::2]
    #energy_step = energy_step[::2]

    plt.plot(qs_step,energy_step,label="$Q_{%d,%d}$"%(helicity_m,helicity_n),marker='s',markersize=10,linewidth=3)
    #plt.fill_between(qs_step,energy_step + std_errs, energy_step - std_errs,alpha=0.5)

plt.ylabel("$\mathcal{J}_{1/4}$")
plt.xlabel("$Q_{m,n}$")
plt.legend(loc='upper left')
plt.grid()
#plt.xscale('symlog',linthresh=1e-5)
plt.show()

print(step_sizes)
