import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import seaborn as sns
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 18})


infile = "quasisymmetry_data_config_A_mmode_3_tmax_0.01.pickle"
#infile = "quasisymmetry_data_config_A_mmode_3_tmax_0.01_step_GD.pickle"
indata = pickle.load(open(infile,"rb"))
# load the keys quick and dirty
for key in list(indata.keys()):
    s  = f"{key} = indata['{key}']"
    print(key)
    exec(s)
n_directions = len(mn_list)



# insert x0 into the point list
step_sizes = np.insert(step_sizes,2,0.0)
idx_sort = np.argsort(step_sizes)
step_sizes = step_sizes[idx_sort]
idx_insert = np.where(step_sizes==0.0)[0] # find where we are putting zero
c_times_plus_all_d = np.insert(c_times_plus_all_d,idx_insert,c_times_x0,axis=1)
energy_plus_all_d = np.insert(energy_plus_all_d,idx_insert,energy_x0,axis=1)
qs_plus_all_d = np.insert(qs_plus_all_d,idx_insert.item(),qs0_all_d,axis=1)

# compute standard deviations of the energy
feat = 3.5*np.exp(-2*c_times_plus_all_d/tmax)
stds = np.std(feat,axis=-1)
std_errs = 1.96*stds/np.sqrt(n_particles)
fig = plt.figure(figsize=(8,6))
print(step_sizes)

# plot the energy 
for ii in range(n_directions):
    plt.plot(step_sizes,energy_plus_all_d[ii],label=f"QS{mn_list[ii]}",marker='s',markersize=10,linewidth=3)
    plt.fill_between(step_sizes,energy_plus_all_d[ii] + std_errs[ii], energy_plus_all_d[ii] - std_errs[ii],alpha=0.5)
plt.scatter([0.0],[energy_x0],color='k')
plt.ylabel("$\mathcal{J}$")
plt.xlabel("$\\alpha$")
plt.legend(loc='upper left')
plt.grid()
plt.xlim([-0.00075,0.0075])
# plt.xticks([-0.0075,0.0,0.0075])\n",
# plt.ylim([0.45,0.9])\n",
# plt.yticks(np.arange(0.45,0.9,0.1))\n",
plt.xscale('symlog',linthresh=1e-4)
plt.show()


percent_change = 100*(qs_plus_all_d.T-qs0_all_d)/qs0_all_d
percent_change = percent_change.T

# now plot the QS objectives
for ii in range(n_directions):
    #plt.plot(step_sizes,qs_plus_all_d[ii],label=f"QS{mn_list[ii]}",marker='s',markersize=10,linewidth=3)
    plt.plot(step_sizes,percent_change[ii],label=f"QS{mn_list[ii]}",marker='s',markersize=10,linewidth=3)
plt.xlabel("$\\alpha$")
plt.ylabel('percent change in QS objective')
plt.xlim([-0.00075,0.0075])
plt.grid()
plt.xscale('symlog',linthresh=1e-5)
plt.legend(loc = 'upper right')
plt.show()
