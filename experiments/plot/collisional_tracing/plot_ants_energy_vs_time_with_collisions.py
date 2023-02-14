#!/usr/bin/env python

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 18})

dirs = [
    #'20220106-01-121_3dlaunch_newprofiles_New_QA',
    #'20220106-01-122_3dlaunch_newprofiles_New_QA_magwell',
    #'20220106-01-123_3dlaunch_newprofiles_New_QA_coils_L24',
    #'20220106-01-124_3dlaunch_newprofiles_New_QA_magwell_coils_L24',
    #'20220106-01-125_3dlaunch_newprofiles_New_QH',
    #'20220106-01-126_3dlaunch_newprofiles_New_QH_magwell',
    '20220106-01-127_3dlaunch_newprofiles_Aten',
    '20220106-01-128_3dlaunch_newprofiles_LHD_R3.60',
    '20220106-01-129_3dlaunch_newprofiles_LHD_R3.75',
    '20220106-01-130_3dlaunch_newprofiles_li383',
    '20220106-01-131_3dlaunch_newprofiles_Garabedian',
    '20220106-01-132_3dlaunch_newprofiles_CFQS',
    '20220106-01-133_3dlaunch_newprofiles_IPP_QA',
    '20220106-01-134_3dlaunch_newprofiles_IPP_QH',
    '20220106-01-135_3dlaunch_newprofiles_HSX',
    #'20220106-01-136_3dlaunch_newprofiles_W7X_no_ripple',
    '20220106-01-137_3dlaunch_newprofiles_W7X_high_narrow_mirror',
    '20220106-01-138_3dlaunch_newprofiles_ARIES-CS',
    #'20220106-01-139_3dlaunch_newprofiles_Giuliani',
    #'20220106-01-140_3dlaunch_newprofiles_Wistell-B',
    #'20220106-01-141_3dlaunch_newprofiles_20220102-01-053_iteratingVmecAndSfincsForBestFrom048_003',
    #'20220106-01-142_3dlaunch_newprofiles_20220102-01-055_QH_A6.5_beta0_aScaling',
    #'20220106-01-143_3dlaunch_newprofiles_multiopt_scan_QH_nfp3_20210924-01-046_QH_nfp3_no_magwell_smaller_6551_aScaling',
    #'20220106-01-144_3dlaunch_newprofiles_multiopt_scan_QH_nfp4_20210924-01-031_target_grad_grad_B_5027_aScaling',
    #'20220106-01-146_3dlaunch_newprofiles_Spong_20160107_ITER_hybridAxisymmFixedBoundary_aScaling',
    #'20220106-01-147_3dlaunch_newprofiles_Spong_20160107_ITER_hybridAxisymmFixedBoundary_lasymF_aScaling',
    #'20220106-01-148_3dlaunch_newprofiles_QI_NFP1_r1_test_aScaling',
]
#        '20220106-01-145_3dlaunch_newprofiles_20220218-01-014-005_QA_nfp2_beta0p03_iotaTarget0p7',


# storage
ts = []
energies = []
filenames = []

# load the data
for j in range(len(dirs)):
    filename = dirs[j] + '/gone_nowhere.3d'
    data = np.loadtxt(filename, skiprows=2)
    ts.append(data[:, 4])
    energies.append(data[:, 5] / (1.602176634e-19))
    filenames.append(filename)



fig,ax = plt.subplots(figsize=(8,7))

# colorblind colors
from matplotlib.colors import ListedColormap
cmap = ListedColormap(sns.color_palette("colorblind",256))
colors = cmap(np.linspace(0,1,len(filenames)))
#colors = ['#377eb8', '#ff7f00', '#4daf4a',
#          '#f781bf', '#a65628', '#984ea3',
#          '#999999', '#e41a1c', '#dede00']

"""
Plot the tracing data
"""

# scatter plot the data
for j,_ in enumerate(ts):
    #plt.semilogy(ts[j], energies[j], ".", label=filenames[j], ms=1)
    #plt.semilogy(ts[j], energies[j], ".",ms=1)
    # rasterize to reduce file size
    plt.semilogy(ts[j], energies[j], ".",ms=1,rasterized=True)

# plot a regression curve
t_data = []
energy_data = []
for j, tj in enumerate(ts):
    t_data = np.append(t_data,tj)
    energy_data = np.append(energy_data,energies[j])
coeffs = np.polyfit(t_data,np.log(energy_data),deg=1)
model  = np.poly1d(coeffs)
times = np.linspace(0,0.1,50)
plt.plot(times,np.exp(model(times)),lw=3,color='k',label='mean')

"""
plot the analytic curves
"""
t_analytic = np.linspace(0, 0.1, 50)

## Slowing-down time based on the max T = 12 keV, max n_e = 4e20 / m^3
#t_s_analytic = 1 / 7.792787330525085
#plt.plot(t_analytic, 3.5e6 * np.exp(-2 * t_analytic / t_s_analytic), "k", lw=3,label="(3.5e6) exp(-2t/t_s), max T and n")

# Slowing-down time based on the avg T = 6 keV, max n_e = (5/6) * 4e20 / m^3
t_s_analytic = 1 / 17.286306947416985
plt.plot(t_analytic, 3.5e6 * np.exp(-2 * t_analytic / t_s_analytic), color="r", ls='--',lw=3,label="energy model")


# axis labels
plt.xlabel('Time [sec]')
plt.ylabel('Energy [eV]')
#plt.title('Alpha particle energy at time of loss')

# legend
plt.legend(loc='upper right', fontsize=14)

# axis limits
plt.xlim([-0.005, 0.1])
plt.ylim([1e4, 5e6])

#plt.figtext(0.5, 0.995, "Run in " + os.getcwd(), ha='center', va='top', fontsize=6)
#plt.figtext(0.5, 0.005, "Plotted by " + os.path.abspath(__file__), ha='center', va='bottom', fontsize=6)

# darken the border
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(2)  

plt.tight_layout()

# save
outfilename = "collisional_energy.pdf"
plt.savefig(outfilename)
#plt.show()
