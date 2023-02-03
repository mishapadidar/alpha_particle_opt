import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
import pandas as pd
import seaborn as sns
import glob

plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 22})

# colorblind colors
colors = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
markers = ['s','o','^','x']
linestyles=['-','--','-.']

# matplotlib.use('TkAgg')
# infile ="./configs/data_opt_nfp4_phase_one_aspect_7.0_iota_-1.043_mean_energy_SAA_surface_0.25_tmax_0.01_bobyqa_mmode_3_iota_None.pickle"
infile ="./configs/data_opt_nfp4_phase_one_aspect_7.0_iota_0.89_mean_energy_SAA_surface_full_tmax_0.01_bobyqa_mmode_3_iota_None.pickle"

# load the data
indata = pickle.load(open(infile,"rb"))
# load the relevant keys
field_line_data = indata['field_line_data']
for key in list(field_line_data.keys()):
    s  = f"{key} = field_line_data['{key}']"
    exec(s)

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(10,10),nrows=2,ncols=2)

p1 = ax1.contour(zeta_mesh,theta_mesh,modB_list[0].reshape((nzeta,ntheta)),linewidths=2)
p2 = ax2.contour(zeta_mesh,theta_mesh,modB_list[1].reshape((nzeta,ntheta)),linewidths=2)
p3 = ax3.contour(zeta_mesh,theta_mesh,modB_list[2].reshape((nzeta,ntheta)),linewidths=2)
p4 = ax4.contour(zeta_mesh,theta_mesh,modB_list[3].reshape((nzeta,ntheta)),linewidths=2)

fontsize=20
t = ax1.text(0.017, 0.93, '$s=$%s'%s_list[0], transform=ax1.transAxes, fontsize=fontsize)
t.set_bbox(dict(facecolor='grey', alpha=0.6, edgecolor='k'))
t = ax2.text(1.89, 0.93, '$s=$%s'%s_list[1], transform=ax1.transAxes, fontsize=fontsize)
t.set_bbox(dict(facecolor='grey', alpha=0.6, edgecolor='k'))
t = ax3.text(0.017, -0.42, '$s=$%s'%s_list[2], transform=ax1.transAxes, fontsize=fontsize)
t.set_bbox(dict(facecolor='grey', alpha=0.6, edgecolor='k'))
t = ax4.text(1.89, -0.42, '$s=$%s'%s_list[3], transform=ax1.transAxes, fontsize=fontsize)
t.set_bbox(dict(facecolor='grey', alpha=0.6, edgecolor='k'))

# Set the ticks and ticklabels for all axes
plt.setp([ax1,ax2,ax3,ax4], xticks=[0,np.pi/4,np.pi/2], xticklabels=['0','$\pi$/4','$\pi$/2'],
        yticks=[0,np.pi,2*np.pi], yticklabels=['0','$\pi$','2$\pi$'])

plt.colorbar(p1,ax=ax1)
plt.colorbar(p2,ax=ax2)
plt.colorbar(p3,ax=ax3)
plt.colorbar(p4,ax=ax4)

ax1.set_xlabel('$\zeta$')
ax2.set_xlabel('$\zeta$')
ax3.set_xlabel('$\zeta$')
ax4.set_xlabel('$\zeta$')
ax1.set_ylabel('$\\theta$')
ax2.set_ylabel('$\\theta$')
ax3.set_ylabel('$\\theta$')
ax4.set_ylabel('$\\theta$')

ax1.patch.set_edgecolor('black')  
ax1.patch.set_linewidth('2')  
ax2.patch.set_edgecolor('black')  
ax2.patch.set_linewidth('2')  
ax3.patch.set_edgecolor('black')  
ax3.patch.set_linewidth('2')  
ax4.patch.set_edgecolor('black')  
ax4.patch.set_linewidth('2')  

plt.tight_layout()
