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

#infile ="./nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_-1.043_field_strength.pickle"
infile ="./nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_1.043_field_strength.pickle"
#infile ="./nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_0.89_field_strength.pickle"

# load the data
field_line_data = pickle.load(open(infile,"rb"))
# load the relevant keys
for key in list(field_line_data.keys()):
    s  = f"{key} = field_line_data['{key}']"
    exec(s)

fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(figsize=(8,8),nrows=2,ncols=2)

# plot the data
p1 = ax1.contour(zeta_mesh,theta_mesh,modB_list[0].reshape((nzeta,ntheta)),levels=13,linewidths=2)
p2 = ax2.contour(zeta_mesh,theta_mesh,modB_list[1].reshape((nzeta,ntheta)),levels=13,linewidths=2)
p3 = ax3.contour(zeta_mesh,theta_mesh,modB_list[2].reshape((nzeta,ntheta)),levels=13,linewidths=2)
p4 = ax4.contour(zeta_mesh,theta_mesh,modB_list[3].reshape((nzeta,ntheta)),levels=13,linewidths=2)

# make the grey box
fontsize=18
t = ax1.text(0.030, 0.915, '$s=$%s'%s_list[0], transform=ax1.transAxes, fontsize=fontsize)
t.set_bbox(dict(facecolor='grey', alpha=0.6, edgecolor='k'))
t = ax2.text(1.81, 0.915, '$s=$%s'%s_list[1], transform=ax1.transAxes, fontsize=fontsize)
t.set_bbox(dict(facecolor='grey', alpha=0.6, edgecolor='k'))
t = ax3.text(0.030, -0.38, '$s=$%s'%s_list[2], transform=ax1.transAxes, fontsize=fontsize)
t.set_bbox(dict(facecolor='grey', alpha=0.6, edgecolor='k'))
t = ax4.text(1.81, -0.38, '$s=$%s'%s_list[3], transform=ax1.transAxes, fontsize=fontsize)
t.set_bbox(dict(facecolor='grey', alpha=0.6, edgecolor='k'))

# Set the ticks and ticklabels for all axes
plt.setp([ax1,ax2,ax3,ax4], xticks=[0,np.pi/2], xticklabels=['0','$\pi$/2'],
        yticks=[0,2*np.pi], yticklabels=['0','2$\pi$'])

# set the colobar
from matplotlib import ticker
tick_font_size = 16
for ax in (ax1,ax2,ax3,ax4):
  cbar = plt.colorbar(p1,ax=ax)
  cbar.locator = ticker.MaxNLocator(nbins=6)
  cbar.ax.tick_params(labelsize=tick_font_size)
  cbar.update_ticks()

# make the axis labels
for ax in (ax1,ax2,ax3,ax4):
  ax.set_xlabel('$\zeta$')
  ax.set_ylabel('$\\theta$')
  ax.xaxis.set_label_coords(.5,-0.025)
  ax.yaxis.set_label_coords(-0.025,.5)

# darken the border
for ax in (ax1,ax2,ax3,ax4):
  ax.patch.set_edgecolor('black')  
  ax.patch.set_linewidth(2)  

plt.tight_layout()
filename = infile[:-7] # remove the .pickle
filename = filename + ".pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
#plt.show()
