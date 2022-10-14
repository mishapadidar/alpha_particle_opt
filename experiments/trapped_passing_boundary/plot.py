import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use('TkAgg')

"""
pull data with
rsync -avz map454@g2-login.coecis.cornell.edu:/home/map454/alpha_particle_opt/experiments/trapped_passing_boundary/_batch*/*data* ./
"""


surf = 0.8
phi = 0.5
#infile = f"./_batch_surf_{surf}_{phi}/plot_data_s_{surf}_phi_{phi}.pickle"
infile = f"./plot_data_s_{surf}_phi_{phi}.pickle"
indata = pickle.load(open(infile,'rb'))
stp_inits = indata['stp_inits']
theta_inits  = stp_inits[:,1]
vpar_inits = indata['vpar_inits']
FX = indata['FX']
n_directions = len(FX)
T = indata['T']
tmax = indata['tmax']

# map -inf to inf
FX[~np.isfinite(FX)]= np.inf


# 1d plots
colors = cm.jet(np.linspace(0, 1, n_directions))
for ii in range(n_directions):
  #plt.plot(T,np.quantile(FX[ii],q=0.2,axis=1),'-o',markersize=5,linewidth=2,color=colors[ii],label=f'dir {ii}')
  plt.plot(T,np.mean(FX[ii][:,::],axis=1),'-o',linewidth=2,markersize=5,color=colors[ii],label=f'dir {ii}')
plt.ylabel('function value')
plt.xlabel('distance from x0')
#plt.xscale('symlog',linthresh=1e-9)
plt.yscale('log')
plt.legend()
plt.show()


# select the direction for plotting
direction_idx = 6

## plot the scatter plots along the slice
#for point_idx in range(len(FX[direction_idx])):
#  plt.figure()
#
#  fX = FX[direction_idx,point_idx,:]
#  plt.scatter(theta_inits,vpar_inits,c=fX)
#
#  #plt.clim(0, tmax) # colorbar range
#  plt.colorbar()
#  plt.ylabel('vpar')
#  plt.xlabel('theta')
#  #plt.xscale('symlog',linthresh=1e-9)
#  #plt.yscale('log')
#  plt.legend()
#  plt.show()

# plot the difference plots along the slice
n_points = len(FX[direction_idx])
for point_idx in range(n_points-1):
  plt.figure()
  diff = np.abs(FX[direction_idx,point_idx+1,:] - FX[direction_idx,point_idx,:])

  diff = diff>0.1*tmax # large deviations
  
  plt.scatter(theta_inits,vpar_inits,c=diff)
  #plt.scatter(theta_inits,vpar_inits,c=diff,norm=matplotlib.colors.LogNorm())
  #plt.clim(1e-16, 5*tmax) # colorbar range
  plt.colorbar()
  plt.ylabel('vpar')
  plt.xlabel('theta')
  plt.title("change in confinement time")
  #plt.xscale('symlog',linthresh=1e-9)
  #plt.yscale('log')
  plt.show()
