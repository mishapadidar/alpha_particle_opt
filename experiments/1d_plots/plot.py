import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use('TkAgg')


infile = "1dplot_data.pickle"
indata = pickle.load(open(infile,'rb'))
T = indata['T']
FX = indata['FX']
n_directions = indata['n_directions']

# which objective of confinement time to plot
def objective(c_times):
  #return np.std(c_times,axis=1)
  #return np.mean(c_times[0:10],axis=1)
  return np.mean(c_times,axis=1)

#colors = cm.rainbow(np.linspace(0, 1, n_directions))
colors = cm.jet(np.linspace(0, 1, n_directions))
for ii in range(n_directions):
  plt.plot(T,objective(FX[ii]),linewidth=2,color=colors[ii],label=f'dir {ii}')
plt.ylabel('function value')
plt.xlabel('distance from x0')
#plt.xscale('symlog',linthresh=1e-9)
plt.yscale('log')
plt.legend()
plt.show()
