import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use('TkAgg')


infile = "plot_data_s_0.6_phi_0.0.pickle"
indata = pickle.load(open(infile,'rb'))
stp_inits = indata['stp_inits']
theta_inits  = stp_inits[:,1]
vpar_inits = indata['vpar_inits']
FX = indata['FX']

colors = cm.jet(np.linspace(0, 1, len(FX)))
plt.scatter(theta_inits,vpar_inits,c=FX)
plt.colorbar()
plt.ylabel('vpar')
plt.xlabel('theta')
#plt.xscale('symlog',linthresh=1e-9)
#plt.yscale('log')
plt.legend()
plt.show()
