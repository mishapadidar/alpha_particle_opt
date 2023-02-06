
from radial_density import RadialDensity
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
from scipy.integrate import simpson
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 22})
#from matplotlib.colors import ListedColormap
#cmap = ListedColormap(sns.color_palette("colorblind",256))
#colors = cmap(np.linspace(0,1,n_configs))

# load the angle density data
indata = pickle.load(open("angle_density_data.pickle","rb"))
stz_grid = indata['stz_grid']
thetas = indata['thetas']
zetas = indata['zetas']
s_label = indata['s_label']
ntheta = indata['ntheta']
nzeta = indata['nzeta']
detjac = indata['detjac'].reshape((nzeta,ntheta))
# integrate the |det(jac)|
theta_lin = thetas[0]
zeta_lin = zetas[:,0]
total = simpson(detjac,x = theta_lin)
total = simpson(total,x = zeta_lin)
# normalize to get a density
detjac = detjac/total


fig,ax_both = plt.subplots(figsize=(10,6), ncols=2)
ax1,ax2 = ax_both

# plot the angle density
cont = ax2.contour(zetas,thetas,detjac,linewidths=2,levels=17)

# set the colobar
#plt.colorbar(cont)
from matplotlib import ticker
tick_font_size = 16
cbar = plt.colorbar(cont,ax=ax2)
cbar.locator = ticker.MaxNLocator(nbins=8)
cbar.ax.tick_params(labelsize=tick_font_size)
cbar.update_ticks()

# plot the radial density
n_points = 1000
sampler = RadialDensity(n_points)
x = np.linspace(0,1,1000)
y = sampler._pdf(x)
ax1.plot(x,y,linewidth=3)

# axis labels
ax1.set_xlabel("$s$")
ax2.set_xlabel('$\zeta$')
ax2.set_ylabel('$\\theta$')
ax1.xaxis.set_label_coords(.5,-0.025)
ax2.xaxis.set_label_coords(.5,-0.025)
ax2.yaxis.set_label_coords(-0.025,.5)

# ticks
ax1.set_xticks([0,1])
ax2.set_xticks([0,np.pi/2],[0,"$\pi/2$"])
ax2.set_yticks([0,2*np.pi],[0,"$2\pi$"])

# titles
ax1.set_title("$p(s)$")
ax2.set_title(f"$p(\\theta,\zeta \ |\  s={s_label})$")

# darken the border
ax1.patch.set_edgecolor('black')  
ax1.patch.set_linewidth('2')  
ax2.patch.set_edgecolor('black')  
ax2.patch.set_linewidth('2')  

plt.tight_layout()

filename = "density_plot.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
