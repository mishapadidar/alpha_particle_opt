import sys
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
from scipy.io import netcdf_file

plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 18})


filelist = ["configs/wout_nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_0.89.nc","configs/wout_nfp4_QH_cold_high_res_phase_one_mirror_1.35_aspect_7.0_iota_1.043.nc"]

    
n_files = len(filelist)
ntheta = 200
nphi = 4
R = np.zeros((n_files,ntheta,nphi))
Z = np.zeros((n_files,ntheta,nphi))

for ifile,infile in enumerate(filelist):
    
    f = netcdf_file(infile,'r',mmap=False)
    ns = f.variables['ns'][()]
    nfp = f.variables['nfp'][()]
    xn = f.variables['xn'][()]
    xm = f.variables['xm'][()]
    rmnc = f.variables['rmnc'][()]
    zmns = f.variables['zmns'][()]
    bmnc = f.variables['bmnc'][()]
    lasym = f.variables['lasym__logical__'][()]
    if lasym==1:
        rmns = f.variables['rmns'][()]
        zmnc = f.variables['zmnc'][()]
    else:
        rmns = 0*rmnc
        zmnc = 0*rmnc
    f.close()
    nmodes = len(xn)
    
    theta = np.linspace(0,2*np.pi,num=ntheta)
    phi = np.linspace(0,2*np.pi/nfp,num=nphi,endpoint=False)
    iradius = ns-1
    for itheta in range(ntheta):
        for iphi in range(nphi):
            for imode in range(nmodes):
                angle = xm[imode]*theta[itheta] - xn[imode]*phi[iphi]
                R[ifile,itheta,iphi] = R[ifile,itheta,iphi] + rmnc[iradius,imode]*np.cos(angle) + rmns[iradius,imode]*np.sin(angle)
                Z[ifile,itheta,iphi] = Z[ifile,itheta,iphi] + zmns[iradius,imode]*np.sin(angle) + zmnc[iradius,imode]*np.cos(angle)
    
    
labels=[r'$\phi=0$',r'1/4 period: $\phi=\pi/8$',r'1/2 period: $\phi=\pi/4$',r'3/4 period: $\phi=3\pi/8$']

# colorblind colors
colors = ['#377eb8', '#ff7f00', '#4daf4a',
          '#f781bf', '#a65628', '#984ea3',
          '#999999', '#e41a1c', '#dede00']
#from matplotlib.pyplot import cm
#from matplotlib.colors import ListedColormap
#cmap = ListedColormap(sns.color_palette("colorblind",256))
#colors = cmap(np.linspace(0,1,nphi))

fig,ax = plt.subplots(figsize=(8,8))

for ii in range(nphi):
    # plot config 0
    ax.plot(R[0,:,ii], Z[0,:,ii], linewidth=2,linestyle='-',color=colors[ii],label=labels[ii])
    # plot config 1
    ax.plot(R[1,:,ii], Z[1,:,ii], linewidth=2,linestyle='--',color=colors[ii])

# legend
plt.legend(loc='upper left',fontsize=14)

# axis labels
plt.xlabel('R [meters]')
plt.ylabel('Z [meters]')

# darken the border
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(2)  
    
plt.tight_layout()
filename = "cross_sections.pdf"
plt.savefig(filename, format="pdf", bbox_inches="tight")
#plt.show()
    
