#!/usr/bin/env python

import numpy as np
import sys, os
from scipy.io import netcdf
import glob

makePDF = len(sys.argv) > 1

filenames = [
   "20230415-01-001_configuration_A/neo_out.neo.00000",
   "20230415-01-002_configuration_B/neo_out.neo.00000",
]

labels = ["Config. A", "Config. B"]

radii = []
eps_eff32s = []

no32 = True
for j in range(2):
   filename = filenames[j]
   # Note: we can't use np.loadtxt since there are fortran-style "d+" exponents, which python doesn't recognize.
   f = open(filename,'r')
   lines = f.readlines()
   f.close()
   jradius = []
   eps_eff32 = []
   for line in lines:
      splitline = line.split()
      this_eps_eff32 = float(splitline[1].replace('D','E'))
      if this_eps_eff32 != 0.0:
         jradius.append(int(splitline[0]))
         eps_eff32.append(this_eps_eff32)

   (head,tail) = os.path.split(filename)
   booz_files = glob.glob(os.path.join(head,'boozmn_*.nc'))
   print("booz_files:",booz_files)
   if len(booz_files) > 1:
      # There are multiple boozmn files, so see if there is one with the same extension as the neo_out file:
      booz_filename = os.path.join(head,'boozmn_'+tail[8:]+'.nc')
   else:
      booz_filename = booz_files[0]
   print("About to read booz_filename=",booz_filename)

   f = netcdf.netcdf_file(booz_filename,'r',mmap=False)
   ns = f.variables['ns_b'][()]
   f.close()

   print("Read file "+filename)
   print(" jradius=",jradius)
   print(" eps_eff32=",eps_eff32)

   # I assume the booz_xform / neo surfaces are the vmec half mesh?
   jradius = np.array(jradius)
   radius = (jradius-0.5)/(ns-1)

   radii.append(radius)
   eps_eff32s.append(np.array(eps_eff32))

import matplotlib.pyplot as plt
import matplotlib
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 18})

#plt.figure(figsize=(4, 3))
fig,ax = plt.subplots(figsize=(7,5))
for j in range(len(radii)):
   if no32:
      plt.semilogy(radii[j],eps_eff32s[j] ** (2.0/3),label=labels[j],linewidth=3)
   else:
      #plt.semilogy(radii[j],eps_eff32s[j],label=sys.argv[j+1])
      plt.loglog(radii[j],eps_eff32s[j],label=sys.argv[j+1])

plt.xlabel("Normalized toroidal flux $s$")
if no32:
   #plt.ylabel('eps_eff (NOT raised to the 3/2 power)')
   plt.ylabel(r"$\epsilon_{eff}$", fontsize=18, labelpad=-3, rotation=0)
else:
   plt.ylabel('eps_eff ^ (3/2)', fontsize=19, labelpad=-3)
plt.xlim([0, 1])
plt.ylim([1e-4, 2e-2])
plt.legend(loc="lower right", fontsize=13)

# darken the border
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(2)  

plt.tight_layout()
#plt.subplots_adjust(left=0.15, bottom=0.15, right=0.97, top=0.99)
plt.savefig("epsilon_effective.pdf")
