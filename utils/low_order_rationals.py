import numpy as np
import matplotlib.pyplot as plt

"""
Compute and plot the low order rationals.
"""

num = np.arange(0,12,1)
den = np.arange(1,7,1)
print(num)
print(den)

rat = []
for n in num:
  for d in den:
    rat.append(n/d)

rat = np.array(rat)
plt.plot(rat,0.0*rat,marker='o',markersize=5)
plt.xlim(0,2)
plt.show()
