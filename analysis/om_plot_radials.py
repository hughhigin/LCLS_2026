import sys, os
import numpy as np
import h5py
import matplotlib.pyplot as plt

file = sys.argv[1]
fname_nopath = os.path.basename(file)
basename, ext = os.path.splitext(fname_nopath)
do_norm = 0 # 1 for true, normalize traces

if do_norm:
    basename += "_normalized"

h5 = h5py.File(file)
q = h5['data/q'][:]
q = q.mean(0)

radials = h5['data/radial'][:]
for i in range(0,radials.shape[0],1000):
    # Bonus: normalize radials
    if do_norm:
        norm_val = np.max(radials[i])
    else:
        norm_val = 1
    norm_radial = radials[i] / norm_val
    plt.plot(q,norm_radial)

mean_rad = radials.mean(0)
if do_norm:
    norm_val = np.max(mean_rad)
else:
    norm_val = 1

plt.plot(q,mean_rad / norm_val,'k-',lw=3,label='mean')
plt.legend()
plt.xlabel('q')
plt.ylabel('I(q)')
plt.title(file)
plt.savefig(basename+'_radials.png',dpi=150)
plt.show()
