import matplotlib.pyplot as plt
import numpy as np

sphere = ds.sphere(c, (100.0, "pc"))
x_lum = sphere[('gas', 'xray_luminosity_0.5_2_keV')]
vol = sphere[('gas', 'cell_volume')]

idx = np.argsort(x_lum)
cum_vol = np.cumsum(vol[idx[::-1]]) / np.sum(vol)
cum_lum = np.cumsum(x_lum[idx[::-1]]) / np.sum(x_lum)

fig, ax = plt.subplots(figsize=(4, 3))  
ax.semilogy(cum_lum, cum_vol)
ax.set_xlim(0.6, 0.95)
ax.set_ylim(1e-3, 2e-2)
ax.set_xlabel('X-ray Luminosity fraction')
ax.set_ylabel('Cell Volume fraction')
ax.set_title('Sorted X-ray Luminosity vs Sorted Cell Volume')

plt.savefig('vol_frac.pdf', format='pdf', bbox_inches='tight')  