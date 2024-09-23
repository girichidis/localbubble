import numpy as np
import matplotlib.pyplot as plt

sphere = ds.sphere(c, (100.0, "pc"))
x_lum = sphere[('gas', 'xray_luminosity_0.5_2_keV')]
dens = sphere[("gas", "density")]
idx = np.argsort(dens)
sorted_dens = dens[idx]
y = np.cumsum(x_lum[idx])

fig, ax = plt.subplots(figsize=(4, 3))  
ax.plot(np.log10(sorted_dens), y / np.sum(x_lum))
ax.set_xlabel('Logρ')
ax.set_ylabel('Cumulative of x_lum')
ax.set_xlim(-27, -23)

plt.tight_layout()  
plt.savefig('cum_xlum_dens.pdf', bbox_inches='tight')  