import numpy as np
import matplotlib.pyplot as plt

temp = sphere[('gas', 'temperature')]
sorted_temp = np.sort(temp)
idx = np.argsort(temp)
y = np.cumsum(x_lum[idx])

fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(np.log10(sorted_temp), y / np.sum(x_lum))
ax.set_xlabel('LogT')
ax.set_ylabel('Cumulative of x_lum')
ax.set_xlim(5, 7.5)

plt.savefig('cum_xlum_temp.pdf', bbox_inches='tight')
