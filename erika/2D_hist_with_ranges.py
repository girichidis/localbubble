import yt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

c = ([-80, -150, 0], "pc")
sphere = ds.sphere(c, (100.0, "pc"))

tot_lum = sphere[("gas", "xray_luminosity_0.5_2_keV")].v.sum()

density = sphere[("gas", "density")].to("g/cm**3").v
temperature = sphere[("gas", "temperature")].to("K").v
xray_luminosity = sphere[("gas", "xray_luminosity_0.5_2_keV")].v / tot_lum

density_bins = np.logspace(np.log10(density.min()), np.log10(density.max()), 64)
temperature_bins = np.logspace(np.log10(temperature.min()), np.log10(temperature.max()), 64)

hist, xedges, yedges = np.histogram2d(density, temperature, bins=[density_bins, temperature_bins], weights=xray_luminosity)

X, Y = np.meshgrid(xedges, yedges)

plt.figure(figsize=(6, 4))
plt.pcolormesh(X, Y, hist.T, shading='auto', cmap='viridis', norm=LogNorm(vmin=1e-6, vmax=1e-1))
plt.colorbar(label='Normalized X-ray Luminosity')

# add vertical and horizontal lines to highlight density and temperature ranges
plt.axvline(x=4.231630470797093e-26, color='blue', linestyle='--', linewidth=1)
plt.axvline(x=1.6352602401850995e-24, color='blue', linestyle='--', linewidth=1)
plt.axhline(y=712480.25, color='green', linestyle='--', linewidth=1)
plt.axhline(y=10061067.0, color='green', linestyle='--', linewidth=1)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Density (g/cm³)')
plt.ylabel('Temperature (K)')
plt.title('2D Histogram of Density vs. Temperature\nwith overlaid ranges contributing to 90% of emission', fontsize=10)

plt.savefig("density_temperature_histogram.pdf")
