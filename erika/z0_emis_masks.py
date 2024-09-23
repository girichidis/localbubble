#slice emiss + supernova
import yt
import matplotlib.pyplot as plt
import numpy as np

c = ([-80, -150, 0], "pc")

def plot_slice_with_emissivity_masks(ds, center, radius, z_value, field, vmin, vmax, fname):
    c_z = list(center[0])
    c_z[2] = z_value
    c_new = (c_z, center[1])
    
    p = yt.SlicePlot(ds, "z", center=c_new, fields=field)
    p.set_zlim(field, vmin, vmax)
    
    # extract slice data
    frb = p.data_source.to_frb((2*radius, 'pc'), 800)
    emissivity_data = frb[field]
    
    # total emissivity for the slice
    total_emission = np.sum(emissivity_data)
    
    # emissivity values corresponding to 95%, 90%, and 80% of total emission in the slice
    sorted_emissivity_data = np.sort(emissivity_data.flatten())[::-1]
    cumsum_emissivity = np.cumsum(sorted_emissivity_data)
    threshold_95 = sorted_emissivity_data[np.argmax(cumsum_emissivity >= 0.95 * total_emission)]
    threshold_90 = sorted_emissivity_data[np.argmax(cumsum_emissivity >= 0.90 * total_emission)]
    threshold_80 = sorted_emissivity_data[np.argmax(cumsum_emissivity >= 0.80 * total_emission)]
    

    mask_95 = emissivity_data >= threshold_95
    mask_90 = emissivity_data >= threshold_90
    mask_80 = emissivity_data >= threshold_80

    vmin = -40
    vmax = -22
    
    plt.figure(figsize=(4, 4))
    
    img = plt.imshow(np.log10(emissivity_data), cmap='magma', origin='lower', extent=[-radius, radius, -radius, radius],vmin=vmin, vmax=vmax)
    
    contour_95 = plt.contour(mask_95, levels=[0.5], colors='red', linewidths=2, extent=[-radius, radius, -radius, radius], label='95% Emissivity')
    contour_90 = plt.contour(mask_90, levels=[0.5], colors='blue', linewidths=2, extent=[-radius, radius, -radius, radius], label='90% Emissivity')
    contour_80 = plt.contour(mask_80, levels=[0.5], colors='green', linewidths=2, extent=[-radius, radius, -radius, radius], label='80% Emissivity')

    plt.text(0.05, 0.85, "Regions with 95% of emissivity", color='red', transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(0.05, 0.80, "Regions with 90% of emissivity", color='blue', transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(0.05, 0.75, "Regions with 80% of emissivity", color='green', transform=plt.gca().transAxes, ha='left', va='top')
    plt.xlabel('x (pc)')
    plt.ylabel('y (pc)')
    point_x =(-1.03971680e+20/3.086e+18 +80)
    point_y =(-5.53008789e+20/3.086e+18 +150)

    
    plt.scatter(point_x, point_y, color='white', s=75, edgecolor='white', label='Specific Point')
    
    cbar = plt.colorbar(img, label='Log(X-Ray Emissivity (erg cm$^{-3}$ s$^{-1}$))')
    plt.title(f"Slice at z = {z_value} pc")
    
    plt.savefig(fname)
    plt.close()
    print(f"Plot saved as {fname}")

z_value = 0
radius = 120
fname = f"emissivity_slice_masks_z_{z_value}_sn.pdf"

plot_slice_with_emissivity_masks(ds, c, radius, z_value, ("gas", "xray_emissivity_0.5_2_keV"), vmin=1e-27, vmax=1e-22, fname=fname)