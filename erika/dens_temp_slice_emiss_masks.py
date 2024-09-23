#density and temperature
import yt
import matplotlib.pyplot as plt
import numpy as np

c = ([-80, -150, 0], "pc")

def plot_slice_with_emissivity_masks(ds, center, radius, z_value, emissivity_field, density_field, vmin, vmax, fname):
    c_z = list(center[0])
    c_z[2] = z_value
    c_new = (c_z, center[1])
    
    p_emissivity = yt.SlicePlot(ds, "z", center=c_new, fields=emissivity_field)
    p_emissivity.set_zlim(emissivity_field, vmin, vmax)
    
    frb_emissivity = p_emissivity.data_source.to_frb((2*radius, 'pc'), 800)
    emissivity_data = frb_emissivity[emissivity_field]
    
    total_emission = np.sum(emissivity_data)
    
    sorted_emissivity_data = np.sort(emissivity_data.flatten())[::-1]
    cumsum_emissivity = np.cumsum(sorted_emissivity_data)
    threshold_95 = sorted_emissivity_data[np.argmax(cumsum_emissivity >= 0.95 * total_emission)]
    threshold_90 = sorted_emissivity_data[np.argmax(cumsum_emissivity >= 0.90 * total_emission)]
    threshold_80 = sorted_emissivity_data[np.argmax(cumsum_emissivity >= 0.80 * total_emission)]
    
    mask_95 = emissivity_data >= threshold_95
    mask_90 = emissivity_data >= threshold_90
    mask_80 = emissivity_data >= threshold_80

    vmin = 2  
    vmax = 7.5
    #vmin = -26  
    #vmax = -21
    p_density = yt.SlicePlot(ds, "z", center=c_new, fields=density_field)
    
    frb_density = p_density.data_source.to_frb((2*radius, 'pc'), 800)
    density_data = frb_density[density_field]
    
    plt.figure(figsize=(4, 4))
    
    img = plt.imshow(np.log10(density_data), cmap='inferno', origin='lower', extent=[-radius, radius, -radius, radius], vmin=vmin, vmax=vmax)
    
    contour_95 = plt.contour(mask_95, levels=[0.5], colors='red', linewidths=2, extent=[-radius, radius, -radius, radius], label='95% Emissivity')
    contour_90 = plt.contour(mask_90, levels=[0.5], colors='blue', linewidths=2, extent=[-radius, radius, -radius, radius], label='90% Emissivity')
    contour_80 = plt.contour(mask_80, levels=[0.5], colors='green', linewidths=2, extent=[-radius, radius, -radius, radius], label='80% Emissivity')
    
    plt.text(0.05, 0.85, "Regions with 95% of emissivity", color='red', transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(0.05, 0.80, "Regions with 90% of emissivity", color='blue', transform=plt.gca().transAxes, ha='left', va='top')
    plt.text(0.05, 0.75, "Regions with 80% of emissivity", color='green', transform=plt.gca().transAxes, ha='left', va='top')
    plt.xlabel('x (pc)')
    plt.ylabel('y (pc)')
    point_x = (-1.03971680e+20 / 3.086e+18 + 80)
    point_y = (-5.53008789e+20 / 3.086e+18 + 150)
    
    plt.scatter(point_x, point_y, color='white', s=75, edgecolor='white', label='Specific Point')
    
    cbar = plt.colorbar(img, label='Log(Density(g cm$^{-3}$))') 
    #cbar = plt.colorbar(img, label='Log(Temperature (K))')
    plt.title(f"Slice at z = {z_value} pc")
    
    plt.savefig(fname)
    plt.close()
    print(f"Plot saved as {fname}")

z_value = 0
radius = 120
emissivity_field = ("gas", "xray_emissivity_0.5_2_keV")
temperature_field= ("gas", "temperature")
density_field = ("gas", "density")
fname_dens = f"emissivity_slice_masks_z_{z_value}_density.pdf"
fname_temp = f"emissivity_slice_masks_z_{z_value}_temperature.pdf"
#plot_slice_with_emissivity_masks(ds, c, radius, z_value, emissivity_field, density_field, vmin=1e-27, vmax=1e-22, fname=fname_dens)
plot_slice_with_emissivity_masks(ds, c, radius, z_value, emissivity_field, temperature_field, vmin=1e-27, vmax=1e-22, fname=fname_temp)