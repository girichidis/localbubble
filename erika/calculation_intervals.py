import yt
import numpy as np

def calculate_xray_emission(file_path, percentage):
    c = ([-80, -150, 0], "pc")
    r = (100.0, "pc")
    sphere = ds.sphere(c, r)

    # quantities of interest
    dens = sphere[("gas", "density")]
    temp = sphere[("gas", "temperature")]
    x_lum = sphere[("gas", "xray_luminosity_0.5_2_keV")]

    # total Luminosity
    total_x_lum = np.sum(x_lum)

    # cumulative fraction of X-ray luminosity
    sorted_indices = np.argsort(x_lum)[::-1]
    sorted_x_lum = x_lum[sorted_indices]
    cumulative_fraction = np.cumsum(sorted_x_lum) / total_x_lum

    # index where the cumulative fraction exceeds the specified percentage
    idx_threshold = np.argmax(cumulative_fraction >= percentage / 100.0)

    # cells contributing to the specified percentage of the emission
    selected_indices = sorted_indices[:idx_threshold]
    selected_densities = dens[selected_indices]
    selected_temperatures = temp[selected_indices]

    # density and temperature ranges
    dens_min, dens_max = selected_densities.min(), selected_densities.max()
    temp_min, temp_max = selected_temperatures.min(), selected_temperatures.max()

    print("Density range:", dens_min, "-", dens_max)
    print("Temperature range:", temp_min, "-", temp_max)

calculate_xray_emission("/home/erea/data/B6-1pc/SILCC_hdf5_plt_cnt_1080", 90)
