import yt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm

def calculate_xray_emission(file_path):  
    c = ([-80, -150, 0], "pc")
    r = (100.0, "pc")

    sphere = ds.sphere(c, r)

    emitting_x = sphere['gas', 'x'].to('pc').value
    emitting_y = sphere['gas', 'y'].to('pc').value
    emitting_z = sphere['gas', 'z'].to('pc').value

    dens = sphere[("gas", "density")].to('g/cm**3').value
    temp = sphere[("gas", "temperature")].value

    mask = np.logical_and.reduce([
        dens >= 4.231630470797093e-26,
        dens <= 1.6352602401850995e-24,
        temp >= 712480.25,
        temp <= 10061067.0
    ])

    emitting_x = emitting_x[mask]
    emitting_y = emitting_y[mask]
    emitting_z = emitting_z[mask]
    dens = dens[mask]

    dot_x = 46.30859364873623 - 80
    dot_y = -29.19921872974723 - 150
    dot_z = 0.4882812508101102

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')

    norm = LogNorm(vmin=dens.min(), vmax=dens.max())
    cmap = plt.get_cmap('viridis')

    marker_size = 100

    sc = ax.scatter(emitting_x, emitting_y, emitting_z, s=1e-1, c=dens, cmap=cmap, norm=norm, alpha=0.4)

    ax.scatter(dot_x, dot_y, dot_z, color='red', s=marker_size, edgecolor='black', zorder=5,alpha=1)  

    ax.set_xlabel('X [pc]')
    ax.set_ylabel('Y [pc]')
    ax.set_zlabel('Z [pc]')

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = c[0][0] + 100 * np.outer(np.cos(u), np.sin(v))
    y = c[0][1] + 100 * np.outer(np.sin(u), np.sin(v))
    z = c[0][2] + 100 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='b', alpha=0.2)

    cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.03, pad=0.1)
    cbar.set_label('Density [g/cm**3]')

    plt.savefig("/home/erea/localbubble/scripts/3D_cells_density_sn.pdf")

calculate_xray_emission("/home/erea/data/B6-1pc/SILCC_hdf5_plt_cnt_1080")
