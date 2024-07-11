import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import yt
from mpi4py import MPI
import argparse
from matplotlib.colors import LogNorm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nside = 64
pc = 3.085678e+18
cx = -80.
cy = -150.
cz = 0.0
min = 18.
max = 32.
rmin_pc=5.

# command line arguments
parser = argparse.ArgumentParser(description='Command line arguments')
parser.add_argument('files', nargs='+', help='List of files')
args = parser.parse_args()


Rmax_values = [100.0]

lum_tot = np.zeros(len(Rmax_values))


for idx, Rmax_pc in enumerate(Rmax_values):
    c = ([cx, cy, cz], "pc")

    for files in args.files:
        plt2 = files[-4:]  # Extract last 4 characters of filename
        print(plt2)
        
        ds = yt.load(files)
        ad = ds.all_data()

        def _nuclei_density(field, data):
            return data[("gas", "number_density")] * data[("flash","ihp ")]
        
        ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
        yt.add_xray_emissivity_field(ds, 0.5, 2, metallicity=1.0, data_dir="/home/erea/data/xray-tables/")

        comm.Barrier()
        yt.load(files)

        sp = ds.sphere(c, (Rmax_pc, "pc"))

        # positions relative to center
        posx_ctr = sp[("gas", "x")].in_units("pc").v - cx
        posy_ctr = sp[("gas", "y")].in_units("pc").v - cy
        posz_ctr = sp[("gas", "z")].in_units("pc").v - cz

        if rank == 0:
            print("min/max x: ", np.min(posx_ctr), np.max(posx_ctr))
            print("min/max y: ", np.min(posy_ctr), np.max(posy_ctr))
            print("min/max z: ", np.min(posz_ctr), np.max(posz_ctr))

        # radial distance
        rad_ctr = np.sqrt(posx_ctr ** 2 + posy_ctr ** 2 + posz_ctr ** 2)

        if rank == 0:
            print("min/max r: ", np.min(rad_ctr), np.max(rad_ctr))

        # normalized direction vectors
        vec_norm_x = posx_ctr / rad_ctr
        vec_norm_y = posy_ctr / rad_ctr
        vec_norm_z = posz_ctr / rad_ctr

        # angular size of cells
        R = (3 * sp[("gas", "cell_volume")].v / (4 * np.pi)) ** (1. / 3.)
        angle = np.arctan2(R / pc, rad_ctr)

        if rank == 0:
            print("min/max R: ", np.min(R), np.max(R))
            print("min/max p: ", np.min(angle), np.max(angle))

        # HEALPix map arrays initialization
        NPIX = hp.nside2npix(nside)
        im_loc = np.zeros(NPIX)
        im_glb = np.zeros(NPIX)

        Ltot_loc = 0.0
        Ltot_glb = 0.0

        N = rad_ctr.size

        N0 = np.zeros(size, dtype=np.int64)
        N1 = np.zeros(size, dtype=np.int64)

        # split indices for parallelization
        for i in range(size):
            delta = (N // size)
            N0[i] = rank * delta
            if rank < size - 1:
                N1[i] = (i + 1) * delta
            else:
                N1[i] = N

        time0_loop = MPI.Wtime()

        print("loop over cells: ", N0[rank], "to", N1[rank])

        # Loop over cells and fill HEALPix map
        for i in range(N0[rank], N1[rank]):
            Ltot_loc = Ltot_loc + sp[('gas', 'xray_luminosity_0.5_2_keV')][i].v

            d=np.maximum(rad_ctr[i],rmin_pc)
            if rad_ctr[i]<rmin_pc:
                im_loc += (sp[('gas', 'xray_luminosity_0.5_2_keV')][i].v / (d ** 2)) / len(im_loc)
            else:    
            # Query HEALPix pixels covered by the cell
                pixels = hp.query_disc(nside, [vec_norm_x[i], vec_norm_y[i], vec_norm_z[i]], angle[i], inclusive=True)
                for pixel in pixels:
                    im_loc[pixel] += (sp[('gas', 'xray_luminosity_0.5_2_keV')][i].v / (d ** 2)) / len(pixels)
                    

        comm.Barrier()
        comm.Allreduce(im_loc, im_glb, op=MPI.SUM)
        Ltot_glb = comm.allreduce(Ltot_loc, op=MPI.SUM)

        lum_tot[idx] = Ltot_glb

        if rank == 0:
            print(sp[("gas", "xray_luminosity_0.5_2_keV")].sum())
            lmin = np.log10(np.min(im_glb))
            lmax = np.log10(np.max(im_glb))

            # time from filename
            t = float(plt2.split("_")[-1]) / 100.0

            # plot HEALPix map of X-ray luminosity
            fig = plt.figure(figsize=(10, 5))
            hp.mollview(np.log10(im_glb), return_projected_map=True, min=min, max=max,
                        title=f'X-ray Luminosity Map (0.5-2 keV) - Time: {t} Myr - Radius: {Rmax_pc} pc', cbar=False)
            hp.graticule()

            ax = plt.gca()
            image = ax.get_images()[0]
            cbar = fig.colorbar(image, ax=ax, orientation='vertical', shrink=0.8)
            cbar.set_label('log(X-ray Luminosity (erg/s/sr))', rotation=270, labelpad=20)

            plt.savefig(files + f"-mollweide-x_lum_{Rmax_pc}__.pdf",  bbox_inches="tight")
            plt.close()

            # X-ray emissivity vs density phase plot
            phase_density = yt.PhasePlot(sp, ("gas", "density"), ("gas", "xray_emissivity_0.5_2_keV"), ("gas", "cell_volume"))
            phase_density.set_xlim(1e-28,1e-21)
            phase_density.set_ylim(1e-33,1e-20)
            phase_density.annotate_title(f'Phase Plot 1: X-ray Emissivity vs. Density\nRmax: {Rmax_pc} pc')
            phase_density.save(files + f"-phase_plot_xray_emissivity_density_{Rmax_pc}.pdf")

            # X-ray emissivity vs temperature phase plot
            phase_temperature = yt.PhasePlot(sp, ("gas", "temperature"), ("gas", "xray_emissivity_0.5_2_keV"), ("gas", "cell_volume"))
            phase_temperature.set_xlim(1e5,1e8)
            phase_temperature.set_ylim(1e-33,1e-20)
            phase_temperature.annotate_title(f'Phase Plot 2: X-ray Emissivity vs. Temperature\nRmax: {Rmax_pc} pc')
            phase_temperature.save(files + f"-phase_plot_xray_emissivity_temperature_{Rmax_pc}.pdf")

            plt.close('all')


            # 2D Histogram of Density vs Temperature
            tot_lum = sp[("gas", "xray_luminosity_0.5_2_keV")].v.sum()

            density = sp[("gas", "density")].to("g/cm**3").v
            temperature = sp[("gas", "temperature")].to("K").v
            xray_luminosity = sp[("gas", "xray_luminosity_0.5_2_keV")].v / tot_lum

            # bins
            density_bins = np.logspace(np.log10(density.min()), np.log10(density.max()), 64)
            temperature_bins = np.logspace(np.log10(temperature.min()), np.log10(temperature.max()), 64)

            hist, xedges, yedges = np.histogram2d(density, temperature, bins=[density_bins, temperature_bins], weights=xray_luminosity)

            X, Y = np.meshgrid(xedges, yedges)

            plt.figure(figsize=(8, 6))
            plt.pcolormesh(X, Y, hist.T, shading='auto', cmap='viridis', norm=LogNorm(vmin=1e-6, vmax=1e-1))
            plt.colorbar(label='Normalized X-ray Luminosity')

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Density (g/cm³)')
            plt.ylabel('Temperature (K)')
            plt.title(f'2D Histogram of Density vs. Temperature - Rmax: {Rmax_pc} pc')

            plt.savefig(files + f"-histogram_density_temperature_{Rmax_pc}.pdf", bbox_inches="tight")
            plt.close('all')






            # Cumulative plot of X-ray luminosity - density
            x_lum = sp[('gas', 'xray_luminosity_0.5_2_keV')].v
            dens = sp[("gas", "density")].v
            sorted_dens=np.sort(dens)
            idx_dens = np.argsort(dens)

            cum_x_lum = np.cumsum(x_lum[idx_dens]) / np.sum(x_lum)

            plt.figure(figsize=(8, 6))
            plt.plot(np.log10(sorted_dens), cum_x_lum)
            plt.xlabel('Log Density (g/cm³)')
            plt.ylabel('Cumulative X-ray Luminosity')
            plt.title(f'Cumulative X-ray Luminosity vs. Density - Rmax: {Rmax_pc} pc')
            #plt.xlim(-27, -23)
            plt.savefig(files + f"-cumulative_x_lum_density_{Rmax_pc}.pdf", bbox_inches="tight")
            plt.close()

            
            # Cumulative plot of X-ray luminosity - temperature
            temp = sp[('gas', 'temperature')].v
            sorted_temp=np.sort(temp)
            idx_temp = np.argsort(temp)

            cum_x_lum_temp = np.cumsum(x_lum[idx_temp]) / np.sum(x_lum)

            plt.figure(figsize=(8, 6))
            plt.plot(np.log10(sorted_temp), cum_x_lum_temp)
            plt.xlabel('Log Temperature (K)')
            plt.ylabel('Cumulative X-ray Luminosity')
            plt.title(f'Cumulative X-ray Luminosity vs. Temperature - Rmax: {Rmax_pc} pc')
            #plt.xlim(5, 7.5)
            plt.savefig(files + f"-cumulative_x_lum_temperature_{Rmax_pc}.pdf", bbox_inches="tight")
            plt.close()

# Print total X-ray luminosities for each Rmax
if rank == 0:
    print("Total X-ray luminosities for Rmax values:")
    for idx, Rmax_pc in enumerate(Rmax_values):
        print(f"Rmax = {Rmax_pc} pc, lum_tot = {lum_tot[idx]} erg/s")
