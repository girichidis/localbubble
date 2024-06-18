import numpy as np
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import yt
from mpi4py import MPI
import argparse

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nside = 64

min = 18.
max = 32.

pc = 3.085678e+18
kpc = 1e3 * pc

cx = -80.
cy = -150.
cz = 0.0

parser = argparse.ArgumentParser(description='cmd line args')
parser.add_argument('files', nargs='+', help='files')
args = parser.parse_args()

Rmax_values = [40.0, 80.0, 100.0, 120.0]

lum_tot = np.zeros(len(Rmax_values))

for idx, Rmax_pc in enumerate(Rmax_values):
    c = ([cx, cy, cz], "pc")
    c = [cx * pc, cy * pc, cz * pc]

    for files in args.files:
        for f in args.files:
            plt2 = f[-4:]
            print(plt2)
            ds = yt.load(f)
            ad = ds.all_data()

            def _nuclei_density(field, data):
                return data[("gas", "number_density")]
            ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)

            yt.add_xray_emissivity_field(ds, 0.5, 2, metallicity=1.0, data_dir="/home/erea/data/xray-tables/")

            comm.Barrier()
            yt.load(files)

            sp = ds.sphere(c, (Rmax_pc, "pc"))

            posx_ctr = sp[("gas", "x")].in_units("pc").v - cx
            posy_ctr = sp[("gas", "y")].in_units("pc").v - cy
            posz_ctr = sp[("gas", "z")].in_units("pc").v - cz

            if rank == 0:
                print("min/max x: ", np.min(posx_ctr), np.max(posx_ctr))
                print("min/max y: ", np.min(posy_ctr), np.max(posy_ctr))
                print("min/max z: ", np.min(posz_ctr), np.max(posz_ctr))

            rad_ctr = np.sqrt(posx_ctr ** 2 + posy_ctr ** 2 + posz_ctr ** 2)

            if rank == 0:
                print("min/max r: ", np.min(rad_ctr), np.max(rad_ctr))

            vec_norm_x = posx_ctr / rad_ctr
            vec_norm_y = posy_ctr / rad_ctr
            vec_norm_z = posz_ctr / rad_ctr

            R = (3 * sp[("gas", "cell_volume")].v / (4 * np.pi)) ** (1. / 3.)
            angle = np.arctan2(R / pc, rad_ctr)

            if rank == 0:
                print("min/max R: ", np.min(R), np.max(R))
                print("min/max p: ", np.min(angle), np.max(angle))

            NPIX = hp.nside2npix(nside)
            im_loc = np.zeros(NPIX)
            im_glb = np.zeros(NPIX)

            Ltot_loc = 0.0
            Ltot_glb = 0.0

            N = rad_ctr.size

            N0 = np.zeros(size, dtype=np.int64)
            N1 = np.zeros(size, dtype=np.int64)

            for i in range(size):
                delta = (N // size)
                N0[i] = rank * delta
                if rank < size - 1:
                    N1[i] = (i + 1) * delta
                else:
                    N1[i] = N

            time0_loop = MPI.Wtime()

            print("loop over cells: ", N0[rank], "to", N1[rank])

            for i in range(N0[rank], N1[rank]):
                Ltot_loc = Ltot_loc + sp[('gas', 'xray_luminosity_0.5_2_keV')][i].v

                pixels = hp.query_disc(nside, [vec_norm_x[i], vec_norm_y[i], vec_norm_z[i]], angle[i], inclusive=True,)
                for pixel in pixels:
                    im_loc[pixels] += (sp[('gas', 'xray_luminosity_0.5_2_keV')][i].v / (rad_ctr[i] ** 2)) / len(pixels)

            comm.Barrier()
            comm.Allreduce(im_loc, im_glb, op=MPI.SUM)
            Ltot_glb = comm.allreduce(Ltot_loc, op=MPI.SUM)

            lum_tot[idx] = Ltot_glb

            if rank == 0:
                print(sp[("gas", "xray_luminosity_0.5_2_keV")].sum())
                lmin = np.log10(np.min(im_glb))
                lmax = np.log10(np.max(im_glb))

                # Calcola il tempo
                t = float(plt2.split("_")[-1]) / 100.0

                fig = plt.figure(figsize=(10, 5))
                hp.mollview(np.log10(im_glb), return_projected_map=True, min=min, max=max, title="X-ray Luminosity Map (0.5-2 keV) - Time: " + str(t) + " Myr - Radius: " + str(Rmax_pc) + " pc", cbar=False)
                hp.graticule()

                ax = plt.gca()
                image = ax.get_images()[0]
                cbar = fig.colorbar(image, ax=ax, orientation='vertical', shrink=0.8)
                cbar.set_label('log(X-ray Luminosity (erg/s/sr))', rotation=270, labelpad=20)

                plt.savefig(files + "-mollweide-x_lum_" + str(Rmax_pc) + "_weighted.pdf", bbox_inches="tight")

# Stampa il vettore delle luminosità totali
if rank == 0:
    print("lum_tot_Rmax_40, lum_tot_Rmax_80, lum_tot_Rmax_100, lum_tot_Rmax_120:")
    print(lum_tot)
