# create Mollweide projections of the Halpha emission
# for the SILCC simulations
# Philipp Girichidis, 2025

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import yt

import argparse
from tqdm import tqdm
import pickle
#from unyt import cm

from astropy import units as u, constants  as cc

pc   = cc.pc.cgs.value
kB   = cc.k_B.cgs.value
Msun = cc.M_sun.cgs.value
G    = cc.G.cgs.value
Myr  = u.Myr.in_units("s")

parser = argparse.ArgumentParser(description='cmd line args')
parser.add_argument('files', nargs='+', help='files')

parser.add_argument('-radius', help='radius of the sphere in parsec', default=120.0, type=float)
parser.add_argument('-rad_res', help='minimal radius to include', default=4.0, type=float)

parser.add_argument('-nside', help='nside for healpix', default=64, type=int)

parser.add_argument('-odir', help='output directory', default=".", type=str)
parser.add_argument('-odir_data', help='output directory for data files', default="", type=str)

args = parser.parse_args()

if args.odir_data == "":
    args.odir_data = args.odir

radius_str = str(int(args.radius)).zfill(4)

file_suffix = f"r{radius_str}"
print(file_suffix)


# numerical parameters for healpix
nside = args.nside

cx = -80.
cy = -150.
cz = 0.0
c = ([cx, cy, cz], "pc")          # new yt version
c = [cx*pc, cy*pc, cz*pc] # old yt version
print("center of the bubble at (pc) :", cx, cy, cz)

# loop over all files
for files in args.files:
    print("processing file", files)
    ds = yt.load(files)

    # select the region around the bubble
    sp = ds.sphere(c, (args.radius, "pc"))

    print()
    print("min and max positions of the selected region")
    print("min/max x (pc): ", np.min(sp[("gas", "x")].in_units("pc").v), np.max(sp[("gas", "x")].in_units("pc").v))
    print("min/max y (pc): ", np.min(sp[("gas", "y")].in_units("pc").v), np.max(sp[("gas", "y")].in_units("pc").v))
    print("min/max z (pc): ", np.min(sp[("gas", "z")].in_units("pc").v), np.max(sp[("gas", "z")].in_units("pc").v))

    # box size in pc
    Lx = ds.domain_width[0].in_units("pc").v
    Ly = ds.domain_width[1].in_units("pc").v
    Lz = ds.domain_width[2].in_units("pc").v

    # find coordinates with respect to the center of the sphere taking into account periodic boundaries in x and y
    posx_ctr = (sp[("gas", "x")].in_units("pc").v - cx + Lx/2.) % Lx - Lx/2.
    posy_ctr = (sp[("gas", "y")].in_units("pc").v - cy + Ly/2.) % Ly - Ly/2.
    posz_ctr =  sp[("gas", "z")].in_units("pc").v - cz
    print()
    print("min and max positions of the selected region - shifted to center of the sphere")
    print("min/max x (pc): ", np.min(posx_ctr), np.max(posx_ctr))
    print("min/max y (pc): ", np.min(posy_ctr), np.max(posy_ctr))
    print("min/max z (pc): ", np.min(posz_ctr), np.max(posz_ctr))
    # radius based on centre of mass
    rad_ctr = np.sqrt(posx_ctr**2 + posy_ctr**2 + posz_ctr**2)
    # print min max of radius
    print()
    print("min/max r (pc): ", np.min(rad_ctr), np.max(rad_ctr))

    # cell volume
    volu = sp[("gas", "cell_volume")].in_units("pc**3").v

    # create normal vectors for the healpix map
    vec_norm_x = posx_ctr / rad_ctr
    vec_norm_y = posy_ctr / rad_ctr
    vec_norm_z = posz_ctr / rad_ctr

    # compute angular size of all cells at the respective distance to the observer
    R = (3 * volu / (4 * np.pi))**(1./3.)
    dx = np.power(volu, 1./3.)

    # compute the angle of the cell with respect to the observer    
    angle = np.arctan2(R, rad_ctr)

    # number of cells
    Ncell = rad_ctr.size


    print()
    # create three healpix maps
    NPIX = hp.nside2npix(nside)
    print("create healpix maps for various quantities with nside", nside, "and NPIX", NPIX)
    coldens_map     = np.zeros(NPIX) # column density
    coldens_ihp_map = np.zeros(NPIX) # column density (ionized gas)
    coldens_iha_map = np.zeros(NPIX) # column density (neutral gas)
    coldens_ih2_map = np.zeros(NPIX) # column density (molecular gas)
    Halpha_map      = np.zeros(NPIX) # Halpha emission

    total_mass = 0.0
    total_volu = 0.0
    total_halpha = 0.0

    print()
    for i in tqdm(range(Ncell)):

        # find the pixels in the healpix map
        pix = hp.query_disc(nside, [vec_norm_x[i], vec_norm_y[i], vec_norm_z[i]], angle[i], inclusive=True)

        # total column density
        coldens_map[pix]   += sp[("gas", "cell_mass")][i].v / (np.pi*(R[i]*pc)**2) / len(pix)
        # column density of ionized gas
        coldens_ihp_map[pix] += sp[("gas", "cell_mass")][i].v * sp[("flash", "ihp ")][i].v / (np.pi*(R[i]*pc)**2) / len(pix)


        total_mass += sp[("gas", "cell_mass")][i].v
        total_volu += volu[i]

    print()
    print("done with the loop over cells, make plots and write data to disk")


    # prepare the data field to write to disk
    data = {}

    data["simtime"] = ds.current_time.v
    data["simtime_Myr"] = ds.current_time.in_units("Myr").v

    
    for name, field, norm in zip(["coldens", "coldens_ihp", "coldens_iha", "coldens_ih2", "Halpha"], \
                            [coldens_map, coldens_ihp_map, coldens_iha_map, coldens_ih2_map, Halpha_map], \
                            [True, True, True, True, True]):

        # create a mollweide projection for all fields
        fig = plt.figure(figsize=(10, 5))
        if norm:
            plt_field = np.log10(field).copy()
        else:
            plt_field = field.copy()

        hp.mollview(plt_field, return_projected_map=True, title=name, cbar=False)
        hp.graticule()

        ax = plt.gca()
        image = ax.get_images()[0]
        cbar = fig.colorbar(image, ax=ax, orientation='vertical', shrink=0.8)
        cbar.set_label(name, rotation=270, labelpad=20)
        plt.savefig(f'{args.odir}/{files}-{file_suffix}-{name}.pdf', bbox_inches="tight")
        plt.close()

        # dave the data in global dictionary
        data[name] = field

    # add global quantities to the data dictionary
    data["total_mass"] = total_mass
    data["total_volu"] = total_volu

    # write data to disk using pickle
    with open(f"{args.odir_data}/{files}-{file_suffix}-data.pkl", "wb") as f:
        pickle.dump(data, f)
