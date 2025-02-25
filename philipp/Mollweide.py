import numpy as np
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import yt
#from mpi4py import MPI
import argparse
from tqdm import tqdm
import pickle
from unyt import cm

from astropy import units as u, constants  as c

pc = c.pc.cgs.value
kB  = c.k_B.cgs.value
Msun = c.M_sun.cgs.value
G = c.G.cgs.value
Myr = u.Myr.in_units("s")

# Macbook
dat_path = "/Users/girichidis/Physics/Tables/yt-python/"
# helix
dat_path = "/home/hd/hd_hd/hd_ud081/work/SILCC/2018-SILCC-Clouds/localbubble/tables/"

parser = argparse.ArgumentParser(description='cmd line args')
parser.add_argument('files', nargs='+', help='files')

#parser.add_argument('-rad_max_pc', help='max radius around the local bubble in parsec', default=120.0, type=float)
#parser.add_argument('-rad_max_pc', help='radius around the local bubble in parsec', default=120.0, type=float)
#parser.add_argument('-Nrad_pc', help='number of radii to investigate', default=10, type=int)
parser.add_argument('-radius', help='radius of the sphere in parsec', default=120.0, type=float)
parser.add_argument('-rad_res', help='minimal radius to include', default=4.0, type=float)

parser.add_argument('-LB', help='chose local bubble', action='store_true')

parser.add_argument('-nside', help='nside for healpix', default=64, type=int)

parser.add_argument('-odir', help='output directory', default=".", type=str)
parser.add_argument('-odir_data', help='output directory for data files', default="", type=str)

parser.add_argument('-Sigma_crit', help='max critical coldens', default=1e20, type=float)

args = parser.parse_args()

if args.odir_data == "":
    args.odir_data = args.odir

radius_str = str(int(args.radius)).zfill(4)
coldens_str = str(int(args.Sigma_crit*1e6)).zfill(6)

file_suffix = f"r{radius_str}-c{coldens_str}"
print(file_suffix)


# numerical parameters
nside = args.nside

# center of the local bubble
if args.LB:
    cx = -80.
    cy = -150.
    cz = 0.0
    c = ([cx, cy, cz], "pc")          # new yt version
    c = [cx*pc, cy*pc, cz*pc] # old yt version
else:
    #c = ([0, 0, 0], "pc")          # new yt version
    c = [0, 0, 0]


def _activate_xray_fields(field, data):
    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
    ds.add_field(('gas', 'El_number_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
    yt.add_xray_emissivity_field(ds, 0.1, 2, metallicity=1.0, data_dir=dat_path, table_type="apec")
    sp = ds.sphere(ctr, (args.radius, "pc"))


print("center at (pc) :", cx, cy, cz)

# loop over all files
for files in args.files:
    print("processing file", files)
    ds = yt.load(files)
    #ad = ds.all_data()

    def _nuclei_density(field, data):
        return data[("gas", "number_density")]*data[("flash", "ihp ")]
    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, \
                sampling_type="local", units="cm**(-3)", force_override=True)

    def _electron_density(field, data):
        return data[("gas", "number_density")]*data[("flash", "ihp ")]
    ds.add_field(("gas", "electron_density"), function=_electron_density, \
                sampling_type="local", units="cm**-3", force_override=True)
    
    def _electron_density_squared(field, data):
        return data[("gas", "electron_density")]**2
    ds.add_field(("gas", "electron_density_squared"), function=_electron_density_squared, \
                sampling_type="local", units="cm**-6", force_override=True)
    
    def _electron_density_squared_hot(field, data):
        return np.where(data[("gas", "temperature")] > 8e5, data[("gas", "electron_density")].v**2, 1e-50) / (cm**6)
    ds.add_field(("gas", "electron_density_squared_hot"), function=_electron_density_squared_hot, \
                sampling_type="local", units="cm**-6", force_override=True)

    yt.add_xray_emissivity_field(ds, 0.1, 2, metallicity=1.0, data_dir=dat_path, table_type="apec")

    # loop over radii
    #radii = np.linspace(args.rad_min_pc, args.rad_max_pc, args.Nrad_pc)

    # buffer all fields in separate arrays
    print("buffer all fields in separate arrays")

    # select different phases for the X-ray luminosity
    # total luminosity
    _activate_xray_fields()
    Ltot = sp[("gas", "xray_luminosity_0.1_2_keV")].copy()

    # fixed number density
    def _nuclei_density(field, data):
        return args.fixed_n / (cm**3)
    _activate_xray_fields()
    Lfix = sp[("gas", "xray_luminosity_0.1_2_keV")].copy()

    # only hot gas
    def _nuclei_density(field, data):
        return np.where(data[("gas", "temperature")] > 8e5, \
                        data[("gas", "number_density")].v * data[("flash", "ihp ")], \
                        1e-50) / (cm**3)
    _activate_xray_fields()
    Lhot = sp[("gas", "xray_luminosity_0.1_2_keV")].copy()

    # hot and fixed number density
    def _nuclei_density(field, data):
        return np.where(data[("gas", "temperature")] > 8e5, \
                        args.fixed_n, \
                        1e-50) / (cm**3)
    _activate_xray_fields()
    Lhotfix = sp[("gas", "xray_luminosity_0.1_2_keV")].copy()



    print()
    print("min and max positions of the selected region")
    print("min/max x (pc): ", np.min(sp[("gas", "x")].in_units("pc").v), np.max(sp[("gas", "x")].in_units("pc").v))
    print("min/max y (pc): ", np.min(sp[("gas", "y")].in_units("pc").v), np.max(sp[("gas", "y")].in_units("pc").v))
    print("min/max z (pc): ", np.min(sp[("gas", "z")].in_units("pc").v), np.max(sp[("gas", "z")].in_units("pc").v))

    Lx = ds.domain_width[0].in_units("pc").v
    Ly = ds.domain_width[1].in_units("pc").v
    Lz = ds.domain_width[2].in_units("pc").v

    # find the center of the sphere taking into account periodic boundaries in x and y
    posx_ctr = (sp[("gas", "x")].in_units("pc").v - cx + Lx/2.) % Lx - Lx/2.
    posy_ctr = (sp[("gas", "y")].in_units("pc").v - cy + Ly/2.) % Ly - Ly/2.
    posz_ctr =  sp[("gas", "z")].in_units("pc").v - cz

    print()
    print("min and max positions of the selected region - shifted to center of the sphere")
    print("min/max x (pc): ", np.min(posx_ctr), np.max(posx_ctr))
    print("min/max y (pc): ", np.min(posy_ctr), np.max(posy_ctr))
    print("min/max z (pc): ", np.min(posz_ctr), np.max(posz_ctr))

    volu = sp[("gas", "cell_volume")].in_units("pc**3").v

    print()
    # create three healpix maps
    print("create healpix maps for coldens, Xraylum, Xrayflx, emmeasure, radius with nside", nside, "and NPIX", hp.nside2npix(nside))
    NPIX = hp.nside2npix(nside)
    coldens_map     = np.zeros(NPIX) # column density
    coldens_map2    = np.zeros(NPIX) # check column density

    Xraylum_map     = np.zeros(NPIX) # X-ray luminosity (full luminosity)
    Xrayflx_map     = np.zeros(NPIX) # X-ray flux
    Xraylumhot_map  = np.zeros(NPIX) # X-ray luminosity (hot gas)
    Xrayflxhot_map  = np.zeros(NPIX) # X-ray flux
    Xraylumfixn_map = np.zeros(NPIX) # X-ray luminosity (fixed number density)
    Xrayflxfixn_map = np.zeros(NPIX) # X-ray flux
    Xraylumhfn_map  = np.zeros(NPIX) # X-ray luminosity (hot gas, fixed number density)
    Xrayflxhfn_map  = np.zeros(NPIX) # X-ray flux
    emmeasure_map   = np.zeros(NPIX) # emission measure
    EM_hot_map      = np.zeros(NPIX) # map for emission measure of hot gas (T>8e5K)

    radius_map      = np.zeros(NPIX) # radius
    bubble_open_map = np.zeros(NPIX) # map to check if the bubble is open

    # create a mask for the pixels with "pix" and check if the column density is less than Sigma_crit
    idxmap = np.zeros(NPIX, dtype=bool)

    # radius based on centre of mass
    rad_ctr = np.sqrt(posx_ctr**2 + posy_ctr**2 + posz_ctr**2)
    
    Ncell = rad_ctr.size

    # print min max of radius
    print()
    print("min/max r (pc): ", np.min(rad_ctr), np.max(rad_ctr))
    
    # and normalise it
    vec_norm_x = posx_ctr / rad_ctr
    vec_norm_y = posy_ctr / rad_ctr
    vec_norm_z = posz_ctr / rad_ctr

    # compute angular size of all cells at the respective distance to the observer
    R = (3 * volu / (4 * np.pi))**(1./3.)
    dx = np.power(volu, 1./3.)

    #R_radians = R**2 / ( 4. * rad_ctr**2 )
    
    angle = np.arctan2(R, rad_ctr)

    total_mass = 0.0
    total_volu = 0.0

    total_lumi = 0.0
    total_flux = 0.0
    total_lumi_hot = 0.0
    total_flux_hot = 0.0
    total_lumi_fixn = 0.0
    total_flux_fixn = 0.0
    total_lumi_hfn = 0.0
    total_flux_hfn = 0.0

    # sort cells based on radius
    idx = np.argsort(rad_ctr)

    print()
    for i in tqdm(range(Ncell)):

        # for every cell compute the healpix index
        if rad_ctr[idx[i]] > args.rad_res:

            # reset the mask
            idxmap[:] = False

            # find the pixels in the healpix map
            pix = hp.query_disc(nside, [vec_norm_x[idx[i]], vec_norm_y[idx[i]], vec_norm_z[idx[i]]], angle[idx[i]], inclusive=True)
            idxmap[pix] = True

            # check where the column density is less than Sigma_crit
            idx2 = np.where((coldens_map < args.Sigma_crit) & (idxmap == True))

            # total column density to check
            coldens_map2[pix]   += sp[("gas", "cell_mass")][idx[i]].v / (np.pi*(R[idx[i]]*pc)**2) / len(pix)

            # fill the maps
            radius_map[idx2]     = rad_ctr[idx[i]]
            coldens_map[idx2]   += sp[("gas", "cell_mass")][idx[i]].v / (np.pi*(R[idx[i]]*pc)**2) / len(pix)

            Xraylum_map[idx2]     += Ltot[idx[i]].v / len(pix)
            Xrayflx_map[idx2]     += Ltot[idx[i]].v / len(pix) / (4. * np.pi * rad_ctr[idx[i]]**2)
            Xraylumhot_map[idx2]  += Lhot[idx[i]].v / len(pix)
            Xrayflxhot_map[idx2]  += Lhot[idx[i]].v / len(pix) / (4. * np.pi * rad_ctr[idx[i]]**2)
            Xraylumfixn_map[idx2] += Lfix[idx[i]].v / len(pix)
            Xrayflxfixn_map[idx2] += Lfix[idx[i]].v / len(pix) / (4. * np.pi * rad_ctr[idx[i]]**2)
            Xraylumhfn_map[idx2]  += Lhotfix[idx[i]].v / len(pix)
            Xrayflxhfn_map[idx2]  += Lhotfix[idx[i]].v / len(pix) / (4. * np.pi * rad_ctr[idx[i]]**2)

            emmeasure_map[idx2] += sp[("gas", "electron_density_squared")][idx[i]].v * dx[idx[i]]
            EM_hot_map[idx2]    += sp[("gas", "electron_density_squared_hot")][idx[i]].v * dx[idx[i]]

            f_idx_pix   = len(idx2[0]) / len(pix)
            total_mass += sp[("gas", "cell_mass")][idx[i]].v * f_idx_pix

            total_lumi      += Ltot[idx[i]].v * f_idx_pix
            total_flux      += Ltot[idx[i]].v * f_idx_pix / (4. * np.pi * rad_ctr[idx[i]]**2)
            total_lumi_hot  += Lhot[idx[i]].v * f_idx_pix
            total_flux_hot  += Lhot[idx[i]].v * f_idx_pix / (4. * np.pi * rad_ctr[idx[i]]**2)
            total_lumi_fixn += Lfix[idx[i]].v * f_idx_pix
            total_flux_fixn += Lfix[idx[i]].v * f_idx_pix / (4. * np.pi * rad_ctr[idx[i]]**2)
            total_lumi_hfn  += Lhotfix[idx[i]].v * f_idx_pix
            total_flux_hfn  += Lhotfix[idx[i]].v * f_idx_pix / (4. * np.pi * rad_ctr[idx[i]]**2)

            total_volu += volu[idx[i]] * f_idx_pix

        else:

            radius_map[:]     = args.rad_res
            coldens_map[:]   += sp[("gas", "cell_mass")][idx[i]].v / NPIX / (4. * np.pi * (args.rad_res*pc)**2)
            coldens_map2[:]  += sp[("gas", "cell_mass")][idx[i]].v / NPIX / (4. * np.pi * (args.rad_res*pc)**2)

            Xraylum_map[:]     += Ltot[idx[i]].v / NPIX
            Xrayflx_map[:]     += Ltot[idx[i]].v / NPIX / (4. * np.pi * args.rad_res**2)
            Xraylumhot_map[:]  += Lhot[idx[i]].v / NPIX
            Xrayflxhot_map[:]  += Lhot[idx[i]].v / NPIX / (4. * np.pi * args.rad_res**2)
            Xraylumfixn_map[:] += Lfix[idx[i]].v / NPIX
            Xrayflxfixn_map[:] += Lfix[idx[i]].v / NPIX / (4. * np.pi * args.rad_res**2)
            Xraylumhfn_map[:]  += Lhotfix[idx[i]].v / NPIX
            Xrayflxhfn_map[:]  += Lhotfix[idx[i]].v / NPIX / (4. * np.pi * args.rad_res**2)

            emmeasure_map[:] += sp[("gas", "electron_density_squared")][idx[i]].v * dx[idx[i]]
            EM_hot_map[:]    += sp[("gas", "electron_density_squared_hot")][idx[i]].v * dx[idx[i]]

            total_mass      += sp[("gas", "cell_mass")][idx[i]].v
            total_lumi      += Ltot[idx[i]].v
            total_flux      += Ltot[idx[i]].v / (4. * np.pi * args.rad_res**2)
            total_lumi_hot  += Lhot[idx[i]].v
            total_flux_hot  += Lhot[idx[i]].v / (4. * np.pi * args.rad_res**2)
            total_lumi_fixn += Lfix[idx[i]].v
            total_flux_fixn += Lfix[idx[i]].v / (4. * np.pi * args.rad_res**2)
            total_lumi_hfn  += Lhotfix[idx[i]].v
            total_flux_hfn  += Lhotfix[idx[i]].v / (4. * np.pi * args.rad_res**2)

            total_volu += volu[idx[i]]

   # bubble open map
    bubble_open_map = np.where(radius_map > 0.9*args.radius, 1, 0)

    print()
    print("done with the loop over cells, make plots")

    # prepare the data field to write to disk
    data = {}

    data["simtime"] = ds.current_time.v
    data["simtime_Myr"] = ds.current_time.in_units("Myr").v

    
    for name, field, norm in zip(["coldens", "coldens2", "Xraylum", "Xrayflx", "Xraylum_hot", "Xrayflx_hot", "Xraylum_fixn", "Xrayflx_fixn", \
        "Xraylum_hfn", "Xrayflx_hfn", "emmeasure", "radius", "bubble_open", "EM_hot"], \
                            [coldens_map, coldens_map2, Xraylum_map, Xrayflx_map, Xraylumhot_map, Xrayflxhot_map, Xraylumfixn_map, Xrayflxfixn_map, \
                                Xraylumhfn_map, Xrayflxhfn_map, emmeasure_map, radius_map, bubble_open_map, EM_hot_map], \
                            [True, True, True, True, True, True, True, True, True, True, True, False, False, True]):

        fig = plt.figure(figsize=(10, 5))
        if norm:
            plt_field = np.log10(field).copy()
        else:
            plt_field = field.copy()

        hp.mollview(plt_field, return_projected_map=True, title=name, cbar=False)
        hp.graticule()

        #hp.write_map(f"{args.odir_data}/{files}-r{R:.1f}-{f}.fits", locals()[f"{f}_map"], overwrite=True)
        ax = plt.gca()
        image = ax.get_images()[0]
        cbar = fig.colorbar(image, ax=ax, orientation='vertical', shrink=0.8)
        cbar.set_label(name, rotation=270, labelpad=20)
        plt.savefig(f'{args.odir}/{files}-{file_suffix}-{name}.pdf', bbox_inches="tight")
        plt.close()

        # dave the data in global dictionary
        data[name] = field

    # add global quantities to the data dictionary
    data["radius_max"] = args.radius
    data["total_mass"] = total_mass
    data["total_lumi"] = total_lumi
    data["total_flux"] = total_flux
    data["total_volu"] = total_volu

    # write data to disk using pickle
    with open(f"{args.odir_data}/{files}-{file_suffix}-data.pkl", "wb") as f:
        pickle.dump(data, f)


