import matplotlib.pyplot as plt
import numpy as np
import yt
import pickle

from astropy import units as u, constants  as const

pc = const.pc.cgs.value
kB  = const.k_B.cgs.value
Msun = const.M_sun.cgs.value
G = const.G.cgs.value
Myr = u.Myr.in_units("s")

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble" : r'\boldmath'
})

K_to_keV = 8.61732814974057e-8
K_to_keV = 8.625e-8

# radius of the bubble and center
radius=120
# c = ([-80, -150, 0], "pc") # only new version of YT
c = [-80*pc, -150*pc, 0*pc]

fileno_active = [1080, 1140] #np.arange(1070, 1100, 2)

all_data_active = {}

for files in fileno_active:
    ds = yt.load(f"sim-files/SILCC_hdf5_plt_cnt_{files:04d}")
    print(ds.current_time.in_units("Myr"))
    all_data_active[files] = {}
    all_data_active[files]["time"] = ds.current_time.in_units("Myr")

    # create a sphere object
    sphere = ds.sphere(c, (radius, "pc"))

    # get data fields
    dens = sphere[("gas","number_density")]
    edens= sphere[("gas","number_density")]*sphere[("flash","ihp ")]
    temp = sphere[("gas","temperature")]
    tkeV = temp*K_to_keV
    vol  = sphere[("gas","cell_volume")]

    # select hot gas
    idx = np.where(temp > 1e4)
    np_med = np.median(dens[idx])
    ne_med = np.median(edens[idx])
    Hp, edp = np.histogram(np.log10(dens[idx]),  bins=100, weights=vol[idx]/np.sum(vol[idx]))
    He, ede = np.histogram(np.log10(edens[idx]), bins=100, weights=vol[idx]/np.sum(vol[idx]))
    ctrp = 0.5*(edp[1:]+edp[:-1])
    ctre = 0.5*(ede[1:]+ede[:-1])

    # select underdense gas
    idx = np.where(dens < 5e-3)
    T_med = np.median(tkeV[idx])
    HT, edT = np.histogram(np.log10(tkeV[idx]),  bins=100, weights=vol[idx]/np.sum(vol[idx]))
    ctrT = 0.5*(edT[1:]+edT[:-1])

    all_data_active[files]["np"] = ctrp
    all_data_active[files]["ne"] = ctre
    all_data_active[files]["Hp"] = Hp
    all_data_active[files]["He"] = He
    all_data_active[files]["T"] = ctrT
    all_data_active[files]["HT"] = HT

# save all_data_active as a pickle file
with open("datafiles/all_data_active.pkl", "wb") as f:
    pickle.dump(all_data_active, f)