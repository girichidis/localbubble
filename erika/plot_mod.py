import pickle
import numpy as np
import matplotlib.pyplot as plt

def _read_Ltot(pickle_path):
    """Function to read total luminosity from a pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data["simtime_Myr"], data["total_lumi"]

def _read_flux(pickle_path):
    """Function to read total flux from a pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data["simtime_Myr"], data["total_flux"]

cmap = plt.get_cmap('twilight')
colors = cmap(np.linspace(0.1, 0.9, 6))

fig, ax = plt.subplots(figsize=(12, 5))

cd_names = ["c000050", "c000100", "c000167", "c000500", "c001000", "c001670"]
#cd_labels = ["3e19 cm^{-2}", "6e19 cm^{-2}", "1e20 cm^{-2}", "3e20 cm^{-2}", "6e20 cm^{-2}", "1e21 cm^{-2}"]
cd_labels = [r"3$\cdot10^{19}\mathrm{cm}^{-2}$",
             r"6$\cdot10^{19}\mathrm{cm}^{-2}$",
             r"1$\cdot10^{20}\mathrm{cm}^{-2}$",
             r"3$\cdot10^{20}\mathrm{cm}^{-2}$",
             r"6$\cdot10^{20}\mathrm{cm}^{-2}$",
             r"1$\cdot10^{21}\mathrm{cm}^{-2}$"]

# vertical lines times
#orange_times = [8.532943785668992, 8.86621629676601, 9.46610215599239, 9.532684083703234, 9.932201109701966, 12.065611223842739]
#red_times = [8.200511414077361, 8.666315218769816, 8.93249930247305, 9.19900773620799, 10.132647717184527, 10.19926537729867,10.265438459099556, 10.466229866835764, 10.73220989220038, 10.932442454026633, 11.46545269499049, 11.865518199112238,12.198622828154724]
#sn_times_150 = [8.195938, 8.528185, 8.661482, 8.728356, 8.861272, 8.927518, 9.193877, 9.460823, 9.527368, 9.926662, 10.126997, 10.193577, 10.259713, 10.460393, 10.726224, 10.926345, 11.459058, 11.858901, 12.058882, 12.191820, 12.858581]
#sn_times_180 = [8.195938, 8.528185, 8.661482, 8.728356, 8.861272, 8.927518, 9.193877, 9.394164, 9.460823, 9.527368, 9.794173, 9.926662, 10.060073, 10.126997, 10.193577, 10.259713, 10.460393, 10.726224, 10.859818, 10.926345, 11.260382, 11.459058, 11.858901, 11.991983, 12.058882, 12.191820, 12.259387, 12.858581]
#sn_times_200 = [8.195938, 8.327677, 8.461352, 8.528185, 8.661482, 8.728356, 8.861272, 8.927518, 9.193877, 9.260736, 9.394164, 9.460823, 9.527368, 9.794173, 9.926662, 10.060073, 10.126997, 10.193577, 10.259713, 10.460393, 10.726224, 10.859818, 10.926345, 11.260382, 11.459058, 11.793207, 11.858901, 11.991983, 12.058882, 12.191820, 12.259387, 12.724948, 12.858581, 12.924980]



#____
sn_internal_200_180_150=[
    8.200511414077361,
    8.666315218769816,
    8.93249930247305,
    9.19900773620799,
    10.132647717184527,
    10.19926537729867,
    10.265438459099556,
    10.466229866835764,
     10.73220989220038,
    10.932442454026633,
    11.46545269499049,
    11.865518199112238,
   12.198622828154724,
    12.865756658211794
]

sn_external_200 = [
    8.332324350031707,
    8.466073462270133,
    8.532943785668992,
    8.733226949904882,
    8.86621629676601,
    9.265903677869373,
    9.399406150919468,
    9.46610215599239,
    9.532684083703234,
    9.799638205453393,
    9.932201109701966,
    10.065686968928345,
    10.865878503487634,
    11.266665377298668,
     11.79978807863031,
    11.998674889029804,
    12.065611223842739,    
    12.266228281547242,
    12.732049112238428,
    12.932192485732402
]

sn_external_180 = [
    8.532943785668992,
    8.733226949904882,
    8.86621629676601,
    9.399406150919468,
    9.46610215599239,
    9.532684083703234,
    9.799638205453393,
    9.932201109701966,
    10.065686968928345,
    10.865878503487634,
    11.266665377298668,
    11.998674889029804,
    12.065611223842739,   
    12.266228281547242
]



sn_external_150= [
    8.532943785668992,
    8.733226949904882,
    8.86621629676601,
    9.46610215599239,
    9.532684083703234,
    9.932201109701966,
    12.065611223842739
]

sn_internal_130=[
    8.200511414077361,
    8.666315218769816,
    8.93249930247305,
    9.19900773620799,
    10.132647717184527,
    10.19926537729867,
    10.265438459099556,
    10.466229866835764,
     10.73220989220038,
    10.932442454026633,
    11.46545269499049,
    11.865518199112238,
   12.198622828154724,
]
sn_external_130= [
    8.532943785668992,
    8.86621629676601,
    9.46610215599239,
    9.532684083703234,
    9.932201109701966,
    12.065611223842739
]


for i, cd, label in zip(range(6), cd_names, cd_labels):
    print(cd)
    global_times_lum = []
    global_Xlum = []
    global_times_flux = []
    global_flux = []

    for file_no in range(800, 1300, 2):
        pickle_path = "data-float32/SILCC_hdf5_plt_cnt_" + str(file_no).zfill(4) + "-r0150-" + cd + "-data-float32.pkl"

        t_lum, L = _read_Ltot(pickle_path)
        t_flux, F = _read_flux(pickle_path)

        global_times_lum.append(t_lum)
        global_Xlum.append(L)
        global_times_flux.append(t_flux)
        global_flux.append(F)

    ax.semilogy(global_times_lum, global_Xlum, label=f"Lum ({label})", color=colors[i], linestyle='-')
    ax.semilogy(global_times_flux, global_flux, label=f"Flux ({label})", color=colors[i], linestyle='--')

for time in sn_external_130:
    ax.axvline(x=time, color='orange', linestyle='--', linewidth=1, label='External sn' if time == sn_external_130[0] else "")

for time in sn_internal_130:
    ax.axvline(x=time, color='red', linestyle='--', linewidth=1, label='Internal sn' if time == sn_internal_130[0] else "")

#for time in sn_times_200:
   # ax.axvline(x=time, color='blue', linestyle='--', linewidth=1, label='SN line' if time == sn_times_200[0] else "")

ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=12)

ax.set_xlabel("Time (Myr)", fontsize=14)
ax.set_ylabel("Top: Luminosity (erg/s) and bottom: Flux (erg/s/sr)", fontsize=14)

ax.tick_params(axis='both', which='major', labelsize=12)

fig.tight_layout()
fig.savefig("lum_flux_sn_130_small.pdf", bbox_inches="tight")
