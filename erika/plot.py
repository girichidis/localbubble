import pickle
import numpy as np
import matplotlib.pyplot as plt

def _read_Ltot(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data["simtime_Myr"], data["total_lumi"]

def _read_Rmed(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data["simtime_Myr"], np.median(data["radius"][data["bubble_open"]==0])

def _read_fopen(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data["simtime_Myr"], np.median(np.sum(data["bubble_open"])/data["bubble_open"].size)


cmap = plt.get_cmap('twilight')
colors = cmap(np.linspace(0.1,0.9,6))

fig, ax = plt.subplots(figsize=(4,3))

cd_names  = ["c000050", "c000100", "c000167", "c000500", "c001000", "c001670"]
#cd_labels = ["3e19", "6e19", "1e20", "3e20", "6e20", "1e21"]
cd_labels = [r"3$\cdot10^{19}\mathrm{cm}^{-2}$",
             r"6$\cdot10^{19}\mathrm{cm}^{-2}$",
             r"1$\cdot10^{20}\mathrm{cm}^{-2}$",
             r"3$\cdot10^{20}\mathrm{cm}^{-2}$",
             r"6$\cdot10^{20}\mathrm{cm}^{-2}$",
             r"1$\cdot10^{21}\mathrm{cm}^{-2}$"]
for i, cd, label in zip(range(6), cd_names, cd_labels):
    print(cd)
    global_times = []
    global_Xlum = []
    
    for file_no in range(800,1300,2):
        pickle_path = "data-float32/SILCC_hdf5_plt_cnt_"+str(file_no).zfill(4)+"-r0150-"+cd+"-data-float32.pkl"
        #pickle_path2 = "data-float32/SILCC_hdf5_plt_cnt_"+str(file_no).zfill(4)+"-r0150-c000050-data-float32.pkl"

        #t, L = _read_Ltot(pickle_path)
        t, L = _read_Rmed(pickle_path)
        #t, L = _read_fopen(pickle_path)
        global_times.append(t)
        global_Xlum.append(L)

    ax.semilogy(global_times, global_Xlum, label=label, color = colors[i])

ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=12)
ax.set_xlabel("Time (Myr)")
#ax.set_ylabel("L_X (erg/s)")
ax.set_ylabel("R_cd (pc)")
#ax.set_ylabel("Fraction open")
#fig.savefig("fraction-open-all-files.png", bbox_inches="tight")
#fig.savefig("total-Xlumi-all-files.png", bbox_inches="tight")
fig.savefig("total-R_cd-all-files.pdf", bbox_inches="tight")





