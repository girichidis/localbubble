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

fig, ax = plt.subplots(figsize=(10,5))

cd_names  = ["c000050", "c000100", "c000167", "c000500", "c001000", "c001670"]
cd_labels = ["3e19", "6e19", "1e20", "3e20", "6e20", "1e21"]
for i, cd, label in zip(range(6), cd_names, cd_labels):
    print(cd)
    global_times = []
    global_Xlum = []
    
    for file_no in range(800,1300,2):
        pickle_path = "data-float32/SILCC_hdf5_plt_cnt_"+str(file_no).zfill(4)+"-r0150-"+cd+"-data-float32.pkl"
        #pickle_path2 = "data-float32/SILCC_hdf5_plt_cnt_"+str(file_no).zfill(4)+"-r0150-c000050-data-float32.pkl"

        #t, L = _read_Ltot(pickle_path)
        #t, L = _read_Rmed(pickle_path)
        t, L = _read_fopen(pickle_path)
        global_times.append(t)
        global_Xlum.append(L)

    ax.semilogy(global_times, global_Xlum, label=label, color = colors[i])

ax.legend()
ax.set_xlabel("time (Myr)")
#ax.set_ylabel("L_X (erg/s)")
#ax.set_ylabel("R_cd (pc)")
ax.set_ylabel("fraction open")
fig.savefig("fraction-open-all-files.pdf", bbox_inches="tight")
