import pickle
import numpy as np
import matplotlib.pyplot as plt

global_times = []
global_Xlum = []
global_Xlum2 = []

def _read(pickle_path):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data["simtime_Myr"], data["total_mass"]

for file_no in range(1000,1100,2):
    print(file_no)
    pickle_path = "data-float32/SILCC_hdf5_plt_cnt_"+str(file_no).zfill(4)+"-r0150-c001670-data-float32.pkl"
    pickle_path2 = "data-float32/SILCC_hdf5_plt_cnt_"+str(file_no).zfill(4)+"-r0150-c000050-data-float32.pkl"

    t, L = _read(pickle_path)
    global_times.append(t)
    global_Xlum.append(L)
    t, L2 = _read(pickle_path2)
    global_Xlum2.append(L2)
    print(L, L2)
fig, ax = plt.subplots(figsize=(10,5))
ax.semilogy(global_times, global_Xlum, label="1e20")
ax.semilogy(global_times, global_Xlum2, label="3e19")
ax.legend()
fig.savefig("total-Xlumi-all-files.pdf", bbox_inches="tight")
