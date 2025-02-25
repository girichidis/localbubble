import yt
import numpy as np
import argparse
from astropy import units as u, constants  as c

pc = c.pc.cgs.value
kB  = c.k_B.cgs.value
Msun = c.M_sun.cgs.value
G = c.G.cgs.value
Myr = u.Myr.in_units("s")

dat_path = "/home/hd/hd_hd/hd_ud081/work/SILCC/2018-SILCC-Clouds/localbubble/tables/"

parser = argparse.ArgumentParser(description='cmd line args')
parser.add_argument('files', nargs='+', help='files')
parser.add_argument('-suff', default="", help='file suffix', type=str)
parser.add_argument('-radius', default=120, help='file suffix', type=float)
args = parser.parse_args()


def _nuclei_density(field, data):
    return data[("gas", "number_density")] * data[("flash", "ihp ")]

suff = ""
if args.suff != "":
    suff = "-"+args.suff
    print(suff)
    
ctr = [-80*pc, -150*pc, 0*pc]

time = []
F60 = []
F70 = []
F80 = []
F90 = []
F95 = []
F99 = []
Ltot = []

for f in args.files:
    ds = yt.load(f)

    time.append(ds.current_time.in_units("Myr"))

    # register data fields
    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
    ds.add_field(('gas', 'El_number_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)    

    yt.add_xray_emissivity_field(ds, 0.1, 2, metallicity=1.0, data_dir=dat_path, table_type="apec")

    # create sphere around ctr
    sphere = ds.sphere(ctr, (args.radius, "pc"))
    x_lum = sphere[('gas', 'xray_luminosity_0.1_2_keV')]
    vol = sphere[('gas', 'cell_volume')]

    idx = np.argsort(x_lum)

    cv = np.cumsum(vol[idx[::-1]]) / np.sum(vol)
    cl = np.cumsum(x_lum[idx[::-1]]) / np.sum(x_lum)

    F60.append(cv[np.where(cl > 0.60)[0][0]])
    F70.append(cv[np.where(cl > 0.70)[0][0]])
    F80.append(cv[np.where(cl > 0.80)[0][0]])
    F90.append(cv[np.where(cl > 0.90)[0][0]])
    F95.append(cv[np.where(cl > 0.95)[0][0]])
    F99.append(cv[np.where(cl > 0.99)[0][0]])
    Ltot.append(np.sum(x_lum))

            
data = np.vstack((time,Ltot,F60,F70,F80,F90,F95,F99)).T
np.savetxt("datafiles/time-evol-emission-volume-R"+str(int(args.radius))+suff+".txt", data, \
           header="time   Ltot   f_vol(60%em)   f_vol(70%em)   f_vol(80%em)   f_vol(90%em)   f_vol(95%em)   f_vol(99%em)")


