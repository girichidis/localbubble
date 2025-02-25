import yt
import numpy as np
import argparse
from astropy import units as u, constants  as c

pc = c.pc.cgs.value
kB  = c.k_B.cgs.value
Msun = c.M_sun.cgs.value
G = c.G.cgs.value
Myr = u.Myr.in_units("s")

parser = argparse.ArgumentParser(description='cmd line args')
parser.add_argument('files', nargs='+', help='files')
parser.add_argument('-suff', default="", help='file suffix', type=str)
parser.add_argument('-radius', default=120, help='file suffix', type=float)
args = parser.parse_args()

suff = ""
if args.suff != "":
    suff = "-"+args.suff
    print(suff)
    
ctr = [-80*pc, -150*pc, 0*pc]
#radius=120

time = []
Tvw = []
nvw = []
vff = []

for f in args.files:
    ds = yt.load(f)

    time.append(ds.current_time.in_units("Myr"))
    # create sphere around ctr
    sp = ds.sphere(ctr, (args.radius, "pc"))
    
    vol = sp[("gas", "cell_volume")]
    Vtot = np.sum(vol)
    
    # get temperature and density data
    for field in [("gas", "number_density"), ("gas", "temperature")]:
        dat = sp[field]
        vw  = np.sum(dat*vol)/Vtot
        if field == ("gas", "number_density"):
            nvw.append(vw)
        if field == ("gas", "temperature"):
            Tvw.append(vw)
            idx = np.where(dat > 8e5)
            vff.append(np.sum(vol[idx])/Vtot)
            
data = np.vstack((time,Tvw,nvw,vff)).T
np.savetxt("datafiles/time-evol-temp-dens-R"+str(int(args.radius))+suff+".txt", data, header="time   temp(vw)   n(vw)   vff(T>8e5K)")


