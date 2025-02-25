import yt
import numpy as np
import argparse
from unyt import cm
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
parser.add_argument('-radius', default=120, help='radius in pc', type=float)
parser.add_argument('-fixed_n', default=0.01, help='fixed number density', type=float)
args = parser.parse_args()

suff = ""
if args.suff != "":
    suff = "-"+args.suff
    print(suff)
    
ctr = [-80*pc, -150*pc, 0*pc]

time = []
Ltot = []
Lnfix   = []
Lnfix3  = []
Lnfix13 = []
LtotT8e5    = []
LnfixT8e5   = []
Lnfix3T8e5  = []
Lnfix13T8e5 = []

for f in args.files:
    ds = yt.load(f)

    time.append(ds.current_time.in_units("Myr"))
    
    # version 1: normal number density
    def _nuclei_density(field, data):
        return data[("gas", "number_density")] * data[("flash", "ihp ")]

    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
    ds.add_field(('gas', 'El_number_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)    
    yt.add_xray_emissivity_field(ds, 0.1, 2, metallicity=1.0, data_dir=dat_path, table_type="apec")
    sphere = ds.sphere(ctr, (args.radius, "pc"))
    x_lum = np.sum(sphere[('gas', 'xray_luminosity_0.1_2_keV')])
    Ltot.append(x_lum)

    # version 2: fixed number density everywhere
    def _nuclei_density(field, data):
        return args.fixed_n / (cm**3)

    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
    ds.add_field(('gas', 'El_number_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)    
    yt.add_xray_emissivity_field(ds, 0.1, 2, metallicity=1.0, data_dir=dat_path, table_type="apec")
    sphere = ds.sphere(ctr, (args.radius, "pc"))
    x_lum = np.sum(sphere[('gas', 'xray_luminosity_0.1_2_keV')])
    Lnfix.append(x_lum)

    # version 3: fixed number density everywhere * 3
    def _nuclei_density(field, data):
        return 3. * args.fixed_n / (cm**3)

    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
    ds.add_field(('gas', 'El_number_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)    
    yt.add_xray_emissivity_field(ds, 0.1, 2, metallicity=1.0, data_dir=dat_path, table_type="apec")
    sphere = ds.sphere(ctr, (args.radius, "pc"))
    x_lum = np.sum(sphere[('gas', 'xray_luminosity_0.1_2_keV')])
    Lnfix3.append(x_lum)

    # version 4: fixed number density everywhere / 3
    def _nuclei_density(field, data):
        return 1./3. * args.fixed_n / (cm**3)

    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
    ds.add_field(('gas', 'El_number_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)    
    yt.add_xray_emissivity_field(ds, 0.1, 2, metallicity=1.0, data_dir=dat_path, table_type="apec")
    sphere = ds.sphere(ctr, (args.radius, "pc"))
    x_lum = np.sum(sphere[('gas', 'xray_luminosity_0.1_2_keV')])
    Lnfix13.append(x_lum)

    # now only apply density to hot gas
    
    # version 5: only consider hot gas (T>8e5), full density
    def _nuclei_density(field, data):
        return np.where(data[("gas", "temperature")] > 8e5, \
                        data[("gas", "number_density")].v * data[("flash", "ihp ")], \
                        1e-50) / (cm**3)

    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
    ds.add_field(('gas', 'El_number_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)    
    yt.add_xray_emissivity_field(ds, 0.1, 2, metallicity=1.0, data_dir=dat_path, table_type="apec")
    sphere = ds.sphere(ctr, (args.radius, "pc"))
    x_lum = np.sum(sphere[('gas', 'xray_luminosity_0.1_2_keV')])
    LtotT8e5.append(x_lum)

    # version 6: only consider hot gas (T>8e5), fixed n
    def _nuclei_density(field, data):
        return np.where(data[("gas", "temperature")] > 8e5, \
                        args.fixed_n, \
                        1e-50) / (cm**3)

    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
    ds.add_field(('gas', 'El_number_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)    
    yt.add_xray_emissivity_field(ds, 0.1, 2, metallicity=1.0, data_dir=dat_path, table_type="apec")
    sphere = ds.sphere(ctr, (args.radius, "pc"))
    x_lum = np.sum(sphere[('gas', 'xray_luminosity_0.1_2_keV')])
    LnfixT8e5.append(x_lum)

    # version 7: only consider hot gas (T>8e5), 3 fixed n
    def _nuclei_density(field, data):
        return np.where(data[("gas", "temperature")] > 8e5, \
                        3. * args.fixed_n, \
                        1e-50) / (cm**3)

    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
    ds.add_field(('gas', 'El_number_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)    
    yt.add_xray_emissivity_field(ds, 0.1, 2, metallicity=1.0, data_dir=dat_path, table_type="apec")
    sphere = ds.sphere(ctr, (args.radius, "pc"))
    x_lum = np.sum(sphere[('gas', 'xray_luminosity_0.1_2_keV')])
    Lnfix3T8e5.append(x_lum)

    # version 8: only consider hot gas (T>8e5), 1/3 fixed n
    def _nuclei_density(field, data):
        return np.where(data[("gas", "temperature")] > 8e5, \
                        1./3. * args.fixed_n, \
                        1e-50) / (cm**3)

    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)
    ds.add_field(('gas', 'El_number_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)    
    yt.add_xray_emissivity_field(ds, 0.1, 2, metallicity=1.0, data_dir=dat_path, table_type="apec")
    sphere = ds.sphere(ctr, (args.radius, "pc"))
    x_lum = np.sum(sphere[('gas', 'xray_luminosity_0.1_2_keV')])
    Lnfix13T8e5.append(x_lum)
            
data = np.vstack((time,Ltot,Lnfix,Lnfix3,Lnfix13,LtotT8e5,LnfixT8e5,Lnfix3T8e5,Lnfix13T8e5)).T
np.savetxt("datafiles/time-evol-emission-test-fixed-n-R"+str(int(args.radius))+suff+".txt", data, \
           header="(0)time   (1)Ltot   (2)Ltotnfix   (3)Ltotnfix3   (4)Ltotnfix13   (5)LtotT8e5   (6)LtotnfixT8e5   (7)Ltotnfix3T8e5   (8)Ltotnfix13T8e5")


