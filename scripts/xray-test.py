import yt
import numpy as np

c=([-80,-150,0], "pc")

def _plot_proj(ds, field, vmin, vmax, fname):
    p = yt.ProjectionPlot(ds, "z", center=c, fields=field)
    p.set_zlim(field,vmin,vmax)
    p.save(fname)
def _plot_slice(ds, field, vmin, vmax, fname):
    p = yt.SlicePlot(ds, "z", center=c, fields=field)
    p.set_zlim(field,vmin,vmax)
    p.save(fname)

for f in ["SILCC_hdf5_plt_cnt_1080"]: #, "SILCC_hdf5_plt_cnt_3000"]:
    
    plt = f[-4:]
    print(plt)
    ds = yt.load(f)
    ad = ds.all_data()
    
    def _nuclei_density(field, data):
        return data[("gas", "number_density")]
    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, \
                 sampling_type="local", units="cm**(-3)", force_override=True)

    yt.add_xray_emissivity_field(ds, 0.3, 2, metallicity=1.0, \
                                 data_dir="/Users/girichidis/Physics/Tables/yt-python/")
    
    # coldens
    _plot_proj(ds, field=("gas", "density"), vmin = 1e-5, vmax = 1e-1, fname=plt+"-Proj-z-dens.pdf")
    _plot_slice(ds, field=("gas", "density"), vmin = 1e-28, vmax = 1e-20, fname=plt+"-Slc-z-dens.pdf")
    _plot_slice(ds, field=("gas", "temperature"), vmin = 100, vmax = 1e8, fname=plt+"-Slc-z-temp-o.pdf")
    _plot_proj(ds, field=("gas", "xray_emissivity_0.3_2_keV"), vmin = 1e-17, vmax = 1e-2, fname=plt+"-Proj-z-Xray-To.pdf")

