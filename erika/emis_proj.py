import yt

ds = yt.load("/home/erea/data/B6-1pc/SILCC_hdf5_plt_cnt_1080")

c = ([-80, -150, 0], "pc")

def _plot_proj(ds, field, vmin, vmax, fname):
    p = yt.ProjectionPlot(ds, "z", center=c, fields=field)
    p.set_figure_size(5)
    p.set_zlim(field, vmin, vmax)
    p.set_xlabel('x(pc)')
    p.set_ylabel('y(pc)')
    p.set_cmap(field, 'magma') 
    p.save(fname)

def _plot_slice(ds, field, vmin, vmax, fname):
    p = yt.SlicePlot(ds, "z", center=c, fields=field)
    p.set_figure_size(5)
    p.set_zlim(field, vmin, vmax)
    p.save(fname)

for f in ["/home/erea/data/B6-1pc/SILCC_hdf5_plt_cnt_1080"]:
    plt = f[-4:]
    print(plt)
    ds = yt.load(f)
    
    def _nuclei_density(field, data):
        return data[("gas", "number_density")] * data[("flash", "ihp ")]

    ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, sampling_type="local", units="cm**(-3)", force_override=True)

    yt.add_xray_emissivity_field(ds, 0.5, 2, metallicity=1.0, data_dir="/home/erea/data/xray-tables/")

    _plot_proj(ds, field=("gas", "xray_emissivity_0.5_2_keV"), vmin=1e-17, vmax=1e-2, fname=plt + "-Proj-z-Xray-To.pdf")
