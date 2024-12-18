import numpy as np
import yt

# define center of the bubble (in pc)
cx = -80.
cy = -150.
cz = 0.0

for files in ["SILCC_hdf5_plt_cnt_1080"]:
    ds = yt.load(files)

    # define radius of the bubble (in pc)
    simtime = ds.current_time.in_units("Myr").v
    R = 80.0+5.*(simtime-7.0)
    print("sim time: ", simtime, "radius: ", R)
    w = 30

    # set domain size (in pc)
    Lx = ds.domain_width[0].in_units("pc").v
    Ly = ds.domain_width[1].in_units("pc").v
    Lz = ds.domain_width[2].in_units("pc").v

    # define positions relative to the bubble center (including periodic BC in x, y)
    # need to define this in dimensionless units, otherwise I get YT unit errors,
    # which I was unable to solve
    def _posx_ctr(field, data):
        return (data[("gas", "x")].in_units("pc").v - cx + Lx/2.) % Lx - Lx/2.
    ds.add_field(name=("gas", "xctr"), function=_posx_ctr, sampling_type="local", units="", force_override=True)
    def _posy_ctr(field, data):
        return (data[("gas", "y")].in_units("pc").v - cy + Ly/2.) % Ly - Ly/2.
    ds.add_field(name=("gas", "yctr"), function=_posy_ctr, sampling_type="local", units="", force_override=True)
    def _posz_ctr(field, data):
        return (data[("gas", "z")].in_units("pc").v - cz)
    ds.add_field(name=("gas", "zctr"), function=_posz_ctr, sampling_type="local", units="", force_override=True)

    # radius function w.r.t. bubble center
    def _rad_ctr(field, data):
        return np.sqrt(data[("gas", "xctr")]**2 + data[("gas", "yctr")]**2 + data[("gas", "zctr")]**2)
    ds.add_field(name=("gas", "rctr"), function=_rad_ctr, sampling_type="local", units="", force_override=True)

    slc = yt.SlicePlot(ds, "z", ("gas", "rctr"), center=([cx, cy, cz], "pc")).save()

    def _dens_cut1(field, data):
        return data["gas", "density"]*(0.5-np.arctan((data["gas", "rctr"]-R)/w)/np.pi)
    def _dens_cut2(field, data):
        return data["gas", "density"] * np.where(data["gas", "rctr"] <= R, \
                                                1.0, np.where(data["gas", "rctr"] >= R+w, 0.0, \
                                                0.5 + np.cos((data["gas", "rctr"] - R) * np.pi / w) / 2.0))
    def _dens_cut3(field, data):
        return data["gas", "density"] * np.where(data["gas", "rctr"] <= R, 1.0, np.power(data["gas", "rctr"]-R+1,-1.5))

    ds.add_field(name=("gas", "dens_cut1"), function=_dens_cut1, sampling_type="local", units="g*cm**-3", force_override=True)
    ds.add_field(name=("gas", "dens_cut2"), function=_dens_cut2, sampling_type="local", units="g*cm**-3", force_override=True)
    ds.add_field(name=("gas", "dens_cut3"), function=_dens_cut3, sampling_type="local", units="g*cm**-3", force_override=True)

    for f in ["dens_cut1", "dens_cut2", "dens_cut3"]:
        slc = yt.SlicePlot(ds, "z", ("gas", f), center=([cx, cy, cz], "pc"))
        slc.set_zlim(("gas", f), 1e-27, 1e-22)
        slc.save()

