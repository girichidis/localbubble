
import numpy as np
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import yt
from mpi4py import MPI
import argparse




comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nside = 64

min=18.
max=32.


pc = 3.085678e+18
kpc = 1e3 * pc

cx = -80.
cy = -150.
cz = 0.0

#cx = 0.0
#cy = 0.0
#cz = 0.0
c = ([cx, cy, cz], "pc")          # new yt version
c = [cx*pc, cy*pc, cz*pc] # old yt version
#c = [0.0*pc, 0.0*pc, 0.0*pc] # old yt version


Rmax_pc = 120.0

# parse command line arguments
parser = argparse.ArgumentParser(description='cmd line args')
parser.add_argument('files', nargs='+', help='files')
args = parser.parse_args()

 

for files in args.files:

    #_______________________________
    c=([-80,-150,0], "pc")
    def _plot_proj(ds, field, vmin, vmax, fname):
        p = yt.ProjectionPlot(ds, "z", center=c, fields=field)
        p.set_zlim(field,vmin,vmax)
        p.save(fname)
    def _plot_slice(ds, field, vmin, vmax, fname):
        p = yt.SlicePlot(ds, "z", center=c, fields=field)
        p.set_zlim(field,vmin,vmax)
        p.save(fname)

    for f in ["/home/erea/data/B6-1pc/SILCC_hdf5_plt_cnt_1080"]:
    
        plt2 = f[-4:]
        print(plt2)
        ds = yt.load(f)
        ad = ds.all_data()
    
        def _nuclei_density(field, data):
            return data[("gas", "number_density")]
        ds.add_field(('gas', 'H_nuclei_density'), _nuclei_density, \
                 sampling_type="local", units="cm**(-3)", force_override=True)

        yt.add_xray_emissivity_field(ds, 0.5, 2, metallicity=1.0, \
                                 data_dir="/home/erea/data/xray-tables/")
  


#_______________________________


    
    comm.Barrier()

    yt.load(files)
    
    sp = ds.sphere(c, (Rmax_pc, "pc"))
    
    
    posx_ctr = sp[("gas", "x")].in_units("pc").v - cx
    posy_ctr = sp[("gas", "y")].in_units("pc").v - cy
    posz_ctr = sp[("gas", "z")].in_units("pc").v - cz
    
    # print min max of positions
    if rank == 0:
        print("min/max x: ", np.min(posx_ctr), np.max(posx_ctr))
        print("min/max y: ", np.min(posy_ctr), np.max(posy_ctr))
        print("min/max z: ", np.min(posz_ctr), np.max(posz_ctr))
    
    # radius based on centre of mass
    rad_ctr = np.sqrt(posx_ctr**2 + posy_ctr**2 + posz_ctr**2)
    
    # print min max of radius
    if rank == 0:
        print("min/max r: ", np.min(rad_ctr), np.max(rad_ctr))

    # and normalise it
    vec_norm_x = posx_ctr / rad_ctr   
    vec_norm_y = posy_ctr / rad_ctr   
    vec_norm_z = posz_ctr / rad_ctr   

    # compute angular size of all cells at the respective distance to the observer
    R = (3 * sp[("gas", "cell_volume")].v / (4 * np.pi))**(1./3.)
    #R_radians = R**2 / ( 4. * rad_ctr**2 )
    
    angle = np.arctan2(R/pc, rad_ctr)
    
    # print min max of R
    if rank == 0:
        print("min/max R: ", np.min(R), np.max(R))
        print("min/max p: ", np.min(angle), np.max(angle))

    # set up healpix map
    NPIX = hp.nside2npix(nside)
    im_loc = np.zeros(NPIX)
    im_glb = np.zeros(NPIX)
        
    # local and total mass in map
    Ltot_loc = 0.0
    Ltot_glb = 0.0
    
    N = rad_ctr.size

    N0 = np.zeros(size, dtype=np.int64)
    N1 = np.zeros(size, dtype=np.int64)
    
    # define split indices for parallelisation
    for i in range(size):
        delta = (N//size)
        N0[i] = rank*delta
        if rank < size-1:
            N1[i] = (i+1)*delta
        else:
            N1[i] = N

    # measure time for reading data
    time0_loop = MPI.Wtime()

    # loop over cells and fill HP map
    print("loop over cells: ", N0[rank], "to", N1[rank])

    for i in range(N0[rank], N1[rank]):
        
        Ltot_loc = Ltot_loc + sp[('gas','xray_luminosity_0.5_2_keV')][i].v
       
        pixels = hp.query_disc(nside, [vec_norm_x[i], vec_norm_y[i], vec_norm_z[i]], angle[i], inclusive=True,)  
        for pixel in pixels:
            im_loc[pixels] += (sp[('gas','xray_luminosity_0.5_2_keV')][i].v/(rad_ctr[i] ** 2))/len(pixels)




    


    comm.Barrier()
    comm.Allreduce(im_loc,   im_glb,   op=MPI.SUM)
    # sum up total mass
    Ltot_glb = comm.allreduce(Ltot_loc, op=MPI.SUM)
    if rank == 0:
        #print(Ltot_loc, Ltot_glb)
        print(sp[("gas", "xray_luminosity_0.5_2_keV")].sum())
    if rank == 0:
        lmin = np.log10(np.min(im_glb))
        lmax = np.log10(np.max(im_glb))
        
        fig = plt.figure(figsize=(10, 5))
        hp.mollview(np.log10(im_glb), return_projected_map=True, min=min, max=max, title="X-ray Luminosity Map (0.5-2 keV) - Time: " + str(t) + " Myr - Radius: " + str(Rmax_pc) + " pc", cbar=False)
        hp.graticule()

        # Create a custom colorbar
        ax = plt.gca()
        image = ax.get_images()[0]
        cbar = fig.colorbar(image, ax=ax, orientation='vertical', shrink=0.8)
        cbar.set_label('log(X-ray Luminosity (erg/s/sr))', rotation=270, labelpad=20)

        plt.savefig(files + "-mollweide-x_lum_" + str(Rmax_pc) + "_weighted.pdf", bbox_inches="tight")

