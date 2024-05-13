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

min=20.
max=30.

pc = 3.085678e+18
kpc = 1e3 * pc

cx = -80.
cy = -150.
cz = 0.0

c = ([cx, cy, cz], "pc")          # new yt version
c = [-80.0*pc, -150.0*pc, 0.0*pc] # old yt version

Rmax_pc = 20.

# parse command line arguments
parser = argparse.ArgumentParser(description='cmd line args')
parser.add_argument('files', nargs='+', help='files')
args = parser.parse_args()


for files in args.files:

    comm.Barrier()

    ds = yt.load(files)
    
    sp = ds.sphere(c, (Rmax_pc, "pc"))
    
    posx_ctr = sp[("gas", "x")].in_units("pc").v - cx
    posy_ctr = sp[("gas", "y")].in_units("pc").v - cy
    posz_ctr = sp[("gas", "z")].in_units("pc").v - cz
    
    # radius based on centre of mass
    rad_ctr = np.sqrt(posx_ctr**2 + posy_ctr**2 + posz_ctr**2)
    
    # and normalise it
    vec_norm_x = posx_ctr / rad_ctr   
    vec_norm_y = posy_ctr / rad_ctr   
    vec_norm_z = posz_ctr / rad_ctr   
        
    # compute angular size of all cells at the respective distance to the observer
    R = (3 * sp[("gas", "cell_volume")].v / (4 * np.pi))**(1./3.)
    angle = np.arctan2(R, rad_ctr)
    
    # set up healpix map
    NPIX = hp.nside2npix(nside)
    im_loc = np.zeros(NPIX)
    im_glb = np.zeros(NPIX)
        
    # local and total mass in map
    Mtot_loc = 0.0
    Mtot_glb = 0.0
    
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

    # loop over cells and fill HP map
    print("loop over cells: ", N0[rank], "to", N1[rank])
    for i in range(N0[rank], N1[rank]):
        if rank == 0:
            if i%100 == 0:
                print(i)
        Mtot_loc = Mtot_loc + sp[("gas","cell_mass")][i].v
        pixels = hp.query_disc(nside, [vec_norm_x[i], vec_norm_y[i], vec_norm_z[i]], angle[i], inclusive=True,)  
        for p in pixels:
            im_loc[p] += sp[("gas","cell_mass")][i].v/len(pixels)

    comm.Barrier()
    comm.Allreduce(im_loc, im_glb, op=MPI.SUM)
    
    if rank == 0:
        min = np.min(im_glb)
        max = np.max(im_glb)
        hp.mollview(np.log10(im_glb), return_projected_map=True, min=min, max=max)
        hp.graticule()
        plt.savefig(files+"-mollweide-coldens.pdf", bbox_inches="tight")
        plt.savefig(files+"-mollweide-coldens.png", bbox_inches="tight", dpi=300)

