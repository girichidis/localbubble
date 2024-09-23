import numpy as np
import healpy as hp
import scipy.special as sps
# only used for the sanity check and demo at the bottom of the file
import argparse


# computes radius of a structure defined by a series of spherical harmonics
# alm is an array containing the spherical harmonic coefficients
# it is ordered the way healpy orders these things by default
def sph_radius(theta, phi, alm):
    lmax = hp.Alm.getlmax(len(alm))

    r = 0
    for l in range(lmax+1):
        for m in np.arange(-l,l+1):
            # these coefficient arrays are ordered very weirdly
            # hp.Alm.getidx returns the index in the array based on the l and m
            index_alm = hp.Alm.getidx(lmax=lmax, l=l, m=np.abs(m))
            a = alm[index_alm]
            # alm are only stored for positive m because they can be calculated like so
            # a_l,-m = complex conjuate(a_l,+m)
            if m<0:
                a = (-1)**(-m) * np.conj(a)
            r += np.real(a*sps.sph_harm(m, l, phi, theta))

    return r

# vs being a numpy array containing [ radius, polar angle theta, azimuth angle phi], angles in units of pi
def spherical_to_cartesian(vs):
    r, theta, phi = vs
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x, y, z])

# vc being a numpy array containing [ x, y, z]
def cartesian_to_spherical(vc):
    x, y, z = vc
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)

    return np.array([r, theta, phi])

# point is the point in question
# alm is the array with spherical harmonic coefficients
# center is the bubble center
# IMPORTANT: the alm are computed in relation to the center
# do not change the center unless you also change the alm
def in_sph(point, alm, center=np.array([-80,-160,0])):
    # tranform point into coordinate system centered around sph center
    _point = point - center

    # transform to spherical coordinates
    rp, thetap, phip = cartesian_to_spherical(_point)


    # compute spherical harmonic structure radius with thetap, phip
    rsph = sph_radius(theta=thetap, phi=phip, alm=alm)

    # point is to be considered inside the structure if rp is less or equal rsph
    return (rp<=rsph)

### THIS CONCLUDES ALL THE RELEVANT FUNCTIONS ###






# demonstration/sanity check
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check if a point is inside the bubble, requires an alm file')

    parser.add_argument('-a', type=str, help='Path to the alm file which contains the spherical harmonic coefficients')
    parser.add_argument('-c', nargs=3, default=[-80,-160,0], type=float, help='Bubble center coordinates, keep in mind that the alm coefficients were calculated in relation to a given center, so only change this if you have corresponding coefficients!')
    parser.add_argument('-p', nargs=3, type=float, help='Coordinates of the point you want to check')

    args = parser.parse_args()
    c = np.array(args.c)
    p = np.array(args.p)

    # each alm file comes with two set of coefficients
    # one for the inner shell boundary, one for the outer
    # the inner is more reliable, I recommend to use that
    alm_i, alm_o = np.loadtxt(args.a, dtype=np.complex_)
    inside = in_sph(point=p, alm=alm_i, center=c)

    if inside:
        print(f'The point (x={p[0]}, y={p[1]}, z={p[2]}) is inside the bubble.')
    else:
        print(f'The point (x={p[0]}, y={p[1]}, z={p[2]}) is not inside the bubble.')
