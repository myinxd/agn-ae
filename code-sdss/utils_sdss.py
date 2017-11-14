# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>
"""
Some utilities to handle SDSS data
"""

#import kcorrect as kc
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as au
import math

def calc_maggies(mag, ext=None, band_id=None):
    """Calc flux from magnitude with Asinh calibration"""
    # extinction correction
    if ext is not None:
        mag = mag - ext
    if band_id is not None:
        # AB calibration
        ab_coeff = [-0.04, 0, 0, 0, 0.02]
        mag = mag + mag * ab_coeff[band_id]
        # mag to maggie
        f0 = 3631 #[Jy]
        b_coeff = [1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10]
        b = b_coeff[band_id]
        maggie = math.sinh((mag / (-2.5/math.log(10)) - math.log(b))) * (2*b) * f0
        #maggie = 10 ** (mag / -2.5)
    else:
        maggie = 10 ** (mag / -2.5)
    
    return maggie

def calc_maggies_ivar(maggie, magerr):
    "Calc maggie inverse variance"
    maggie_ivar = (0.4 * math.log(10) * maggie * magerr)**(-2)
    return maggie_ivar

def get_sample_params(mags,exts,magerrs,z):
    """Generate the parameters for kcorrection estimation"""
    param = np.zeros((11,))
    param[0] = z
    # calc maggies
    for i, mag in enumerate(mags):
        param[i+1] = calc_maggies(mag=mag, ext=exts[i], band_id=i)
    # calc maggie_ivar
    for i, magerr in enumerate(magerrs):
        param[i+6] = calc_maggies_ivar(maggie=param[i+1], magerr=magerr)
    
    return param

def calc_reconmag(sample_params):
    """calc reconstructed_mag"""
    kc.load_templates() # load the default template
    kc.load_filters() # load the SDSS filters
    # get the coeffs
    coeff = kc.fit_coeffs(sample_params)
    reconmag = kc.reconstruct_maggies(coeff)
    return reconmag

def calc_reconmag_batch(parampath,coeffpath,reconpath):
    """calc reconstructed_mag"""
    kc.load_templates() # load the default template
    kc.load_filters() # load the SDSS filters
    # get the coeffs
    kc.fit_coeffs_from_file(parampath, outfile=coeffpath)
    kc.reconstruct_maggies_from_file(coeffpath, outfile=reconpath)

def calc_luminosity_distance(redshift):
    """Calculate the rate, kpc/px."""
    # Init
    # Hubble constant at z = 0
    H0 = 71.0
    # Omega0, total matter density
    Om0 = 0.27
    # Cosmo
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    # Angular diameter distance, [Mpc]
    dL = cosmo.luminosity_distance(redshift)

    return dL.to(au.pc)

def calc_absmag(mag_r,dl,kcrt):
    """Calculate the absolute magnitude
    
    Reference
    =========
    [1] k correction
    https://en.wikipedia.org/wiki/K_correction
    """
    mag_abs = mag_r - 5*(math.log10(dl) - 1) - kcrt
    return mag_abs

def flux_to_luminosity(redshift,flux,alpha=2):
    dL = calc_luminosity_distance(redshift=redshift)
    dL = dL.to(au.m)
    flux_mJy = au.mJy*flux # mJy
    flux = flux_mJy.to(au.Jy) # Jy 
    lumo = (flux * 4 * np.pi * dL**2) / (1+redshift)**(1+alpha)
    return lumo.value * 10e-26 #* au.W / (au.m**2 * au.Hz)

def draw_step(nums,bin_edge,step=0.01):
    x = np.arange(bin_edge[0],bin_edge[-1],step)
    y = np.zeros(x.shape)
    for i in range(len(bin_edge)-1):
        idx_l = np.where(x >= bin_edge[i]) 
        idx_r = np.where(x < bin_edge[i+1])
        idx = np.intersect1d(idx_l, idx_r)
        y[idx]= nums[i]/nums.sum()
    
    return x,y
