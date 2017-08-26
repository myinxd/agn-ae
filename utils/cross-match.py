# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Do cross mathcing between labeled samples and the unlabled ones.

methods
=======
icrs2galactic: J2000 sky coordinate to splitted coordinate
galactic2icrs: splitted to J2000 sky coordinate
save_result: save the result to txt
"""

import os
import numpy as np
import argparse
import time
from astropy import units as u
from astropy.coordinates import SkyCoord
from pandas import read_csv
from pandas import read_excel

def coord2split(ra,dec):
    """
    Transform coordinates from icrs to galactic

    inputs
    ======
    ra: float
    dec: float

    outputs
    =======
    coord_str
    """
    c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    ra_rms = tuple(c.ra.hms)
    dec_dms = tuple(c.dec.dms)
    ra_h = "%02d" % (int(ra_rms[0]))
    ra_m = "%02d" % (int(ra_rms[1]))
    ra_s_i = np.fix(np.round(ra_rms[2]*100)/100)
    ra_s_f = np.round(ra_rms[2]*100)/100 - ra_s_i
    ra_s = "%02d.%02d" % (int(ra_s_i),int(ra_s_f*100))
    if dec_dms[0] > 0:
        de_d = "+%02d" % (int(dec_dms[0]))
    else:
        de_d = "-%02d" % (abs(int(dec_dms[0])))
    de_m = "%02d" % (abs(int(dec_dms[1])))
    de_s_i = np.fix(np.abs(np.round(dec_dms[2]*10)/10))
    de_s_f = np.abs(np.round(dec_dms[2]*10)/10) - de_s_i
    de_s = "%02d.%01d" % (int(de_s_i),np.round(de_s_f*10))
    coord_str = 'J' + ''.join([ra_h,ra_m,ra_s,de_d,de_m,de_s]) + '.fits'

    return coord_str

def split2coord(ra_hms, dec_dms):
    """
    Transform from icrs to galactic

    inputs
    ======
    ra_hms: str
    dec_dms: str

    outputs
    =======
    ra: float
    dec: float
    """
    c = SkyCoord(ra=ra_hms, dec=dec_dms, frame='icrs')
    return c.ra.value, c.dec.value

def save2csv(samplelist,csvname):
    """
    Save result to csv

    inputs
    ======
    samplelist: dict
        The cross matched samplelist
        {"labeled": [], "unlabeled": [], "RA": [], "DEC": [], "idx": []}

    csvname: str
        path to save the result
    """
    from pandas import DataFrame
    numsamples = len(samplelist['labeled'])
    sample_frame = DataFrame(samplelist, index=range(0,numsamples))
    # write
    sample_frame.to_csv(csvname, sep=' ')

def findmatch(s_match,s_all, bias=0.001):
    """
    Find matched samples

    s_match: tuple
        icrs coordinate of the sample to be matched, (ra, dec)
    s_all: list
        The list of all the unlabeled samples
    bias: float
        The largest bias between two matched sample
    """
    s_ra = s_all[0,:] - s_match[0]
    s_dec = s_all[1,:] - s_match[1]
    s_dist = np.sqrt(s_ra*s_ra + s_dec*s_dec)
    s_dist_min = s_dist.min()

    if s_dist_min > bias:
        s_idx = [-1]
        fname_all = "nomatching"
    else:
        s_idx = np.where(s_dist == s_dist_min)[0]
        fname_all = coord2split(ra=s_all[0, s_idx],dec=s_all[1,s_idx[0]])

    # get coord strings
    fname_match = coord2split(ra=s_match[0],dec=s_match[1])

    return [fname_match, fname_all, s_match[0], s_match[1], s_idx[0]]


def main():
    # Init
    parser = argparse.ArgumentParser(description="Cross matching samples")
    # parameters
    parser.add_argument("pathmatch",help="Path of samples to be cross matched.")
    parser.add_argument("pathall",help="path of the samplelist.")
    parser.add_argument("csvpath",help="path to save the matched result.")
    args = parser.parse_args()

    pathmatch = args.pathmatch
    pathall = args.pathall
    csvpath = args.csvpath

    # sample_frame
    sample_frame = {"labeled": [],
                    "unlabeled": [],
                    "RA": [],
                    "DEC": [],
                    "idx": []}


    # read list of all
    listall = read_csv(pathall, sep=' ')
    s_all_ra = listall['RAJ2000']
    s_all_dec = listall['DEJ2000']
    s_all = np.vstack([s_all_ra, s_all_dec])
    print(s_all.shape)

    # read list of samples to be matched
    # FRICAT
    """
    f = read_excel(pathmatch)
    samples = f.get_values()
    ra = samples[:,1]
    for i in range(len(ra)):
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        # get parameters
        s_ra = "%sh%sm%ss" % (ra[i][2:4], ra[i][4:6], ra[i][6:11])
        s_dec= "%sd%sm%ss" % (ra[i][11:14],ra[i][14:16],ra[i][16:20])
        print("[%s] Matching sample locates at %s\t%s" % (t,s_ra,s_dec))
        match_str = findmatch(s_match=split2coord(s_ra,s_dec), s_all=s_all, bias=5)
        sample_frame["labeled"].append(match_str[0])
        sample_frame["unlabeled"].append(match_str[1])
        sample_frame["RA"].append(match_str[2])
        sample_frame["DEC"].append(match_str[3])
        sample_frame["idx"].append(match_str[4])
    """

    # FRIICAT
    """
    fp = open(pathmatch, 'r')
    samples = fp.readlines()
    for i in range(len(samples)):
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        s = samples[i].split(" ")
        ra = s[1]
        # get parameters
        s_ra = "%sh%sm%ss" % (ra[1:3], ra[3:5], ra[5:10])
        s_dec= "%sd%sm%ss" % (ra[10:13],ra[13:15],ra[15:19])
        print("[%s] Matching sample locates at %s\t%s" % (t,s_ra,s_dec))
        match_str = findmatch(s_match=split2coord(s_ra,s_dec), s_all=s_all, bias=5)
        sample_frame["labeled"].append(match_str[0])
        sample_frame["unlabeled"].append(match_str[1])
        sample_frame["RA"].append(match_str[2])
        sample_frame["DEC"].append(match_str[3])
        sample_frame["idx"].append(match_str[4])
    """
    # X-shaped
    f = read_excel(pathmatch)
    samples = f.get_values()
    ra = samples[:,1]
    dec = samples[:,2]
    for i in range(len(ra)):
        t = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        RA = ra[i][1:-2].split(' ')
        DEC = dec[i][1:-2].split(' ')
        # get parameters
        if DEC[0][0] == '−':
            DEC[0] = '-'+DEC[0][1:]
        s_ra = "%sh%sm%ss" % (RA[0], RA[1], RA[2])
        s_dec= "%sd%sm%ss" % (DEC[0], DEC[1], DEC[2])
        print("[%s] Matching sample locates at %s\t%s" % (t,s_ra,s_dec))
        match_str = findmatch(s_match=split2coord(s_ra,s_dec), s_all=s_all, bias=5)
        sample_frame["labeled"].append(match_str[0])
        sample_frame["unlabeled"].append(match_str[1])
        sample_frame["RA"].append(match_str[2])
        sample_frame["DEC"].append(match_str[3])
        sample_frame["idx"].append(match_str[4])



    # save
    save2csv(sample_frame, csvpath)

if __name__ == "__main__":
    main()


