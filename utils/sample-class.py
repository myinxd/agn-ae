# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Rename samples of Best into the JHHMMSS.ss+/-DDMMSS.s style

Reference
=========
[1] math.modf
    http://www.runoob.com/python/func-number-modf.html
"""

import os
import math
import numpy as np
import time
import argparse
import pickle

def batch_class_csv(listpath,batch):
    """Batchly rename the samples

    Inputs
    ======
    listpath: str
        The path of the data list
    """
    from pandas import read_csv
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import time

    sample_cls = {"1": [],
                  "2": [],
                  "3": [],
                  "4": []}
    names = []
    # load csv
    f = read_csv(listpath, sep=' ')
    ra = f['RAJ2000'] # RA
    dec = f['DEJ2000'] # DEC
    rcl = f['RCl']
    # regularize the batch
    if batch[1] > len(f):
        batch[1] = len(f)
    # log file optional
    fl = open('log.txt', 'a')
    # Iteration body
    for i in range(batch[0], batch[1]+1):
        # timestamp
        t = time.strftime('%Y-%m-%d',time.localtime(time.time()))
        # get params
        temp_c = SkyCoord(ra=ra[i]*u.degree, dec=dec[i]*u.degree, frame='icrs')
        # Coordinate transform
        ra_rms = tuple(temp_c.ra.hms)
        dec_dms = tuple(temp_c.dec.dms)
        # save name
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
        fname = 'J' + ''.join([ra_h,ra_m,ra_s,de_d,de_m,de_s]) + '.fits'
        names.append(fname)
        try:
            print("[%s] s: %s type: %s" % (t, fname, rcl[i]))
            sample_cls[str(rcl[i])].append(i)
        except:
            fl.write("%d: %s" % (i, fname))
            continue
        # print log
        # print('[%s]: Fetching %s' % (t, fname))
    fl.close()

    return sample_cls,names

def main():
    # Init
    parser = argparse.ArgumentParser(description="Rename FIRST observations.")
    # Parameters
    # parser.add_argument("url", help="URL of the archive'")
    parser.add_argument("listpath", help="Path of the sample list.")
    args = parser.parse_args()

    listpath = args.listpath
    batch = [0,18284]


    sample_cls,names = batch_class_csv(listpath=listpath,
                     batch=batch)

    with open("../sample_cls_names.pkl",'wb') as fp:
        pickle.dump({"names":names, "labels":sample_cls},fp)

if __name__ == "__main__":
    main()
