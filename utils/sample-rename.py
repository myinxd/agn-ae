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

def batch_rename_csv(listpath, batch, fromfolder,savefolder):
    """Batchly rename the samples

    Inputs
    ======
    listpath: str
        The path of the data list
    batch: tuple
        The region of indices w.r.t. samples to be fetched.
    fromfolder: str
        Folder saved the samples to be renamed
    savefolder: str
        Folder to save the fetched sample files
    """
    from pandas import read_csv
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import time

    # load csv
    f = read_csv(listpath, sep=' ')
    ra = f['RAJ2000'] # RA
    dec = f['DEJ2000'] # DEC
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
        ra_h = str(int(ra_rms[0]))
        ra_m = str(int(ra_rms[1]))
        ra_s = str(np.round(ra_rms[2]*1000)/1000)
        de_d = str((dec_dms[0]))
        de_m = str(int(np.abs(dec_dms[1])))
        de_s = str(np.abs(np.round(dec_dms[2]*1000)/1000))
        # download file
        fname_from = 'J' + ''.join([ra_h,ra_m,ra_s,de_d,de_m,de_s]) + '.fits'
        frompath = os.path.join(fromfolder,fname_from)
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
        fname_save = 'J' + ''.join([ra_h,ra_m,ra_s,de_d,de_m,de_s]) + '.fits'
        savepath = os.path.join(savefolder,fname_save)
        try:
            print("[%s] f: %s\t s: %s" % (t, fname_from, fname_save))
            os.system("cp %s %s" % (frompath, savepath))
        except:
            fl.write("%d: %s" % (i, fname))
            continue
        # print log
        # print('[%s]: Fetching %s' % (t, fname))
    fl.close()

def main():
    # Init
    parser = argparse.ArgumentParser(description="Rename FIRST observations.")
    # Parameters
    # parser.add_argument("url", help="URL of the archive'")
    parser.add_argument("listpath", help="Path of the sample list.")
    parser.add_argument("batchlow", help="Begin index of the batch.")
    parser.add_argument("batchhigh",help="End index of the batch.")
    parser.add_argument("fromfolder", help="Path saved samples to be renamed.")
    parser.add_argument("savefolder", help="The folder to save files.")
    args = parser.parse_args()

    listpath = args.listpath
    batch = [int(args.batchlow),int(args.batchhigh)]
    savefolder = args.savefolder
    fromfolder = args.fromfolder

    if not os.path.exists(savefolder):
        os.mkdir(savefolder)

    batch_rename_csv(listpath=listpath,
                     batch=batch,
                     fromfolder=fromfolder,
                     savefolder=savefolder)

if __name__ == "__main__":
    main()
