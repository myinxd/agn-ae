# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Do augmentations

update
======
[2017-10-08]: Add AGN restriction
"""
import numpy as np
import pickle
import time
import os
import math
import pandas as pd

import sys
sys.setrecursionlimit(1000000)

import utils

def main():
    # save folder
    foldname = "../BH12/BH12_fits"

    # load sample list
    clsdict = pd.read_excel("../BH12/BH12_AGN.xlsx")
    samplelist = clsdict["Name-J2000"]
    redshift = clsdict["z"]
    snvss = clsdict["S_NVSS"]

    numsamples = len(samplelist)

    # parameters
    num_aug = 1

    # crop
    # crop_box = (120,120)
    # rez_box = (120,120)
    crop_box = (120, 120)
    rez_box = (120, 120)
    clipflag = True
    clipparam = [3, 50]

    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("[%s] Totally %d samples, and %d augmentation for each." % (t, numsamples, num_aug) )
    # Do augmentation
    samples = np.zeros((num_aug*numsamples,
                           rez_box[0]*rez_box[1]))
    for i in range(numsamples):
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        f = samplelist[i]+".fits"
        print("[%s] Processing on %s..." % (t, f))
        img = os.path.join(foldname, f)
        # augmentation
        img_aug = utils.get_augmentation_single(img=img,
                                         crop_box=crop_box,
                                         rez_box = rez_box,
                                         num_aug = num_aug,
                                         clipflag = clipflag,
                                         clipparam = clipparam)
        samples[i*num_aug: (i+1)*num_aug, :] = img_aug.reshape(num_aug,
                                                               rez_box[0]*rez_box[1])

    # save result
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("[%s] Saving results ..." % (t))

    savefolder = "../BH12/data/BH12_171017"
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    savepath = os.path.join(savefolder,"sample-BH-120-120-10-c3.pkl")
    with open(savepath, 'wb') as fp:
        sample_dict = {"data": samples,
                       "name": samplelist,
                       "redshift": redshift,
                       "snvss": snvss}
        pickle.dump(sample_dict, fp)

if __name__ == "__main__":
    main()

