# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Do augmentations on the labeled samples, augmentation is optional.
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
    # from folder
    foldname = "../images/"

    # load sample list
    sampleinfo = pd.read_csv("../sample-labeled-list.csv", sep=" ")
    samplelist = sampleinfo["Name"]
    sampletypes = sampleinfo["Type"]
    numsamples = len(samplelist)
    # parameters
    num_samples = [284, 585, 431, 392, 37, 100, 65]
    # num_aug = np.array([6, 3, 3, 4, 45, 15, 24])
    num_aug = np.array([8, 4, 6, 6, 0, 24, 36])
    mask = np.array([1,1,1,1,0,1,1])

    num_aug = num_aug * mask

    # crop
    crop_box = (120,120)
    rez_box = (120,120)
    clipflag = True
    clipparam = [3, 100]

    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("[%s] Totally %d samples, and augmentations for each type." % (t, numsamples))
    # Do augmentation
    samples = np.zeros(((num_aug*num_samples).sum(),
                           rez_box[0]*rez_box[1]))
    labels = np.zeros(((num_aug*num_samples).sum(),))
    begin_idx = 0
    for i in range(numsamples):
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        f = samplelist[i]+".jpeg"
        f_t = int(sampletypes[i])
        print("[%s] Processing on %s of type %d..." % (t, f, f_t))
        img = os.path.join(foldname, f)
        # augmentation
        img_aug = utils.get_augmentation(img=img,
                                         crop_box=crop_box,
                                         rez_box = rez_box,
                                         num_aug = num_aug[f_t-1],
                                         clipflag = clipflag,
                                         clipparam = clipparam)
        end_idx = begin_idx + num_aug[f_t-1]
        samples[begin_idx : end_idx, :] = img_aug.reshape(num_aug[f_t-1],
                                                          rez_box[0]*rez_box[1])
        labels[begin_idx : end_idx] = f_t - 1 # Begin from zero...
        begin_idx = end_idx

    # save result
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("[%s] Saving results ..." % (t))
    # split for saving result
    num_tot = (num_samples * num_aug).sum()
    num_per_grp = 20000
    num_grp = math.ceil(num_tot / num_per_grp)

    savefold = "../sample_170927"
    try:
        os.mkdir(savefold)
    except:
        pass
    for i in range(num_grp):
        savepath = os.path.join(savefold,"sample-labeled-120-c3-sel-10x-gr{0}.pkl".format(i))
        with open(savepath, 'wb') as fp:
            sample_dict = {"data": samples[i*num_per_grp:(i+1)*num_per_grp,:],
                           "label": labels[i*num_per_grp:(i+1)*num_per_grp]}
            pickle.dump(sample_dict, fp)
        print("I am sleeping...")
        time.sleep(5)

if __name__ == "__main__":
    main()
