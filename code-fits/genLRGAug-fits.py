# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Do augmentations

update
======
[2017-10-08]: Add training and test
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
    foldname = "../LRG/combine_fits"

    # load sample list
    clsdict = pd.read_excel("../LRG-filter/LRG_combine_filtered.xlsx")
    samplelist = np.array(clsdict["Name-J2000"])
    sampletypes = np.array(clsdict["Type"])

    numsamples = len(samplelist)

    # training and test
    idx = np.random.permutation(numsamples)
    test_rate = 0.1
    numtest = int(numsamples * test_rate)
    numtrain = numsamples - numtest
    # test
    idx_test = idx[0:numtest]
    name_test = samplelist[idx_test]
    label_test = sampletypes[idx_test]
    # train
    idx_train = idx[numtest:]
    name_train = samplelist[idx_train]
    # print(name_train)
    label_train = sampletypes[idx_train]

    num_train = [len(np.where(label_train == 1)[0]),
                 len(np.where(label_train == 2)[0]),
                 len(np.where(label_train == 3)[0]),
                 len(np.where(label_train == 4)[0]),
                 len(np.where(label_train == 5)[0]),
                 len(np.where(label_train == 6)[0]),
                 len(np.where(label_train == 7)[0]),
                ]

    # parameters
    # num_samples = np.array([392, 284, 580, 430, 100, 65, 7, 37])
    num_aug = np.array([64, 28, 17, 20, 37, 94, 17])
    mask = np.array([1,1,1,1,1,1,1])

    num_aug = num_aug * mask

    # crop
    # crop_box = (120,120)
    # rez_box = (120,120)
    crop_box = (120,120)
    rez_box = (120,120)
    clipflag = True
    clipparam = [3, 50]

    # train
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("[%s] Totally %d samples, and augmentation for each." % (t, numsamples) )
    print("[%s] Number of train %d and Number of test %d" % (t,numtrain, numtest))
    # Do augmentation
    samples = np.zeros(((num_aug*num_train).sum(),
                           rez_box[0]*rez_box[1]))
    labels = np.zeros(((num_aug*num_train).sum(),))
    # names = np.zeros(((num_aug*num_train).sum(),))
    begin_idx = 0
    for i in range(numtrain):
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        f = name_train[i].split(" ")[0]+".fits"
        f_t = int(label_train[i])
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
        samples[begin_idx:end_idx, :] = img_aug.reshape(num_aug[f_t-1],
                                                               rez_box[0]*rez_box[1])
        # names[begin_idx:end_idx] = name_train[i]
        labels[begin_idx:end_idx] = f_t - 1
        begin_idx = end_idx

    # save result
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print("[%s] Saving results ..." % (t))
    # split for saving result
    num_tot = (num_train * num_aug).sum()
    num_per_grp = 20000
    num_grp = math.ceil(num_tot / num_per_grp)

    savefolder = "../LRG-fits/data/lrg_171020"
    if not os.path.exists(savefolder):
        os.mkdir(savefolder)
    for i in range(num_grp):
        savepath = os.path.join(savefolder, "sample-lrg-train-120-120-c3-gr{0}.pkl".format(i))
        with open(savepath, 'wb') as fp:
            sample_dict = {"data": samples[i*num_per_grp:(i+1)*num_per_grp,:],
                           "label":labels[i*num_per_grp:(i+1)*num_per_grp]}
            pickle.dump(sample_dict, fp)
        print("I am sleeping...")
        time.sleep(5)

    # test
    num_aug = 10
    samples = np.zeros(((num_aug*numtest),
                           rez_box[0]*rez_box[1]))
    for i in range(numtest):
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        f = name_test[i].split(" ")[0]+".fits"
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

    savefolder = "../LRG-fits/data/lrg_171020"
    savepath = os.path.join(savefolder,"sample-lrg-test-120-120-c3.pkl")
    with open(savepath, 'wb') as fp:
        sample_dict = {"data": samples,
                       "name": name_test,
                       "label": label_test}
        pickle.dump(sample_dict, fp)

if __name__ == "__main__":
    main()

