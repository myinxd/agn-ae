# Copyright (C) 2017 Zhixian MA <zx@mazhixian.me>

import numpy as np
import pickle
import time
import os

import sys
sys.setrecursionlimit(1000000)

import ConvAE_FT
import ConvNet
import utils

def main():
    # load data
    X_raw = []
    num_grp = 8
    for i in range(num_grp):
        fname = '../BH12/data/BH12_171020/sample-BH-120-120-10-c3-gr{0}.pkl'.format(i)
        with open(fname, 'rb') as fp:
            datadict = pickle.load(fp)
            X_raw.append(datadict['data'])
        time.sleep(3)

    # Combine and normalization
    X_pre = np.vstack(X_raw)
    del(X_raw)

    '''
    # Reshape and generate train and test dataset
    rs = 120
    # normalization and whitening
    X_min = X_pre.min()
    X_max = X_pre.max()
    X_train_pre = (X_pre - X_min) / (X_max - X_min)
    X_in = X_train_pre.reshape(-1,rs,rs,1)
    X_mean = np.mean(X_train_pre)
    X_tr = X_in - X_mean # Whitening?

    # save PS
    normdict = {"X_min": X_min,
                "X_max": X_max,
                "X_mean": X_mean}
    with open("./nets/norm_params.pkl", 'wb') as fp:
        pickle.dump(normdict,fp)
    '''
    rs = 120
    X_tr = X_pre.reshape(-1,rs,rs,1).astype('float32')

    # Construct the network
    numclass = 3
    encode_nodes = 64
    cae = ConvAE_FT.ConvAE(input_shape=X_tr.shape,
                kernel_size=[3,3,3,3,3],
                kernel_num = [8,8,16,32,32],
                fc_nodes=[], encode_nodes=encode_nodes,
                padding=('SAME','SAME'),
                stride=(2,2),
                numclass = numclass)
    cae.cae_build()
    cae.cnn_build(learning_rate=0.001) # In order to init the weights

    # train
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 100
    droprate = 0.5
    cae.cae_train(data=X_tr, num_epochs=num_epochs, learning_rate=learning_rate,
                batch_size=batch_size, droprate=droprate)

    # save the pre-trained net
    foldname = "./nets/pretrain-171020-3cls"
    name = "pretrain-120-171020-3cls.pkl"
    netname = "model-pretrain-120-3cls.ckpt"
    if os.path.exists(foldname):
        os.system("rm -r %s" % (foldname))
    os.mkdir(foldname)
    cae.cae_save(namepath=os.path.join(foldname, name),
                netpath=os.path.join(foldname, netname))

if __name__ == "__main__":
    main()
