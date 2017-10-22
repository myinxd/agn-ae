# Copyright (C) 2017 Zhixian MA <zx@mazhixian.me>

import sys
import os
import pickle
import numpy as np

import utils
import ConvNet
import ConvAE_FT

def pos_to_line(label_pos):
    return np.max(label_pos,axis=1)

def main(argv):

    # folder for saving
    subfold = argv[1]
    if not os.path.exists(subfold):
        os.mkdir(subfold)
        os.mkdir(os.path.join(subfold, "est_labeled"))
    # load data
    fname = '../LRG-fits/data/lrg_171020/sample-LRG-120-120-10-c3.pkl'
    with open(fname, 'rb') as fp:
        datadict = pickle.load(fp)
        X_raw = datadict['data']

    # Reshape and generate train and test dataset
    rs = 120
    # normalization and whitening
    '''
    with open("../nets/norm_params.pkl", 'rb') as fp:
        normparam = pickle.load(fp)
    X_max = normparam["X_max"]
    X_min = normparam["X_min"]
    X_mean = normparam["X_mean"]
    X_train_pre = X_raw
    X_train_pre = (X_train_pre - X_min) / (X_max - X_min)
    X_in = X_train_pre.reshape(-1,rs,rs,1)
    X_mean = np.mean(X_train_pre)
    X_w = X_in - X_mean # Whitening?
    '''
    X_w = X_raw.reshape(-1,rs,rs,1).astype('float32')

    numclass = 2
    encode_nodes = 64
    cnn = ConvNet.ConvNet(input_shape=X_w.shape,
                kernel_size=[3,3,3,3,3],
                kernel_num = [8,8,16,32,32],
                fc_nodes=[], encode_nodes=encode_nodes,
                padding=('SAME','SAME'),
                stride=(2,2),
                numclass = numclass,
                sess = None,
                name = None)
    cnn.cae_build()
    cnn.cnn_build(learning_rate=0.001) # In order to init the weights


    foldname = "{0}/net_l1_140".format(subfold)
    name = "net_l1.pkl"
    cnn.sess, cnn.name = utils.load_net(os.path.join(foldname, name))

    # estimate
    label, label_pos = cnn.cnn_predict(img=X_w)
    label_new_pos = pos_to_line(label_pos)

    # save result
    savefold = "{0}/est_labeled".format(subfold)
    if not os.path.exists(savefold):
        os.mkdir(savefold)
    savepath = "est_l1.pkl"
    savedict = {"label_raw": np.array(datadict['type']),
                "z": np.array(datadict['redshift']),
                "snvss": np.array(datadict['snvss']),
                "name": np.array(datadict['name']),
                "label_est": label,
                "label_pos": label_new_pos}
    with open(os.path.join(savefold,savepath), 'wb') as fp:
        pickle.dump(savedict, fp)

if __name__ == "__main__":
    main(sys.argv)
