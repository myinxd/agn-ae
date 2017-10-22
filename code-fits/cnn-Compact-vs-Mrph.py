# Copyright (C) 2017 Zhixian MA <zx@mazhixian.me>

import numpy as np
import pickle
import time
import os

import sys
sys.setrecursionlimit(1000000)

# from ConvAE_FT import ConvAE
import ConvAE_FT
import ConvNet
import utils

def sub2triple(data,label,mask):
    """Subtypes to triple categories according to provided mask."""
    label_bin = label
    for i,m in enumerate(mask):
        label_bin[np.where(label == i)] = m
    # remove samples of label larger than 1, i.e., discarded samples
    idx = np.where(label_bin <= 2)[0]
    return data[idx,:],label[idx]

def vec2onehot(label,numclass):
    label_onehot = np.zeros((len(label),numclass))
    for i,l in enumerate(label):
        label_onehot[i, int(l)] = 1

    return label_onehot

def main(argv):
    # folder for saving
    subfold = argv[1]
    if not os.path.exists(subfold):
        os.mkdir(subfold)
        os.mkdir(os.path.join(subfold,"features"))

    # load data
    X_cnn_raw = []
    labels_cnn_raw = []
    num_grp = 3
    for i in range(num_grp):
        fname = '../LRG-fits/data/lrg_171020/sample-lrg-train-120-120-c3-gr{0}.pkl'.format(i)
        with open(fname, 'rb') as fp:
            datadict = pickle.load(fp)
            X_cnn_raw.append(datadict['data'])
            labels_cnn_raw.append(datadict['label'])
        time.sleep(3)

    X_test = []
    labels_test = []
    fname = '../LRG-fits/data/lrg_171020/sample-lrg-test-120-120-c3.pkl'
    with open(fname, 'rb') as fp:
        datadict = pickle.load(fp)
        X_test.append(datadict['data'])
        labels_test.append(datadict['label'])

    # Combine and normalization
    sample_mat = np.vstack(X_cnn_raw)
    del(X_cnn_raw)
    labels_cnn = np.hstack(labels_cnn_raw)
    del(labels_cnn_raw)

    # sample_mat = np.nan_to_num(sample_mat)

    rs = 120
    '''
    with open("../nets/norm_params.pkl", 'rb') as fp:
        normparam = pickle.load(fp)
    X_max = normparam["X_max"]
    X_min = normparam["X_min"]
    X_mean = normparam["X_mean"]
    X_train_cnn = (sample_mat - X_min) / (X_max - X_min)
    # X_norm = sample_mat
    X_w_cnn = X_train_cnn - X_mean
    X_tr_cnn = X_w_cnn.reshape(-1, rs, rs, 1).astype('float32')
    '''
    X_tr_cnn = sample_mat.reshape(-1,rs,rs,1).astype('float32')

    idx = np.random.permutation(len(labels_cnn))
    numsamples = 100000
    X_in = X_tr_cnn[idx[0:numsamples],:,:,:]
    # get labels
    X_out = labels_cnn[idx[0:numsamples]].astype('int32')

    mask_layer1 = [0,1,1,1,1,1,1,100]
    data_layer1,label_layer1 = sub2triple(data=X_in, label=X_out, mask=mask_layer1)
    label_layer1_hotpot = vec2onehot(label=label_layer1, numclass=2)

    numclass = 2
    encode_nodes = 64
    cnn = ConvNet.ConvNet(input_shape=data_layer1.shape,
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


    foldname = "./nets/pretrain-171020-2cls/"
    name = "pretrain-120-171020-2cls.pkl"
    cnn.sess, cnn.name = utils.load_net(os.path.join(foldname, name))

    # train
    num_epochs = 100
    learning_rate = 0.001
    batch_size = 100
    droprate = 0.5
    cnn.cnn_train(data=data_layer1, label=label_layer1_hotpot, num_epochs=num_epochs, learning_rate=learning_rate,
                batch_size=batch_size, droprate=droprate)

    # save features
    fname = "code_l1.pkl"
    folder = "{0}/features/".format(subfold)
    if not os.path.exists(folder):
        os.mkdir(folder)
    numsample = data_layer1.shape[0]
    numone = numsample // 10
    code = np.zeros((numsample, encode_nodes))
    for i in range(10):
        code[i*numone:(i+1)*numone] = cnn.cae_encode(data_layer1[i*numone:(i+1)*numone,:,:,:])
    # code = cnn.cae_encode(data_layer1)
    label = label_layer1
    with open(os.path.join(folder, fname), 'wb') as fp:
        code_dict = {"code": code,
                    "label": label}
        pickle.dump(code_dict, fp)

    # save net
    foldname = "{0}/net_l1_140".format(subfold)
    name = "net_l1.pkl"
    netname = "model-l1.ckpt"
    if os.path.exists(foldname):
        os.system("rm -r %s" % (foldname))
    os.mkdir(foldname)
    cnn.cnn_save(namepath=os.path.join(foldname, name),
                netpath=os.path.join(foldname, netname))

if __name__ == "__main__":
    main(sys.argv)
