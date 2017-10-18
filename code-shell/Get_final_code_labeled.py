# Copyright (C) 2017 Zhixian MA <zx@mazhixian.me>

# Final estimation
# 1. load all estimated labels w.r.t. nodes
# 2. combine to get the code
# 3. save as a table

import sys
import os
import pickle
import numpy as np

def main(argv):
    # folder for saving
    subfold = argv[1]
    foldidx = argv[2]
    if not os.path.exists(subfold):
        os.mkdir(subfold)
    # load labels
    foldname = "{0}/est_labeled".format(subfold)
    fnames = ['est_l1.pkl','est_l2.pkl','est_l3.pkl']

    # l1
    with open(os.path.join(foldname, fnames[0]), 'rb') as fp:
        dict_l1 = pickle.load(fp)

    # l2
    with open(os.path.join(foldname, fnames[1]), 'rb') as fp:
        dict_l2 = pickle.load(fp)

    # l3
    with open(os.path.join(foldname, fnames[2]), 'rb') as fp:
        dict_l3 = pickle.load(fp)

    # Judge
    numsample = len(dict_l1['label_pos'])
    codes = []
    types = []
    pos = []
    pos_l1 = dict_l1['label_pos']
    pos_l2 = dict_l2['label_pos']
    pos_l3 = dict_l2['label_pos']
    for i in range(numsample):
        c = "".join([str(dict_l1['label_est'][i]),
                    str(dict_l2['label_est'][i]),
                    str(dict_l3['label_est'][i]),
                    ])
        if c[0] == '0':
            t = 1
            pos.append(pos_l1[i])
        elif c[0:2] == "10":
            t = 2
            pos.append(pos_l1[i]*pos_l2[i])
        elif c[0:2] == "11":
            t = 3
            pos.append(pos_l1[i]*pos_l2[i])
        elif c[0:3] == "120":
            t = 4
            pos.append(pos_l1[i]*pos_l2[i]*pos_l3[i])
        elif c[0:3] == "121":
            t = 5
            pos.append(pos_l1[i]*pos_l2[i]*pos_l3[i])
        elif c[0:3] == "122":
            t = 6
            pos.append(pos_l1[i]*pos_l2[i]*pos_l3[i])
        else:
            t = 9
            pos.append(pos_l1[i]*pos_l2[i]*pos_l3[i])
        codes.append(c)
        types.append(t)

    label_real = dict_l1["label_raw"]
    z = dict_l1["z"]
    snvss = dict_l1['snvss']
    name = dict_l1['name']

    from pandas import DataFrame
    labeldict = {"Real_types": label_real, "code":codes,"types":types,
                "z": z, "S_NVSS": snvss, "Name-J2000": name, "Possibility": pos,
                "Poss_l1": pos_l1, "Poss_l2": pos_l2, "Poss_l3": pos_l3}
    dframe = DataFrame(labeldict)
    dframe.to_excel("%s/code_LRG_%s.xlsx" % (subfold,foldidx))

if __name__ == "__main__":
    main(sys.argv)
