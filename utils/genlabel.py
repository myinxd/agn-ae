# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Save pairs of sample name and its label

Label
=====
I: FRI 
II: FRII
C*: Compact source
S* CCS steep sources
U: Uncertain sources
Iw, IIw: Wide Angle Tail
Ic, IIc: core-jet sources
X: X shaped samples
B: Bent-tailed samples, WAT, NAT or Head-tail
R: Ring samples
"""

import os
import pickle
import argparse
import time
import numpy as np


# label dictionary
label_dict = {"I":  1,
              "Iw": 3,
              "Ic": 4,
              "II": 2,
              "IIw": 3,
              "IIc": 4,
              "C*": 5,
              "C":  5,
              "S*": 6,
              "S":  6,
              "U":  7,
              "X":  8,
              "B":  3,
              "R":  9,
              }

label = {}
id_len = 3

def main():
    # Init
    parser = argparse.ArgumentParser(description="File renaming and label generating.")
    # parameters
    parser.add_argument("foldname", help="Path to save the sample files.")
    parser.add_argument("listpath", help="Path of the sample infos.")
    parser.add_argument("labelpath", help="Path to save the labels")
    args = parser.parse_args()

    foldname = args.foldname
    listpath = args.listpath
    labelpath = args.labelpath

    # load list
    with open(listpath, 'r') as f:
        samples = f.readlines()

    for i in range(len(samples)):
        # timestamp
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        # get params
        s_id = "0"*(id_len-len(str(i))) + str(i)
        sample_line = samples[i].split(' ')
        RA_h = "%02d" % int(sample_line[4])
        RA_m = "%02d" % int(sample_line[5])
        RA_s = float(sample_line[6])
        RA_s_i = np.fix(np.round(RA_s*100)/100)
        RA_s_f = np.round(RA_s*100)/100 - RA_s_i
        RA_s_new = "%02d.%02d" % (int(RA_s_i), int(RA_s_f*100))
        DEC_d = sample_line[8]
        if DEC_d[0] == '+':
            DEC_d = "+%02d" % (int(DEC_d[1:]))
        else:
            DEC_d = "-%02d" % (int(DEC_d[1:]))
        DEC_m = "%02d" % int(sample_line[9])
        DEC_s = float(sample_line[10])
        DEC_s_i = np.fix(np.round(DEC_s*10)/10)
        DEC_s_f = np.round(DEC_s*10)/10 - DEC_s_i
        DEC_s_new = "%02d.%01d" % (int(DEC_s_i), int(DEC_s_f*10))
        fname = 'J' + ''.join([RA_h, RA_m, RA_s_new, DEC_d, DEC_m, DEC_s_new])
        # get labels
        s_label = "R"
        label.update({fname: [s_id, label_dict[s_label]]})
        # print log
        print("[%s] processing on sample %s" % (t, fname))

    # save labels
    # labelpath = os.path.join(foldsave, "label.pkl")
    with open(labelpath,'wb') as fp:
        pickle.dump(label, fp)

if __name__ == "__main__":
    main()
