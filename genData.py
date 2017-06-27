# Copyright (C) 2017 Zhixian mA <zxma_sjtu@qq.com>

"""
Generate the samples for training the network, the raw image are simulated by
Weitian Li.

Two main steps
1. load the large 600 x 600 image
2. split it into patches
3. augmentation <TODO>

"""

import os
import sys
import numpy as np
import argparse
import pickle
from astropy.io import fits

import utils

sys.setrecursionlimit(1000000)

def main():
    """The main function"""

    # Init
    parser = argparse.ArgumentParser(
                    description="Generate samples for cae.")
    # Parameters
    parser.add_argument("foldname", help="Path of the fits samples.")
    parser.add_argument("suffix", help="The suffix of files to eliminate.")
    parser.add_argument("boxsize", help="Size of the patch")
    parser.add_argument("stride", help="Number of stride pixels")
    parser.add_argument("savepath", help="Path to save the data")

    args = parser.parse_args()

    foldname = args.foldname
    suffix = args.suffix
    savepath = args.savepath
    boxsize = int(args.boxsize)
    stride = int(args.stride)

    img_stack = None
    # Load filenames
    try:
        filelist = os.listdir(foldname)
        filelist.sort()
    except:
        return

    # load image and split
    for f in filelist:
        print(f.split('.')[-1])
        if f.split('.')[-1] == 'fits':
            if f.split('.')[-2] == suffix:
                continue
            else:
                print("Processing on %s" % f)
                fname = os.path.join(foldname, f)
                with fits.open(fname) as hdu:
                    img = hdu[0].data
                    img = np.squeeze(img)
                # split
                img_sub = utils.gen_splits(img, boxsize=boxsize, stride=stride)

            # stack
            if img_stack is None:
                img_stack = img_sub
            else:
                img_stack = np.vstack((img_stack, img_sub))
        else:
            continue

    # save
    with open(savepath, 'wb') as fp:
        sample_dict = {'data': img_stack,
                        'boxsize': boxsize,
                        'stride': stride,
                        'foldname':foldname}
        pickle.dump(sample_dict, fp)

if __name__ == "__main__":
    main()
