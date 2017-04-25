# copyright (C) 2017 zxma_sjtu@qq.com

"""
The utils for construcing our convolutional auto-encoder (CAE) model
"""

import os
import pickle
import numpy as np
import scipy.io as sio
from scipy.misc import imread
from skimage import transform

def load_sample(folder, ftype='jpg', savepath=None,
                crop_box=(200, 200), res_box=(50, 50)):
    """
    Load the sample images and reshape to required structure

    input
    =====
    folder: str
        Name of the folder, i.e., the path
    ftype: str
        Type of the images, default as 'jpg'
    savepath: str
        Path to save the reshaped sample mat
        default as None
    crop_box: tuple
        Boxsize of the cropping of the center region
    res_box: tuple
        Scale of the resized image

    output
    ======
    sample_mat: np.ndarray
        The sample matrix
    """
    # Init
    if os.path.exists(folder):
        sample_list = os.listdir(folder)
    else:
        return

    sample_mat = np.zeros((len(sample_list),
                           res_box[0]*res_box[1]))

    # load images
    idx = 0
    for fname in sample_list:
        fpath = os.path.join(folder,fname)
        if fpath.split('.')[-1] == ftype:
            #read image
            img = imread(name=fpath, flatten=True)
            # crop
            rows, cols = img.shape
            row_cnt = int(np.round(rows/2))
            col_cnt = int(np.round(cols/2))
            row_crop_half = int(np.round(crop_box[0]/2))
            col_crop_half = int(np.round(crop_box[1]/2))
            img_crop = img[row_cnt-row_crop_half:
                        row_cnt+row_crop_half,
                        col_cnt-col_crop_half:
                        col_cnt+col_crop_half]
            # resize
            img_rsz = transform.resize(
                img_crop.astype('uint8'),res_box)
            # push into sample_mat
            img_vec = img_rsz.reshape((res_box[0]*res_box[1],))
            sample_mat[idx,:] = img_vec
            idx = idx + 1
        else:
            continue

    # save
    if not savepath is None:
        stype = savepath.split('.')[-1]
        if stype == 'mat':
            # save as mat
            sample_dict = {'data':sample_mat,
                           'name':sample_list}
            sio.savemat(savepath,sample_dict)
        elif stype == 'pkl':
            fp = open(savepath,'wb')
            sample_dict = {'data':sample_mat,
                           'name':sample_list}
            pickle.dump(sample_dict,fp)
            fp.close()

    return sample_mat

