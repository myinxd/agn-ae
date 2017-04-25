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

def gen_sample(folder, ftype='jpg', savepath=None,
                crop_box=(200, 200), res_box=(50, 50)):
    """
    Read the sample images and reshape to required structure

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
                img_crop/255,res_box,mode='reflect')
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

def load_sample(samplepath):
    """Load the sample matrix

    input
    =====
    samplepath: str
        Path to save the samples
    """
    ftype = samplepath.split('.')[-1]
    if ftype == 'pkl':
        try:
            fp = open(samplepath, 'rb')
        except:
            return None
        sample_dict = pickle.load(fp)
        sample_mat = sample_dict['data']
        sample_list = sample_dict['name']
    elif ftype == 'mat':
        try:
            sample_dict = sio.loadmat(samplepath)
        except:
            return None
        sample_mat = sample_dict['data']
        sample_list = sample_dict['name']

    return sample_mat, sample_list

def get_predict(cae,img):
    """
    Predict the output of the input image

    input
    =====
    img: np.ndarray
        The image matrix, (r,c)

    output
    ======
    img_pred: np.ndarray
        The predicted image matrix
    """
    if img.dtype != 'float32':
        img = img.astype('float32')

    if len(img.shape) == 4:
        rows = img.shape[2]
        cols = img.shape[3]
    elif len(img.shape) == 3:
        rows = img.shape[1]
        cols = img.shape[2]
        img = img.reshape(img.shape[0],1,rows,cols)
    elif len(img.shape) == 2:
        rows,cols = img.shape
        img = img.reshape(1,1,rows,cols)
    else:
        print("The shape of image should be 2 or 3 d")
    img_pred = cae.predict(img).reshape(-1, rows, cols)
    img_pred = np.rint(256. * img_pred).astype(int)
    img_pred = np.clip(img_pred, a_min=0, a_max=255)
    img_pred = img_pred.astype('uint8')
    

    return img_pred

def get_encode(cae, img):
    """Encode or compress on the sample

    input
    =====
    img: np.ndarray
        The sample matrix

    output
    ======
    img_en: np.ndarray
        The encoded matrix
    """
    from lasagne.layers import get_output

    if len(img.shape) == 4:
        rows = img.shape[2]
        cols = img.shape[3]
    elif len(img.shape) == 3:
        rows = img.shape[1]
        cols = img.shape[2]
        img = img.reshape(img.shape[0],1,rows,cols)
    elif len(img.shape) == 2:
        rows,cols = img.shape
        img = img.reshape(1,1,rows,cols)
    else:
        print("The shape of image should be 2 or 3 d")

    def get_layer_by_name(net, name):
        for i, layer in enumerate(net.get_all_layers()):
            if layer.name == name:
                return layer, i
        return None, None

    encode_layer, encode_layer_index = get_layer_by_name(cae, 'encode')
    img_en =  get_output(encode_layer, inputs=img).eval()

    return img_en

def get_decode(cae, img_en):
    """Decode to output the recovered image

    input
    =====
    img_en: np.ndarray
        The encoded matrix

    output
    ======
    img_de: np.ndarray
        The recovered or predicted image matrix
    """
    from lasagne.layers import get_output, InputLayer

    def get_layer_by_name(net, name):
        for i, layer in enumerate(net.get_all_layers()):
            if layer.name == name:
                return layer, i
        return None, None

    encode_layer, encode_layer_index = get_layer_by_name(cae,'encode')
    # decoder
    new_input = InputLayer(shape=(None,encode_layer.num_units))
    layer_de_input = cae.get_all_layers()[encode_layer_index + 1]
    layer_de_input.input_layer = new_input
    layer_de_output = cae.get_all_layers()[-1]

    img_de = get_output(layer_de_output, img_en).eval()

    return img_de

def load_net(netpath):
    """
    Load the cae network

    input
    =====
    netpath: str
        Path to save the trained network

    output
    ======
    cae: learn.lasagne.base.NeuralNet
       The trained network
    """
    try:
        fp = open(netpath,'rb')
    except:
        return None

    cae = pickle.load(fp)

    return cae
