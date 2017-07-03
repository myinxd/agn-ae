# copyright (C) 2017 zxma_sjtu@qq.com

"""
The utils for construcing our convolutional auto-encoder (CAE) model
"""

import os
import pickle
import numpy as np
import scipy.io as sio
from astropy.io import fits
from scipy.misc import imread
from skimage import transform
from scipy.signal import convolve2d as conv2
from astropy.stats import sigma_clip

def gen_splits(img, boxsize=200, stride=50):
    """
    Generate samples by splitting the large image into patches
    Input
    -----
    img: np.ndarray
        The 2D raw image
    boxsize: integer
        Size of the box, default as 200
    stride: integer
        Shifted pixels, default as 50
    Output
    ------
    data: np.ndarray
        The matrix holding samples, each slice represents one sample
    """
    # Init
    rows, cols = img.shape
    # Number of boxes
    box_rows = int(np.round((rows - boxsize - 1) / stride)) + 1
    box_cols = int(np.round((cols - boxsize - 1) / stride)) + 1
    # init data and label
    data = np.zeros((box_rows * box_cols, boxsize*boxsize))

    # Split
    for i in range(box_rows):
        for j in range(box_cols):
            sample = img[i * stride:i * stride + boxsize,
                         j * stride:j * stride + boxsize]
            data[i * box_rows + j, :] = sample.reshape((boxsize*boxsize,))

    return data

def gen_sample(folder, ftype='jpg', savepath=None,
                crop_box=(200, 200), res_box=(50, 50), clipflag=False, clipparam=None):
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
    clipflag: booling
        The flag of sigma clipping, default as False
    clipparam: list
        Parameters of the sigma clipping, [sigma, iters]

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

    def read_image(fpath,ftype):
        if ftype == 'fits':
            h = fits.open(fpath)
            img = h[0].data
        else:
            img = imread(name=fpath, flatten=True)
        return img

    # load images
    idx = 0
    for fname in sample_list:
        fpath = os.path.join(folder,fname)
        if fpath.split('.')[-1] == ftype:
            #read image
            img = read_image(fpath=fpath, ftype=ftype)
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
            if clipflag:
                img_rsz = get_sigma_clip(img_rsz,
                                         sigma=clipparam[0],
                                         iters=clipparam[1])
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

def get_concate(cae, layer_idx, savefolder, perline=4):
    """
    Concate the feature maps into a large image

    Reference
    =========
    [1] scipy.signal.convolve2d
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html
    inputs
    ======
    cae: nolearn.lasagne.NeuralNet
        The network
    layer_idx: int
        The corresponding index of the ConvLayer
    savefolder: str
        The folder name to save images
    """
    from scipy.misc import imsave
    # get layer
    layer = cae.get_all_layers()[layer_idx]
    if str(type(layer)).split('.')[-1] == "Conv2DLayer'>":
        weights = layer.get_params()[0]
        maps = weights.get_value()
    else:
        print("The layer does not a conv layer.")
        return

    # Normalization
    maps_max = np.max(maps)
    maps_min = np.min(maps)
    maps_norm = (maps - maps_min) / (maps_max - maps_min)
    # maps_norm = np.clip(np.rint(maps_norm) * 255, a_min=0, a_max=255)
    # maps_norm = maps_norm.astype('uint8')
    # save maps
    if os.path.exists(savefolder):
        os.system("rm -r %s" % savefolder)
        os.mkdir(savefolder)
    else:
        os.mkdir(savefolder)

    for i in range(maps_norm.shape[0]):
        fname = ('map_%d.png' % i)
        f = os.path.join(savefolder,fname)
        # resize: zoom in
        map_res = maps_norm[i,0,:,:]
        map_res = transform.resize(map_res,(50,50),mode="reflect")
        imsave(f,map_res)

    # get concate
    print("Concating the maps")
    pathcon = os.path.join(savefolder,'map_con.png')
    print("montage -mode concatenate -tile %dx %s/*.png %s" % (perline, savefolder,pathcon))
    os.system("montage -mode concatenate -tile %dx %s/*.png %s" % (perline, savefolder,pathcon))

def get_conv(cae, layer_idx, img, savefolder=None, perline=4):
    """A naive convolution method

    inputs
    ======
    cae: nolearn.lasagne.NeuralNet
        The trained cae network
    layer_idx: int
        The index w.r.t. the network
    img: np.ndarray (r,c)
        The image to be convolved

    outputs
    =======
    img_conv: np.ndarray (w,r+s-1,c+s-1)
    """
    from scipy.misc import imsave
    # Judge image dimenstion
    if len(img.shape) > 2:
        if img.shape[0] > 1:
            print("Only a single image can be processed...sorry")
            return
        else:
            rows_img = img.shape[-2]
            cols_img = img.shape[-1]
            img = img.reshape(rows_img, cols_img)
    else:
        rows_img,cols_img = img.shape
    # get layer
    layer = cae.get_all_layers()[layer_idx]
    if str(type(layer)).split('.')[-1] == "Conv2DLayer'>":
        weights = layer.get_params()[0]
        maps = weights.get_value()
    else:
        print("The layer does not a conv layer.")
        return
    # convolution
    rows_map = maps.shape[-2]
    cols_map = maps.shape[-1]
    img_conv = np.zeros((maps.shape[0],
                        rows_img-rows_map+1,
                        cols_img-cols_map+1))
    for i in range(maps.shape[0]):
        img_conv[i,:,:] = conv2(img, maps[i,0,:,:], mode='valid')

    # Normalization
    conv_max = np.max(img_conv)
    conv_min = np.min(img_conv)
    img_norm = (img_conv - conv_min) / (conv_max - conv_min)
    # maps_norm = np.clip(np.rint(maps_norm) * 255, a_min=0, a_max=255)
    # maps_norm = maps_norm.astype('uint8')
    # save maps
    if os.path.exists(savefolder):
        os.system("rm -r %s" % savefolder)
        os.mkdir(savefolder)
    else:
        os.mkdir(savefolder)

    for i in range(img_norm.shape[0]):
        fname = ('conv_%d.png' % i)
        f = os.path.join(savefolder,fname)
        # resize: zoom in
        conv_res = img_norm[i,:,:]
        conv_res = transform.resize(conv_res,(50,50),mode="reflect")
        imsave(f,conv_res)
    # get concate
    print("Concating the convolved maps")
    pathcon = os.path.join(savefolder,'conv_con.png')
    print("montage -mode concatenate -tile %dx %s/*.png %s" % (perline, savefolder,pathcon))
    os.system("montage -mode concatenate -tile %dx %s/*.png %s" % (perline, savefolder,pathcon))

    return img_conv

def get_sigma_clip(img,sigma=3,iters=100):
    """
    Do sigma clipping on the raw images to improve constrast of
    target regions.

    Reference
    =========
    [1] sigma clip
        http://docs.astropy.org/en/stable/api/astropy.stats.sigma_clip.html
    """
    img_clip = sigma_clip(img, sigma=sigma, iters=iters)
    img_mask = img_clip.mask.astype(float)
    img_new = img * img_mask

    return img_new

def get_sample_info(infopath):
    """
    Read samples' infomation, in this work, they are mannually
    classified into odd (1), noisy (2), and normal (3)
    """
    from scipy.io import loadmat
    data = loadmat(infopath)

    info = data['info']

    return info[:,0]

def sort_sample(inpath, savepath=None):
    """
    Sort the samples' names which are namely with integers, into descending
    sort, as well as the corresponding image data.

    Inputs
    ======
    inpath: str
        Path of the unsorted samples
    outpath: str
        Path of the sorted ones, if set as None, the inpath will be used.
    """
    with open(inpath, 'rb') as fp:
        datadict = pickle.load(fp)

    data = datadict['data']
    name = datadict['name']

    # rename the files with equai-length name
    numsamples = data.shape[0]
    # maximun number of characters in a sample name
    maxch = len(str(numsamples))
    # rename
    for i in range(numsamples):
        s = name[i]
        n = s.split('.')
        # fill
        n[0] = '0'*(maxch - len(n[0])) + n[0]
        # rename
        name[i] = '.'.join(n)

    # sort
    idx_sort = np.argsort(name)
    name_array = np.array(name)

    name_sort = name_array[idx_sort]
    data_sort = data[idx_sort]

    # save
    datadict['name'] = name_sort
    datadict['data'] = data_sort

    if savepath is None:
        savepath = inpath

    with open(savepath, 'wb') as f:
        pickle.dump(datadict, f)

