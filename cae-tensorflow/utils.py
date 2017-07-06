# copyright (C) 2017 zxma_sjtu@qq.com

"""
The utils for construcing our convolutional auto-encoder (CAE) model
"""

import os
import pickle
import numpy as np
import tensorflow as tf
import scipy.io as sio
from astropy.io import fits
from scipy.misc import imread
from skimage import transform
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
    cae: ConvAE class
        The trained cae network.
    img: np.ndarray
        The image matrix, (r,c)

    output
    ======
    img_pred: np.ndarray
        The predicted image matrix
    """
    if img.dtype != 'float32':
        img = img.astype('float32')

    # params
    depth = cae.X_in.shape[3]
    rows = cae.X_in.shape[1]
    cols = cae.X_in.shape[2]
    # Reshape the images
    shapes = img.shape
    if len(shapes) == 2:
        if shapes[0] != rows or shapes[1] != cols:
            print('The shape of the test images do not match the network.')
            return None
        img_te = img.reshape(1,rows,cols,depth)
    elif len(shapes) == 3:
        if shapes[0] != rows or shapes[1] != cols or shapes[2] != depth:
            print('The shape of the test images do not match the network.')
            return None
        img_te = img.reshape(1,rows,cols,depth)
    elif len(shapes) == 4:
        if shapes[1] != rows or shapes[2] != cols or shapes[3] != depth :
            print('The shape of the test images do not match the network.')
            return None
        img_te = img.reshape(shapes[0],rows,cols,depth)

    # generate predicted images
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img_pred = sess.run(cae.l_de, feed_dict={cae.l_in: img_te})

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
    if img.dtype != 'float32':
        img = img.astype('float32')

    # params
    depth = cae.X_in.shape[3]
    rows = cae.X_in.shape[1]
    cols = cae.X_in.shape[2]
    # Reshape the images
    shapes = img.shape
    if len(shapes) == 2:
        if shapes[0] != rows or shapes[1] != cols:
            print('The shape of the test images do not match the network.')
            return None
        img_te = img.reshape(1,rows,cols,depth)
    elif len(shapes) == 3:
        if shapes[0] != rows or shapes[1] != cols or shapes[2] != depth:
            print('The shape of the test images do not match the network.')
            return None
        img_te = img.reshape(1,rows,cols,depth)
    elif len(shapes) == 4:
        if shapes[1] != rows or shapes[2] != cols or shapes[3] != depth :
            print('The shape of the test images do not match the network.')
            return None
        img_te = img.reshape(shapes[0],rows,cols,depth)

    # generate predicted images
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img_en = sess.run(cae.l_en, feed_dict={cae.l_in: img_te})

    return img_en

def get_decode(cae, img):
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
    img_de = get_predict(cae=cae, img=img)

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

def down_dimension(code, method='PCA', params=None):
    """
    Do dimension decreasing of the codes, so as to evaluate samples'
    distributions.

    Inputs
    ======
    code: np.ndarray
        The estimated codes by the cae net on the samples.
    method: str
        The method of dimension decreasing, could be PCA, tSNE or Kmeans,
        default as PCA.
    params: dict
        Corresponding parameters to the method.

    Output
    ======
    code_dim: np.ndarray
    The dimension decreased matrix.
    """
    if method == 'PCA':
        from sklearn.decomposition import PCA
        code_dim = PCA().fit_transform(code)
    elif method == 'tSNE':
        from sklearn.manifold import TSNE
        tsne = TSNE()
        for key in params.keys():
            try:
                setattr(tsne, key, params['key'])
            except:
                continue
        code_dim = tsne.fit_transform(code)
    elif method == 'Kmeans':
        from sklearn.cluster import KMeans
        code_dim = KMeans()
        for key in params.keys():
            try:
                setattr(code_dim, key, params['key'])
            except:
                continue
        code_dim.fit(code)
    else:
        print("The method %s is not supported at present." % method)

    return code_dim

