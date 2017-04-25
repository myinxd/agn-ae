# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
Do MNIST feature learning by our code agn-ae
"""

import numpy as np
import matplotlib
matplotlib.use('Agg') # Change matplotlib backend, in case we have no X server running..
import matplotlib.pyplot as plt
from IPython.display import Image as IPImage
from PIL import Image

import sys
sys.setrecursionlimit(1000000)

from ConvAE import ConvAE
import utils

# load data
import pickle
fname = 'mnist/mnist.pkl'
fp = open(fname, 'rb')
train,valid,test = pickle.load(fp,encoding='latin1')
fp.close()

X_train, y_train = train
X_test, y_test = test

print('X_train type and shape:', X_train.dtype, X_train.shape)
print('X_train.min():', X_train.min())
print('X_train.max():', X_train.max())

print('X_test type and shape:', X_test.dtype, X_test.shape)
print('X_test.min():', X_test.min())
print('X_test.max():', X_test.max())

# define the net
# randomly select 10000 samples
idx = np.random.permutation(X_train.shape[0])
X = X_train[idx[0:10000],:]
X_in = X.reshape(-1,1,28,28)
X_out = X
kernel_size = [3, 3, 3]
kernel_num = [16, 16, 32]
pool_flag = [False, True, True]
fc_nodes = [128]
encode_nodes = 16
net = ConvAE(X_in=X_in, X_out=X_out, kernel_size=kernel_size, pool_flag=pool_flag,
             kernel_num=kernel_num, fc_nodes=fc_nodes, encode_nodes = 16)

# generate layers
net.gen_layers()
net.layers

# Build the network and initilization
net.cae_build(learning_rate=0.01, momentum=0.975)

# Train the network
net.cae_train()

# save result
net.cae_save('mnist/net.pkl')

# Plot the loss curve
# net.cae_eval()

# Test the network
imgs = X_test.reshape(-1,28,28)
img_small = imgs[30,:,:]

# encode
img_en = utils.get_encode(net.cae, img_small)
# decode
img_de = utils.get_decode(net.cae, img_en)

# Compare
img_pre = np.rint(img_de.reshape(28,28) * 256).astype(int)
img_pre = np.clip(img_pre, a_min = 0, a_max = 255)
img_pre = img_pre.astype('uint8')
plt.imshow(img_pre)

def get_picture_array(X, rescale=4):
    array = X.reshape(28,28)
    array = np.clip(array, a_min = 0, a_max = 255)
    return  array.repeat(rescale, axis = 0).repeat(rescale, axis = 1).astype(np.uint8())

def compare_images(img, img_pre):
    original_image = Image.fromarray(get_picture_array(255 * img))
    new_size = (original_image.size[0] * 2, original_image.size[1])
    new_im = Image.new('L', new_size)
    new_im.paste(original_image, (0,0))
    rec_image = Image.fromarray(get_picture_array(img_pre))
    new_im.paste(rec_image, (original_image.size[0],0))
    new_im.save('mnist/test.png', format="PNG")
    return IPImage('mnist/test.png')

compare_images(img_small, img_pre)
