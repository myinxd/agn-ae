# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
A class to define the convolutional generative adverasial network (CGAN)
under the structure of tensorflow.

Reference
=========
[1] tensorflow_tutorials
    https://github.com/pkmital/tensorflow_tutorials/blob/master/
    python/07_autoencoder.py
[2] GCGAN
    https://arxiv.org/pdf/1511.06434.pdf
[3] GAN example on MNIST
    http://blog.csdn.net/sparkexpert/article/details/70147409

Methods
=======
"""

import tensorflow as tf
import numpy as np
import time
import math

class ConvGAN():
    """
    A convolutional generative adverasial netowk (CGAN) constructor

    inputs
    ======
    X_in: np.ndarray
        The sample matrix, whose size is (s,d,r,c).
        s: number of samples, d: dimensions,
        r: rows, c: cols
    kernel_size: list
        Window sizes of the kernels in each ConvLayer
    Kernel_num: list
        Number of kernels in each ConvLayer
    fc_nodes: list
        The dense layers after the fully connected layer
        of the last ConvLayer or pooling layer
    encode_nodes: int
        Number of nodes as the input of the generator
    droprate: float
        Dropout rate, belongs to [0,1]
    dropflag: bool
        Flag of dropout

    Methods
    =======
    gen_generator
    gen_descriminator
    get_decode

    """

    def __init__(self, img_shape=(28,28,1), kernel_size=[3,3,3],
                 kernel_num=[10,10,10],
                 fc_nodes=[64], encode_nodes=10,
                 pad = 'VALID',
                 stride = 2,
                 batch_size = 100,
                 ):
        """
        The initializer
        """
        self.img_shape = img_shape
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.fc_nodes = fc_nodes
        self.encode_nodes = encode_nodes
        self.pad = pad
        self.stride = stride
        self.batch_size = batch_size
        self.gen_Tensors()
        # self.gen_discriminator()
        # self.gen_generator()


    def gen_Tensors(self):
        """Generate the placeholded tensors"""
        self.x_smp = tf.placeholder(tf.float32, [self.batch_size,
                                                 self.img_shape[0],
                                                 self.img_shape[1],
                                                 self.img_shape[2]], name="x_smp")
        self.x_gen = tf.placeholder(tf.float32, [self.batch_size,
                                                 self.img_shape[0],
                                                 self.img_shape[1],
                                                 self.img_shape[2]], name="x_gen")
        self.y_smp = tf.placeholder(tf.float32, [self.batch_size,], name="y_smp")
        self.y_gen = tf.placeholder(tf.float32, [self.batch_size,], name="y_gen")
        self.z_en = tf.placeholder(tf.float32,[self.batch_size, self.encode_nodes], name="z_en")
        self.keep_rate = tf.placeholder(tf.float32, name="keep_rate")
        # self.batchsize = tf.placeholder(tf.int31, name="batchsize")


    def gen_BatchIterator(self, x_in, shuffle=True):
        """
        Return the next 'batch_size' examples
        from the X_in dataset

        Reference
        =========
        [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
        Input
        =====
        x_in: np.ndarray
            data to be batched, [numsamples, row, col, channel]
        shuffle: bool
            Whether shuffle the indices.

        Output
        ======
        Yield a batch generator
        """
        if shuffle:
            indices = np.arange(len(x_in))
            np.random.shuffle(indices)
        for start_idx in range(0, len(x_in) - self.batch_size + 1, self.batch_size):
            if shuffle:
                excerpt = indices[start_idx: start_idx + self.batch_size]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
            yield x_in[excerpt]

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def gen_layers(self):
        """
        Generate layer parameters according to provided
        kernel and full connected parameters.

        Layers
        ======
        InputLayer: [NumSamples, rows, cols, ch_in, ch_out]
        ConvLayer: [NumSamples, rows, cols, ch_in, ch_out]
        Denselayer: [NumSamples, None, Nodes]
        FCLayer: [NumSamples, Nodes_in, Nodes_out]
        """
        # Init
        self.layers = {"InputLayer": [self.batch_size,
                                      self.img_shape[0],
                                      self.img_shape[1],
                                      None,
                                      self.img_shape[2]],
                       "ConvLayer": [],
                       "DenseLayer": [],
                       "FCLayer": [],
                       "OutputLayer": []}

        shape_input = self.layers['InputLayer']
        # ConvLayers
        if self.pad == 'SAME':
            for i, depth_output in enumerate(self.kernel_num):
                new_row = math.ceil(shape_input[1]/self.stride)
                new_col = math.ceil(shape_input[2]/self.stride)
                shape_input = [shape_input[0],
                               new_row,
                               new_col,
                               shape_input[4],
                               depth_output]
                self.layers['ConvLayer'].append(shape_input)
        elif self.pad == 'VALID':
            for i, depth_output in enumerate(self.kernel_num):
                new_row = math.ceil((shape_input[1]-self.kernel_size[i]+1)/self.stride)
                new_col = math.ceil((shape_input[2]-self.kernel_size[i]+1)/self.stride)
                shape_input = [shape_input[0],
                               new_row,
                               new_col,
                               shape_input[4],
                               depth_output]
                self.layers['ConvLayer'].append(shape_input)
        else:
            print("Padding mode should be 'SAME' or 'VALID'")
            return None

        # DenseLayer
        shape_dense = self.layers['ConvLayer'][-1]
        self.layers['DenseLayer'] = [shape_dense[0],
                                     None,
                                     shape_dense[1]*shape_dense[2]*shape_dense[4]]

        # Fully connected layers
        shape_input = self.layers['DenseLayer']
        for i, depth_output in enumerate(self.fc_nodes):
            shape_input = [shape_input[0],
                           shape_input[2],
                           depth_output]
            self.layers['FCLayer'].append(shape_input)

        # OuputLayer
        if len(self.fc_nodes):
            self.layers['OutputLayer'] = [-1, self.fc_nodes[-1], -1]
        else:
            self.layers['OutputLayer'] = [-1, self.layers['DenseLayer'][-1],-1]

    def gen_generator(self):
        """Construct the generator network."""

        self.param_gen = []

        current_input = self.z_en
        depth_input = self.encode_nodes
        # Ouput layer to FC or dense
        depth_output = self.layers['OutputLayer'][1]
        W_out = self.weight_variable(shape=[depth_input, depth_output])
        b_out = self.bias_variable(shape=[depth_output])
        self.param_gen.append(W_out)
        self.param_gen.append(b_out)
        output = tf.nn.relu(tf.matmul(current_input, W_out) + b_out)
        current_input = output

        # Dense or Full
        # FC layers
        for i in range(len(self.layers['FCLayer'])-1, -1, -1):
            layer_fc = self.layers['FCLayer'][i]
            W = self.weight_variable(shape=[layer_fc[2], layer_fc[1]])
            b = self.bias_variable(shape=[layer_fc[1]])
            self.param_gen.append(W)
            self.param_gen.append(b)
            output = tf.nn.relu(tf.matmul(current_input, W) + b)
            current_input = output

        # reshape
        shape_conv = self.layers['ConvLayer'][-1]
        current_input = tf.reshape(current_input,
                                    [-1,
                                    shape_conv[1],
                                    shape_conv[2],
                                    shape_conv[4]])

        print(current_input.get_shape())
        # ConvLayers
        layer_convs= self.layers['ConvLayer']
        layer_convs.insert(0, self.layers['InputLayer'])
        print(layer_convs)

        for i in range(len(layer_convs)-2, -1, -1):
            layer_in = layer_convs[i+1]
            layer_out = layer_convs[i]
            W = self.weight_variable(shape=[self.kernel_size[i],
                                            self.kernel_size[i],
                                            layer_in[3],
                                            layer_in[4]])
            b = self.bias_variable(shape=[layer_in[3]])
            self.param_gen.append(W)
            self.param_gen.append(b)
            output = tf.add(tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([layer_out[0],layer_out[1],layer_out[2],layer_out[4]]),
                strides = [1, self.stride, self.stride, 1],
                padding = self.pad,
            ), b)
            output = tf.nn.relu(output)
            current_input = output

        # Generator output
        self.x_gen = current_input


    def gen_discriminator(self):
        """Construct the discriminator network, including ConvLayers and FC layers."""

        self.param_dis = []

        # Conv layers
        current_input = tf.concat([self.x_smp, self.x_gen], 0)
        for i, depth_output in enumerate(self.kernel_num):
            depth_input = current_input.get_shape().as_list()[3]
            W = self.weight_variable(shape=[self.kernel_size[i],self.kernel_size[i], depth_input, depth_output])
            b = self.bias_variable(shape=[depth_output])
            self.param_dis.append(W)
            self.param_dis.append(b)
            # self.encoder.append(W) # share
            output = tf.add(tf.nn.conv2d(current_input,
                                            W, strides=[1,self.stride,self.stride,1],
                                            padding=self.pad), b)
            output = tf.nn.relu(output)
            current_input = output

        # Dense layer
        shape_conv = current_input.get_shape().as_list()
        depth_dense = shape_conv[1] * shape_conv[2] * shape_conv[3]
        current_input = tf.reshape(current_input, [-1, depth_dense])

        # fully connected
        depth_input = depth_dense
        for i, depth_output in enumerate(self.fc_nodes):
            W = self.weight_variable(shape=[depth_input, depth_output])
            b = self.bias_variable(shape=[depth_output])
            self.param_dis.append(W)
            self.param_dis.append(b)
            output = tf.nn.relu(tf.matmul(current_input, W) + b)
            # dropout
            output = tf.nn.dropout(output, self.keep_rate)
            current_input = output
            depth_input = depth_output

        # compare layer
        W_en = self.weight_variable(shape=[depth_input, 1])
        b_en = self.bias_variable(shape=[1])
        self.param_dis.append(W_en)
        self.param_dis.append(b_en)
        output = tf.nn.relu(tf.matmul(current_input, W_en) + b_en)

        self.y_smp = tf.slice(output, [0, 0], [self.batch_size, -1])
        self.y_gen = tf.slice(output, [self.batch_size, 0], [-1, -1])


    def gen_loss(self):
        """Generate the loss functions."""
        # loss of generator
        self.loss_dis = - (tf.log(self.y_smp) + tf.log(1-self.y_gen))
        self.loss_gen = - (tf.log(self.y_gen))


    def gen_optimizer(self, learning_rate=0.01):
        """Generate the optimizer to minimize the cose"""
        self.gen_loss()
        self.optimizer_dis = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_dis, var_list=self.param_dis)
        self.optimizer_gen = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_gen, var_list=self.param_gen)

    def gan_train(self, x_in, num_epochs=20, learning_rate=0.01, keep_rate=0.5):
        """Train the net"""
        self.gen_optimizer(learning_rate = learning_rate)
        init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init_op)

        # generate optimizer
        for epoch in range(num_epochs):
            for batch in self.gen_BatchIterator(x_in = x_in):
                z_sample = np.random.normal(0, 1, size=(self.batch_size, self.encode_nodes)).astype(np.float32)
                # train generator
                # x_generator = self.sess.run(self.x_generator, feed_dict={self.z_en: z_sample, self.keep_rate: keep_rate})
                self.sess.run(self.optimizer_gen,
                              feed_dict={self.x_smp: batch,
                                         self.z_en: z_sample,
                                         # self.x_gen: x_generator,
                                         self.keep_rate: keep_rate})

                # train discriminator
                self.sess.run(self.optimizer_dis,
                              feed_dict={self.x_smp: batch,
                                         self.z_en: z_sample,
                                         # self.x_gen: x_generator,
                                         self.keep_rate: keep_rate})

            timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("[%s] Epoch: %d / %d" % (timestamp, epoch+1, num_epochs))
            # print(epoch+1, self.sess.run(self.cost, feed_dict={self.l_in: batch, self.droprate: droprate})/batch_size)

    def gan_test(self, img):
        """Test the trained network,

        Input
        =====
        img: np.ndarray
            The images to be test.

        Output
        ======
        img_pre: np.ndarray
            The predicted images.
        """
        # params
        depth = self.X_in.shape[3]
        rows = self.X_in.shape[1]
        cols = self.X_in.shape[2]
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
        if hasattr(self, 'sess'):
            img_pre = self.sess.run(self.l_de, feed_dict={self.l_in: img_te, self.droprate: 0.0})
        else:
            print("You should train the network.")
            return None

        return img_pre

    def cae_encode(self, img):
        """Test the trained network,

        Input
        =====
        img: np.ndarray
            The images to be test.

        Output
        ======
        code: np.ndarray
            The encoded codes.
        """
        # params
        depth = self.X_in.shape[3]
        rows = self.X_in.shape[1]
        cols = self.X_in.shape[2]
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
        if hasattr(self, 'sess'):
            code = self.sess.run(self.l_en, feed_dict={self.l_in: img_te, self.droprate: 0.0})
        else:
            print("You should train the network.")
            return None

        return code


    def gan_decode(self, code):
        """Generate agn images with provided codes.

        Input
        =====
        code: np.ndarray
            The code to be decoded.

        Output
        ======
        img_de: np.ndarray
            The decoded and reconstructed images.
        """
        # Compare code length
        code_len = self.l_en.get_shape()[1]
        if code.shape[1] != code_len:
            print("The length of provided codes should be equal to the network's")
            return None
        else:
            # decoding
            if hasattr(self, 'sess'):
                l_in_shape = self.l_in.get_shape().as_list()
                l_in_shape[0] = code.shape[0]
                p_in = np.zeros(l_in_shape) # pseudo input
                img_de = self.sess.run(self.l_de,
                                       feed_dict={self.l_in: p_in,
                                                  self.l_en: code,
                                                  self.droprate: 0.0})
            else:
                print("You should train the network firstly.")
                return None

        return img_de

    def gan_save(self, namepath, netpath):
        """Save the net"""
        import pickle
        import sys
        sys.setrecursionlimit(1000000)

        # save the names
        savedict = {'x_smp': self.x_smp.name,
                    'x_gen': self.x_gen.name,
                    'y_smp': self.y_smp.name,
                    'y_gen': self.y_gen.name,
                    'keep_rate': self.keep_rate.name,
                    'z_en':self.z_en.name,
                    'netpath': netpath}
        with open(namepath, 'wb') as fp:
            pickle.dump(savedict, fp)

        # save the net
        saver = tf.train.Saver()
        saver.save(self.sess, netpath)
