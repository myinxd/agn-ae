# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
A class to define the convolutional neural network under
the structure of tensorflow.

Reference
=========
[1] tensorflow_tutorials
    https://github.com/pkmital/tensorflow_tutorials/blob/master/
    python/07_autoencoder.py
[2] agn-ae
    https://github.com/myinxd/agn-ae/blob/master/ConvAE.py

Methods
=======
"""

import tensorflow as tf
import numpy as np

class ConvAE():
    """
    A convolutional autoencoder (CAE) constructor

    inputs
    ======
    X_in: np.ndarray
        The sample matrix, whose size is (s,d,r,c).
        s: number of samples, d: dimensions,
        r: rows, c: cols
    X_out: np.ndarray
        The real output, which is as the same as X_in
    kernel_size: list
        Window sizes of the kernels in each ConvLayer
    Kernel_num: list
        Number of kernels in each ConvLayer
    pool_flag: list of bool values
        Flags of pooling layer w.r.t. to the ConvLayer
    pool_size: list
        List of the pooling kernels' size
    fc_nodes: list
        The dense layers after the fully connected layer
        of the last ConvLayer or pooling layer
    encode_nodes: int
        Number of nodes in the encode layer
    droprate: float
        Dropout rate, belongs to [0,1]
    dropflag: bool
        Flag of dropout

    Methods
    =======
    gen_layers:
    """

    def __init__(self, X_in, kernel_size=[3,3,3],
                 kernel_num=[10,10,10], pool_flag=[True,True,True],
                 pool_size=[2,2,2], fc_nodes=[64], encode_nodes=10,
                 ):
        """
        The initializer
        """
        self.X_in = X_in
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.pool_flag=pool_flag
        self.pool_size=pool_size
        self.fc_nodes = fc_nodes
        self.encode_nodes = encode_nodes
        self.droprate = tf.placeholder(tf.float32, name="droprate")


    def gen_BatchIterator(self, batch_size=100, shuffle=True):
        """
        Return the next 'batch_size' examples
        from the X_in dataset

        Reference
        =========
        [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
        Input
        =====
        batch_size: int
            Size of a single batch.
        shuffle: bool
            Whether shuffle the indices.

        Output
        ======
        Yield a batch generator
        """
        if shuffle:
            indices = np.arange(len(self.X_in))
            np.random.shuffle(indices)
        for start_idx in range(0, len(self.X_in) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield self.X_in[excerpt]

    def cae_build(self):
        """Construct the network, including ConvLayers and FC layers."""

        # Useful methods
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        '''
        # Convolution and pooling
        def conv2d_en(x, W, pad):
            return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding=pad)

        def max_pool_en(x, pool_size=2):
            return tf.nn.max_pool(x,
                                    ksize=[1,pool_size,pool_size,1],
                                    strides=[1,pool_size,pool_size,1],
                                    padding='VALID')

        def conv2d_de(x, W,):
            return tf.nn.conv2d_transpose(x,
                    [M`'9'`]                        )
        '''

        # Init
        pad_en = 'SAME'
        pad_de = 'SAME'
        self.encoder = []
        self.shapes_en = []

        # Input layer
        in_depth = self.X_in.shape[3]
        in_row = self.X_in.shape[1]
        in_col = self.X_in.shape[2]
        self.l_in = tf.placeholder(tf.float32,
                                [None,in_row,in_col,in_depth],
                                name='l_in')

        # Encoder layers
        current_input = self.l_in
        for i, depth_output in enumerate(self.kernel_num):
            depth_input = current_input.get_shape().as_list()[3]
            self.shapes_en.append(current_input.get_shape().as_list())
            W = weight_variable(shape=[self.kernel_size[i],
                                       self.kernel_size[i],
                                       depth_input,
                                       depth_output,
                                       ])
            b = bias_variable(shape=[depth_output])
            self.encoder.append(W) # share
            output = tf.add(tf.nn.conv2d(current_input,
                                            W, strides=[1,2,2,1],
                                            padding=pad_en), b)
            output = tf.nn.relu(output)
            current_input = output

        if self.encode_nodes is not None:
            # Dense layer
            shape_conv = current_input.get_shape().as_list()
            depth_dense = shape_conv[1] * shape_conv[2] * shape_conv[3]
            l_en_dense = tf.reshape(current_input, [-1, depth_dense])
            # dropout layer
            # keep_prob = tf.placeholder(tf.float32)
            # fully connected
            depth_input = depth_dense
            current_input = l_en_dense
            self.en_fc = [] # save and share weights of the fc layers
            self.fc_depth=[] # depth_input of the encoder
            for i, depth_output in enumerate(self.fc_nodes):
                self.fc_depth.append(depth_input)
                W = weight_variable(shape=[depth_input, depth_output])
                b = bias_variable(shape=[depth_output])
                self.en_fc.append(W) # share weight
                output = tf.nn.relu(tf.matmul(current_input, W) + b)
                # dropout
                output = tf.nn.dropout(output, self.droprate)
                current_input = output
                depth_input = depth_output

            # encode layer
            W_en = weight_variable(shape=[depth_input, self.encode_nodes])
            b_en = bias_variable(shape=[self.encode_nodes])
            output = tf.nn.relu(tf.matmul(current_input, W_en) + b_en)
            current_input = output
            self.l_en = current_input

            # decoder layers
            W_de = tf.transpose(W_en)
            if len(self.fc_nodes):
                depth_output = self.fc_nodes[-1]
            else:
                depth_output = depth_dense
            b_de = bias_variable(shape=[depth_output])
            output = tf.nn.relu(tf.matmul(current_input, W_de) + b_de)
            current_input = output

            # fc layers
            for i in range(len(self.fc_nodes)-1, -1, -1):
                depth_output = self.fc_depth[i]
                W = tf.transpose(self.en_fc[i])
                b = bias_variable(shape=[depth_output])
                output = tf.nn.relu(tf.matmul(current_input, W) + b)
                # dropout
                output = tf.nn.dropout(output, self.droprate)
                current_input = output

            # Dense layer
            # W_de = weight_variable(shape=[depth_input, depth_dense])
            # b_de = bias_variable(shape=[depth_dense])
            # output = tf.nn.relu(tf.matmul(current_input, W_de) + b_de)
            # current_input = tf.nn.dropout(output, self.droprate)
            # reshape
            current_input = tf.reshape(current_input,
                                    [-1,
                                        shape_conv[1],
                                        shape_conv[2],
                                        shape_conv[3]])
        else:
            self.l_en = current_input

        self.decoder = self.encoder.copy()
        self.decoder.reverse()
        self.shapes_de = self.shapes_en.copy()
        self.shapes_de.reverse()

        for i, shape in enumerate(self.shapes_de):
            W = self.decoder[i]
            b = bias_variable(shape=(W.get_shape().as_list()[2],))
            output = tf.add(tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(self.l_in)[0], shape[1], shape[2], shape[3]]),
                strides = [1,2,2,1], padding=pad_de
            ), b)
            output = tf.nn.relu(output)
            current_input = output

        # Decoder output
        self.l_de = current_input

    def gen_cost(self):
        """Generate the cost, usually be the mean square error."""
        # cost function
        self.cost = tf.reduce_sum(tf.square(self.l_de-self.l_in))


    def gen_optimizer(self, learning_rate=0.01):
        """Generate the optimizer to minimize the cose"""
        self.gen_cost()
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def cae_train(self, num_epochs=20, learning_rate=0.01, batch_size=100, droprate=0.5):
        """Train the net"""
        self.gen_optimizer(learning_rate = learning_rate)
        init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init_op)

        # generate optimizer
        for epoch in range(num_epochs):
            for batch in self.gen_BatchIterator(batch_size=batch_size):
                self.sess.run(self.optimizer, feed_dict={self.l_in: batch, self.droprate: droprate})

            print(epoch, self.sess.run(self.cost, feed_dict={self.l_in: batch, self.droprate: droprate}))

    def cae_test(self, img):
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


    def cae_decode(self, code):
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

    def cae_save(self, namepath, netpath):
        """Save the net"""
        import pickle
        import sys
        sys.setrecursionlimit(1000000)

        # save the names
        savedict = {'l_in': self.l_in.name,
                   'l_en': self.l_en.name,
                   'l_de': self.l_de.name,
                    'droprate': self.droprate.name,
                    'netpath': netpath}
        with open(namepath, 'wb') as fp:
            pickle.dump(savedict, fp)

        # save the net
        saver = tf.train.Saver()
        saver.save(self.sess, netpath)
