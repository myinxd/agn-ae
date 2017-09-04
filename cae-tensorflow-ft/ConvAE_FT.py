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
<TODO>

Update
======
[2017-09-04] Add methods to fine-tuning the network.
[2017-09-04] Change padding and stride strtegy to custom providing.
[2017-09-04] Remove pooling staffs
"""

import tensorflow as tf
import numpy as np
import time

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

    Methods
    =======
    gen_layers:
    """

    def __init__(self, input_shape,
                 kernel_size=[3,3,3], kernel_num=[10,10,10],
                 fc_nodes=[64],
                 encode_nodes=10,
                 padding = ('SAME','SAME'),
                 stride = (2, 2),
                 numclass = 5,
                 ):
        """
        The initializer
        """
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.fc_nodes = fc_nodes
        self.encode_nodes = encode_nodes
        self.pad_en = padding[0]
        self.pad_de = padding[1]
        self.stride = stride
        self.numclass = numclass # used for fine tuning, maybe need reconsidered.
        self.y_ = tf.placeholder(tf.float32, shape=[None, numclass], name="cnn-labels") # placeholder of y
        self.droprate = tf.placeholder(tf.float32, name="droprate")


    def gen_BatchIterator(self, data, batch_size=100, shuffle=True):
        """
        Return the next 'batch_size' examples
        from the X_in dataset

        Reference
        =========
        [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
        Input
        =====
        data: 4d np.ndarray
            The samples to be batched
        batch_size: int
            Size of a single batch.
        shuffle: bool
            Whether shuffle the indices.

        Output
        ======
        Yield a batch generator
        """
        if shuffle:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
        for start_idx in range(0, len(data) - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx: start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)
            yield data[excerpt]

    def gen_cnn_BatchIterator(self, data, label, batch_size=100, shuffle=True):
        """
        Return the next 'batch_size' examples
        from the X_in dataset

        Reference
        =========
        [1] tensorflow.examples.tutorial.mnist.input_data.next_batch
        Input
        =====
        data: 4d np.ndarray
            The samples to be batched
        label: np.ndarray
            The labels to be batched w.r.t. data
        batch_size: int
            Size of a single batch.
        shuffle: bool
            Whether shuffle the indices.

        Output
        ======
        Yield a batch generator
        """
        if shuffle:
            indices = np.arange(len(data))
            np.random.shuffle(indices)
        else:
            indices = np.arange(len(data))
        data = data[indices]
        label = label[indices]
        numbatch = len(indices) // batch_size
        return data,label,numbatch


    def cae_build(self):
        """Construct the network, including ConvLayers and FC layers."""

        # Useful methods
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        # Init
        self.encoder = []
        self.shapes_en = []

        # Input layer
        in_depth = self.input_shape[3]
        in_row = self.input_shape[1]
        in_col = self.input_shape[2]
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
                                            W, strides=[1,self.stride[0],self.stride[1],1],
                                            padding=self.pad_en), b)
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
                strides = [1,self.stride[0],self.stride[1],1], padding=self.pad_de),
                b)
            output = tf.nn.relu(output)
            current_input = output

        # Decoder output
        self.l_de = current_input

    def cnn_build(self, learning_rate):
        """Build the cnn for fine-tuning, just add a softmax layer after the encoder layer.
           Since the weights are shared between encoder and decoder, the fine-tuned weights
           can be reused.
        """
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        # add a softmax layer
        W_soft = weight_variable([self.encode_nodes, self.numclass])
        b_soft = bias_variable([self.numclass])

        self.l_cnn = tf.matmul(self.l_en, W_soft) + b_soft
        # generate the optimizer
        self.gen_cnn_optimizer(learning_rate = learning_rate)

    def gen_cost(self):
        """Generate the cost, usually be the mean square error."""
        # cost function
        self.cost = tf.reduce_mean(tf.square(self.l_de-self.l_in))

    def gen_cnn_cost(self):
        """Generate the cost function for the added softmax layer at the end of encoder."""
        # cost function
        self.cnn_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.l_cnn))

    def gen_optimizer(self, learning_rate=0.01):
        """Generate the optimizer to minimize the cose"""
        self.gen_cost()
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

    def gen_cnn_optimizer(self, learning_rate = 0.01):
        """Generate the optimizer for the fine-tuning cnn.

        Reference
        =========
        [1] Tensorflow: No gradients provided for any variable
            https://stackoverflow.com/questions/38778760/tensorflow-no-gradients-provided-for-any-variable
        """
        self.gen_cnn_cost()
        self.cnn_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cnn_cost)

    def cae_train(self, data, num_epochs=20, learning_rate=0.01, batch_size=100, droprate=0.5):
        """Train the net"""
        timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training parameters\n" % (timestamp))
        print("[%s] Epochs: %d\tLearning rate: %.2f\n" % (timestamp, num_epochs, learning_rate))
        print("[%s] Batch size: %d\tDrop rate: %.2f\n" % (timestamp, batch_size, droprate))
        self.gen_optimizer(learning_rate = learning_rate)
        init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(init_op)

        # generate optimizer
        for epoch in range(num_epochs):
            for batch in self.gen_BatchIterator(data=data, batch_size=batch_size):
                self.sess.run(self.optimizer, feed_dict={self.l_in: batch, self.droprate: droprate})

            timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
            print("[%s] Epoch: %03d\tAverage loss: %.3f" %
                (timestamp, epoch+1,
                 self.sess.run(self.cost,
                               feed_dict={self.l_in: batch, self.droprate: droprate})))

    def cnn_train(self, data, label, num_epochs=20, learning_rate=0.01, batch_size=100, droprate=0.5):
        """Train the net"""
        timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training parameters\n" % (timestamp))
        print("[%s] Epochs: %d\tLearning rate: %.2f\n" % (timestamp, num_epochs, learning_rate))
        print("[%s] Batch size: %d\tDrop rate: %.2f\n" % (timestamp, batch_size, droprate))
        # init_op = tf.global_variables_initializer()
        # self.sess = tf.InteractiveSession()
        # self.sess.run(init_op)

        # generate optimizer
        for epoch in range(num_epochs):
            data_batch,label_batch,numbatch = self.gen_cnn_BatchIterator(data=data,
                                                                         label=label,
                                                                         batch_size=batch_size)
            for i in range(numbatch):
                batch = data_batch[i*batch_size:(i+1)*batch_size,:,:,:]
                batch_label = label_batch[i*batch_size:(i+1)*batch_size,:]
                self.sess.run(self.cnn_optimizer,
                              feed_dict={self.l_in: batch,
                                         self.y_: batch_label,
                                         self.droprate: droprate})

            # <TODO> validation
            timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
            print("[%s] Epoch: %03d\tAverage loss: %.3f" %
                (timestamp, epoch+1,
                 self.sess.run(self.cnn_cost, feed_dict={self.l_in: batch, self.y_: batch_label, self.droprate: droprate})))


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
        depth = self.input_shape[3]
        rows = self.input_shape[1]
        cols = self.input_shape[2]
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

    def cnn_predict(self, img):
        """Predict types of the sample with the
           trained fine-tuned network.

        Input
        =====
        img: np.ndarray
            The images to be test.

        Output
        ======
        label_pre: np.ndarray
            The predicted label.
        """
        # params
        depth = self.input_shape[3]
        rows = self.input_shape[1]
        cols = self.input_shape[2]
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
            label_pre = self.sess.run(self.l_cnn, feed_dict={self.l_in: img_te, self.droprate: 0.0})
        else:
            print("You should train the network.")
            return None

        return label_pre


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
        depth = self.input_shape[3]
        rows = self.input_shape[1]
        cols = self.input_shape[2]
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
                    'l_cnn': self.l_cnn.name,
                    'y_':self.y_.name,
                    'droprate': self.droprate.name,
                    'netpath': netpath}
        with open(namepath, 'wb') as fp:
            pickle.dump(savedict, fp)

        # save the net
        saver = tf.train.Saver()
        saver.save(self.sess, netpath)
