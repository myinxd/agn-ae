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

[2017-09-07] Add validation
[2017-09-07] Add accuracy evalutation of CNN, i.e., fine-tuning
[2017-09-07] Add calculation of PSNR <pending>
[2017-09-07] Add draw_training_curve()
[2017-09-07] Add learning_rate decreasing strategy

[2017-09-13] A class only processing on the CNN net
"""

from ConvAE_FT import ConvAE
import tensorflow as tf
import numpy as np
import time

class ConvNet(ConvAE):
    """
    A convolutional neural network (CNN)

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
                 valrate = 0.2,
                 sess = None,
                 name = None
                 ):
        """
        The initializer
        """
        super().__init__(input_shape=input_shape,
                         kernel_size=kernel_size,
                         kernel_num=kernel_num,
                         fc_nodes=fc_nodes,
                         encode_nodes=encode_nodes,
                         padding=padding,
                         stride=stride,
                         numclass=numclass,
                         valrate=valrate,
                         )
        self.sess = sess
        if self.sess is not None:
            try:
                self.l_in = sess.graph.get_tensor_by_name(name['l_in'])
                self.l_en = sess.graph.get_tensor_by_name(name['l_en'])
                self.l_de = sess.graph.get_tensor_by_name(name['l_de'])
            # self.l_cnn= sess.graph.get_tensor_by_name(name['l_cnn'])
            # self.y_   = sess.graph.get_operation_by_name('cnn-labels')
            # self.y_   = sess.graph.get_tensor_by_name(name['y_']),
            # self.cnn_cost = sess.graph.get_tensor_by_name(name['cnn_cost']),
            # self.cnn_optimizer = sess.graph.get_operation_by_name('cnn-optimizer')
            # self.droprate = sess.graph.get_tensor_by_name(name['droprate'])
            except:
                print("Some thing wrong when load the pretrained network...")
    '''
    def cnn_build(self, learning_rate):
        """Build the cnn for fine-tuning, just add a softmax layer after the encoder layer.
           Since the weights are shared between encoder and decoder, the fine-tuned weights
           can be reused.
        """
        def weight_variable(shape, name):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        # For instance the session is None,i.e. a new CNN network
        if self.sess is None:
            current_input = self.l_in
            for i, depth_output in enumerate(self.kernel_num):
                depth_input = current_input.get_shape().as_list()[3]
                W_name = "Conv_W{0}".format(i)
                b_name = "Conv_b{0}".format(i)
                W = weight_variable(shape=[self.kernel_size[i],
                                           self.kernel_size[i],
                                           depth_input,
                                           depth_output,
                                           ],
                                    name=W_name)
                b = bias_variable(shape=[depth_output], name=b_name)
                output = tf.add(tf.nn.conv2d(current_input,
                                             W, strides=[1,self.stride[0],self.stride[1],1],
                                             padding=self.pad_en), b)
                output = tf.nn.relu(output)
                current_input = output

            # Dense layer
            shape_conv = current_input.get_shape().as_list()
            depth_dense = shape_conv[1] * shape_conv[2] * shape_conv[3]
            l_en_dense = tf.reshape(current_input, [-1, depth_dense])
            # dropout layer
            # keep_prob = tf.placeholder(tf.float32)
            # fully connected
            depth_input = depth_dense
            current_input = l_en_dense
            for i, depth_output in enumerate(self.fc_nodes):
                W_name = "FC_W{0}".format(i)
                b_name = "FC_b{0}".format(i)
                W = weight_variable(shape=[depth_input, depth_output], name=W_name)
                b = bias_variable(shape=[depth_output], name=b_name)
                output = tf.nn.relu(tf.matmul(current_input, W) + b)
                # dropout
                output = tf.nn.dropout(output, self.droprate)
                current_input = output
                depth_input = depth_output

            # encode layer
            W_en = weight_variable(shape=[depth_input, self.encode_nodes], name="En_W")
            b_en = bias_variable(shape=[self.encode_nodes], name="En_b")
            output = tf.nn.relu(tf.matmul(current_input, W_en) + b_en)
            current_input = output
            self.l_en = current_input

        # add a softmax layer
        W_soft = weight_variable([self.encode_nodes, self.numclass], name='cnn_W')
        b_soft = bias_variable([self.numclass], name='cnn_b')

        self.l_cnn = tf.nn.softmax(tf.matmul(self.l_en, W_soft) + b_soft)
        # generate the optimizer
        self.gen_cnn_optimizer(learning_rate = learning_rate)
        # generate the accuracy estimator
        self.gen_accuracy()
    '''
    def gen_cnn_cost(self):
        """Generate the cost function for the added softmax layer at the end of encoder."""
        # cost function
        self.cnn_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.l_cnn))

    def gen_cnn_optimizer(self, learning_rate = 0.01):
        """Generate the optimizer for the fine-tuning cnn.

        Reference
        =========
        [1] Tensorflow: No gradients provided for any variable
            https://stackoverflow.com/questions/38778760/tensorflow-no-gradients-provided-for-any-variable
        """
        self.gen_cnn_cost()
        self.cnn_optimizer = tf.train.AdamOptimizer(learning_rate,name='cnn-optimizer').minimize(self.cnn_cost)

    def cnn_train(self, data, label, num_epochs=20, learning_rate=0.01, batch_size=100, droprate=0.5):
        """Train the net"""
        timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
        print("[%s] Training parameters\n" % (timestamp))
        print("[%s] Epochs: %d\tLearning rate: %.2f\n" % (timestamp, num_epochs, learning_rate))
        print("[%s] Batch size: %d\tDrop rate: %.2f\n" % (timestamp, batch_size, droprate))
        if self.sess is None:
            init_op = tf.global_variables_initializer()
            self.sess = tf.InteractiveSession()
            self.sess.run(init_op)

        # save loss for drawing
        self.cnn_trainloss = []
        self.cnn_valloss = []
        self.cnn_trainacc = []
        self.cnn_valacc = []
        # generate training and validation samples
        data_train,data_val = self.gen_validation(data=data, label=label)
        for epoch in range(num_epochs):
            cost = 0
            acc = 0
            data_batch,label_batch,numbatch = self.gen_cnn_BatchIterator(data=data_train["data"],
                                                                         label=data_train["label"],
                                                                         batch_size=batch_size)
            for i in range(numbatch):
                batch = data_batch[i*batch_size:(i+1)*batch_size,:,:,:]
                batch_label = label_batch[i*batch_size:(i+1)*batch_size,:]
                # print(batch_label.shape)
                self.sess.run(self.cnn_optimizer,
                              feed_dict={self.l_in: batch,
                                         self.y_: batch_label,
                                         self.droprate: droprate})
                cost += self.sess.run(self.cnn_cost,
                                      feed_dict={self.l_in: batch, self.y_: batch_label, self.droprate: 0.0})
                acc += self.cnn_accuracy(img=batch, label=batch_label)
            # validation
            valloss = self.sess.run(self.cnn_cost,
                                    feed_dict={self.l_in: data_val["data"],
                                               self.y_: data_val["label"],
                                               self.droprate: 0.0})
            valacc = self.cnn_accuracy(img=data_val["data"], label=data_val["label"])
            timestamp = time.strftime('%Y-%m-%d: %H:%M:%S', time.localtime(time.time()))
            print("[%s] Epoch: %03d Trn loss: %.6f Trn acc: %.3f Val loss: %.6f Val acc: %.3f" %
                (timestamp, epoch+1, cost/numbatch, acc/numbatch, valloss, valacc))
            self.cnn_trainloss.append(cost/numbatch)
            self.cnn_valloss.append(valloss)
            self.cnn_trainacc.append(acc/numbatch)
            self.cnn_valacc.append(valacc)

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
            label_pos = self.sess.run(self.l_cnn, feed_dict={self.l_in: img_te, self.droprate: 0.0})
            self.label_pre = tf.argmax(label_pos, 1, name="label_pre")
        else:
            print("You should train the network.")
            return None

        return self.label_pre.eval(),label_pos

    def gen_accuracy(self):
        """Generate the accuracy tensor"""
        self.correction_prediction = tf.equal(tf.argmax(self.l_cnn, 1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correction_prediction, tf.float32))

    def cnn_accuracy(self, img, label):
        """Calculate the classification accuracy of the cnn network.

        inputs
        ======
        img: np.ndarray
            The images to be classified
        label: np.ndarray, one hot
            The labels w.r.t. the images

        output
        ======
        acc: float
            Classification accuracy
        """
        return self.accuracy.eval(feed_dict={self.l_in: img, self.y_: label, self.droprate: 0.0})

    def cnn_save(self, namepath, netpath):
        """Save the net"""
        import pickle
        import sys
        sys.setrecursionlimit(1000000)

        # save the names
        savedict = {'l_in': self.l_in.name,
                    'l_cnn': self.l_cnn.name,
                    'y_':self.y_.name,
                    'droprate': self.droprate.name,
                    'netpath': netpath}
        with open(namepath, 'wb') as fp:
            pickle.dump(savedict, fp)

        # save the net
        saver = tf.train.Saver()
        saver.save(self.sess, netpath)
