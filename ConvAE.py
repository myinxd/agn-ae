# Copyright (C) 2017 Zhixian MA <zxma_sjtu@qq.com>

"""
A convolutional autoencoder (CAE) constructor based on theano, lasagne, and
nolearn

Reference
=========
[1] convolutional_autoencoder (mikesj)
    https://github.com/mikesj-public/convolutional_autoencoder
[2] nolearn.lasagne
    http://pythonhosted.org/nolearn/lasagne.html
"""

import lasagne
from lasagne.layers import get_output
from lasagne.layers import InputLayer, DenseLayer, Upscale2DLayer, ReshapeLayer
from lasagne.layers import Conv2DLayer, MaxPool2DLayer
from lasagne.layers import DropoutLayer
from nolearn.lasagne import NeuralNet, BatchIterator, PrintLayerInfo

class ConvAE():
    """
    A convolutional autoencoder (CAE) constructor

    inputs
    ======
    X_in: np.ndarray
        The sample matrix, whose size is (s,d,r,c).
        s: number of samples, d: dimensions
        r: rows, c: cols
    X_out: np.ndarray
        It usually equals to X_in.
    kernel_size: list
        Box sizes of the kernels in each ConvLayer
    kernel_num: list
        Number of kernels in each ConvLayer
    pool_flag: list of bool values
        Flags of pooling layer w.r.t. to the ConvLayer
    fc_nodes: list
        The dense layers after the full connected layer
        of last ConvLayer or pooling layer.
    encode_nodes: int
        Number of nodes in the final encoded layer

    methods
    =======
    gen_layers: construct the layers
    gen_encoder: generate the encoder
    gen_decoder: generate the decoder
    cae_build: build the cae network
    cae_train: train the cae network
    cae_eval: evaluate the cae network
    cae_save: save the network
    """

    def __init__(self, X_in, X_out, kernel_size=[3,5,3],
                 kernel_num=[12,12,24], pool_flag=[True,True,True],
                 fc_nodes=[128], encode_nodes=16, 
                 droprate=0.5, dropflag=True):
        """
        The initializer
        """
        self.X_in = X_in
        self.X_out = X_out
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.pool_flag = pool_flag
        self.pool_size = 2
        self.fc_nodes = fc_nodes
        self.encode_nodes = encode_nodes
        self.droprate = droprate
        self.dropflag = dropflag

    def gen_BatchIterator(self,batch_size=100):
        """Generate the batch iterator"""
        B = BatchIterator(batch_size=batch_size, shuffle=True)
        return B

    def gen_layers(self):
        """Construct the layers"""

        # Init <TODO>
        pad_in = 'valid'
        pad_out = 'full'
        self.layers = []
        # input layer
        l_input = (InputLayer,
                   {'shape': (None, self.X_in.shape[1],
                              self.X_in.shape[2],
                              self.X_in.shape[3])})
        self.layers.append(l_input)
        # Encoder: Conv and pool layers
        rows,cols = self.X_in.shape[2:]
        for i in range(len(self.kernel_size)):
            # conv
            l_en_conv = (Conv2DLayer,
                         {'num_filters': self.kernel_num[i],
                          'filter_size': self.kernel_size[i],
                          'nonlinearity': lasagne.nonlinearities.rectify,
                          'W': lasagne.init.GlorotUniform(),
                          'b': lasagne.init.Constant(0.),
                          'pad': pad_in})
            self.layers.append(l_en_conv)
            rows = rows - self.kernel_size[i] + 1
            cols = cols - self.kernel_size[i] + 1
            # pool
            if self.pool_flag[i]:
                l_en_pool = (MaxPool2DLayer,
                             {'pool_size': self.pool_size})
                self.layers.append(l_en_pool)
                rows = rows // 2
                cols = cols // 2
        # drop
        if not self.dropflag:
            self.droprate = 0
        l_drop = (DropoutLayer, {'p': self.droprate})
        # full connected layer
        num_en_fc = rows * cols * self.kernel_num[-1]
        l_en_fc = (ReshapeLayer, 
                    {'shape': (([0], -1))})
        self.layers.append(l_en_fc)
        self.layers.append(l_drop)
        # dense
        for i in range(len(self.fc_nodes)):
            l_en_dense = (DenseLayer, 
                    {'num_units': self.fc_nodes[i],
                     'nonlinearity': lasagne.nonlinearities.rectify,
                     'W': lasagne.init.GlorotUniform(),
                     'b': lasagne.init.Constant(0.)}
                    )
            self.layers.append(l_en_dense)
            self.layers.append(l_drop)
        # encoder layer
        l_en = (DenseLayer,
                {'name': 'encode', 
                 'num_units': self.encode_nodes,
                 'nonlinearity': lasagne.nonlinearities.rectify,
                 'W': lasagne.init.GlorotUniform(),
                 'b': lasagne.init.Constant(0.)})
        self.layers.append(l_en)
        self.layers.append(l_drop)

        # Decoder: reverse
        # dense
        for i in range(len(self.fc_nodes)-1, -1, -1):
            l_de_dense = (DenseLayer, 
                    {'num_units': self.fc_nodes[i],
                     'nonlinearity': lasagne.nonlinearities.rectify,
                     'W': lasagne.init.GlorotUniform(),
                     'b': lasagne.init.Constant(0.)})
            self.layers.append(l_de_dense)
            self.layers.append(l_drop)
        # fc
        l_de_fc = (DenseLayer, 
                {'num_units': num_en_fc,
                 'nonlinearity': lasagne.nonlinearities.rectify,
                 'W': lasagne.init.GlorotUniform(),
                 'b': lasagne.init.Constant(0.)})
        self.layers.append(l_de_fc)
        self.layers.append(l_drop)
        # fc to kernels
        l_de_fc_m = (ReshapeLayer,
                     {'shape': (([0], self.kernel_num[-1], rows, cols))})
        self.layers.append(l_de_fc_m)
        # Conv and pool
        for i in range(len(self.kernel_size)-1, -1, -1):
            # pool
            if self.pool_flag[i]:
                l_de_pool = (Upscale2DLayer,
                             {'scale_factor': self.pool_size})
                self.layers.append(l_de_pool)
            # conv
            if i > 0:
                l_de_conv = (Conv2DLayer,
                             {'num_filters': self.kernel_num[i],
                              'filter_size': self.kernel_size[i],
                              'nonlinearity': lasagne.nonlinearities.rectify,
                              'W': lasagne.init.GlorotUniform(),
                              'b': lasagne.init.Constant(0.),
                              'pad': pad_out})
            else:
                l_de_conv = (Conv2DLayer,
                             {'num_filters': 1,
                              'filter_size': self.kernel_size[i],
                              'nonlinearity': lasagne.nonlinearities.rectify,
                              'W': lasagne.init.GlorotUniform(),
                              'b': lasagne.init.Constant(0.),
                              'pad': pad_out})
            self.layers.append(l_de_conv)
        # output
        self.layers.append((ReshapeLayer, {'shape': (([0], -1))}))

    def cae_build(self, max_epochs=20, batch_size=100,
                  learning_rate=0.001, momentum=0.9,
                  verbose=1):
        """Build the network"""
        if batch_size is None:
            self.cae = NeuralNet(
                layers = self.layers,
                max_epochs = max_epochs,
                update=lasagne.updates.nesterov_momentum,
                update_learning_rate = learning_rate,
                update_momentum = momentum,
                regression = True,
                verbose = verbose)
        else:
            # batch iterator
            batch_iterator = self.gen_BatchIterator(batch_size=batch_size)
            self.cae = NeuralNet(
                layers = self.layers,
                batch_iterator_train = batch_iterator,
                max_epochs = max_epochs,
                update=lasagne.updates.nesterov_momentum,
                update_learning_rate = learning_rate,
                update_momentum = momentum,
                regression = True,
                verbose = verbose)

    def cae_train(self):
        """Train the cae net"""
        print("Training the network...")
        self.cae.fit(self.X_in, self.X_out)
        print("Training done.")
 
    def cae_eval(self):
        """Draw evaluation lines
        <TODO>
        """
        from nolearn.lasagne.visualize import plot_loss
        plot_loss(self.cae)

    def cae_predict(self,img):
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
        img_pred = self.cae.precidt(img).reshape(-1, rows, cols)

        return img_pred

    def get_encode(self, img):
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

        encode_layer, encode_layer_index = get_layer_by_name(self.cae,
                                                             'encode')
        img_en =  get_output(encode_layer, inputs=img).eval()

        return img_en

    def get_decode(self, img_en):
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
        def get_layer_by_name(net, name):
            for i, layer in enumerate(net.get_all_layers()):
                if layer.name == name:
                    return layer, i
            return None, None

        encode_layer, encode_layer_index = get_layer_by_name(self.cae,
                                                             'encode')
        # decoder
        new_input = InputLayer(shape=(None,encode_layer.num_units))
        layer_de_input = self.cae.get_all_layers()[encode_layer_index + 1]
        layer_de_input.input_layer = new_input
        layer_de_output = self.cae.get_all_layers()[-1]

        img_en = get_output(layer_de_output, img_en).eval()

        return img_en

    def cae_save(self,savepath='cae.pkl'):
        """Save the trained network

        input
        =====
        savepath: str
            Path of the net to be saved
        """
        import pickle
        try:
            fp = open(savepath, 'wb')
        except FileNotFoundError:
            import os
            os.system("touch %s" % savepath)
            fp = open(savepath, 'wb')
        # write
        pickle.dump(self.cae, fp)
        fp.close()
