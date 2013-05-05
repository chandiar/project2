"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import collections
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.ifelse import ifelse

from mlp import SoftmaxOutputLayer, HiddenLayer, DropoutHiddenLayer, MaxPoolingHiddenLayer, MaxoutHiddenLayer

import util
from util import dump_tar_bz2, get_theano_constant, save_model_info, save_model_losses_and_costs, save_model_params


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


class CNN(object):
    """Convolutional Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function thanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, inputs, n_in, hidden_sizes, n_out, activation=None,
                 dropout_p=False, maxout_k=False, nkerns=[20, 50], batch_size=500):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        :type nkerns: list of ints
        :param nkerns: number of kernels on each layer

        """

        print 'building CNN...'

        if dropout_p == -1 and maxout_k == -1:
            # No dropout or maxout.
            print 'No dropout or maxout!'
            self._init_ordinary(rng, inputs, n_in, hidden_sizes,
                                n_out, activation, nkerns, batch_size)
        else:
            # Dropout-only or maxout.
            self._init_with_dropout(rng, inputs, n_in, hidden_sizes, n_out,
                                    activation, dropout_p, maxout_k, nkerns,
                                    batch_size)


    def _init_ordinary(self,rng, inputs, n_in, hidden_sizes, n_out,
                       activation, nkerns, batch_size):

        self.layers = []
        # Build first the convolutional pooling layers.
        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # TODO: image shape hardcoded to 28 x 28 for mnist.
        layer0_input = inputs.reshape((batch_size, 1, 28, 28))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                image_shape=(batch_size, 1, 28, 28),
                filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

        self.layers.append(layer0)

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                image_shape=(batch_size, nkerns[0], 12, 12),
                filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

        self.layers.append(layer1)

        # the TanhLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        layer2_input = layer1.output.flatten(2)

        inputs = layer2_input
        n_in = nkerns[1] * 4 * 4
        if hidden_sizes[0] == 0:
            print 'No hidden layer is built.'
            print 'Ouput layer is a softmax.'
            output_layer = SoftmaxOutputLayer(
                inputs=inputs,
                n_in=n_in, n_out=n_out)
        else:
            print 'constructing ordinary %d layers...'%len(hidden_sizes)

            layer_sizes = [n_in] + hidden_sizes + [n_out]
            weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
            next_layer_input = inputs

            for n_in, n_out in weight_matrix_sizes[:-1]:
                print 'building hidden layer of size (%d,%d)'%(n_in, n_out)
                next_layer = HiddenLayer(rng=rng,
                                         input=next_layer_input,
                                         activation=activation,
                                         n_in=n_in, n_out=n_out,
                                     )
                self.layers.append(next_layer)
                next_layer_input = next_layer.output

            # The logistic regression (softmax) layer
            n_in, n_out = weight_matrix_sizes[-1]
            # TODO: number of output hardcoded to 10 for mnnist.
            assert n_out == 10
            print 'init softmax with NLL output layer...'
            output_layer = SoftmaxOutputLayer(
                        inputs=next_layer_input,
                        n_in=n_in, n_out=n_out)

        self.layers.append(output_layer)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L2 = 0
        #for layer in self.layers:
        #    self.L1 += abs(layer.W).sum()
        #    self.L2 += (layer.W ** 2).sum()

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression (softmax) layer
        self.nll = self.layers[-1].cost
        # same holds for the function computing the number of errors
        self.errors = self.layers[-1].cls_error

        # Parameters of the model consist of the parameters in every
        # layer (hidden and softmax output).
        self.params = [ param for layer in self.layers
                        for param in layer.params]


    def _init_with_dropout(self, rng, inputs, n_in, hidden_sizes, n_out,
                           activation, dropout_p, maxout_k, nkerns, batch_size):

        try:
            assert hidden_sizes[0] != 0
        except:
            raise AssertionError('You should not try dropout/maxout without '
                                 ' hidden layers')
        self.layers = []
        self.dropout_layers = []

        # Build first the convolutional pooling layers.
        # Reshape matrix of rasterized images of shape (batch_size,28*28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # TODO: image shape hardcoded to 28 x 28 for mnist.
        layer0_input = inputs.reshape((batch_size, 1, 28, 28))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
        # maxpooling reduces this further to (24/2,24/2) = (12,12)
        # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
        layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                image_shape=(batch_size, 1, 28, 28),
                filter_shape=(nkerns[0], 1, 5, 5), poolsize=(2, 2))

        self.layers.append(layer0)

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
        # maxpooling reduces this further to (8/2,8/2) = (4,4)
        # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
        layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                image_shape=(batch_size, nkerns[0], 12, 12),
                filter_shape=(nkerns[1], nkerns[0], 5, 5), poolsize=(2, 2))

        self.layers.append(layer1)

        # the TanhLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20,32*4*4) = (20,512)
        layer2_input = layer1.output.flatten(2)

        inputs = layer2_input
        n_in = nkerns[1] * 4 * 4

        layer_sizes = [n_in] + hidden_sizes + [n_out]
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        next_layer_input = inputs
        next_dropout_layer_input = DropoutHiddenLayer.apply_dropout(rng, inputs, 0.2)

        # Construct all the layers (input+hidden) except the top logistic
        # regression layer which is done afterwards.
        for n_in, n_out in weight_matrix_sizes[:-1]:
            if maxout_k == -1:
                print 'building Dropout hidden layer of size (%d,%d)'%(
                    n_in, n_out)
                next_dropout_layer = DropoutHiddenLayer(rng=rng,
                                input=next_dropout_layer_input,
                                activation=activation, n_in=n_in,
                                n_out=n_out, dropout_p=dropout_p
                                )
            else:
                print 'building Maxout hidden layer of size (%d,%d)'%(
                    n_in, n_out)
                next_dropout_layer = MaxoutHiddenLayer(rng=rng,
                                input=next_dropout_layer_input,
                                activation=activation, n_in=n_in, n_out=n_out,
                                dropout_p=dropout_p, maxout_k=maxout_k,
                                )
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output
            if maxout_k == -1:
                next_layer = HiddenLayer(rng=rng,
                                     input=next_layer_input,
                                     activation=activation,
                                     W=next_dropout_layer.W * 0.5,
                                     b=next_dropout_layer.b,
                                     n_in=n_in, n_out=n_out
                                    )
            else:
                next_layer = MaxPoolingHiddenLayer(rng=rng,
                                     input=next_layer_input,
                                     activation=activation,
                                     W=next_dropout_layer.W * 0.5,
                                     b=next_dropout_layer.b,
                                     n_in=n_in, n_out=n_out,
                                     maxout_k=maxout_k
                                    )
            self.layers.append(next_layer)
            next_layer_input = next_layer.output


        # The logistic regression (softmax) layer.
        n_in, n_out = weight_matrix_sizes[-1]
        print 'building output layer of size (%d,%d)'%(n_in, n_out)
        assert n_out == 10
        print 'init softmax with NLL output layer...'
        dropout_output_layer = SoftmaxOutputLayer(
                        inputs=next_dropout_layer_input,
                        n_in=n_in, n_out=n_out)

        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse parameters in the dropout output.
        # TODO: the number of ouput units is hardcoded to 10 for mnist.
        assert n_out == 10
        # Ouput layer is a softmax layer.
        output_layer = SoftmaxOutputLayer(
            inputs=next_layer_input,
            W=dropout_output_layer.W * 0.5,
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)

        self.layers.append(output_layer)

        self.dropout_nll = self.dropout_layers[-1].cost
        self.dropout_errors = self.dropout_layers[-1].cls_error

        self.nll = self.layers[-1].cost
        self.errors = self.layers[-1].cls_error

        # Get the params on the convolutional layers.
        self.params = layer0.params + layer1.params

        # Get the params on the drouput layers.
        self.params += [ param for layer in self.dropout_layers
                         for param in layer.params ]

        # Note: we will take the gradient of the cost w.r.t to the params
        # on the convolutional and dropout layers.

        self.L1 = 0
        self.L2 = 0


def train(state, channel, train_x, train_y, valid_x, valid_y, test_x, test_y):
    print 'train_x shape: ', train_x.get_value(borrow=True).shape
    print 'valid_x shape: ', valid_x.get_value(borrow=True).shape
    print 'test_x shape: ', test_x.get_value(borrow=True).shape

    # Compute number of minibatches for training, validation and testing.
    n_train_batches = train_x.get_value(borrow=True).shape[0] / state.batch_size
    n_valid_batches = valid_x.get_value(borrow=True).shape[0] / state.batch_size
    n_test_batches = test_x.get_value(borrow=True).shape[0] / state.batch_size

    # TODO: quiet display.
    print 'number of train batches: ', n_train_batches
    print 'number of valid batches: ', n_valid_batches
    print 'number of test batches: ', n_test_batches

    ######################
    # BUILD ACTUAL MODEL #
    ######################

    rng = numpy.random.RandomState(state.seed)


    ####### THEANO VARIABLES #######
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    tepoch = T.iscalar()
    tepoch.tag.test_value = 1

    # Regularization terms.
    state.L1 = numpy.array(state.L1,dtype='float32')
    state.L2 = numpy.array(state.L2,dtype='float32')
    L1_reg = get_theano_constant(state.L1, 'float32', ())
    L2_reg = get_theano_constant(state.L2, 'float32', ())

    # Compute momentum for the current epoch.
    # TODO: check momentum update formula.
    if state.mom == 0:
        # Do not use mom
        print 'Momentum is not used.'
        momentum = T.cast(state.mom * tepoch, 'float32')
    else:
        # TODO: find what is momentum and how it is computed.
        # https://github.com/mdenil/dropout/blob/master/mlp.py
        '''
        momentum = theano.ifelse.ifelse(tepoch < 500,
                    T.cast(state.mom*(1. - tepoch/500.) + 0.7*(tepoch/500.),
                    'float32'),T.cast(0.7, 'float32'))
        '''
        momentum = theano.ifelse.ifelse(tepoch < 500,
                    T.cast(state.mom*(1. - tepoch/500.) + 0.99*(tepoch/500.),
                    'float32'),T.cast(0.99, 'float32'))

    # end of theano variables initialization


    ####### BUILDING THE CNN MODEL #######
    # Construct the CNN class
    # TODO: Input and output size hardcoded for mnist.
    # TODO: n_in not necessary.
    classifier = CNN(rng=rng, inputs=x, n_in=28 * 28,
                     hidden_sizes=state.hidden_sizes, n_out=10,
                     activation=state.activation, dropout_p=state.dropout_p,
                     maxout_k=state.maxout_k, nkerns=state.nkerns,
                     batch_size=state.batch_size)


    # Get the number of model parameters.
    n_params = len(classifier.params)
    # Get right number of learning rates.
    state.init_lr *= n_params/2

    lr = theano.shared(numpy.asarray(state.init_lr,dtype='float32'))
    lr_0 = theano.shared(numpy.asarray(state.init_lr,dtype='float32'))
    new_lr = T.cast(lr_0 / (1.0 + state.decrease_constant*tepoch), 'float32')


    ####### Cost functions #######
    # the cost we minimize during training is the NLL of the model
    ### Cost function + regularization terms. ###
    cost_reg = L1_reg * classifier.L1 + L2_reg * classifier.L2
    if state.dropout_p == -1 and state.maxout_k == -1:
        cost_train = classifier.nll(y)
        cost_valid = classifier.nll(y)
    else:
        cost_train = classifier.dropout_nll(y)
        cost_valid = classifier.nll(y)

    cost = cost_reg + cost_train
    train_fn_output = [cost_train, cost_reg]

    # end of cost function


    ### Gradient of cost w.r.t. parameters ###
    # Compute the gradient of cost with respect to theta (stored in params).
    # The resulting gradients will be stored in a list gparams.
    gparams = []
    for param in classifier.params:
        gparam = T.grad(cost, param)
        gparams.append(gparam)

    # ... and allocate memory for momentum'd versions of the gradient
    gparams_mom = []
    for param in classifier.params:
        gparam_mom = theano.shared(
            numpy.zeros(param.get_value(borrow=True).shape,
            dtype=theano.config.floatX))
        gparams_mom.append(gparam_mom)

    # end of gradient of cost


    ### Updates rules for parameters ###
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = collections.OrderedDict()

    # Update the step direction using momentum
    # TODO: not need to include in the zip the init_lr.
    for gparam_mom, gparam in zip(gparams_mom, gparams):
        # the following is not consistent with Hinton's paper
        updates[gparam_mom] = momentum * gparam_mom + (1. - momentum) * gparam

    # ... and take a step along that direction
    for param, gparam_mom, gparam, i in zip(classifier.params, gparams_mom,
                                            gparams, range(len(state.init_lr))):
        # the following is not consistent with Hinton's paper
        # TODO: check this equation:
        stepped_param = param - lr[i] * gparam_mom
        #stepped_param = param - (1. - momentum) * lr[i] * gparam_mom

        # This is a silly hack to constrain the norms of the rows of the weight
        # matrices. This just checks if there are two dimensions to the
        # parameter and constrains it if so... maybe this is a bit silly but it
        # should work for now.
        # TODO: test this norms constraint.
        if param.get_value(borrow=True).ndim == 2:
            squared_norms = T.sum(stepped_param**2, axis=1).reshape((
                stepped_param.shape[0],1))
            scale = T.clip(T.sqrt(state.filter_square_limit / squared_norms), 0., 1.)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param

    # end of updates


    ####### THEANO FUNCTIONS #######
    print 'Compiling theano functions...'
    t0 = time.time()

    momentum_fn = theano.function(inputs=[tepoch],
                            outputs=T.as_tensor_variable(momentum),
                            name='returns new momentum')

    # use 1/t decay where t is epoch
    decay_learning_rate_fn = theano.function(inputs=[tepoch],
            outputs=T.as_tensor_variable(lr),
            name='decay learning rate with 1/t',
            updates=collections.OrderedDict({lr: new_lr}))

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model_on_batch_fn = theano.function(inputs=[index, tepoch], outputs=train_fn_output,
            updates=updates,
            name='computes the train cost with params updates',
            givens={
                x: train_x[index * state.batch_size:(index + 1) * state.batch_size],
                y: train_y[index * state.batch_size:(index + 1) * state.batch_size]},
            on_unused_input='ignore')


    # Classification errors and NLL functions.
    train_error_and_cost_on_batch_fn = theano.function(inputs=[index],
        outputs=[classifier.errors(y), cost_train],
        name='returns train classification errors and NLL for a given mini-batch',
        givens={
            x: train_x[index * state.batch_size:(index + 1) * state.batch_size],
            y: train_y[index * state.batch_size:(index + 1) * state.batch_size]})

    valid_error_and_cost_on_batch_fn = theano.function(inputs=[index],
            outputs=[classifier.errors(y), cost_valid],
            name='returns validation classification errors and NLL for a given mini-batch',
            givens={
                x: valid_x[index * state.batch_size:(index + 1) * state.batch_size],
                y: valid_y[index * state.batch_size:(index + 1) * state.batch_size]})

    test_error_and_cost_on_batch_fn = theano.function(inputs=[index],
            outputs=[classifier.errors(y), cost_valid],
            name='returns test classification errors and NLL for a given mini-batch',
            givens={
                x: test_x[index * state.batch_size:(index + 1) * state.batch_size],
                y: test_y[index * state.batch_size:(index + 1) * state.batch_size]})


    # NNET outputs.
    train_output_on_batch_fn = theano.function(inputs=[index],
        outputs=[classifier.layers[-1].p_y_given_x],
        name='returns train nnet output for a given mini-batch',
        givens={
            x: train_x[index * state.batch_size:(index + 1) * state.batch_size]})

    valid_output_on_batch_fn = theano.function(inputs=[index],
            outputs=classifier.layers[-1].p_y_given_x,
            name='returns validation nnet output for a given mini-batch',
            givens={
                x: valid_x[index * state.batch_size:(index + 1) * state.batch_size]})

    test_output_on_batch_fn = theano.function(inputs=[index],
            outputs=classifier.layers[-1].p_y_given_x,
            name='returns test nnet output for a given mini-batch',
            givens={
                x: test_x[index * state.batch_size:(index + 1) * state.batch_size]})


    # Predictions and Targets functions.
    train_pred_and_targ_on_batch_fn = theano.function(inputs=[index],
            outputs=[classifier.layers[-1].y_pred,
                        train_y[index * state.batch_size:(index + 1) * state.batch_size]],
            name='returns the train predictions and targets for a given mini-batch',
            givens={
                x: train_x[index * state.batch_size:(index + 1) * state.batch_size]})

    valid_pred_and_targ_on_batch_fn = theano.function(inputs=[index],
            outputs=[classifier.layers[-1].y_pred,
                        valid_y[index * state.batch_size:(index + 1) * state.batch_size]],
            name='returns the validation predictions and targets for a given mini-batch',
            givens={
                x: valid_x[index * state.batch_size:(index + 1) * state.batch_size]})

    test_pred_and_targ_on_batch_fn = theano.function(inputs=[index],
            outputs=[classifier.layers[-1].y_pred,
                    test_y[index * state.batch_size:(index + 1) * state.batch_size]],
            name='returns the test predictions and targets for a given minbi-batch',
            givens={
                x: test_x[index * state.batch_size:(index + 1) * state.batch_size]})

    t1 = time.time()
    print 'Compilation takes %d seconds' %(t1-t0)

    # end of theano functions compilation


    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    validation_frequency = min(n_train_batches, state.patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    best_epoch = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    # ALL losses (classification errors).
    all_train_losses = {}
    all_valid_losses = {}
    all_test_losses = {}
    all_train_costs_with_reg = {}
    # ALL costs with NO regularization.
    all_train_costs_without_reg = {}
    all_valid_costs = {}
    all_test_costs = {}
    # Best validation and test predictions based on validation nll.
    best_valid_pred_and_targ = []
    best_test_pred_and_targ = []
    # Best validation and test nnet outputs (vector of probabilities)
    # based on validation nll.
    best_valid_p_y_given_x = []
    best_test_p_y_given_x = []
    # Best model params so far based on validation nll.
    best_params = []

    vpredictions = []
    tpredictions = []

    print 'Initial learning rate: ', state.init_lr
    if state.lr_decay:
        print 'Learning decay enabled'
    else:
        print 'No learning decay!'

    print 'C-c to skip'
    try:
        while (epoch < state.n_epochs) and (not done_looping):
            train_costs_with_reg = []
            train_costs_without_reg = []
            train_losses = []

            for minibatch_index in xrange(n_train_batches):

                # Train cost with regularization.
                minibatch_avg_cost = sum(train_model_on_batch_fn(minibatch_index, epoch))
                train_costs_with_reg.append(minibatch_avg_cost)

                if state.save_losses_and_costs:
                    # Train costs with no regularization.
                    train_error, train_cost = train_error_and_cost_on_batch_fn(minibatch_index)
                    train_losses.append(train_error)
                    train_costs_without_reg.append(train_cost)

                # iteration number
                iter = epoch * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss and the nll on the validation set.
                    v_losses = []
                    v_costs = []
                    for i in xrange(n_valid_batches):
                        loss, cost = valid_error_and_cost_on_batch_fn(i)
                        v_losses.append(loss)
                        v_costs.append(cost)

                    this_validation_loss = numpy.mean(v_losses)
                    this_validation_cost = numpy.mean(v_costs)

                    if state.save_losses_and_costs:
                        all_valid_losses.setdefault(epoch, 0)
                        all_valid_losses[epoch] = this_validation_loss

                        all_valid_costs.setdefault(epoch, 0)
                        all_valid_costs[epoch] = this_validation_cost

                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch, minibatch_index + 1, n_train_batches,
                        this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        # improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                            state.improvement_threshold:
                            state.patience = max(state.patience, iter * state.patience_increase)

                        best_validation_loss = this_validation_loss
                        v_p_y_given_x = []
                        for i in xrange(n_valid_batches):
                            p_y_given_x = valid_output_on_batch_fn(i)
                            v_p_y_given_x.append(p_y_given_x)

                        # Best nnet ouputs on validation set.
                        best_valid_p_y_given_x = v_p_y_given_x

                        if state.save_model_info:
                            # Best predictions on validation set.
                            best_valid_pred_and_targ = numpy.array([valid_pred_and_targ_on_batch_fn(i)
                                                               for i in xrange(n_valid_batches)])

                        best_iter = iter
                        best_epoch = epoch
                        # Best model params so far.
                        best_params = classifier.params

                        # test it on the test set
                        # compute zero-one loss and the nll on the test set.
                        t_losses = []
                        t_costs = []
                        t_p_y_given_x = []

                        for i in xrange(n_test_batches):
                            loss, cost = test_error_and_cost_on_batch_fn(i)
                            p_y_given_x = test_output_on_batch_fn(i)
                            t_losses.append(loss)
                            t_costs.append(cost)
                            t_p_y_given_x.append(p_y_given_x)

                        test_score = numpy.mean(t_losses)
                        this_test_cost = numpy.mean(t_costs)
                        # Best nnet ouputs on test set.
                        best_test_p_y_given_x = t_p_y_given_x

                        if state.save_losses_and_costs:
                            all_test_losses.setdefault(epoch, 0)
                            all_test_losses[epoch] = test_score

                            all_test_costs.setdefault(epoch, 0)
                            all_test_costs[epoch] = this_test_cost

                        if state.save_model_info:
                            # Best predictions on test set.
                            best_test_pred_and_targ = numpy.array([test_pred_and_targ_on_batch_fn(i)
                                                                   for i in xrange(n_test_batches)])

                        print(('     epoch %i, minibatch %i/%i, test error of '
                            'best model %f %%') %
                            (epoch, minibatch_index + 1, n_train_batches,
                            test_score * 100.))

                if state.patience <= iter:
                    done_looping = True
                    break

            # Update learning rate.
            if state.lr_decay:
                new_learning_rate = decay_learning_rate_fn(epoch)
                print 'New learning rate: ', lr.get_value()

            # Update momentum.
            new_momentum = momentum_fn(epoch)
            print 'New momentum: ', new_momentum

            if state.save_losses_and_costs and False:
                all_train_costs_with_reg.setdefault(epoch, 0)
                all_train_costs_with_reg[epoch] = numpy.mean(train_costs_with_reg)
                all_train_costs_without_reg.setdefault(epoch, 0)
                all_train_costs_without_reg[epoch] = numpy.mean(train_costs_without_reg)
                all_train_losses.setdefault(epoch, 0)
                all_train_losses[epoch] = numpy.mean(train_losses)

            # Update epoch counter.
            epoch = epoch + 1
    except KeyboardInterrupt:
        print '\n\nskip !'


    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
        'obtained at iteration %i (epoch %i), with test performance %f %%') %
        (best_validation_loss * 100., best_iter + 1, best_epoch, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                        os.path.split(__file__)[1] +
                        ' ran for %.2fm' % ((end_time - start_time) / 60.))


    if state.save_losses_and_costs:
        save_model_losses_and_costs(state.model, all_train_losses, all_valid_losses,
            all_test_losses, all_train_costs_with_reg, all_train_costs_without_reg,
            all_valid_costs, all_test_costs)

    if state.save_model_info:
        # Save the best model valid/test predictions, targets and nnet outputs.
        save_model_info(state.model, best_valid_pred_and_targ,
                        best_test_pred_and_targ, best_valid_p_y_given_x,
                        best_test_p_y_given_x)

    if state.save_model_params:
        save_model_params(state.model, best_params)

    if state.save_state:
        print 'We will save the experiment state'
        dump_tar_bz2(state, 'state.tar.bz2')

    try:
        channel.save()
    except:
        print 'Not in experiment, done!'

    return 0

'''
http://arxiv.org/pdf/1102.0183v1.pdf
http://research.microsoft.com/pubs/152133/DeepConvexNetwork-Interspeech2011-pub.pdf

http://axon.cs.byu.edu/papers/IstookIJNS.pdf
http://www.cse.unsw.edu.au/~billw/mldict.html

https://www.lri.fr/~xlzhang/KAUST/CS229_slides/c10_NN.pdf
http://clopinet.com/isabelle/Projects/ETH/lecture12.pdf
http://courses.cs.tamu.edu/choe/08spring/lectures/slide07.pdf
http://www.cse.unsw.edu.au/~cs9444/Notes09/9444-06.pdf
http://www.cs.bham.ac.uk/~jxb/NN/l15.pdf
http://books.google.ca/books?id=OrJsIULAH7MC&pg=PA85&lpg=PA85&dq=committee+machine+ensemble+averaging&source=bl&ots=yXi2iuZ9Hf&sig=szoiS5RS8Lj572DvdesYVNA899E&hl=en&sa=X&ei=wKeCUZvBKsGmqAHL_oHIDw&redir_esc=y#v=onepage&q=committee%20machine%20ensemble%20averaging&f=false
http://en.wikipedia.org/wiki/Ensemble_averaging#cite_note-haykin-1

maxout
http://arxiv.org/pdf/1302.4389.pdf
https://github.com/lisa-lab/pylearn2/tree/master/pylearn2/scripts/papers/maxout

dropout
http://arxiv.org/pdf/1207.0580.pdf

backpropagation
http://www.stanford.edu/group/pdplab/pdphandbook/handbookch6.html
http://engineeronadisk.com/V2/hugh_jack_masters/engineeronadisk-10.html
http://www.nnwj.de/backpropagation.html
http://www.willamette.edu/~gorr/classes/cs449/backprop.html
http://www.ra.cs.uni-tuebingen.de/SNNS/UserManual/node15.html
http://clemens.bytehammer.com/papers/BackProp/
http://msdn.microsoft.com/en-us/magazine/jj190808.aspx

IIya thesis
http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf

http://tams.informatik.uni-hamburg.de/lectures/2012ss/vorlesung/al/folien/svm2.pdf
http://leon.bottou.org/papers/loosli-canu-bottou-2006
http://research.microsoft.com/pubs/68920/icdar03.pdf
http://arxiv.org/pdf/1103.4487v1.pdf
https://docs.google.com/viewer?a=v&q=cache:S7w2b6m9pN4J:research.microsoft.com/~jplatt/ICDAR03.pdf+&hl=en&pid=bl&srcid=ADGEESiHAFlh2Gja8yvhk8k-2VDw7cz_68Jax3SmMOm57aJD5asdAf0BqF1RpFzBNwmoe3ozH5iCjK5qD0BZcpJxU6dfuRp_s4SeZWDfhfeGOEw8xCKe3vi8ia5myiWPRIhCs-KwnrF7&sig=AHIEtbRODNYcpDzPruhXRutKHNLhDF1sgg

http://www.dmi.usherb.ca/~larocheh/publications/deep-nets-icml-07.pdf
http://www.dmi.usherb.ca/~larocheh/mlpython/datasets.html
Mnist variations
http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/MnistVariations

using gpu theano
http://deeplearning.net/software/theano/tutorial/using_gpu.html

http://www.idsia.ch/~juergen/icdar2011b.pdf

python job
http://pythonhosted.org/joblib/

sift
http://www.aishack.in/2010/05/sift-scale-invariant-feature-transform/














'''