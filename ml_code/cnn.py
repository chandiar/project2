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

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer


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


def train(state, channel, train_x, train_y, valid_x, valid_y, test_x, test_y):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
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
    print '... building the model'

    rng = numpy.random.RandomState(state.seed)

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    tepoch = T.iscalar()
    tepoch.tag.test_value = 1
    lr = theano.shared(numpy.asarray(state.init_lr,dtype='float32'))
    lr_0 = theano.shared(numpy.asarray(state.init_lr,dtype='float32'))
    new_lr = T.cast(lr_0 / (1.0 + state.decrease_constant*tepoch), 'float32')

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((state.batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(state.batch_size, 1, 28, 28),
            filter_shape=(state.nkerns[0], 1, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(state.batch_size, state.nkerns[0], 12, 12),
            filter_shape=(state.nkerns[1], state.nkerns[0], 5, 5), poolsize=(2, 2))

    # the TanhLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=state.nkerns[1] * 4 * 4,
                         n_out=500, activation='tanh')

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
             givens={
                x: test_x[index * state.batch_size: (index + 1) * state.batch_size],
                y: test_y[index * state.batch_size: (index + 1) * state.batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
            givens={
                x: valid_x[index * state.batch_size: (index + 1) * state.batch_size],
                y: valid_y[index * state.batch_size: (index + 1) * state.batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i, i in zip(params, grads, range(len(state.init_lr))):
        updates.append((param_i, param_i - lr[i] * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_x[index * state.batch_size: (index + 1) * state.batch_size],
            y: train_y[index * state.batch_size: (index + 1) * state.batch_size]})

    # use 1/t decay where t is epoch
    decay_learning_rate_fn = theano.function(inputs=[tepoch],
            outputs=T.as_tensor_variable(lr),
            name='decay learning rate with 1/t',
            updates=collections.OrderedDict({lr: new_lr}))

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    validation_frequency = min(n_train_batches, state.patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    best_epoch = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    all_train_losses = {}
    all_valid_losses = {}
    all_test_losses = {}
    all_train_costs_with_reg = {}
    # Train costs with NO regularization.
    all_train_costs_without_reg = {}
    all_valid_costs = {}
    all_test_costs = {}
    # Best validation and test nnet outputs (vector of probabilities).
    best_valid_p_y_given_x = []
    best_test_p_y_given_x = []

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
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                iter = (epoch - 1) * n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter
                cost_ij = train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                        in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                        (epoch, minibatch_index + 1, n_train_batches, \
                        this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                        state.improvement_threshold:
                            state.patience = max(state.patience, iter * state.patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)
                        print(('     epoch %i, minibatch %i/%i, test error of best '
                            'model %f %%') %
                            (epoch, minibatch_index + 1, n_train_batches,
                            test_score * 100.))

                if state.patience <= iter:
                    done_looping = True
                    break
    except KeyboardInterrupt:
        print '\n\nskip !'

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


    return 0

    if state.save_model:
        # Save the model.
        print 'Saving model not implemented.'

    if state.save_model_info:
        # Save the model valid/test predictions and targets.
        print 'Saving the model predictions and targets'
        path = '%s_pred_and_targ.npz'%state.model
        valid_pred = valid_pred_and_targ[:,0,:]
        valid_targ = valid_pred_and_targ[:,1,:]
        test_pred = test_pred_and_targ[:,0,:]
        test_targ = test_pred_and_targ[:,1,:]
        numpy.savez_compressed(path, valid_pred=valid_pred,
                                     valid_targ=valid_targ,
                                     test_pred=test_pred,
                                     test_targ=test_targ)

    if state.save_losses_and_costs:
        print 'Saving losses and costs'
        path = '%s_losses.npz'%state.model
        numpy.savez_compressed(path, all_train_losses=all_train_losses,
                                     all_valid_losses=all_valid_losses,
                                     all_test_losses=all_test_losses)

        path = '%s_costs.npz'%state.model
        numpy.savez(path, train_costs_with_reg=all_train_costs_with_reg,
                          train_costs_without_reg=all_train_costs_without_reg,
                          valid_costs=all_valid_costs,
                          test_costs=all_test_costs)

    try:
        channel.save()
    except:
        print 'Not in experiment, done!'

    return 0