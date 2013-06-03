"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import collections
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.ifelse import ifelse

import util
from util import dump_tar_bz2, get_theano_constant, save_model_info, save_model_losses_and_costs, save_model_params

rectified_linear_activation = lambda x: T.maximum(0.0, x)


class LinearOutputLayer(object):
    def __init__(self, inputs, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the linear output layer.

        :type inputs: theano.tensor.TensorType
        :param inputs: symbolic variable that describes the input of the
                       architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        self.n_in = n_in
        self.n_out = n_out

        if self.n_out == 1:
            self.W = theano.shared(value=numpy.zeros((n_in,1),
                                dtype='float32'),
                                name='W')
            # initialize the biases b as a vector of n_out 0s
            self.b = theano.shared(value=numpy.asarray(0).astype('float32'),
                                   name='b')
        else:
            # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
            self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                dtype='float32'),name='W')
            # initialize the baises b as a vector of n_out 0s
            self.b = theano.shared(value=numpy.zeros((n_out,),
                                dtype='float32'), name='b')

        if W is not None:
            self.W = W
        if b is not None:
            # dropout will pass b from its dropout_layers[-1]
            self.b = b

        self.output = T.dot(inputs, self.W) + self.b

        # parameters of the model
        self.params = [self.W, self.b]


class CrossEntropyOutputLayer(LinearOutputLayer):
    def __init__(self, inputs, n_in, n_out, W=None, b=None):
        assert n_out == 1
        super(CrossEntropyOutputLayer, self).__init__(inputs, n_in, n_out, W, b)
        self.p_y_given_x = T.nnet.sigmoid(self.output)
        self.y_pred = T.switch(T.ge(self.p_y_given_x, 0.5), 1, 0)

    def cost(self, y):
        """
        y should be a vector representing probabilities
        """
        predicts = self.p_y_given_x
        predicts = predicts.flatten()
        rval = T.mean(- y*T.log(predicts) - (1-y)*T.log(1-predicts))

        return rval

    def cls_error(self, y):
        # Note: if y is a binary target, we don't have to divide the target
        # into 2 parts. However, when classifying the fun ratings, we will
        # have to this division.
        if y.dtype.startswith('float'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            y = T.switch(T.lt(y,0.5), 0, 1)
        #else:
            #raise NotImplementedError()
        return T.mean(T.neq(self.y_pred.flatten(), y))


class SoftmaxOutputLayer(LinearOutputLayer):
    def __init__(self, inputs, n_in, n_out, W=None, b=None):
        assert n_out > 1
        super(SoftmaxOutputLayer, self).__init__(inputs, n_in, n_out, W, b)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(inputs, self.W) + self.b)
        self.output = self.p_y_given_x

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def cost(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        rval = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        return rval

    def cls_error(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation='tanh'):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: string
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.n_in = n_in
        self.n_out = n_out
        self.input = input
        self.activation = activation

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # float32 so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype='float32')
            #if self.activation == theano.tensor.nnet.sigmoid:
            if self.activation == 'sigmoid':
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype='float32')
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        linear_output = T.dot(input, self.W) + self.b

        if self.activation == 'rectifier':
            self.output = rectified_linear_activation(linear_output)
        elif self.activation == 'linear':
            self.output = linear_output
        elif self.activation == 'sigmoid':
            self.output = T.nnet.sigmoid(linear_output)
        elif self.activation == 'tanh':
            self.output = T.tanh(linear_output)
        elif self.activation == 'softplus':
            self.output = T.nnet.softplus(linear_output)
        else:
            raise NotImplementedError('activation function %s not supported.'%self.activation)

        # parameters of the model
        self.params = [self.W, self.b]


class DropoutHiddenLayer(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out,
                 activation, W=None, b=None, dropout_p=0.5):

        print 'constructing the dropout layer with p=%f'%dropout_p

        super(DropoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out, W=W, b=b,
                activation=activation)

        self.output = self.apply_dropout(rng, self.output, p=dropout_p)

    @classmethod
    def apply_dropout(cls, rng, inputs, p):
        """p is the probablity of dropping a unit
        """
        #srng = theano.tensor.shared_randomstreams.RandomStreams(
        #        rng.randint(999999))
        # the following version is much faster
        srng = MRG_RandomStreams(rng.randint(999999))
        # p=1-p because 1's indicate keep and p is prob of dropping
        mask = srng.binomial(n=1, p=1-p, size=inputs.shape)
        # The cast is important because
        # int * float32 = float64 which pulls things off the gpu
        output = inputs * T.cast(mask, 'float32')

        return output


class MaxPoolingHiddenLayer(HiddenLayer):
    """
    with pooling on the linear, followed by max
    """
    # TODO : is activation being used?
    def __init__(self, rng, input, n_in, n_out, activation,
                 W=None, b=None, maxout_k=5):
        self.n_in = n_in
        self.n_out = n_out
        self.input = input
        if W is None:
            #W_values = numpy.asarray(rng.uniform(
            #        low=-numpy.sqrt(6. / (n_in + n_out*maxout_k)),
            #        high=numpy.sqrt(6. / (n_in + n_out*maxout_k)),
            #        size=(n_in, n_out*maxout_k)), dtype=float32)
            W_values = numpy.asarray(rng.normal(
                    loc=0, scale=0.005,
                    size=(n_in, n_out*maxout_k)), dtype='float32')
            W = theano.shared(value=W_values, name='W')
        # rebuild b with proper shape
        if b is None:
            b_values = numpy.zeros((n_out,), dtype='float32')
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        self.output = T.dot(input, self.W)
        n_samples = self.output.shape[0]
        self.output = self.output.reshape((n_samples, n_out, maxout_k))
        self.output = T.max(self.output, axis=2) + self.b

        self.params = [self.W, self.b]


class MaxoutHiddenLayer(MaxPoolingHiddenLayer):
    # TODO: is activation being used?
    def __init__(self, rng, input, n_in, n_out, activation,
                 W=None, b=None, dropout_p=0.5, maxout_k=5):
        print 'constructing the Maxout layer with p=%f and k=%f'%(
            dropout_p, maxout_k)

        super(MaxoutHiddenLayer, self).__init__(
                rng=rng, input=input, n_in=n_in, n_out=n_out,
                maxout_k=maxout_k, W=W, b=b, activation=activation)

        self.output = DropoutHiddenLayer.apply_dropout(rng, self.output, p=dropout_p)


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``SigmoidalLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, inputs, n_in, hidden_sizes, n_out, output_activation,
                 hidden_activation=None, dropout_p=False, maxout_k=False):
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

        """

        print 'building MLP...'

        self.output_activation = output_activation

        if dropout_p == -1 and maxout_k == -1:
            # No dropout or maxout.
            print 'No dropout or maxout!'
            self._init_ordinary(rng, inputs, n_in, hidden_sizes,
                                n_out, hidden_activation)
        else:
            # Dropout-only or maxout.
            self._init_with_dropout(rng, inputs, n_in, hidden_sizes, n_out,
                                    hidden_activation, dropout_p, maxout_k)

    def _init_ordinary(self,rng, inputs, n_in, hidden_sizes, n_out, hidden_activation):

        self.layers = []
        if hidden_sizes[0] == 0:
            print 'No hidden layer is built.'
            if self.output_activation == 'regression':
                print 'Output layer is a regression layer.'
                raise NotImplementedError('Not yet supported!')
            elif self.output_activation == 'binary_cross_entropy':
                print 'Ouput layer is the binary cross entropy layer.'
                output_layer = CrossEntropyOutputLayer(
                    inputs=inputs,
                    n_in=n_in, n_out=n_out)
            elif self.output_activation == 'softmax':
                print 'Ouput layer is a softmax layer.'
                output_layer = SoftmaxOutputLayer(
                    inputs=inputs,
                    n_in=n_in, n_out=n_out)
            elif self.output_activation == 'sigmoid':
                # NOTE: check if the sigmoid output activation is the same as
                # the binary cross entropy output activation.
                print 'Output layer is a sigmoid layer.'
                raise NotImplementedError('Not yet supported!')
            else:
                raise NotImplementedError('%s not supported!'%self.output_activation)
        else:
            print 'constructing ordinary %d layers...'%len(hidden_sizes)

            layer_sizes = [n_in] + hidden_sizes + [n_out]
            weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
            next_layer_input = inputs
            for n_in, n_out in weight_matrix_sizes[:-1]:
                print 'building hidden layer of size (%d,%d)'%(n_in, n_out)
                next_layer = HiddenLayer(rng=rng,
                                         input=next_layer_input,
                                         activation=hidden_activation,
                                         n_in=n_in, n_out=n_out,
                                     )
                self.layers.append(next_layer)
                next_layer_input = next_layer.output

            # Output layer.
            n_in, n_out = weight_matrix_sizes[-1]
            if n_out == 1:
                if self.output_activation == 'regression':
                    print 'init regression with MSE output layer...'
                    raise NotImplementedError('Not yet supported!')
                elif self.output_activation == 'binary_cross_entropy':
                    print 'init sigmoid with CE output layer...'
                    output_layer = CrossEntropyOutputLayer(
                                    inputs=next_layer_input,
                                    n_in=n_in, n_out=n_out)
                elif self.output_activation == 'sigmoid':
                    print 'init sigmoid with NLL ouput layer...'
                    raise NotImplementedError('Not yet supported!')
                    output_layer = SigmoidOutputLayer(
                        inputs=next_layer_input,
                        n_in=n_in, n_out=n_out)
                else:
                    raise NotImplementedError('%s not supported!'%self.output_activation)
            else:
                if self.output_activation == 'softmax':
                    print 'init softmax with NLL output layer...'
                    output_layer = SoftmaxOutputLayer(
                                inputs=next_layer_input,
                                n_in=n_in, n_out=n_out)
                else:
                    raise NotImplementedError('%s not supported!'%self.output_activation)

        self.layers.append(output_layer)

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L2 = 0
        for layer in self.layers:
            self.L1 += abs(layer.W).sum()
            self.L2 += (layer.W ** 2).sum()

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
                           hidden_activation, dropout_p, maxout_k):

        try:
            assert hidden_sizes[0] != 0
        except:
            raise AssertionError('You should not try dropout/maxout without '
                                 'hidden layers')
        self.layers = []
        self.dropout_layers = []
        layer_sizes = [n_in] + hidden_sizes + [n_out]
        weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])

        next_layer_input = inputs
        next_dropout_layer_input = DropoutHiddenLayer.apply_dropout(rng, inputs, 0.2)
        # Construct all the layers (input+hidden) except the top layer which is
        # done afterwards.
        for n_in, n_out in weight_matrix_sizes[:-1]:
            if maxout_k == -1:
                print 'building Dropout hidden layer of size (%d,%d)'%(
                    n_in, n_out)
                next_dropout_layer = DropoutHiddenLayer(rng=rng,
                                input=next_dropout_layer_input,
                                activation=hidden_activation, n_in=n_in,
                                n_out=n_out, dropout_p=dropout_p
                                )
            else:
                print 'building Maxout hidden layer of size (%d,%d)'%(
                    n_in, n_out)
                next_dropout_layer = MaxoutHiddenLayer(rng=rng,
                                input=next_dropout_layer_input,
                                activation=hidden_activation, n_in=n_in, n_out=n_out,
                                dropout_p=dropout_p, maxout_k=maxout_k,
                                )
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output
            if maxout_k == -1:
                next_layer = HiddenLayer(rng=rng,
                                     input=next_layer_input,
                                     activation=hidden_activation,
                                     W=next_dropout_layer.W * 0.5,
                                     b=next_dropout_layer.b,
                                     n_in=n_in, n_out=n_out
                                    )
            else:
                next_layer = MaxPoolingHiddenLayer(rng=rng,
                                     input=next_layer_input,
                                     activation=hidden_activation,
                                     W=next_dropout_layer.W * 0.5,
                                     b=next_dropout_layer.b,
                                     n_in=n_in, n_out=n_out,
                                     maxout_k=maxout_k
                                    )
            self.layers.append(next_layer)
            next_layer_input = next_layer.output

        # Output layer.
        n_in, n_out = weight_matrix_sizes[-1]
        print 'building output layer of size (%d,%d)'%(n_in, n_out)

        if n_out == 1:
            if self.output_activation == 'regression':
                print 'init linear regression with MSE output layer...'
                raise NotImplementedError('Not yet supported!')
            elif self.output_activation == 'binary_cross_entropy':
                print 'init sigmoid with CE output layer...'
                dropout_output_layer = CrossEntropyOutputLayer(
                                            inputs=next_dropout_layer_input,
                                            n_in=n_in, n_out=n_out)
            elif self.output_activation == 'sigmoid':
                print 'init sigmoid with NLL ouput layer...'
                raise NotImplementedError('Not yet supported!')
                dropout_output_layer = SigmoidOutputLayer(
                                            inputs=next_dropout_layer_input,
                                            n_in=n_in, n_out=n_out)
            else:
                raise NotImplementedError('%s not supported!'%self.output_activation)
        else:
            if self.output_activation == 'softmax':
                print 'init softmax with NLL output layer...'
                dropout_output_layer = SoftmaxOutputLayer(
                                            inputs=next_dropout_layer_input,
                                            n_in=n_in, n_out=n_out)
            else:
                raise NotImplementedError('%s not supported!'%self.output_activation)

        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse parameters in the dropout output.
        if n_out == 1:
            if self.output_activation == 'regression':
                print 'init linear regression with MSE output layer...'
                raise NotImplementedError('Not yet supported!')
            elif self.output_activation == 'binary_cross_entropy':
                print 'init sigmoid with CE output layer...'
                output_layer = CrossEntropyOutputLayer(
                                    inputs=next_layer_input,
                                    W=dropout_output_layer.W * 0.5,
                                    b=dropout_output_layer.b,
                                    n_in=n_in, n_out=n_out)
            elif self.output_activation == 'sigmoid':
                print 'init sigmoid with NLL ouput layer...'
                raise NotImplementedError('Not yet supported!')
                output_layer = SigmoidOutputLayer(
                                    inputs=next_layer_input,
                                    W=dropout_output_layer.W * 0.5,
                                    b=dropout_output_layer.b,
                                    n_in=n_in, n_out=n_out)
            else:
                raise NotImplementedError('%s not supported!'%self.output_activation)
        else:
            if self.output_activation == 'softmax':
                # Ouput layer is a softmax layer.
                output_layer = SoftmaxOutputLayer(
                    inputs=next_layer_input,
                    W=dropout_output_layer.W * 0.5,
                    b=dropout_output_layer.b,
                    n_in=n_in, n_out=n_out)
            else:
                raise NotImplementedError('%s not supported!'%self.output_activation)

        self.layers.append(output_layer)

        self.dropout_nll = self.dropout_layers[-1].cost
        self.dropout_errors = self.dropout_layers[-1].cls_error

        self.nll = self.layers[-1].cost
        self.errors = self.layers[-1].cls_error

        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers
                        for param in layer.params ]

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
    x = T.matrix('x')  # the data is presented as rasterized images
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
        momentum = theano.ifelse.ifelse(tepoch < 500,
                    T.cast(state.mom*(1. - tepoch/500.) + 0.7*(tepoch/500.),
                    'float32'),T.cast(0.7, 'float32'))

    # end of theano variables initialization


    ####### BUILDING THE MLP MODEL #######
    # Construct the MLP class
    n_in = train_x.get_value().shape[1]
    if state.dataset == 'mnist':
        n_out = 10
    elif 'mq' in state.dataset:
        n_out = 1
    else:
        raise NotImplementedError('Dataset %s not supported!'%state.dataset)

    classifier = MLP(rng=rng, inputs=x, n_in= n_in,
                     hidden_sizes=state.hidden_sizes, n_out=n_out,
                     hidden_activation=state.hidden_activation,
                     output_activation=state.output_activation,
                     dropout_p=state.dropout_p,
                     maxout_k=state.maxout_k)


    # Get the number of model parameters.
    n_params = len(classifier.params)
    # Get right number of learning rates.
    state.init_lr *= n_params/2

    lr = theano.shared(numpy.asarray(state.init_lr,dtype='float32'))
    lr_0 = theano.shared(numpy.asarray(state.init_lr,dtype='float32'))
    new_lr = T.cast(lr_0 / (1.0 + state.decrease_constant*tepoch), 'float32')


    ####### Cost functions #######
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    ### Cost function + regularization terms. ###
    cost_reg = L1_reg * classifier.L1 + L2_reg * classifier.L2
    if state.dropout_p != -1 or state.maxout_k != -1:
        cost_train = classifier.dropout_nll(y)
        cost_valid = classifier.nll(y)
    else:
        cost_train = classifier.nll(y)
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
            dtype='float32'))
        gparams_mom.append(gparam_mom)

    # end of gradient of cost


    ### Updates rules for parameters ###
    # Specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs.
    updates = collections.OrderedDict()
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]

    # Update the step direction using momentum
    # TODO: not need to include in the zip the init_lr.
    for gparam_mom, gparam, i in zip(gparams_mom, gparams, range(len(state.init_lr))):
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
                y: train_y[index * state.batch_size:(index + 1) * state.batch_size]})


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
                                # minibatches before checking the network
                                # on the validation set; in this case we
                                # check every epoch

    best_validation_score = numpy.inf
    best_iter = 0
    best_epoch = 0
    best_test_score = 0.
    best_test_cost = 0.
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
    this_valid_score = None

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

                # TODO: if we don't want to save the losses and costs, we
                # should still be able to compute the train costs with no reg.
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

                    if state.dataset == 'mnist':
                        this_valid_score = this_validation_loss
                        print('epoch %i, minibatch %i/%i, validation error %f %%' %
                                (epoch, minibatch_index + 1, n_train_batches,
                                this_valid_score * 100.))
                    else:
                        this_valid_score = this_validation_cost
                        print('epoch %i, minibatch %i/%i, train error %f , validation error %f ' %
                                (epoch, minibatch_index + 1, n_train_batches,
                                numpy.mean(train_costs_without_reg),
                                this_valid_score))

                    if state.save_losses_and_costs:
                        all_valid_losses.setdefault(epoch, 0)
                        all_valid_losses[epoch] = this_validation_loss

                        all_valid_costs.setdefault(epoch, 0)
                        all_valid_costs[epoch] = this_validation_cost

                    # if we got the best validation score until now
                    if this_valid_score < best_validation_score:
                        # improve patience if loss improvement is good enough
                        if this_valid_score < best_validation_score *  \
                            state.improvement_threshold:
                            state.patience = max(state.patience, iter * state.patience_increase)

                        best_validation_score = this_valid_score
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

                        best_test_score = numpy.mean(t_losses)
                        best_test_cost = numpy.mean(t_costs)
                        # Best nnet ouputs on test set.
                        best_test_p_y_given_x = t_p_y_given_x

                        if state.save_losses_and_costs:
                            all_test_losses.setdefault(epoch, 0)
                            all_test_losses[epoch] = best_test_score

                            all_test_costs.setdefault(epoch, 0)
                            all_test_costs[epoch] = best_test_cost

                        if state.save_model_info:
                            # Best predictions on test set.
                            best_test_pred_and_targ = numpy.array([test_pred_and_targ_on_batch_fn(i)
                                                                   for i in xrange(n_test_batches)])

                        if state.dataset == 'mnist':
                            print(('     epoch %i, minibatch %i/%i, test error of '
                                'best model %f %%') %
                                (epoch, minibatch_index + 1, n_train_batches,
                                best_test_score * 100.))
                        else:
                            print(('     epoch %i, minibatch %i/%i, test error of '
                                'best model %f ') %
                                (epoch, minibatch_index + 1, n_train_batches,
                                best_test_cost))

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

            if state.save_losses_and_costs:
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
    if state.dataset == 'mnist':
        print(('Optimization complete. Best validation score of %f %% '
            'obtained at iteration %i (epoch %i), with test performance %f %%') %
            (best_validation_score * 100., best_iter + 1, best_epoch, best_test_score * 100.))
    else:
        print(('Optimization complete. Best validation score of %f '
            'obtained at iteration %i (epoch %i), with test performance %f ') %
            (best_validation_score, best_iter + 1, best_epoch, best_test_cost))
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
