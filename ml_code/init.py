# -*- coding: utf-8 -*-

# In order to use gpu:
# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python init.py
# TO check if the gpu is being used:
# nvidia-smi

import collections, cPickle, os, sys, time
import numpy
import sklearn
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import cross_validation
import theano
from theano import tensor as T

# Local imports
from mlp import MLP
import util
from util import dump_tar_bz2, load_data

# Options.
# Data directory:
data_dir = util.get_dataset_base_path()
# Data preprocessing options.
#scale = 255.
random_state = 1234
data_size = 70000
train_size = 50000
valid_size = 10000
test_size = 10000

assert data_size <= 70000
assert train_size + valid_size + test_size <= data_size
assert train_size > 0 and 10000 >= test_size > 0

# Type of cross_validation
# None, 'KFold', 'StratifiedKfold', 'LOO', 'LPO', 'LOLO', 'LPLO', 'ShuffleSplit'
#cv_strategy = None

#if cv_strategy is not None:
#    assert valid_size > 0


def batch_pred(model, data):
    n = 1000
    n_batches = len(data) / n
    rval = []
    for i in range(n_batches+1):
	rval.append(model.predict(data[i * n : (i + 1) * n]))
    rval = numpy.concatenate(rval)
    return rval


def main(state, channel):
    start = time.time()
    # Load the MNIST dataset.
    print 'Loading MNIST from '
    '''
    mnist = fetch_mldata('MNIST original',
        data_home=data_dir)

    # Split the data into train, valid and test sets.
    # TODO: add Scaling, normalization options.
    # reference: https://github.com/rosejn/torch-datasets/blob/master/dataset/mnist.lua
    # scaling: scale values between [0,1] (by default, they are in the range [0, 255])
    # TODO: try a [-1, 1] scaling which according to this post gives better results for
    # the svm: http://peekaboo-vision.blogspot.ca/2010/09/mnist-for-ever.html
    # Test that the test sets is the same as the one found in Yann LeCun's page.
    train_valid_x = mnist.data[:-10000, :] / scale
    train_valid_y = mnist.target[:-10000]
    test_x = mnist.data[-10000:, :] / scale
    test_y = mnist.target[-10000:]

    del mnist

    # Shuffle the train, valid and test sets since they are ordered.
    train_valid_x, train_valid_y = shuffle(train_valid_x, train_valid_y, random_state=random_state)
    test_x, test_y = shuffle(test_x, test_y)
    '''

    data_path = os.path.join(data_dir, 'mnist.pkl.gz')
    print 'Loading the MNIST dataset from %s' %data_path
    if state.model == 'nnet':
        state.gpu = True
    if state.gpu:
        print 'GPU enabled'
    else:
        print 'GPU disabled'
    datasets = load_data(data_path, splits=[train_size, valid_size, test_size], shared=state.gpu)

    train_x, train_y = datasets[0]
    valid_x, valid_y = datasets[1]
    test_x, test_y = datasets[2]

    # Cross-validation.
    '''
    if cv_strategy == 'KFold':
        assert len(valid_x) > 0
        print 'KFold used'
        # Concatenate both the train and validation sets.
        train_valid_x = numpy.concatenate((train_x, valid_x), axis=0)
        train_valid_y = numpy.concatenate((train_y, valid_y), axis=0)
        kf = cross_validation.KFold(len(train_valid_x), n_folds=9)
        for train_index, valid_index in kf:
            train_x, valid_x = train_valid_x[train_index], train_valid_x[valid_index]
            train_y, valid_y = train_valid_y[train_index], train_valid_y[valid_index]
            train(state, channel, train_x, train_y, valid_x, valid_y, test_x, test_y)
    elif cv_strategy is None:
        print 'No cross-validation'
        train(state, channel, train_x, train_y, valid_x, valid_y, test_x, test_y)
    else:
        raise NotImplementedError('Cross-validation type not supported.')
    '''

    if state.model == 'nnet':
        train_mlp(state, channel, train_x, train_y, valid_x, valid_y, test_x, test_y)
    else:
        train(state, channel, train_x, train_y, valid_x, valid_y, test_x, test_y)

    stop = time.time()
    print 'It took %s minutes'%( (stop-start) / float(60) )

    if state.save_state:
        print 'We will save the experiment state'
        dump_tar_bz2(state, 'state.tar.bz2')

    return 0


def train_mlp(state, channel, train_x, train_y, valid_x, valid_y, test_x, test_y):
    print 'train_x shape: ', train_x.get_value(borrow=True).shape
    #print 'train_y shape: ', train_y.get_value(borrow=True).shape
    print 'valid_x shape: ', valid_x.get_value(borrow=True).shape
    #print 'valid_y shape: ', valid_y.get_value(borrow=True).shape
    print 'test_x shape: ', test_x.get_value(borrow=True).shape
    #print 'test_y shape: ', test_y.get_value(borrow=True).shape

    if state.model == 'nnet':
        # compute number of minibatches for training, validation and testing
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

        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        x = T.matrix('x')  # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels
        tepoch = T.iscalar()
        #tepoch.tag.test_value = 1
        lr = theano.shared(numpy.asarray(state.init_lr,dtype='float32'))
        lr_0 = theano.shared(numpy.asarray(state.init_lr,dtype='float32'))

        rng = numpy.random.RandomState(state.seed)
        # construct the MLP class
        classifier = MLP(rng=rng, input=x, n_in=28 * 28,
                        n_hidden=state.n_hidden, n_out=10)

        # the cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed
        # here symbolically
        cost = classifier.negative_log_likelihood(y) \
            + state.L1 * classifier.L1 \
            + state.L2 * classifier.L2_sqr

        # compiling a Theano function that computes the mistakes that are made
        # by the model on a minibatch
        test_model = theano.function(inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: test_x[index * state.batch_size:(index + 1) * state.batch_size],
                    y: test_y[index * state.batch_size:(index + 1) * state.batch_size]})

        validate_model = theano.function(inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: valid_x[index * state.batch_size:(index + 1) * state.batch_size],
                    y: valid_y[index * state.batch_size:(index + 1) * state.batch_size]})

        # compute the gradient of cost with respect to theta (stored in params)
        # the resulting gradients will be stored in a list gparams
        gparams = []
        for param in classifier.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        # specify how to update the parameters of the model as a list of
        # (variable, update expression) pairs
        updates = []
        # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
        # same length, zip generates a list C of same size, where each element
        # is a pair formed from the two lists :
        #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
        for param, gparam in zip(classifier.params, gparams):
            updates.append((param, param - lr * gparam))

        new_lr = T.cast(lr_0 / (1.0 + state.decrease_constant*tepoch), 'float32')
        # use 1/t decay where t is epoch
        decay_learning_rate_fn = theano.function(inputs=[tepoch],
                outputs=T.as_tensor_variable(lr),
                name='decay learning rate with 1/t',
                updates=collections.OrderedDict({lr: new_lr}))

        # compiling a Theano function `train_model` that returns the cost, but
        # in the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(inputs=[index], outputs=cost,
                updates=updates,
                givens={
                    x: train_x[index * state.batch_size:(index + 1) * state.batch_size],
                    y: train_y[index * state.batch_size:(index + 1) * state.batch_size]})

        if state.save_losses:
            get_train_error = theano.function(inputs=[index],
                outputs=classifier.errors(y),
                givens={
                    x: train_x[index * state.batch_size:(index + 1) * state.batch_size],
                    y: train_y[index * state.batch_size:(index + 1) * state.batch_size]})

        if state.save_model_info:
            get_train_targ = theano.function(inputs=[index],
                    outputs=train_y[index * state.batch_size:(index + 1) * state.batch_size])
            get_train_pred = theano.function(inputs=[index],
                    outputs=classifier.y_pred,
                    givens={
                        x: train_x[index * state.batch_size:(index + 1) * state.batch_size]})

            get_valid_targ = theano.function(inputs=[index],
                    outputs=valid_y[index * state.batch_size:(index + 1) * state.batch_size])
            get_valid_pred = theano.function(inputs=[index],
                    outputs=classifier.y_pred,
                    givens={
                        x: valid_x[index * state.batch_size:(index + 1) * state.batch_size]})

            get_test_targ = theano.function(inputs=[index],
                    outputs=test_y[index * state.batch_size:(index + 1) * state.batch_size])
            get_test_pred = theano.function(inputs=[index],
                    outputs=classifier.y_pred,
                    givens={
                        x: test_x[index * state.batch_size:(index + 1) * state.batch_size]})

        if state.save_costs:
            get_train_cost = theano.function(inputs=[index],
                    outputs=cost,
                    givens={
                        x: train_x[index * state.batch_size:(index + 1) * state.batch_size],
                        y: train_y[index * state.batch_size:(index + 1) * state.batch_size]})
            get_valid_cost = theano.function(inputs=[index],
                    outputs=classifier.negative_log_likelihood(y),
                    givens={
                        x: valid_x[index * state.batch_size:(index + 1) * state.batch_size],
                        y: valid_y[index * state.batch_size:(index + 1) * state.batch_size]})
            get_test_cost = theano.function(inputs=[index],
                    outputs=classifier.negative_log_likelihood(y),
                    givens={
                        x: test_x[index * state.batch_size:(index + 1) * state.batch_size],
                        y: test_y[index * state.batch_size:(index + 1) * state.batch_size]})

        ###############
        # TRAIN MODEL #
        ###############
        print '... training'

        # early-stopping parameters
        state.patience = 10000  # look as this many examples regardless
        state.patience_increase = 2  # wait this much longer when a new best is
                            # found
        state.improvement_threshold = 0.995  # a relative improvement of this much is
                                    # considered significant
        validation_frequency = min(n_train_batches, state.patience / 2)
                                    # go through this many
                                    # minibatches before checking the network
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
        all_train_costs = {}
        all_valid_costs = {}
        all_test_costs = {}

        vpredictions = []
        tpredictions = []

        print 'Initial learning rate: ', state.init_lr
        if state.lr_decay:
            print 'Learning decay enabled'
        else:
            print 'No learning decay!'
        while (epoch < state.n_epochs) and (not done_looping):
            train_costs = []
            train_losses = []
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                train_costs.append(minibatch_avg_cost)

                if state.save_losses:
                    train_losses.append(get_train_error(minibatch_index))

                '''
                train_pred = numpy.array(get_train_pred(minibatch_index))
                train_targ = numpy.array(get_train_targ(minibatch_index))
                train_errors = 1 - (sum(train_pred==train_targ)/ float(len(train_targ)))
                '''

                # iteration number
                iter = epoch * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = [validate_model(i) for i
                                        in xrange(n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)

                    if state.save_losses:
                        all_valid_losses.setdefault(epoch, [])
                        all_valid_losses[epoch].append(this_validation_loss)

                    if state.save_costs:
                        # Compute the nll on the validation set.
                        validation_costs = [get_valid_cost(i) for i
                                            in xrange(n_valid_batches)]
                        this_validation_cost = numpy.mean(validation_costs)
                        all_valid_costs.setdefault(epoch, [])
                        all_valid_costs[epoch].append(this_validation_cost)

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
                        if state.save_model_info:
                            # Best predictions on validation set.
                            valid_pred = numpy.array([get_valid_pred(i) for i in xrange(n_valid_batches)]).flatten()
                            valid_targ = numpy.array([get_valid_targ(i) for i in xrange(n_valid_batches)]).flatten()

                        best_iter = iter
                        best_epoch = epoch

                        # test it on the test set
                        test_losses = [test_model(i) for i
                                    in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        if state.save_losses:
                            all_test_losses.setdefault(epoch, [])
                            all_test_losses[epoch].append(test_score)

                        if state.save_costs:
                            # Compute the nll on the test set.
                            test_costs = [get_test_cost(i) for i
                                                in xrange(n_test_batches)]
                            this_test_cost = numpy.array(test_costs).mean()
                            all_test_costs.setdefault(epoch, [])
                            all_test_costs[epoch].append(this_test_cost)

                        if state.save_model_info:
                            # Best predictions on test set.
                            test_pred = numpy.array([get_test_pred(i) for i in xrange(n_test_batches)]).flatten()
                            test_targ = numpy.array([get_test_targ(i) for i in xrange(n_test_batches)]).flatten()

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
            epoch = epoch + 1

            if state.save_costs:
                all_train_costs.setdefault(epoch, [])
                all_train_costs[epoch].append(numpy.mean(train_costs))

            if state.save_losses:
                all_train_losses.setdefault(epoch, [])
                all_train_losses[epoch].append(numpy.mean(train_losses))

        end_time = time.clock()
        print(('Optimization complete. Best validation score of %f %% '
            'obtained at iteration %i (epoch %i), with test performance %f %%') %
            (best_validation_loss * 100., best_iter + 1, best_epoch,test_score * 100.))
        print >> sys.stderr, ('The code for file ' +
                            os.path.split(__file__)[1] +
                            ' ran for %.2fm' % ((end_time - start_time) / 60.))
    else:
        raise NotImplementedError('Model %s not supported.'%state.model)

    if state.save_model:
        # Save the model.
        '''
        print 'Pickling the model'
        path = '%s.tar.bz2'%state.model
        dump_tar_bz2(classifier, path)
        '''
        print 'Saving model not implemented.'

    if state.save_model_info:
        # Save the model valid/test predictions and targets.
        model_info = {'validation': { 'predictions': valid_pred,
                                      'targets'    : valid_targ },
                      'test'      : { 'predictions': test_pred,
                                      'targets'    : test_targ }
                     }

        print 'Saving the model info'
        path = '%s_info.tar.bz2'%state.model
        dump_tar_bz2(model_info, path)

    if state.save_losses:
        print 'Saving train losses'
        path = '%s_losses.tar.bz2'%state.model
        dump_tar_bz2({'train_losses': all_train_losses,
                      'valid_losses': all_valid_losses,
                      'test_losses': all_test_losses
                      },
                     path)

    if state.save_costs:
        print 'Saving train costs'
        path = '%s_costs.tar.bz2'%state.model
        dump_tar_bz2({'train_costs': all_train_costs,
                      'valid_costs': all_valid_costs,
                      'test_costs': all_test_costs
                      },
                     path)

    try:
        channel.save()
    except:
        print 'Not in experiment, done!'

    return 0


def train(state, channel, train_x, train_y, valid_x, valid_y, test_x, test_y):
    print 'train_x shape: ', train_x.shape
    print 'train_y shape: ', train_y.shape
    print 'valid_x shape: ', valid_x.shape
    print 'valid_y shape: ', valid_y.shape
    print 'test_x shape: ', test_x.shape
    print 'test_y shape: ', test_y.shape

    if state.model == 'gdbt':
        print 'Fitting GDBT'
        classifier = GradientBoostingClassifier(
                          n_estimators=state.n_estimators,
                          learning_rate=state.learning_rate,
                          max_depth=state.max_depth)
    elif state.model == 'random_forest':
        print 'Fitting Random Forest'
        classifier = RandomForestClassifier(
                        n_estimators=state.n_estimators,
                        max_depth=state.max_depth,
                        random_state=0)
    elif state.model == 'knn':
        print 'Fitting KNN'
        classifier = KNeighborsClassifier(n_neighbors=state.n_neighbors)
    elif state.model == 'svm':
        print 'Fitting SVM'
        classifier = SVC(
                    C=state.C,
                    kernel=state.kernel,
                    degree=state.degree,
                    gamma=state.gamma,
                    coef0=state.coef0,
                    tol=state.tol,
                    cache_size=state.cache_size)
    elif state.model == 'lsvm':
        print 'Fitting Linear SVM'
        classifier = LinearSVC(
                    C=state.C,
                    loss=state.loss,
                    penalty=state.penalty,
                    dual=state.dual,
                    multi_class=state.multi_class,
                    fit_intercept=state.fit_intercept,
                    intercept_scaling=state.intercept_scaling,
                    tol=state.tol)
    else:
        raise NotImplementedError('Model %s not supported.'%state.model)

    classifier.fit(train_x, train_y)

    print 'Results'

    del train_x
    del train_y

    if state.model == 'knn':
        if len(valid_x):
            print 'Computing valid predictions'
            vpredictions = classifier.predict(valid_x)
        else:
            vpredictions = []
        print 'Computing test predictions'
        tpredictions = classifier.predict(test_x)
    else:
        if len(valid_x):
            print 'Computing valid predictions'
            vpredictions = batch_pred(classifier, valid_x)
        else:
            vpredictions = []
        print 'Computing test predictions'
        tpredictions = batch_pred(classifier, test_x)
    valid_ce = 0
    if len(valid_x):
        diff = vpredictions - valid_y
        errors = diff[diff!=0]
        valid_ce = len(errors) / float(len(valid_y))
    state_valid_ce = valid_ce

    diff = tpredictions - test_y
    errors = diff[diff!=0]
    test_ce = len(errors) / float(len(test_y))
    state.test_ce = test_ce

    print 'Classification errors (ce)'
    print 'valid_ce :', valid_ce
    print 'test_ce :', test_ce

    if state.save_model:
        # Save the model.
        print 'Pickling the model'
        path = '%s.tar.bz2'%state.model
        dump_tar_bz2(classifier, path)

    if state.save_model_info:
        # Save the model valid/test predictions and targets.
        model_info = []
        model_info.append({  'predictions': vpredictions,
                            'targets'    : valid_y })
        model_info.append({  'predictions': tpredictions,
                            'targets'    : test_y  })

        print 'Saving the model info'
        path = '%s_info.tar.bz2'%state.model
        dump_tar_bz2(model_info, path)

    try:
        channel.save()
    except:
        print 'Not in experiment, done!'

    return 0


def experiment(state, channel):
    main(state, channel)
    return channel.COMPLETE

if __name__ == '__main__':
    from jobman import DD, expand
    args = {'model'             : 'nnet',
            'dataset'           : 'mnist',
            'save_model_info'   : True,
            'save_model'        : False,
            'save_state'        : False,
            'save_losses'       : True,
            'save_costs'        : True,
            # If using gpu, we will load the dataset
            # as a shared variable.
            'gpu'               : False,
            ### gdbt and random_forest ###
            'n_estimators'      : 100,
            'learning_rate'     : 0.1,
            'max_depth'         : 3,
            ### knn ###
            'n_neighbors'       : 3,
            ### svm and lsvm ###
            'C'                 : 1,
            'kernel'            : 'rbf',
            'degree'            : 3,
            'gamma'             : 0,
            'coef0'             : 0,
            'tol'               : 1e-3,
            'cache_size'        : 500,
            ### lsvm ###
            'loss'              : 'l2',
            'penalty'           : 'l2',
            'dual'              : False,
            'multi_class'       : 'ovr',
            'fit_intercept'     : True,
            'intercept_scaling' : 1,
            ### nnet ###
            'seed'              : 1234,
            'batch_size'        : 100,
            'lr_decay'          : True,
            'init_lr'           : 1e-1,
            'decrease_constant' : 1e-3,
            'n_epochs'          : 1000,
            # Top layer output activation.
            'output_activation' : 'softmax',
            # Early-stopping.
            # Look at this many examples regardless.
            'patience'          : 10000,
            # Wait this much longer when a new best is found.
            'patience_increase' : 2,
            # A relative improvement of this much is considered significant.
            'improvement_threshold'  : 0.995,
            # regularization terms
            'L1'                : 1e-5,
            'L2'                : 1e-5,
            ## Hidden layers ##
            # set this to [0] to fall back to LR
            #'hidden_sizes'      : [2500, 1000],
            'n_hidden'            : 1000,
            # Hidden output activation:
            # tanh, rectifier, softplus, sigmoid, hard_tanh
            'hidden_activation' : 'tanh',
            }

    '''
    try:
       for arg in sys.argv[1:]:
           k, v = arg.split('=')
           args[k] = v
    except:
       print 'args must be like input=data foo.bar=1'
       print 'Check that you are not using white spaces within an arg.'
       exit(1)
    '''
    # wrap args into a DD object
    state = expand(args)
    main(state, {})
