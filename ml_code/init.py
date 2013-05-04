# -*- coding: utf-8 -*-

# In order to use gpu:
# THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python init.py
# TO check if the gpu is being used:
# nvidia-smi

import os, sys, time
import numpy
import sklearn
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn import cross_validation
import theano

# Local imports
import mlp
import cnn
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
    if state.model in ['nnet', 'cnn']:
        state.gpu = True
        print 'GPU should be enabled'
    # TODO: check how to retrieve the gpu status.
    if state.gpu:
        #print 'GPU enabled'
        print 'Loading dataset in shared variables'
    else:
        #print 'GPU disabled'
        print 'Loading dataset in numpy array'
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

    print 'Confing ', state

    # Start timer for training.
    start = time.time()

    if state.model == 'nnet':
        status = mlp.train(state, channel, train_x, train_y, valid_x, valid_y, test_x, test_y)
    elif state.model == 'cnn':
        status = cnn.train(state, channel, train_x, train_y, valid_x, valid_y, test_x, test_y)
    else:
        status = train(state, channel, train_x, train_y, valid_x, valid_y, test_x, test_y)

    stop = time.time()
    print 'It took %s minutes'%( (stop-start) / float(60) )

    if state.save_state:
        print 'We will save the experiment state'
        dump_tar_bz2(state, 'state.tar.bz2')

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

    if state.save_model_params:
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
    # TODO: use jobman DD instead of dictionnary.
    args = {'model'                 : 'lsvm',
            # TODO: add 'model_type' option.
            'dataset'               : 'mnist',
            'save_losses_and_costs' : True,
            'save_model_params'     : False,
            'save_model_info'       : True,
            'save_state'            : True,
            # If using gpu, we will load the dataset
            # as a shared variable.
            'gpu'               : False,
            ### gdbt and random_forest ###
            'n_estimators'      : 90,
            'learning_rate'     : 1e-1,
            'max_depth'         : 8,
            ### knn ###
            'n_neighbors'       : 30,
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
            'init_lr'           : [1e-1, 1e-1],
            'decrease_constant' : 1e-3,
            'n_epochs'          : 1000,
            # Set dropout_p and maxout_k to -1 to not use them.
            'dropout_p'             : -1,#0.5,
            'maxout_k'              : -1,#2,
            # Set mom to 0 to not use momentum.
            'mom'                   : 0.5,
            'filter_square_limit'   : 15.0,
            # Top layer output activation.
            # TODO: not used yet.
            'output_activation'     : 'softmax',
            # Early-stopping.
            # Look at this many examples regardless.
            'patience'          : 10000,
            # Wait this much longer when a new best is found.
            'patience_increase' : 2,
            # A relative improvement of this much is considered significant.
            'improvement_threshold'  : 0.995,
            # regularization terms
            'L1'                : 0,
            'L2'                : 0,
            ## Hidden layers ##
            # set this to [0] to fall back to LR
            'hidden_sizes'      : [500],
            # Hidden output activation:
            # tanh, rectifier, softplus, sigmoid, hard_tanh
            'activation' : 'tanh',
            ### cnn ###
            # Number of filters.
            'nkerns'            : [20, 50],
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
