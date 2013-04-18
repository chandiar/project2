# -*- coding: utf-8 -*-

import cPickle, sys
import numpy
import sklearn
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split # TODO: not used for the moment.
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC

# Options.
# Data directory:
data_dir = '/home/chandias/data/IFT6141/'
# Data preprocessing options.
scale = 255.
data_size = 70000
train_size = 50000
valid_size = 10000
test_size = 10000

assert data_size <= 70000
assert train_size + valid_size + test_size == data_size

def batch_pred(model, data):
    n = 1000
    n_batches = len(data) / n
    rval = []
    for i in range(n_batches+1):
	rval.append(model.predict(data[i * n : (i + 1) * n]))
    rval = numpy.concatenate(rval)
    return rval


def train(state, channel):
    # Load data
    # Load the original MNIST dataset.
    import pdb; pdb.set_trace()
    mnist = fetch_mldata('MNIST original',
        data_home=data_dir)

    # Split the data into train, valid and test sets.
    # TODO: add Scaling, normalization options.
    # reference: https://github.com/rosejn/torch-datasets/blob/master/dataset/mnist.lua
    # scaling: scale values between [0,1] (by default, they are in the range [0, 255])
    # TODO: try a [-1, 1] scaling which according to this post gives better results for
    # the svm: http://peekaboo-vision.blogspot.ca/2010/09/mnist-for-ever.html
    # Test that the test sets is the same as the one found in Yann LeCun's page.
    train_x = mnist.data[:train_size, :] / scale
    train_y = mnist.target[:train_size]
    valid_x = mnist.data[train_size:train_size+valid_size, :] / scale
    valid_y = mnist.target[train_size:train_size+valid_size]
    test_x = mnist.data[-test_size:, :] / scale
    test_y = mnist.target[-test_size:]
    #X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target)

    del mnist

    # Shuffle the train, valid and test sets since they are ordered.
    train_x, train_y = shuffle(train_x, train_y)
    valid_x, valid_y = shuffle(valid_x, valid_y)
    test_x, test_y = shuffle(test_x, test_y)

    if state.model == 'gbdt':
        print 'Fitting GBDT'
        classifier = GradientBoostingClassifier(
                          n_estimators=state.n_estimators,
                          learning_rate=state.learning_rate,
                          max_depth=state.max_depth)
        classifier.fit(train_x, train_y)
    elif state.model == 'random_forest':
        print 'Random Forest'
        classifier = RandomForestClassifier(
                        n_estimators=state.n_estimators,
                        max_depth=state.max_depth,
                        random_state=0)
        classifier.fit(train_x, train_y)
    elif state.model == 'knn':
        print 'KNN'
        classifier = KNeighborsClassifier(n_neighbors=state.n_neighbors)
        classifier.fit(train_x, train_y)
    elif state.model == 'svm':
        print 'SVM'
        classifier = SVC(
                    C=state.C,
                    kernel=state.kernel,
                    degree=state.degree,
                    gamma=state.gamma,
                    coef0=state.coef0,
                    tol=state.tol,
                    cache_size=state.cache_size)
        classifier.fit(train_x, train_y)
    elif state.model == 'lsvm':
        print 'Linear SVM'
        classifier = LinearSVC(
                    C=state.C,
                    loss=state.loss,
                    penalty=state.penalty,
                    dual=state.dual,
                    multi_class=state.multi_class,
                    fit_intercept=state.fit_intercept,
                    intercept_scaling=state.intercept_scaling,
                    tol=state.tol)
        classifier.fit(train_x, train_y)
    else:
        raise NotImplementedError('Model %s not supported.'%state.model)

    print 'Results'

    del train_x
    del train_y

    if state.model == 'knn':
        vpredictions = classifier.predict(valid_x)
        tpredictions = classifier.predict(test_x)
    else:
        vpredictions = batch_pred(classifier, valid_x)
        tpredictions = batch_pred(classifier, test_x)
    diff = vpredictions - valid_y
    errors = diff[diff!=0]
    valid_ce = len(errors) / float(len(valid_y))
    state.valid_ce = valid_ce

    diff = tpredictions - test_y
    errors = diff[diff!=0]
    test_ce = len(errors) / float(len(test_y))
    state.test_ce = test_ce

    print 'valid_ce :', valid_ce
    print 'test_ce :', test_ce

    # The model pickle
    f = open('%s_classifier.pkl'%state.model, 'w')
    cPickle.dump(classifier, f)
    f.close()

    model_info = []
    model_info.append({  'predictions': vpredictions,
                         'targets'    : valid_y })
    model_info.append({  'predictions': tpredictions,
                         'targets'    : test_y  })

    f = open('classifier_info.pkl','w')
    cPickle.dump(model_info, f)
    f.close()

    try:
        channel.save()
    except:
        print 'Not in experiment, done!'


def experiment(state, channel):
    train(state, channel)
    return channel.COMPLETE

if __name__ == '__main__':
    from jobman import DD, expand
    args = {'model'             : 'knn',
            # gbdt and random_forest
            'n_estimators'      : 15,
            'learning_rate'     : 0.0025,
            'max_depth'         : 20,
            # knn
            'n_neighbors'       : 3,
            # svm and lsvm
            'C'                 : 1,
            'kernel'            : 'rbf',
            'degree'            : 3,
            'gamma'             : 0,
            'coef0'             : 0,
            'tol'               : 1e-3,
            'cache_size'        : 500,
            # lsvm
            'loss'              : 'l2',
            'penalty'           : 'l2',
            'dual'              : False,
            'multi_class'       : 'ovr',
            'fit_intercept'     : True,
            'intercept_scaling' : 1,}

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
    train(state, {})
