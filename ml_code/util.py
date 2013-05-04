# -*- coding: utf-8 -*-

import cPickle
import gzip
import numpy
import os
import StringIO
import tarfile
import theano
from theano import tensor as T


def dump_tar_bz2(obj, path):
    """
    Save object to a .tar.bz2 file.

    The file stored within the .tar.bz2 has the same basename as 'path', but
    ends with '.pkl' instead of '.tar.bz2'.

    :param obj: Object to be saved.

    :param path: Path to the file (must end with '.tar.bz2').
    """
    assert path.endswith('.tar.bz2')
    pkl_name = os.path.basename(path)[0:-8] + '.pkl'
    # We use StringIO to avoid having to write to disk a temporary
    # pickle file.
    obj_io = None
    f_out = tarfile.open(path, mode='w:bz2')
    try:
        obj_str = cPickle.dumps(obj)
        obj_io = StringIO.StringIO(obj_str)
        tarinfo = tarfile.TarInfo(name=pkl_name)
        tarinfo.size = len(obj_str)
        f_out.addfile(tarinfo=tarinfo, fileobj=obj_io)
    finally:
        f_out.close()
        if obj_io is not None:
            obj_io.close()


def get_dataset_base_path():
    data_dir = os.environ.get('ML_DATA_PATH')
    if data_dir is None:
        print 'Environment variable ML_DATA_PATH not set correctly!'
        return 1
    else:
        return data_dir


def get_theano_constant(constant, dtype, bc_pattern):
    # usage: dtype = 'float32', bc_pattern='()'
    # see http://deeplearning.net/software/theano/library/tensor/basic.html for details.
    try:
        rval = theano.tensor.TensorConstant(theano.tensor.TensorType(dtype,
                            broadcastable=bc_pattern), numpy.asarray(constant, 'float32'))
    except TypeError:
        import ipdb; ipdb.set_trace()
    return rval


def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval


def load_tar_bz2(path):
    """
    Load a file saved with `dump_tar_bz2`.
    """
    assert path.endswith('.tar.bz2')
    name = os.path.split(path)[-1].replace(".tar.bz2", ".pkl")
    f = tarfile.open(path).extractfile(name)
    try:
        data = f.read()
    finally:
        f.close()
    return cPickle.loads(data)



def load_data(dataset, splits=[50000, 10000, 10000], shared=False):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    train_set = (train_set[0][:splits[0]], train_set[1][:splits[0]])
    valid_set = (valid_set[0][0:splits[1]], valid_set[1][0:splits[1]])
    test_set = (test_set[0][0:splits[2]], test_set[1][0:splits[2]])

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    if shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)
    else:
        test_set_x, test_set_y = test_set
        valid_set_x, valid_set_y = valid_set
        train_set_x, train_set_y = train_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def save_model_info(model_name, valid_pred_and_targ, test_pred_and_targ,
                    valid_p_y_given_x, test_p_y_given_x):

    print 'Saving the best model predictions and targets'
    path = '%s_pred_and_targ.npz'%model_name
    valid_pred = valid_pred_and_targ[:,0,:]
    valid_targ = valid_pred_and_targ[:,1,:]
    test_pred = test_pred_and_targ[:,0,:]
    test_targ = test_pred_and_targ[:,1,:]
    numpy.savez_compressed(path, valid_pred=valid_pred,
                                 valid_targ=valid_targ,
                                 test_pred=test_pred,
                                 test_targ=test_targ)

    print 'Saving the best model nnet outputs (p_y_given_x) on valid and test'
    path = '%s_p_y_given_x.npz'%model_name
    numpy.savez_compressed(path, valid_p_y_given_x=valid_p_y_given_x,
                                 test_p_y_given_x=test_p_y_given_x)


def save_model_losses_and_costs(model_name, all_train_losses, all_valid_losses,
    all_test_losses, all_train_costs_with_reg, all_train_costs_without_reg,
    all_valid_costs, all_test_costs):

    print 'Saving all losses and costs'
    path = '%s_losses.npz'%model_name
    numpy.savez_compressed(path, all_train_losses=all_train_losses,
                                 all_valid_losses=all_valid_losses,
                                 all_test_losses=all_test_losses)

    path = '%s_costs.npz'%model_name
    numpy.savez(path, train_costs_with_reg=all_train_costs_with_reg,
                      train_costs_without_reg=all_train_costs_without_reg,
                      valid_costs=all_valid_costs,
                      test_costs=all_test_costs)


def save_model_params(model_name, params_to_save):

    # Save the best model params.
    print 'Saving the best model params based on validation set.'
    params = []
    for i, param in enumerate(params_to_save):
        params.append((param.name, param.get_value()))
    path = '%s_params.tar.bz2'%model_name
    dump_tar_bz2(params, path)