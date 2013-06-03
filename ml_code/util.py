# -*- coding: utf-8 -*-

import argparse
import cPickle
import gzip
import itertools
import numpy
import os
import StringIO
import sys
import tarfile
import theano
from theano import tensor as T

log_normalize_fn = lambda x : numpy.cast['float32'](numpy.log(1 + numpy.abs(x)) * 2 * (numpy.cast['float32'](x >= 0) - 0.5))

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



def load_data(dataset=None, data_path=None, splits=[50000, 10000, 10000], shared=False, state=None):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    if dataset is not None:
        # Download the MNIST dataset if it is not present
        data_dir, data_file = os.path.split(dataset)
        if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
            import urllib
            origin = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, dataset)

        print '... loading mnist data'
        f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()

        train_set = (train_set[0][:splits[0]], train_set[1][:splits[0]])
        valid_set = (valid_set[0][0:splits[1]], valid_set[1][0:splits[1]])
        test_set = (test_set[0][0:splits[2]], test_set[1][0:splits[2]])
    else:
        assert data_path is not None
        print '... loading the MQ data'
        train_X     =   numpy.load(os.path.join(data_path, 'train_X.npy'))
        valid_X     =   numpy.load(os.path.join(data_path, 'valid_X.npy'))
        test_X      =   numpy.load(os.path.join(data_path, 'test_X.npy'))

        #import pdb; pdb.set_trace()
        #if state.normalize:
        #    print '... normalizing'
        #    raise NotImplementedError('Option to normalize not supported yet!')
        #    normalize(train_X, valid_X, test_X)

        #if state.log_normalize:
        #    print '... log normalizing'
        #    log_normalize(train_X, valid_X, test_X)

        #if state.top_features:
        #    print '... selecting top k features and exploding their dimensions'
        #    explode_features(train_X, valid_X, test_X)

        #save_data(state.save_path, train_X, valid_X, test_X, train_Y, valid_Y, test_Y)

        train_Y     =   numpy.load(os.path.join(data_path, 'train_Y.npy'))
        valid_Y     =   numpy.load(os.path.join(data_path, 'valid_Y.npy'))
        test_Y      =   numpy.load(os.path.join(data_path, 'test_Y.npy'))

        if 'diff' in state.dataset:
            train_Y = train_Y[:,0]
            valid_Y = valid_Y[:,0]
            test_Y = test_Y[:,0]
            assert (numpy.unique(train_Y)==numpy.array([0, 1])).all()
        elif 'fun' in state.dataset:
            train_Y = train_Y[:,1]
            valid_Y = valid_Y[:,1]
            test_Y = test_Y[:,1]
            assert (numpy.unique(train_Y)==numpy.arange(0, 1.25, 0.25)).all()
        else:
            raise NotImplementedError('MQ dataset not supported!'%state.dataset)

        train_X = train_X.astype(numpy.float32)
        valid_X = valid_X.astype(numpy.float32)
        test_X = test_X.astype(numpy.float32)
        train_Y = train_Y.astype(numpy.float32)
        valid_Y = valid_Y.astype(numpy.float32)
        test_Y = test_Y.astype(numpy.float32)

        train_set = (train_X, train_Y)
        valid_set = (valid_X, valid_Y)
        test_set = (test_X, test_Y)


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
        # lets us get around this issue
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


# Standardization is applied first followed by log normalization.
# nrows x ncols = figure grid dimension
def histograms(which_data='train', transformations=[], save_fig=False, show_graph=True, nrows=4, ncols=4, nbins=100):
    import matplotlib
    if not show_graph:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Order in which the different group of features were added in the raw
    # dataset.
    order = ['CASTLES', 'USERS', 'clayout_fixed', 'clayout_pooled']

    from matplotlib.ticker import MaxNLocator
    # Path to raw dataset.
    data_path = '/data/lisa/exp/chandiar/IFT6141/project2/data/MQ/raw'
    filename = os.path.join(data_path, '%s_X.npy'%which_data)
    data = numpy.load(filename)
    print '... loading raw dataset from %s'%filename

    print '... loading features names'
    f = open('/data/lisa/exp/chandiar/MQ/data11.raw/features_names.pkl', 'rb')
    features_names = cPickle.load(f)
    f.close()

    features_groups = {'CASTLES': 112, 'USERS': 84, 'clayout_fixed': 68, 'clayout_pooled': 231}
    features_names['clayout_fixed'] = numpy.arange(0, features_groups['clayout_fixed'], 1)
    features_names['clayout_pooled'] = numpy.arange(0, features_groups['clayout_pooled'], 1)

    assert sum([len(v) for k,v in features_names.iteritems() if k in order]) == sum(features_groups.values())

    for transf in transformations:
        if transf == 'standardize':
            print '... applying standardization'
            data = standardize(data)
        elif transf == 'log_normalize':
            print '... applying log normalization'
            data = log_normalize(data)
        else:
            raise NotImplementedError('Transformation %s not supported!'%transf)

    start_idx = 0
    end_idx = 0
    total_feature = 0
    # Total number of figures per page.
    total_figures = nrows * ncols

    for type_feature in order:
        dim = features_groups[type_feature]
        end_idx = start_idx + dim
        features = data[:, start_idx:end_idx]
        start_idx  = end_idx
        count = 1
        n_fig = 0
        fig = plt.figure(figsize=(20, 15))
        for i in xrange(features.shape[1]):
            feature_name = features_names[type_feature][i]
            x = features[:,i]
            min_val = x.min()
            max_val = x.max()
            std = x.std()
            mean = x.mean()

            if len(set(x)) <= 2:
                bins = len(set(x))
                if not set(x).symmetric_difference(set([0,1])):
                    num_diff_val = 'binary'
                else:
                    num_diff_val = len(set(x))
            else:
                bins = nbins
                num_diff_val = len(set(x))
                if num_diff_val >= 100:
                    num_diff_val = '>100'
                else:
                    num_diff_val = num_diff_val

            ax = fig.add_subplot(nrows,ncols,count)
            hist, bin_edges, patches = ax.hist(x, bins=bins)
            text = 'diff_values:%s\nmin=%0.2f\nmax=%0.2f\nstd=%0.3f'%(num_diff_val, min_val, max_val, std)
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.text(0.65, 0.70, text, bbox=dict(facecolor='white'), transform=ax.transAxes)
            plt.title('%s (%s, %s)'%(feature_name, i, total_feature))

            if count == total_figures:
                info = {'type_feature': type_feature,
                        'n_fig'       : n_fig,
                        'feature_i'   : i,}
                save_and_show_figure(plt, info, save_fig, show_graph)
                fig = plt.figure(figsize=(20, 15))
                n_fig += 1
                count = 1
            else:
                count += 1
            total_feature += 1

        # Plot the remaining graphs.
        info = {'type_feature': type_feature,
                'n_fig'       : n_fig,
                'feature_i'   : i,}
        save_and_show_figure(plt, info, save_fig, show_graph)


    return 0
    # References: http://matplotlib.org/users/transforms_tutorial.html#transforms-tutorial


# info = {'type_feature': None, 'n_fig': None, 'feature_i': None}
def save_and_show_figure(plt, info, save_fig= False, show_graph=True):
    wspace = 0.30
    hspace = 0.30
    print '\nInfo on figure: type_feature=%s, n_fig=%s, feature_i=%s'%(info['type_feature'], info['n_fig'], info['feature_i'])
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if save_fig:
        filename = 'fig_%s_%s.png'%(info['type_feature'], info['n_fig'])
        print '...saving figure %s'%filename
        plt.savefig(filename)
    if show_graph:
        plt.show()


def rescale(data, filename=None, yes_indices=[], no_indices=[], save_path=''):
    if filename:
        data = load_dataset(filename)

    print '... rescaling features'


def get_good_indices(data, yes_indices=[], no_indices=[]):
    all_indices = numpy.arange(0, data.shape[1])
    if len(no_indices):
        yes_indices = numpy.setdiff1d(all_indices, no_indices)
    elif len(yes_indices):
        yes_indices = len(numpy.intersect1d(yes_indices, all_indices))
    else:
        yes_indices = all_indices
    return yes_indices


def get_bin_features_indices(data):
    rval = []
    for dim in xrange(data.shape[1]):
        x = data[:, dim]
        if not set(x).symmetric_difference(set([0,1])):
            rval.append(dim)
    return rval


def load_dataset(filename):
    print '... loading dataset from %s'%filename
    rval = numpy.load(filename)
    return rval


# TODO: factorize log_normalize and standardize functions.
def log_normalize(data, filename=None, yes_indices=[], no_indices=[],
                    save_path='', apply_to_bin_features=False):
    log_normalized_data = None
    if filename:
        data = load_dataset(filename)

    if not apply_to_bin_features:
        print 'Log-normalization will not be applied on binary features'
        # Get indices of binary features.
        bin_indices = get_bin_features_indices(data)
        # Add these indices to the indices of features we don't want to apply
        # the log-normalization.
        no_indices.extend(bin_indices)
        print 'Found %s binary features'%len(bin_indices)

    if yes_indices or no_indices:
        yes_indices = get_good_indices(data, yes_indices, no_indices)
        print 'In total, there are %s features that will not be log-normalized'%len(no_indices)
        print 'And, there are %s features that will be log-normalized'%len(yes_indices)
        print 'Applying log-normalization to these indices: %s'%yes_indices
        for dim in yes_indices:
            data[:, dim] = log_normalize_fn(data[:, dim])
        log_normalized_data = data
    else:
        print 'Applying log normalization on the whole data'
        log_normalized_data = log_normalize_fn(data)
    return log_normalized_data


def standardize(data, filename=None, yes_indices=[], no_indices=[],
                save_path='', apply_to_bin_features=False, mean=None, std=None):
    # TODO: save_path not used.
    standardized_data = None
    if filename:
        data = load_dataset(filename)

    if not apply_to_bin_features:
        print 'Standardization will not be applied on binary features'
        # Get indices of binary features.
        bin_indices = get_bin_features_indices(data)
        # Add these indices to the indices of features we don't want to apply
        # the standardization.
        no_indices.extend(bin_indices)
        print 'Found %s binary features'%len(bin_indices)

    if yes_indices or no_indices:
        yes_indices = get_good_indices(data, yes_indices, no_indices)
        print 'In total, there are %s features that will not be standardized'%len(no_indices)
        print 'And, there are %s features that will be standardized'%len(yes_indices)
        #print 'Applying standardization to these indices: %s'%yes_indices
        #data[:, yes_indices] = (data[:, yes_indices] - data[:, yes_indices].mean()) / (1e-4 + data[:, yes_indices].std())
        for dim in yes_indices:
            if mean is not None and std is not None:
                data[:, dim] = (data[:, dim] - mean[dim]) / (1e-4 + std[dim])
            elif mean is None and std is None:
                data[:, dim] = (data[:, dim] - data[:, dim].mean()) / (1e-4 + data[:, dim].std())
            else:
                raise RuntimeError('You must specify both the mean and std!')
        standardized_data = data
    else:
        print 'Applying standardization on the whole data'
        if mean is not None and std is not None:
            standardized_data = (data - mean) / (1e-4 + std)
        elif mean is None and std is None:
            standardized_data = (data - data.mean(axis=0)) / (1e-4 + data.std(axis=0))
        else:
            raise RuntimeError('You must specify both the mean and std!')

    return standardized_data


# TODO; factorize get_best_k_features_indices and get_worst_k_features_indices functions.
def get_best_k_features_indices(data, report, k=None, greater_or_equal_than=None):
    features_groups = {'CASTLES': 0, 'USERS': 112, 'clayout_fixed': 112+84, 'clayout_pooled': 112+84+68}
    order = ['CASTLES', 'USERS', 'clayout_fixed', 'clayout_pooled']
    best_k_indices = []
    score_idx = 4

    if k is None and greater_or_equal_than is None:
        return []
    if k == 0 and greater_or_equal_than is not None:
        # We take all the features.
        k = data.shape[1]
        print 'We will get all the features (%s)'%k
    else:
        print 'We will get the best %s features'%k
    best_features_report = report[:k]

    if greater_or_equal_than:
        print 'We  will also get the features with mutual information => %s'%less_than_or_equal
        scores = report[:, score_idx]
        best_features_indices = numpy.where(scores >= greater_or_equal_than)[0]
        best_features_report = report[best_features_indices]

    for tup in best_features_report:
        idx, actual_idx, feature_type, feature_name, score, _ = tup
        if feature_type.startswith('CASTLE'):
            feature_type = 'CASTLES'
        if feature_type.startswith('USER'):
            feature_type = 'USERS'
        if feature_type in order:
            best_k_indices.append(int(actual_idx)+features_groups[feature_type])

    print 'The number of features retrieved is %s'%len(best_k_indices)
    return best_k_indices


def get_worst_k_features_indices(data, report, k=None, less_than_or_equal=None):
    features_groups = {'CASTLES': 0, 'USERS': 112, 'clayout_fixed': 112+84, 'clayout_pooled': 112+84+68}
    order = ['CASTLES', 'USERS', 'clayout_fixed', 'clayout_pooled']
    worst_k_indices = []
    score_idx = 4

    if k is None and less_than_or_equal is None:
        return []
    if k is None and less_than_or_equal is not None:
        # We take all the features for the moment.
        k = 0
    else:
        print 'We will get the worst %s features'%k
    worst_features_report = report[-k:]

    if less_than_or_equal:
        print 'We  will also get the features with mutual information <= %s'%less_than_or_equal
        scores = report[:, score_idx]
        worst_features_indices = numpy.where(scores <= less_than_or_equal)[0]
        worst_features_report = report[worst_features_indices]

    for tup in worst_features_report:
        idx, actual_idx, feature_type, feature_name, score, _ = tup
        if feature_type.startswith('CASTLE'):
            feature_type = 'CASTLES'
        if feature_type.startswith('USER'):
            feature_type = 'USERS'
        if feature_type in order:
            worst_k_indices.append(int(actual_idx)+features_groups[feature_type])

    print 'The number of features to be retrieved is %s'%len(worst_k_indices)
    return worst_k_indices


def save_data(save_path, train_X, valid_X, test_X, train_Y, valid_Y, test_Y):
    pass


def save_model_info(model_name, valid_pred_and_targ, test_pred_and_targ,
                    valid_p_y_given_x, test_p_y_given_x):

    print 'Saving the best model predictions and targets'
    path = '%s_pred_and_targ.npz'%model_name

    if len(valid_pred_and_targ):
        valid_pred = valid_pred_and_targ[:,0,:]
        valid_targ = valid_pred_and_targ[:,1,:]
    else:
        valid_pred = []
        valid_targ = []

    if len(test_pred_and_targ):
        test_pred = test_pred_and_targ[:,0,:]
        test_targ = test_pred_and_targ[:,1,:]
    else:
        test_pred = []
        test_targ = []

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


def combine_features(data, indices):
    # Retrieve the feature vectors associated to the indices given as input.
    features = data[:, indices]
    # Do all combinations of 2 features between these features and take
    # their product between these combinations of 2 features.
    combs = [(i,j) for i,j in
        itertools.combinations_with_replacement(numpy.arange(0,features.shape[1]), 2)]
    extra_features = []
    for comb in combs:
        prod = features[:, comb[0]] * features[:, comb[1]]
        extra_features.append(prod)
    extra_features = numpy.array(extra_features)
    shape = extra_features.shape
    extra_features = extra_features.reshape((shape[1], shape[0]))
    # Add these extra features into the input dataset.
    augmented_data = numpy.hstack((data, extra_features))
    return augmented_data


def remove_features(data, indices):
    print '%s features will be removed'%len(indices)
    valid_indices = get_good_indices(data, no_indices=indices)
    return data[:, valid_indices]


def pipeline_preprocessing(args):
    raw_data_path = args.raw_data_path
    output_data_path = args.output_data_path[0]
    filenames = args.filenames
    topk = args.topk
    to_filter = args.to_filter
    apply_to_bin_features = args.apply_to_bin_features
    transformations = args.transformations
    save_targets = args.save_targets
    report_path = args.report_path
    train_mean = None
    train_std = None
    indices_for_testing = None
    # Get the targets if necessary.
    # TODO: for the moment, the targets filenames are hardcoded.
    targets_filenames = ['train_Y.npy', 'valid_Y.npy', 'test_Y.npy']
    if save_targets:
        for filename in targets_filenames:
            # Get the full path to the targets. They must be in the raw data path.
            targets_path = os.path.join(raw_data_path, filename)
            # Load the targets.
            targets = numpy.load(targets_path)
            # Get the output full path where the targets will be saved.
            output_targets_path = os.path.join(output_data_path, filename)
            # Save the targets.
            print '... saving targets %s' %output_targets_path
            numpy.save(output_targets_path, targets)
    # Get report.
    # Mutual information used as criterion to score each feature.
    report_path = os.path.join(report_path, 'report_MI.npy')
    if os.path.exists(report_path):
        report = numpy.load(report_path)
    else:
        report = None
    import pdb; pdb.set_trace()
    for idx, filename in enumerate(filenames):
        print '\n########################'
        # Get raw dataset to be transformed.
        raw_filename = os.path.join(raw_data_path, filename)
        transformed_data = load_dataset(raw_filename)
        if to_filter:
            assert report is not None, 'Report is not present in %s' %report_path
            # Get indices of features that should be discarded based on their
            # mutual information.
            no_indices = get_worst_k_features_indices(transformed_data, report,
                                                    less_than_or_equal=to_filter)
            # Remove these features associated to the no_indices.
            transformed_data = remove_features(transformed_data, no_indices)
            # Sanity check to make sure that we are removing the same features in
            # train, valid and test sets. There are lots of functionns here that
            # should only be called once (.i.e. when loading the train set).
            if indices_for_testing is None:
                indices_for_testing = numpy.array(no_indices)
            else:
                tmp = numpy.intersect1d(indices_for_testing, no_indices)
                assert len(tmp) == len(indices_for_testing), 'not removing the same features in all datasets'
        if topk:
            assert report is not None, 'Report is not present in %s' %report_path
            # Get the indices of the top 10 features we should use for creating
            # extra features.
            yes_indices = get_best_k_features_indices(transformed_data, report, k=topk)
            # Combine the top k features to create extra features. The extra
            # features will added completely at the end of the raw dataset.
            transformed_data = combine_features(transformed_data, yes_indices)
        if idx == 0:
            # Compute the train set mean and std which will be used for
            # standardisizing all the datasets.
            train_mean, train_std = transformed_data.mean(axis=0), transformed_data.std(axis=0)

        assert train_mean is not None and train_std is not None, 'No train mean or std computed'

        for transformation in transformations:
            if transformation == 'standardize':
                transformed_data = standardize(transformed_data, mean=train_mean,
                    std=train_std, apply_to_bin_features=apply_to_bin_features)
            elif transformation == 'log_normalize':
                transformed_data = log_normalize(transformed_data,
                    apply_to_bin_features=apply_to_bin_features)
            else:
                raise NotImplementedError('Transformation %s not implemented!'%transformation)
        print '... saving the %s set'%filename
        transformed_data_path = os.path.join(output_data_path, filename)
        numpy.save(transformed_data_path, transformed_data)
        print 'Shape of dataset %s saved: %s' %(filename, transformed_data.shape)
        print '########################\n'
    print 'All files have been transformed!'
    return 0


# TODO: factorize the next 4 functions.
# output_data_path = /data/lisa/exp/chandiar/IFT6141/project2/data/MQ/standardized/diff_augmented
# output_data_path: report and datasets (train, valid and test)
# raw_data_path: raw MQ dataset to be transformed.
# Check the README in the output_data_path for more information on how the datasets
# have been transformed.
def generate_dataset_standardized_diff(raw_data_path,
    output_data_path, filenames, topk, to_filter, apply_to_bin_features):

    train_mean = None
    train_std = None
    test_indices = None
    for filename in filenames:
        print '\n########################'
        # Get raw dataset to be transformed.
        raw_filename = os.path.join(raw_data_path, filename)
        raw_data = load_dataset(raw_filename)
        # Get report.
        # Mutual information used as criterion to score each feature.
        report_path = os.path.join(output_data_path, 'report_MI.npy')
        report = numpy.load(report_path)
        if to_filter is not None:
            # Get indices of features that should be discarded based on their
            # mutual information.
            no_indices = get_worst_k_features_indices(raw_data, report,
                                                    less_than_or_equal=to_filter)
            # Remove the features associated to no_indices.
            raw_data = remove_features(raw_data, no_indices)
        if topk is not None:
            # Get the indices of the topk features we should use for creating
            # extra features.
            yes_indices = get_best_k_features_indices(raw_data, report, k=topk)
            # Combine the top k features to create extra features. The extra
            # features will added completely at the end of the raw dataset.
            raw_data = combine_features(raw_data, yes_indices)
        if idx == 0:
            # Compute the raw_data's mean and std which will be used for
            # standardisizing the datasets.
            train_mean, train_std = raw_data.mean(axis=0), raw_data.std(axis=0)
        # Sanity check to make sure that we are removing the same features in
        # train, valid and test sets. There are lots of functionns here that
        # should only be called once (.i.e. when loading the train set).
        if test_indices is None:
            test_indices = numpy.array(no_indices)
        else:
            tmp = numpy.intersect1d(test_indices, no_indices)
            assert len(tmp) == len(test_indices)
        assert train_mean is not None and train_std is not None
        # Standardize the raw dataset but only on the valid features indices.
        standardized_data = standardize(raw_data, mean=train_mean, std=train_std,
            apply_to_bin_features=apply_to_bin_features)
        print '... saving the %s set'%filename
        standardized_data_path = os.path.join(output_data_path, filename)
        numpy.save(standardized_data_path, standardized_data)
        print '########################\n'
    print 'All files have been standardized!'
    return 0


# output_data_path = /data/lisa/exp/chandiar/IFT6141/project2/data/MQ/standardized+log_normalized/diff_augmented
# output_data_path: report and datasets (train, valid and test)
# raw_data_path: raw MQ dataset to be transformed.
# Check the README in the output_data_path for more information on how the datasets
# have been transformed.
def generate_dataset_standardized_log_normalized_diff(raw_data_path,
    output_data_path, filenames, topk, to_filter, apply_to_bin_features):

    train_mean = None
    train_std = None
    test_indices = None
    for idx, filename in enumerate(filenames):
        print '\n########################'
        # Get raw dataset to be transformed.
        raw_filename = os.path.join(raw_data_path, filename)
        raw_data = load_dataset(raw_filename)
        # Get report.
        # Mutual information used as criterion to score each feature.
        report_path = os.path.join(output_data_path, 'report_MI.npy')
        report = numpy.load(report_path)
        if to_filter:
            # Get indices of features that should be discarded based on their
            # mutual information.
            no_indices = get_worst_k_features_indices(raw_data, report,
                                                    less_than_or_equal=to_filter)
            # Remove these features associated to the no_indices.
            raw_extra_data = remove_features(raw_extra_data, no_indices)
        if topk:
            # Get the indices of the top 10 features we should use for creating
            # extra features.
            yes_indices = get_best_k_features_indices(raw_data, report, k=topk)
            # Combine the top k features to create extra features. The extra
            # features will added completely at the end of the raw dataset.
            raw_data = combine_features(raw_data, yes_indices)
        if idx == 0:
            # Compute the raw_data's mean and std which will be used for
            # standardisizing the datasets.
            train_mean, train_std = raw_data.mean(axis=0), raw_data.std(axis=0)
        # Sanity check to make sure that we are removing the same features in
        # train, valid and test sets. There are lots of functionns here that
        # should only be called once (.i.e. when loading the train set).
        if test_indices is None:
            test_indices = numpy.array(no_indices)
        else:
            tmp = numpy.intersect1d(test_indices, no_indices)
            assert len(tmp) == len(test_indices)
        assert train_mean is not None and train_std is not None
        # Standardize the raw dataset but only on the valid features indices.
        standardized_data = standardize(raw_data, mean=train_mean, std=train_std)
        # Log-normalize the standardized dataset but only on the valid features indices.
        standardized_log_normalized_data = log_normalize(standardized_data)
        print '... saving the %s set'%filename
        standardized_log_normalized_data_path = os.path.join(output_data_path, filename)
        numpy.save(standardized_log_normalized_data_path, standardized_log_normalized_data)
        print '########################\n'
    print 'All files have been standardized and log-normalized!'
    return 0


# output_data_path = /data/lisa/exp/chandiar/IFT6141/project2/data/MQ/log_normalized/diff_augmented
# output_data_path: report and datasets (train, valid and test)
# raw_data_path: raw MQ dataset to be transformed.
# Check the README in the output_data_path for more information on how the datasets
# have been transformed.
def generate_dataset_log_normalized_diff(raw_data_path,
    output_data_path, filenames, topk, to_filter):

    test_indices = None
    for filename in filenames:
        print '\n########################'
        # Get raw dataset to be transformed.
        raw_filename = os.path.join(raw_data_path, filename)
        raw_data = load_dataset(raw_filename)
        # Get report.
        # Mutual information used as criterion to score each feature.
        report_path = os.path.join(output_data_path, 'report_MI.npy')
        report = numpy.load(report_path)
        # Get indices of features that should be discarded based on their
        # mutual information.
        no_indices = get_worst_k_features_indices(raw_data, report,
                                                  less_than_or_equal=to_filter)
        # Get the indices of the top 10 features we should use for creating
        # extra features.
        yes_indices = get_best_k_features_indices(raw_data, report, k=topk)
        # Combine the top k features to create extra features. The extra
        # features will added completely at the end of the raw dataset.
        raw_extra_data = combine_features(raw_data, yes_indices)
        # Remove these features associated to the no_indices.
        raw_extra_data = remove_features(raw_extra_data, no_indices)
        # Sanity check to make sure that we are removing the same features in
        # train, valid and test sets. There are lots of functionns here that
        # should only be called once (.i.e. when loading the train set).
        if test_indices is None:
            test_indices = numpy.array(no_indices)
        else:
            tmp = numpy.intersect1d(test_indices, no_indices)
            assert len(tmp) == len(test_indices)
        # Log-normalize the raw dataset but only on the valid features indices.
        log_normalized_data = log_normalize(raw_extra_data)
        print '... saving the %s set'%filename
        log_normalized_data_path = os.path.join(output_data_path, filename)
        numpy.save(log_normalized_data_path, log_normalized_data)
        print '########################\n'
    print 'All files have been log-normalized!'
    return 0


# output_data_path = /data/lisa/exp/chandiar/IFT6141/project2/data/MQ/log_normalized+standardized/diff_augmented
# output_data_path: report and datasets (train, valid and test)
# raw_data_path: raw MQ dataset to be transformed.
# Check the README in the output_data_path for more information on how the datasets
# have been transformed.
def generate_dataset_log_normalized_standardized_diff(raw_data_path,
    output_data_path, filenames, topk, to_filter):

    train_mean = None
    train_std = None
    test_indices = None
    for idx, filename in enumerate(filenames):
        print '\n########################'
        # Get raw dataset to be transformed.
        raw_filename = os.path.join(raw_data_path, filename)
        raw_data = load_dataset(raw_filename)
        # Get report.
        # Mutual information used as criterion to score each feature.
        report_path = os.path.join(output_data_path, 'report_MI.npy')
        report = numpy.load(report_path)
        # Get indices of features that should be discarded based on their
        # mutual information.
        no_indices = get_worst_k_features_indices(raw_data, report,
                                                  less_than_or_equal=to_filter)
        # Get the indices of the top 10 features we should use for creating
        # extra features.
        yes_indices = get_best_k_features_indices(raw_data, report, k=topk)
        # Combine the top k features to create extra features. The extra
        # features will added completely at the end of the raw dataset.
        raw_extra_data = combine_features(raw_data, yes_indices)
        # Remove these features associated to the no_indices.
        raw_extra_data = remove_features(raw_extra_data, no_indices)
        # Sanity check to make sure that we are removing the same features in
        # train, valid and test sets. There are lots of functionns here that
        # should only be called once (.i.e. when loading the train set).
        if test_indices is None:
            test_indices = numpy.array(no_indices)
        else:
            tmp = numpy.intersect1d(test_indices, no_indices)
            assert len(tmp) == len(test_indices)
        # Log-normalize the raw dataset but only on the valid features indices.
        log_normalized_data = log_normalize(raw_extra_data, save_path=output_data_path)
        if idx == 0:
            # Compute the log-normalized data's mean and std which will be used
            # for standardisizing the datasets.
            train_mean, train_std = log_normalized_data.mean(axis=0), log_normalized_data.std(axis=0)
        assert train_mean is not None and train_std is not None
        # Standardize the log-normalized dataset but only on the valid
        # features indices.
        log_normalized_standardized_data  = standardize(log_normalized_data,
                                                mean=train_mean, std=train_std)
        print '... saving the %s set'%filename
        log_normalized_standardized_data_path = os.path.join(output_data_path, filename)
        numpy.save(log_normalized_standardized_data_path, log_normalized_standardized_data)
        print '########################\n'
    print 'All files have been log-normalized!'
    return 0


def main():
    """
    Executable entry point.

    :return: 0 on success.
    # TODO: return non-zero error code on failure (when we raise an error).
    """
    args = parse_args()

    pipeline_preprocessing(args)

    return 0


def parse_args():
    """
    Parse command-line arguments.

    :return: Parsed arguments.
    """
    # The global program parser.
    parser = argparse.ArgumentParser(description=('Pipeline for applying '
        'transformations on MQ raw dataset.'))

    parser.add_argument('-i',  '--raw_data_path', type=str, default='/data/lisa/exp/chandiar/IFT6141/project2/data/MQ/raw')
    parser.add_argument('-o', '--output_data_path', type=str, nargs=1)
    parser.add_argument('-r',  '--report_path', type=str, default='/data/lisa/exp/chandiar/MQ/mutual_info_report/diff')
    parser.add_argument('-n', '--filenames', type=str, default=['train_X.npy', 'valid_X.npy', 'test_X.npy'], nargs=3)
    parser.add_argument('-k', '--topk', type=int)
    parser.add_argument('-f', '--to_filter', type=float)
    parser.add_argument('-b', '--apply_to_bin_features', action='store_true')
    parser.add_argument('-t', '--transformations', choices=['standardize', 'log_normalize'], nargs='+', default=['standardize'])
    parser.add_argument('-s', '--save_targets', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    sys.exit(main())


'''
if __name__ == "__main__":
    import pdb; pdb.set_trace()
    raw_data_path = '/data/lisa/exp/chandiar/IFT6141/project2/data/MQ/raw'
    output_data_path = '/data/lisa/exp/chandiar/IFT6141/project2/data/MQ/standardized/diff'
    topk=None
    to_filter=None
    # NOTE: The order is important since we want to compute the mean and std of
    # the train set that will be used for doing the standardization of all
    # datasets.
    filenames = ['train_X.npy', 'valid_X.npy', 'test_X.npy']
    generate_dataset_standardized_diff(raw_data_path, output_data_path, filenames, topk, to_filter)
    #generate_dataset_log_normalized_diff_augmented()
    #generate_dataset_standardized_log_normalized_diff_augmented(raw_data_path, output_data_path, filenames, topk)
    #generate_dataset_log_normalized_standardized_diff_augmented()
    #histograms(which_data='train', transformations=['log_normalize', 'standardize'], save_fig=True, show_graph=False)
'''