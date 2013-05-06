import os
import numpy
from scipy.stats import mode

import util
from util import load_data

# whic_models[0] = SVM,
# whic_models[1] = CNN,
# whic_models[2] = MLP,
which_models = [0,0,1]
these_k = [1, 5, 10]
all_models = [0, 0, 0]
clr_by_model_type = {'svm':{},'cnn':{},'mlp':{}}
all_models_clr_by_type = {'svm': None, 'cnn': None, 'mlp': None}

train_size = 50000
valid_size = 10000
test_size = 10000

data_dir = util.get_dataset_base_path()
data_path = os.path.join(data_dir, 'mnist.pkl.gz')
datasets = load_data(data_path, splits=[train_size, valid_size, test_size], shared=False)

train_x, train_y = datasets[0]
valid_x, valid_y = datasets[1]
test_x, test_y = datasets[2]

all_models_path = os.path.join(data_dir, 'models_for_committee')

# SVM
if which_models[0]:
    print 'Building SVM committee'
    svm_models_path = os.path.join(all_models_path, 'svm')
    svm_models = []

    for model_path in os.listdir(svm_models_path):
        svm_model = util.load_tar_bz2(os.path.join(svm_models_path, model_path, 'svm.tar.bz2'))
        test_nnet_output = svm_model.predict_proba(test_x)
        pred_targ = util.load_tar_bz2(os.path.join(svm_models_path, model_path, 'svm_info.tar.bz2'))
        valid_targ = pred_targ[0]['targets']
        valid_pred = pred_targ[0]['predictions']
        test_targ = pred_targ[1]['targets']
        test_pred = pred_targ[1]['predictions']

        valid_clr = 1 - (sum(valid_targ == valid_pred) / float(len(valid_targ)))
        test_clr = 1 - (sum(test_targ == test_pred) / float(len(test_targ)))

        svm_models.append( (valid_clr, test_nnet_output, test_pred, test_clr) )

    svm_models.sort(key=lambda tup: tup[0])
    all_models_clr_by_type['svm'] = [(svm[0], svm[3]) for svm in svm_models]

    if all_models[0]:
        print 'all models'
        if len(svm_models) == 1:
            print 'Only 1 model'
            these_k = [1]
        else:
            these_k = numpy.arange(1, len(svm_models))

    for first_k in these_k:

        if first_k > len(svm_models):
            print 'There are %s SVM models and first_k=%s' %(len(svm_models), first_k)
            print 'Thus first_k is set to ', len(svm_models)
            first_k = len(svm_models)

        if first_k in clr_by_model_type['svm']:
            print 'skipping %s because it has already been computed' %first_k
            continue

        all_predictions = []
        all_test_nnet_output = numpy.concatenate([svm[1] for svm in svm_models[0:first_k]])
        for sample in xrange(10000):
            probs_by_model = all_test_nnet_output.take(numpy.arange(start=sample, stop=first_k*10000, step=10000), axis=0)
            preds_by_model = numpy.argmax(probs_by_model, axis=1)
            mean = numpy.mean(probs_by_model, axis=0)
            prediction_by_mean = int(numpy.argmax(mean))
            prediction_by_median = int(numpy.argmax(mean))
            prediction_by_voting = int(mode(preds_by_model)[0][0])

            all_predictions.append((prediction_by_mean, prediction_by_median, prediction_by_voting))
            '''
            test_targ_i = test_targ[sample]
            if test_targ_i != prediction_by_mean:
                import pdb; pdb.set_trace()
                pass
            '''

        all_predictions = numpy.array(all_predictions)
        clr_by_mean = None
        clr_by_median = None
        clr_by_voting = None

        clr_by_mean = 1 - (sum(test_targ == all_predictions[:, 0]) / float(len(test_targ)))
        clr_by_median = 1 - (sum(test_targ == all_predictions[:, 1]) / float(len(test_targ)))
        clr_by_voting = 1 - (sum(test_targ == all_predictions[:, 2]) / float(len(test_targ)))

        clr_by_model_type['svm'][first_k] = (clr_by_mean, clr_by_median, clr_by_voting)

    print all_models_clr_by_type['svm']
    print clr_by_model_type['svm']

import pdb; pdb.set_trace()


# CNN
if which_models[1]:
    print 'Building CNN committee'
    cnn_models_path = os.path.join(all_models_path, 'cnn')
    cnn_models = []

    for model_path in os.listdir(cnn_models_path):
        nnet_output = numpy.load(os.path.join(cnn_models_path, model_path, 'nnet_p_y_given_x.npz'))
        valid_nnet_output = nnet_output['valid_p_y_given_x'].reshape((10000, 10))
        test_nnet_output = nnet_output['test_p_y_given_x'].reshape((10000, 10))
        pred_targ = numpy.load(os.path.join(cnn_models_path, model_path, 'nnet_pred_and_targ.npz'))
        valid_targ = pred_targ['valid_targ'].flatten()
        valid_pred = pred_targ['valid_pred'].flatten()
        test_targ = pred_targ['test_targ'].flatten()
        test_pred = pred_targ['test_pred'].flatten()

        valid_clr = 1 - (sum(valid_targ == valid_pred) / float(len(valid_targ)))
        test_clr = 1 - (sum(test_targ == test_pred) / float(len(test_targ)))

        cnn_models.append( (valid_clr, test_nnet_output, test_pred, test_clr) )

    cnn_models.sort(key=lambda tup: tup[0])
    all_models_clr_by_type['cnn'] = [(cnn[0], cnn[3]) for cnn in cnn_models]

    if all_models[1]:
        print 'all models'
        if len(cnn_models) == 1:
            print 'Only 1 model'
            these_k = [1]
        else:
            these_k = numpy.arange(1, len(cnn_models))

    for first_k in these_k:
        if first_k > len(cnn_models):
            print 'There are %s CNN models and first_k=%s' %(len(cnn_models), first_k)
            print 'Thus first_k is set to ', len(cnn_models)
            first_k = len(cnn_models)
        if first_k in clr_by_model_type['cnn']:
            print 'skipping %s because it has already been computed' %first_k
            continue

        all_predictions = []
        all_test_nnet_output = numpy.concatenate([cnn[1] for cnn in cnn_models[0:first_k]])
        for sample in xrange(10000):
            probs_by_model = all_test_nnet_output.take(numpy.arange(start=sample, stop=first_k*10000, step=10000), axis=0)
            preds_by_model = numpy.argmax(probs_by_model, axis=1)
            mean = numpy.mean(probs_by_model, axis=0)
            prediction_by_mean = int(numpy.argmax(mean))
            prediction_by_median = int(numpy.argmax(mean))
            prediction_by_voting = int(mode(preds_by_model)[0][0])

            all_predictions.append((prediction_by_mean, prediction_by_median, prediction_by_voting))
            '''
            test_targ_i = test_targ[sample]
            if test_targ_i != prediction_by_mean:
                import pdb; pdb.set_trace()
                pass
            '''

        all_predictions = numpy.array(all_predictions)
        clr_by_mean = None
        clr_by_median = None
        clr_by_voting = None

        clr_by_mean = 1 - (sum(test_targ == all_predictions[:, 0]) / float(len(test_targ)))
        clr_by_median = 1 - (sum(test_targ == all_predictions[:, 1]) / float(len(test_targ)))
        clr_by_voting = 1 - (sum(test_targ == all_predictions[:, 2]) / float(len(test_targ)))

        clr_by_model_type['cnn'][first_k] = (clr_by_mean, clr_by_median, clr_by_voting)

    print all_models_clr_by_type['cnn']
    print clr_by_model_type['cnn']

import pdb; pdb.set_trace()


# MLP
if which_models[2]:
    print 'Building MLP committee'
    mlp_models_path = os.path.join(all_models_path, 'nnet')
    mlp_models = []

    for model_path in os.listdir(mlp_models_path):
        nnet_output = numpy.load(os.path.join(mlp_models_path, model_path, 'nnet_p_y_given_x.npz'))
        valid_nnet_output = nnet_output['valid_p_y_given_x'].reshape((10000, 10))
        test_nnet_output = nnet_output['test_p_y_given_x'].reshape((10000, 10))
        pred_targ = numpy.load(os.path.join(mlp_models_path, model_path, 'nnet_pred_and_targ.npz'))
        valid_targ = pred_targ['valid_targ'].flatten()
        valid_pred = pred_targ['valid_pred'].flatten()
        test_targ = pred_targ['test_targ'].flatten()
        test_pred = pred_targ['test_pred'].flatten()

        valid_clr = 1 - (sum(valid_targ == valid_pred) / float(len(valid_targ)))
        test_clr = 1 - (sum(test_targ == test_pred) / float(len(test_targ)))

        mlp_models.append( (valid_clr, test_nnet_output, test_pred, test_clr) )

    mlp_models.sort(key=lambda tup: tup[0])
    all_models_clr_by_type['mlp'] = [(mlp[0], mlp[3]) for mlp in mlp_models]

    if all_models[2]:
        print 'all models'
        if len(mlp_models) == 1:
            print 'Only 1 model'
            these_k = [1]
        else:
            these_k = numpy.arange(1, len(mlp_models))


    for first_k in these_k:
        if first_k > len(mlp_models):
            print 'There are %s MLP models and first_k=%s' %(len(mlp_models), first_k)
            print 'Thus first_k is set to ', len(mlp_models)
            first_k = len(mlp_models)

        if first_k in clr_by_model_type['mlp']:
            print 'skipping %s because it has already been computed' %first_k
            continue

        all_predictions = []
        all_test_nnet_output = numpy.concatenate([mlp[1] for mlp in mlp_models[0:first_k]])
        for sample in xrange(10000):
            probs_by_model = all_test_nnet_output.take(numpy.arange(start=sample, stop=first_k*10000, step=10000), axis=0)
            preds_by_model = numpy.argmax(probs_by_model, axis=1)
            mean = numpy.mean(probs_by_model, axis=0)
            prediction_by_mean = int(numpy.argmax(mean))
            prediction_by_median = int(numpy.argmax(mean))
            prediction_by_voting = int(mode(preds_by_model)[0][0])

            all_predictions.append((prediction_by_mean, prediction_by_median, prediction_by_voting))
            '''
            test_targ_i = test_targ[sample]
            if test_targ_i != prediction_by_mean:
                import pdb; pdb.set_trace()
                pass
            '''

        all_predictions = numpy.array(all_predictions)
        clr_by_mean = None
        clr_by_median = None
        clr_by_voting = None

        clr_by_mean = 1 - (sum(test_targ == all_predictions[:, 0]) / float(len(test_targ)))
        clr_by_median = 1 - (sum(test_targ == all_predictions[:, 1]) / float(len(test_targ)))
        clr_by_voting = 1 - (sum(test_targ == all_predictions[:, 2]) / float(len(test_targ)))

        clr_by_model_type['mlp'][first_k] = (clr_by_mean, clr_by_median, clr_by_voting)

    print all_models_clr_by_type['mlp']
    print clr_by_model_type['mlp']

    import pdb; pdb.set_trace()