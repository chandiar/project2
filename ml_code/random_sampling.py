# -*- coding: utf-8 -*-
import multiprocessing
import numpy
import numpy.random as rng
import os
import socket
import sys
import time


HOST = socket.gethostname()
print HOST

S = int(time.time())
print S
rng.seed(S)

# TODO: do not hardcode this option, should be given in the command line.
with_gpu = False
print 'with_gpu=',with_gpu


model_config = {
        # http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        'gdbt' : {
            'model'             : 'gdbt',
            'features'          : 'hog',
            'n_estimators'      : ((40,500),int),
            'learning_rate'     : ((1e-4,1),float),
            'max_depth'         : ((3,20),int),
            'min_samples_split' : 2,#((2,10),int),
            'min_samples_leaf'  : 1,#((1,10),int),
            'subsample'         : 1,#((1e-2,1.0),float),
            #'max_features'      : ((1e-4,0.5),float),
            #'verbose'           : 0, 1, or > 1
            # TODO: COMMON options.
            'save_model_params' : False,
            'save_model_info'   : True,
            'save_state'        : True,
            'gpu'               : False,
        },

        # http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'random_forest' : {
            'model'             : 'random_forest',
            'features'          : 'hog',
            'n_estimators'      : ((40,500),int),
            'max_depth'         : ((3,20),int),
            #'criterion'         : 'gini',
            #'max_features'      : ((1e-4,0.5),float),
            'min_samples_split' : 2,#((2,10),int),
            'min_samples_leaf'  : 1,#((1,10),int),
            #'min_density'       : ((1e-4,1.0),float),
            #'bootstrap'         : True,
            #'n_jobs'            : 1,
            #'verbose'           : 0, 1, or > 1
            # TODO: COMMON options.
            'save_model_params' : False,
            'save_model_info'   : True,
            'save_state'        : True,
            'gpu'               : False,
        },

        # http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        'knn' : {
            'model'             : 'knn',
            'features'          : 'hog',
            'n_neighbors'       : ((1,10), int),
            #'weights'           : 'uniform' or 'distance' or callable function
            #'p'                 : 2,
            # TODO: COMMON options.
            'save_model_params' : False,
            'save_model_info'   : True,
            'save_state'        : True,
            'gpu'               : False,
        },

        # http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html
        # 'rad_nn' : {
            # 'model'             : 'rad_nn',
            #'radius'            : ((1,100), float),
            #'weights'           : 'uniform' or 'distance' or callable function
            #'p'                 : 2,
            # TODO: COMMON options.
            #'save_model_params' : False,
            #'save_model_info'   : True,
            #'save_state'        : True,
            #'gpu'               : False,
        # },

        # http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.NearestCentroid.html
        # 'nearest_centroid' : {
            # 'model'             : 'nearest_centroid',
            # 'metric'            : 'euclidean',
            # 'shrink_threshold'  : None,
            # TODO: COMMON options.
            #'save_model_params' : False,
            #'save_model_info'   : True,
            #'save_state'        : True,
            #'gpu'               : False,
        # },

        # http://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html
        'svm' : {
            'model'             : 'svm',
            'features'          : 'hog',
            'C'                 : ((1,10000),float),
            'kernel'            : ['poly', 'rbf'], #['linear', 'poly', 'rbf', 'sigmoid'],
            'degree'            : ((3, 10), int),
            'gamma'             : ((1e-5, 1e3), float),
            'coef0'             : ((1e-5, 100), float),
            'tol'               : ((1e-5, 1), float),
            'cache_size'        : 1000,
            'probability'       : True,
            #'max_iter'          : ((100, 1000), int),
            # TODO: COMMON options.
            'save_model_params' : False,
            'save_model_info'   : True,
            'save_state'        : True,
            'gpu'               : False,
        },

        # http://scikit-learn.org/dev/modules/generated/sklearn.svm.LinearSVC.html
        'lsvm' : {
            'model'             : 'lsvm',
            'features'          : 'hog',
            'C'                 : ((1,10000),float),
            'loss'              : 'l2', # 'l1'
            'penalty'           : 'l2', # 'l1'
            'dual'              : False,
            'multi_class'       : 'ovr', # 'crammer_singer'
            'fit_intercept'     : True,
            'intercept_scaling' : 1, # ((1e-5, 1), float),
            'tol'               : ((1e-5, 1), float),

            # TODO: COMMON options.
            'save_model_params' : False,
            'save_model_info'   : True,
            'save_state'        : True,
            'gpu'               : False,
        },

        # MLP
        'nnet' : {
            'model'                 : 'nnet',
            # dataset = ['mnist',
            #            'mq+diff+std+top10', 'mq+diff+log+top10',
            #            'mq+diff+std+log+top10', 'mq+diff+log+std+top10']
            'dataset'               : 'mq+diff+std+log+top10',
            'features'              : None,
            # TODO: COMMON options.
            'save_model_params'     : False,
            'save_model_info'       : True,
            'save_state'            : True,
            'gpu'                   : True,
            'save_losses_and_costs' : True,
            'seed'                  : 1234,
            'batch_size'            : 100,
            'lr_decay'              : True,
            'init_lr'               : [[1e-1, 1e-3], [-1, -1]],
            'decrease_constant'     : ((1e-4, 1e-1), float),
            'n_epochs'              : 300,
            'dropout_p'             : -1,#((0.3, 0.8), float),
            'maxout_k'              : -1,#((2, 5), int),
            'mom'                   : 0,#((0.4, 0.8), float),
            'filter_square_limit'   : 15.0,
            # Top layer output activation.
            # TODO: not used yet.
            'output_activation'     : 'binary_cross_entropy',
            # Early-stopping.
            # Look at this many examples regardless.
            'patience'              : 10000,
            # Wait this much longer when a new best is found.
            'patience_increase'     : 2,
            # A relative improvement of this much is considered significant.
            'improvement_threshold' : 0.99995,
            # regularization terms
            'L1'                    : ((1e-10, 1), float),
            'L2'                    : ((1e-10, 1), float),
            ## Hidden layers ##
            # set this to [0] to fall back to LR
            'hidden_sizes'          : [[500, 2000], [500, 2000]],
            # Hidden output activation:
            # tanh, rectifier, softplus, sigmoid, hard_tanh
            'hidden_activation'     : 'tanh',
        },

        # Convolutional neural net.
        'cnn' : {
            'model'                 : 'cnn',
            'features'              : None,
            # TODO: COMMON options.
            'save_model_params'     : False,
            'save_model_info'       : True,
            'save_state'            : True,
            'gpu'                   : True,
            'save_losses_and_costs' : True,
            'seed'                  : 1234,
            'batch_size'            : 100,
            'lr_decay'              : True,
            'init_lr'               : [[1e-1, 1e-3], [-1, -1]],
            'decrease_constant'     : ((1e-4, 1e-1), float),
            'n_epochs'              : 1000,
            'dropout_p'             : -1,#,((0.3, 0.8), float),
            'maxout_k'              : -1,#((2, 5), int),
            'mom'                   : 0,#0.5,
            'filter_square_limit'   : 15.0,
            # TODO: not used yet.
            'output_activation'     : 'softmax',
            # Early-stopping.
            # Look at this many examples regardless.
            'patience'              : 10000,
            # Wait this much longer when a new best is found.
            'patience_increase'     : 2,
            # A relative improvement of this much is considered significant.
            'improvement_threshold' : 0.99995,
            # regularization terms
            'L1'                    : ((1e-10, 1), float),
            'L2'                    : ((1e-10, 1), float),
            ## Hidden layers ##
            # set this to [0] to fall back to LR
            'hidden_sizes'          : [[500, 2500], [500, 2500]],
            # Hidden output activation:
            # tanh, rectifier, softplus, sigmoid, hard_tanh
            'activation'            : 'tanh',
            # TODO: cnn options only.
            # Number of filters.
            'nkerns'            : [20, 50],
        },
}


def worker(num, cmd):
    """worker function"""
    print 'Worker %s' %num
    os.system(cmd)
    return


def exp_sampling(((low,high),t)):
  low = numpy.log(low)
  high = numpy.log(high)
  return t(numpy.exp(rng.uniform(low,high)))


def cmd_line_embed(config):
  # TODO: do not hardcode the common options!
  if 'briaree' in HOST:
    cmd = 'jobman -r cmdline ml_code.init.experiment '
  else:
    cmd = 'THEANO_FLAGS=floatX=float32 jobman -r cmdline ml_code.init.experiment '

  for key in config:

    if 'hidden_sizes' == key:
        cmd_params_structure = config[key]
        hidden_sizes = []
        for layer in cmd_params_structure:
            v = rng.randint(layer[0], layer[1])
            hidden_sizes.append(v)
        temp = `hidden_sizes`
        # TODO: removing whitespaces should be done for all cases.
        temp = temp.replace(' ', '')
        cmd +=  (key+'='+temp+' ')
    elif 'init_lr' == key:
        # TODO: for the moment, same learning rates for every layers!!!
        cmd_params_lrs = config[key]
        # Set up learning rates for different layers.
        init_lr = []
        candidate = None

        for layer in cmd_params_lrs:
            # TODO: not really a layer!
            if layer == [-1, -1]:
                # different layers will use the same learning rate
                assert candidate is not None
                init_lr.append(candidate)
            else:
                l = numpy.log(layer[0])
                r = numpy.log(layer[1])
                candidate = numpy.exp(rng.uniform(l,r))
                init_lr.append(candidate)

        temp = `init_lr`
        temp = temp.replace(' ', '')
        cmd += (key+'='+temp+' ')

        # TODO: checking not tested.
        #assert len(init_lr) == 2 * len(hidden_sizes) + 2
    elif 'nkerns' == key:
      # TODO: should not harcode the name of the option in this case.
      temp = `config[key]`
      temp = temp.replace(' ', '')
      cmd += (key+'='+temp+' ')
    elif type(config[key])==type(()):
      val = exp_sampling(config[key])
      cmd += (key+'='+`val`+' ')
    elif type(config[key])==type([]):
      cmd += (key+'='+`config[key][rng.randint(len(config[key]))]`+' ')
    else:
      cmd += (key+'='+`config[key]`+' ')
  return cmd


def get_cmd(model, mem):
    cmd = 'jobdispatch --file=commands_%s --mem=%s'%(model, mem)
    if 'umontreal' in HOST:
        cmd += ' --condor '
    elif 'ip05' in HOST:
        cmd += ' --bqtools '
    elif 'briaree' in HOST:
        # Briaree cluster.
        if with_gpu:
            cmd += ' --gpu '
        cmd += ' --env=THEANO_FLAGS=floatX=float32 '
    return cmd


if __name__=='__main__':
    mem = 1000
    models = {'gdbt': (False, 150, 1000),  'random_forest': (False, 150, 1000),
              'svm' : (False, 150, 1500),  'lsvm'         : (False, 150, mem),
              'knn' : (False, 150, 1000),  'nnet'         : (True, 200, 1000),
              'cnn' : (False, 5, 1500)}

    # Do you want to run the jobs locally instead on the cluster.
    run_local = True
    # Number of jobs to be run locally simultaneously.
    n_local_jobs = 25
    cmds = []
    exps_by_model = {}

    for model, (launch, n_exps, mem) in models.iteritems():
        if not launch:
            continue
        cmd = get_cmd(model, mem)
        if not os.path.exists(model):
            os.mkdir(model)
        f = open('%s/commands_%s'%(model, model),'w')
        for i in range(n_exps):
            exp_cmd = cmd_line_embed(model_config[model])
            f.write(exp_cmd+'\n')
            exps_by_model.setdefault(model, [])
            exps_by_model[model].append(exp_cmd)
        f.close()
        cmds.append((model, cmd))

    for model, cmd in cmds:
        os.chdir(model)
        if not run_local:
            os.system(cmd)
        else:
            print 'Jobs will be run locally.'
            print '%s jobs will be run simultaneously.'%n_local_jobs
            import pdb; pdb.set_trace()
            n_jobs = 0
            n_job_simult = 0
            jobs = []
            commands = exps_by_model[model]
            for command in commands:
                if n_job_simult < n_local_jobs:
                    assert len(jobs) <= n_local_jobs
                    p = multiprocessing.Process(target=worker, args=(n_jobs, command))
                    jobs.append((n_jobs, p))
                    p.start()
                    n_jobs += 1
                    n_job_simult += 1
                else:
                    ready_for_more = False
                    while not ready_for_more:
                        for j_i, j in enumerate(jobs):
                            if 'stopped' in str(j[1]):
                                print 'Job %s finished' %j[0]
                                jobs.pop(j_i)
                                n_job_simult -= 1
                                ready_for_more = True
                                break

            more_jobs = True
            while more_jobs:
                for j_i, j in enumerate(jobs):
                    if 'stopped' in str(j[1]):
                        print 'Job %s finished' %j[0]
                        jobs.pop(j_i)
                    if len(jobs) == 0:
                        more_jobs = False
                        break
            print 'All jobs finished running.'


        os.chdir('../')

    '''
    if n_gpu:
    f = open('commands','w')
    for i in range(n_gpu):
        f.write(cmd_line_embed(standard_config)+'\n')
    f.close()

    cmd = 'jobdispatch --gpu --env=THEANO_FLAGS=device=gpu,floatX=float32 --mem=1500 --bqtools --file=commands'
    os.system(cmd)
    '''
