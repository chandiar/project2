# -*- coding: utf-8 -*-
import sys
import numpy
import os
import numpy.random as r
import socket
import time

HOST = socket.gethostname()

S = int(time.time())
print S
r.seed(S)

model_config = {
        # http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
        'gdbt' : {
            'model'             : 'gdbt',
            'n_estimators'      : ((40,500),int),
            'learning_rate'     : ((1e-4,1),float),
            'max_depth'         : ((3,20),int),
            'min_samples_split' : ((2,10),int),
            'min_samples_leaf'  : ((1,10),int),
            'subsample'         : ((1e-2,1.0),float),
            #'max_features'      : ((1e-4,0.5),float),
            #'verbose'           : 0, 1, or > 1
        },

        # http://scikit-learn.org/dev/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        'random_forest' : {
            'model'             : 'random_forest',
            'n_estimators'      : ((40,500),int),
            'max_depth'         : ((3,20),int),
            #'criterion'         : 'gini',
            #'max_features'      : ((1e-4,0.5),float),
            'min_samples_split' : ((2,10),int),
            'min_samples_leaf'  : ((1,10),int),
            #'min_density'       : ((1e-4,1.0),float),
            #'bootstrap'         : True,
            #'n_jobs'            : 1,
            #'verbose'           : 0, 1, or > 1
        },

        # http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        'knn' : {
            'model'             : 'knn',
            'n_neighbors'       : ((1,10), int),
            #'weights'           : 'uniform' or 'distance' or callable function
            #'p'                 : 2,
        },

        # http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.RadiusNeighborsClassifier.html
        # 'rad_nn' : {
            # 'model'             : 'rad_nn',
            #'radius'            : ((1,100), float),
            #'weights'           : 'uniform' or 'distance' or callable function
            #'p'                 : 2,
        # },

        # http://scikit-learn.org/dev/modules/generated/sklearn.neighbors.NearestCentroid.html
        # 'nearest_centroid' : {
            # 'model'             : 'nearest_centroid',
            # 'metric'            : 'euclidean',
            # 'shrink_threshold'  : None,
        # },

        # http://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html
        'svm' : {
            'model'             : 'svm',
            'C'                 : ((1,10000),float),
            'kernel'            : ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree'            : ((3, 10), int),
            'gamma'             : ((1e-5, 1e3), float),
            'coef0'             : ((1e-5, 100), float),
            'tol'               : ((1e-5, 1), float),
            'cache_size'        : 1000,
            #'max_iter'          : ((100, 1000), int),
        },

        # http://scikit-learn.org/dev/modules/generated/sklearn.svm.LinearSVC.html
        'lsvm' : {
            'model'             : 'lsvm',
            'C'                 : ((1,10000),float),
            'loss'              : 'l2', # 'l1'
            'penalty'           : 'l2', # 'l1'
            'dual'              : False,
            'multi_class'       : 'ovr', # 'crammer_singer'
            'fit_intercept'     : True,
            'intercept_scaling' : 1, # ((1e-5, 1), float),
            'tol'               : ((1e-5, 1), float),
        },
}

def exp_sampling(((low,high),t)):
  low = numpy.log(low)
  high = numpy.log(high)
  return t(numpy.exp(r.uniform(low,high)))

def cmd_line_embed(config):
  # TODO: do not hardcode the common options!
  cmd = 'jobman -r cmdline ml_code.init.experiment save_model_params=False save_model_info=True save_state=True gpu=False '

  for key in config:

    if type(config[key])==type(()):
      val = exp_sampling(config[key])
      cmd += (key+'='+`val`+' ')
    elif type(config[key])==type([]):
      cmd += (key+'='+`config[key][r.randint(len(config[key]))]`+' ')
    else:
      cmd += (key+'='+`config[key]`+' ')
  return cmd

def get_cmd(model, mem):
    cmd = 'jobdispatch --file=commands_%s --mem=%s'%(model, mem)
    if 'umontreal' in HOST:
        cmd += ' --condor '
    elif 'ip05' in HOST:
        cmd += ' --bqtools '
    return cmd

if __name__=='__main__':
    mem = 2000
    models = {'gdbt': (True, 150, mem),  'random_forest': (True, 150, mem),
              'svm' : (True, 150, mem),  'lsvm'         : (False, 150, mem),
              'knn' : (False, 150, mem)}

    cmds = []

    for model, (launch, n_exps, mem) in models.iteritems():
        if not launch:
            continue
        cmd = get_cmd(model, mem)
        os.mkdir(model)
        f = open('%s/commands_%s'%(model, model),'w')
        for i in range(n_exps):
            f.write(cmd_line_embed(model_config[model])+'\n')
        f.close()
        cmds.append((model, cmd))

    import pdb; pdb.set_trace()
    for model, cmd in cmds:
        os.chdir(model)
        os.system(cmd)
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


