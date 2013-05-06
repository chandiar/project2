# -*- coding: utf-8 -*-

import cPickle
import gzip
import os
import sys
import numpy
import scipy
import scipy.io

import project2
from project2.ml_code import util

data_dir = util.get_dataset_base_path()


def build_features_matrix(labels, which_data):
    print 'Building features matrix from %s' %which_data
    dir_path = os.path.join(data_dir, 'hog_features_with_numpy', '%s'%which_data)
    hog_features = []
    for idx, label in enumerate(labels):
        sys.stdout.write('\r%2d%%' % int(idx / float(len(labels)) * 100 + 0.5))
        sys.stdout.flush()
        filename = 'image_idx_%s_number_%s.npy'%(idx, label)
        hog = numpy.load(os.path.join(dir_path, filename))
        hog_features.append(hog.flatten())
    hog_features = numpy.array(hog_features)
    numpy.save(os.path.join(data_dir, '%s_hog_features.npy'%which_data), hog_features)

def main():
    # Load the MNIST dataset.
    data_path = os.path.join(data_dir, 'mnist.pkl.gz')
    print 'Loading the MNIST dataset from %s' %data_path
    f = gzip.open(data_path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    import pdb; pdb.set_trace()
    build_features_matrix(train_set[1], 'train_set')
    import pdb; pdb.set_trace()

if __name__=='__main__':
    sys.exit(main())