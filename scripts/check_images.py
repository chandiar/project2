# -*- coding: utf-8 -*-

import cPickle
import gzip
import os
import sys
import numpy
import scipy
import sklearn
from sklearn.datasets import fetch_mldata

import project2
from project2.ml_code import util

data_dir = util.get_dataset_base_path()


def check_images(data):
    dataset = data[0][0]
    label = data[0][1]
    dataset_name = data[1]
    print 'Checking images from %s' %dataset_name
    print '%s shape is %s' %(dataset_name, dataset.shape)
    print 'Label shape is %s' %label.shape
    dir_path = os.path.join(data_dir, 'mnist_images', '%s'%dataset_name)
    filenames = os.listdir(dir_path)
    n_valid_files = 0
    n_invalid_files = 0
    for i, filename in enumerate(filenames):
        if not filename.endswith('.png'):
            n_invalid_files += 1
            continue
        sys.stdout.write('\r%2d%%' % int(i / float(len(filenames)) * 100 + 0.5))
        sys.stdout.flush()
        idx = int(filename.split('idx_')[1].split('_')[0])
        target = int(filename.split('number_')[1].split('.')[0])
        if int(label[idx]) != target:
            import pdb; pdb.set_trace()
        n_valid_files += 1
    print 'Number of valid files: ', n_valid_files
    print 'Number of invalid files: ', n_invalid_files


def main():
    # Load the MNIST dataset.
    data_path = os.path.join(data_dir, 'mnist.pkl.gz')
    print 'Loading the MNIST dataset from %s' %data_path
    f = gzip.open(data_path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    for data in [(train_set, 'train_set'), (valid_set, 'valid_set'), (test_set, 'test_set')]:
        check_images(data)
        import pdb; pdb.set_trace()

if __name__=='__main__':
    sys.exit(main())