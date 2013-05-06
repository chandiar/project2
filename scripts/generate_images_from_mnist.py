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

def generate_images(data):
    dataset = data[0][0]
    label = data[0][1]
    dataset_name = data[1]
    print 'Generating images from %s' %dataset_name
    print '%s shape is %s' %(dataset_name, dataset.shape)
    print 'Label shape is %s' %label.shape
    for idx, sample in enumerate(dataset):
        sys.stdout.write('\r%2d%%' % int(idx / float(len(dataset)) * 100 + 0.5))
        sys.stdout.flush()
        sample = sample.reshape((28,28))
        target = int(label[idx])
        out_dir = os.path.join(data_dir, 'mnist_images', '%s'%dataset_name)
        if not os.path.exists(out_dir):
            print 'creating directory %s' %out_dir
            os.makedirs(out_dir)
        filename = 'image_idx_%s_number_%s.png'%(idx, target)
        out_path = os.path.join(out_dir, filename)
        if not os.path.exists(out_path):
            scipy.misc.imsave(out_path, sample)

def main():
    # Load the MNIST dataset.
    data_path = os.path.join(data_dir, 'mnist.pkl.gz')
    print 'Loading the MNIST dataset from %s' %data_path
    f = gzip.open(data_path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    import pdb; pdb.set_trace()
    generate_images((valid_set, 'valid_set'))
    import pdb; pdb.set_trace()
    for data in [(train_set, 'train_set'), (valid_set, 'valid_set'), (test_set, 'test_set')]:
        generate_images(data)
        import pdb; pdb.set_trace()

if __name__=='__main__':
    sys.exit(main())
