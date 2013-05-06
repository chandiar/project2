# -*- coding: utf-8 -*-

import os
import sys
import numpy
import scipy
import scipy.io

import project2
from project2.ml_code import util

data_dir = util.get_dataset_base_path()


def convert_mat_to_numpy(which_data):
    print 'Converting images from %s' %which_data
    dir_path = os.path.join(data_dir, 'hog_features', '%s'%which_data)
    files = os.listdir(dir_path)
    for idx, file in enumerate(files):
        sys.stdout.write('\r%2d%%' % int(idx / float(len(files)) * 100 + 0.5))
        sys.stdout.flush()
        hog = scipy.io.loadmat(os.path.join(dir_path, file))['hog']
        basename = os.path.splitext(file)[0]
        out_dir = os.path.join(data_dir, 'hog_features_with_numpy', '%s'%which_data)
        if not os.path.exists(out_dir):
            print 'creating directory %s' %out_dir
            os.makedirs(out_dir)
        filename = '%s.npy'%basename
        out_path = os.path.join(out_dir, filename)
        if not os.path.exists(out_path):
            numpy.save(out_path, hog)


def main():
    import pdb; pdb.set_trace()
    convert_mat_to_numpy('train_set')
    import pdb; pdb.set_trace()
    for data in ['train_set', 'valid_set', 'test_set']:
        convert_mat_to_numpy(which_data)
        import pdb; pdb.set_trace()

if __name__=='__main__':
    sys.exit(main())