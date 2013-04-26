# -*- coding: utf-8 -*-

import os
import sys
import numpy
import scipy
import sklearn
from sklearn.datasets import fetch_mldata

from ml_code import util

def main():
    # Load the MNIST dataset.
    data_dir = util.get_dataset_base_path()
    mnist = fetch_mldata('MNIST original',
        data_home=data_dir)
    data = mnist.data
    targets = mnist.target
    import pdb; pdb.set_trace()
    for idx, sample in enumerate(data):
        sys.stdout.write('\r%2d%%' % int(idx / float(len(data)) * 100 + 0.5))
        sys.stdout.flush()
        sample = sample.reshape((28,28))
        target = int(targets[idx])
        path = os.path.join(data_dir, 'mnist_images', )
        filename = 'image_idx_%s_number_%s.png'%(idx, target)
        path = os.path.join(path, filename)
        scipy.misc.imsave(path, sample)
    
if __name__=='__main__':
    sys.exit(main())