# -*- coding: utf-8 -*-

import sys
import sklearn
from sklearn.datasets import fetch_mldata

import util

def main():
    # Load the MNIST dataset.
    data_dir = util.get_dataset_base_path()
    mnist = fetch_mldata('MNIST original',
        data_home=data_dir)


if __name__=='__main__':
    sys.exit(main())