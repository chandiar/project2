# -*- coding: utf-8 -*-

import os
import shutil
import sys
import numpy

from ml_code import util

def main():
    data_dir = util.get_dataset_base_path()
    path = os.path.join(data_dir, 'mnist_images')
    dirs = [os.path.join(path, name) for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    image_count = {}
    for idx, dirname in enumerate(dirs):
        target = int(dirname.split('target_')[-1][0])
        files = [os.path.join(dirname, f) for f in os.listdir(dirname) if os.path.isfile(os.path.join(dirname, f)) and f.endswith('.png')]
        for idx2, filename in enumerate(files):
            sys.stdout.write('\r%2d%%' % int(idx2 / float(len(files)) * 100 + 0.5))
            sys.stdout.flush()
            image_count.setdefault(target, 0)
            image_count[target] += 1
    print image_count
    import pdb; pdb.set_trace()
if __name__=='__main__':
    sys.exit(main())